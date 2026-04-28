"""Middleware that forces Financial Agent requests through the FinMA expert tool first."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)

_FINANCIAL_AGENT_MODEL_NAMES = {"financial-agent"}
_PURE_MODEL_NAMES = {"glm"}
_FINANCE_KEYWORDS = {
    "finance",
    "financial",
    "earnings",
    "guidance",
    "revenue",
    "margin",
    "cash flow",
    "sentiment",
    "risk",
    "valuation",
    "equity",
    "stock",
    "market",
    "macro",
    "commodity",
    "commodities",
    "gold",
    "oil",
    "crude",
    "bond",
    "yield",
    "rate",
    "inflation",
    "fed",
    "federal reserve",
    "company",
    "shares",
    "filing",
    "transcript",
    "财报",
    "金融",
    "估值",
    "情绪",
    "风险",
    "同业",
    "市场",
    "宏观",
    "大宗商品",
    "商品",
    "黄金",
    "金价",
    "原油",
    "油价",
    "债券",
    "收益率",
    "利率",
    "降息",
    "加息",
    "通胀",
    "美联储",
    "央行",
    "汇率",
    "美元",
    "新闻",
    "消息",
    "收益",
    "利润",
    "利润率",
    "收入",
    "指引",
    "现金流",
    "股票",
    "股价",
    "公司",
    "行业",
    "研报",
    "公告",
}
_SENTIMENT_TERMS = {"sentiment", "情绪", "正面", "负面", "中性", "看多", "看空"}
_RISK_TERMS = {"risk", "风险", "regulation", "regulatory", "监管", "合规"}
_TONE_TERMS = {"tone", "语气", "management tone", "management", "管理层"}
_SIGNAL_TERMS = {"signal", "信号", "cash flow", "margin", "guidance", "指引", "现金流", "利润率"}
_EVENT_TERMS = {"impact", "影响", "事件", "event", "reaction", "催化"}
_V3_CONTEXT_TERMS = {
    "news",
    "headline",
    "sentiment",
    "earnings",
    "shares",
    "stock",
    "guidance",
    "revenue",
    "margin",
    "新闻",
    "消息",
    "情绪",
    "财报",
    "股价",
    "股票",
    "盘后",
    "上涨",
    "下跌",
    "超预期",
    "低于预期",
}
_UPPER_TICKER_RE = re.compile(r"\$?([A-Z]{1,5})(?:\b|\))")
_FINANCIAL_ANALYSIS_REWRITE_REMINDER = (
    "<system_reminder>\n上一条回答只暴露了原始分类标签，信息量不够。\n请重写成一段金融分析，而不是只给一个词。\n固定包含：结论、理由、短期影响。\n不要只输出 Positive / Negative / Neutral / Mixed。\n</system_reminder>"
)
_PURE_MODEL_REWRITE_REMINDER = (
    "<system_reminder>\n"
    "上一条回答暴露了内部工具调用标签或伪函数调用格式，例如 <function=...>。\n"
    "当前是纯 GLM 模式，工具已禁用。请不要输出任何 XML、函数调用、参数标签或内部标记。\n"
    "如果问题需要实时数据，请明确说明无法直接获取实时数据，并基于已知信息给出有条件的回答。\n"
    "</system_reminder>"
)
_LABEL_ONLY_RESPONSES = {
    "positive",
    "negative",
    "neutral",
    "mixed",
    "bullish",
    "bearish",
    "confident",
    "cautious",
    "balanced",
    "positive.",
    "negative.",
    "neutral.",
    "mixed.",
}

_LABEL_ZH = {
    "positive": "积极",
    "negative": "消极",
    "neutral": "中性",
    "mixed": "多空交织",
    "bullish": "偏多",
    "bearish": "偏空",
}
_PSEUDO_TOOL_RE = re.compile(
    r"<tool_call>[\s\S]*?(?:</tool_call>|$)"
    r"|<function=[\s\S]*?</function>"
    r"|<parameter=[\s\S]*?</parameter>"
    r"|<arg_key>[\s\S]*?</arg_key>"
    r"|<arg_value>[\s\S]*?</arg_value>"
    r"|</?(?:tool_call|function|parameter|arg_key|arg_value)[^>]*>",
    re.IGNORECASE,
)


def _message_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[str] = []
        for item in content:
            if isinstance(item, str):
                blocks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    blocks.append(text)
        return "\n".join(blocks)
    return str(content)


def _latest_user_message(messages: list[object]) -> tuple[int, HumanMessage] | None:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, HumanMessage):
            continue
        if getattr(msg, "name", None):
            continue
        return idx, msg
    return None


def _is_finance_query(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered or keyword in text for keyword in _FINANCE_KEYWORDS)


def _choose_task(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered or term in text for term in _SENTIMENT_TERMS):
        return "sentiment"
    if any(term in lowered or term in text for term in _RISK_TERMS):
        return "risk_classification"
    if any(term in lowered or term in text for term in _TONE_TERMS):
        return "management_tone"
    if any(term in lowered or term in text for term in _SIGNAL_TERMS):
        return "financial_signal_extraction"
    if any(term in lowered or term in text for term in _EVENT_TERMS):
        return "event_impact"
    if len(text) <= 1800 and any(term in lowered or term in text for term in _V3_CONTEXT_TERMS):
        return "sentiment"
    return "general_financial_analysis"


def _can_use_v3(text: str, task: str) -> bool:
    """Return true for short financial-news style inputs suited to the sentiment LoRA."""
    if task != "sentiment":
        return False
    if len(text) > 1800:
        return False
    lowered = text.lower()
    return any(term in lowered or term in text for term in _V3_CONTEXT_TERMS)


def _extract_ticker(text: str) -> str:
    for match in _UPPER_TICKER_RE.finditer(text):
        candidate = match.group(1)
        if candidate in {"I", "A", "AN", "THE", "USD", "CNY"}:
            continue
        return candidate
    return ""


def _already_routed_since(messages: list[object], start_idx: int) -> bool:
    for msg in messages[start_idx + 1 :]:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if any(tc.get("name") == "financial_analysis" for tc in tool_calls):
                return True
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "financial_analysis":
            return True
    return False


def _should_force_route(request: ModelRequest) -> tuple[bool, str, str, str, str]:
    runtime_context = request.runtime.context or {}
    model_name = runtime_context.get("model_name")
    if model_name not in _FINANCIAL_AGENT_MODEL_NAMES:
        return False, "", "", "", ""

    latest = _latest_user_message(request.messages)
    if latest is None:
        return False, "", "", "", ""

    latest_idx, latest_msg = latest
    if _already_routed_since(request.messages, latest_idx):
        return False, "", "", "", ""

    text = _message_to_text(latest_msg.content).strip()
    if not text or not _is_finance_query(text):
        return False, "", "", "", ""

    task = _choose_task(text)
    model_strategy = "v3_and_base" if _can_use_v3(text, task) else "base_only"
    return True, text, task, _extract_ticker(text), model_strategy


def _is_financial_agent_model(model_name: object) -> bool:
    return isinstance(model_name, str) and model_name in _FINANCIAL_AGENT_MODEL_NAMES


def _is_pure_model(model_name: object) -> bool:
    return isinstance(model_name, str) and model_name in _PURE_MODEL_NAMES


def _rewrite_reminder_count(messages: list[object]) -> int:
    return sum(1 for msg in messages if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "financial_analysis_rewrite")


def _latest_finma_tool_index(messages: list[object]) -> int:
    last_index = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "financial_analysis":
            last_index = idx
    return last_index


def _latest_finma_payload(messages: list[object]) -> dict | None:
    last_index = _latest_finma_tool_index(messages)
    if last_index == -1:
        return None

    content = _message_to_text(messages[last_index].content).strip()
    if not content:
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {"provider": "finma_raw", "raw_output": content}
    return payload if isinstance(payload, dict) else {"provider": "finma_raw", "raw_output": content}


def _find_model_result(payload: dict, keyword: str) -> dict | None:
    results = payload.get("model_results")
    if not isinstance(results, list):
        return None
    for result in results:
        if not isinstance(result, dict):
            continue
        model_used = str(result.get("model_used") or "").lower()
        if keyword in model_used:
            return result
    return None


def _first_model_result(payload: dict) -> dict | None:
    results = payload.get("model_results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                return result
    return payload if payload else None


def _result_label(result: dict | None) -> str:
    if not result:
        return ""
    for key in ("label", "impact_direction", "sentiment"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _result_explanation(result: dict | None) -> str:
    if not result:
        return ""
    for key in ("rationale", "analysis", "raw_output", "summary"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _label_to_zh(label: str) -> str:
    return _LABEL_ZH.get(label.lower(), label or "待判断")


def _generic_reason_from_label(label: str, source_text: str) -> str:
    if label == "positive":
        return "原文包含超预期、上涨、改善或增长类信号，通常说明市场对公司业绩或预期的反应偏正面。"
    if label == "negative":
        return "原文包含低于预期、下跌、压力、亏损或风险类信号，通常会压制投资者情绪或估值预期。"
    if label == "neutral":
        return "原文没有给出足够明确的利好或利空方向，当前更适合作为中性信息处理。"
    if source_text:
        return "原文信号并不完全单一，需要结合事件背景和基本面进一步确认。"
    return "当前信息不足，需要结合原始材料继续核验。"


def _is_internal_placeholder(text: str) -> bool:
    lowered = text.lower()
    return "label-only classification" in lowered or "use the original text" in lowered or "finma returned" in lowered or "mock fallback" in lowered


def _clean_explanation(text: str, label: str, source_text: str) -> str:
    cleaned = text.strip()
    if not cleaned or _is_internal_placeholder(cleaned):
        return _generic_reason_from_label(label, source_text)
    cleaned = re.sub(r"\bFinMA\b", "模型", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbase model\b", "分析结果", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bv3\b", "情绪判断", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_pseudo_tool_markup(text: str) -> str:
    return _PSEUDO_TOOL_RE.sub("", text).strip()


def _contains_pseudo_tool_markup(text: str) -> bool:
    return bool(_PSEUDO_TOOL_RE.search(text))


def _pure_model_rewrite_count(messages: list[object]) -> int:
    return sum(1 for msg in messages if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "pure_model_rewrite")


def _build_direct_financial_answer(messages: list[object]) -> AIMessage | None:
    payload = _latest_finma_payload(messages)
    if not payload:
        return None

    latest = _latest_user_message(messages)
    source_text = _message_to_text(latest[1].content).strip() if latest else ""
    strategy = str(payload.get("model_strategy") or "")
    v3_result = _find_model_result(payload, "sentiment")
    base_result = _find_model_result(payload, "7b")
    primary_result = base_result or _first_model_result(payload)
    v3_label = _result_label(v3_result)
    base_label = _result_label(base_result)
    primary_label = base_label or v3_label or _result_label(primary_result)
    primary_explanation = _clean_explanation(_result_explanation(primary_result), primary_label, source_text)

    conclusion_label = _label_to_zh(primary_label)
    evidence_lines: list[str] = []
    if v3_result:
        evidence_lines.append(f"情绪方向偏{_label_to_zh(v3_label)}。")
    if base_result:
        base_text = _clean_explanation(_result_explanation(base_result), base_label, source_text)
        if base_text:
            evidence_lines.append(base_text)
        elif base_label:
            evidence_lines.append(f"综合判断方向偏{_label_to_zh(base_label)}。")

    if not evidence_lines and primary_explanation:
        evidence_lines.append(primary_explanation)

    if strategy == "v3_and_base":
        conclusion = f"结论：这条金融信息整体偏{conclusion_label}。"
    else:
        conclusion = f"结论：当前判断为{conclusion_label}。"

    reason = "理由：" + (" ".join(evidence_lines) if evidence_lines else primary_explanation)
    impact = "影响：如果该信息与上市公司业绩、股价或行业预期直接相关，短期通常会影响投资者情绪和估值预期；后续还需要结合公司基本面、估值水平、市场环境和是否已有价格反应来判断投资意义。"
    if primary_label == "positive":
        impact = "影响：短期可能提振股价和风险偏好，但仍需要确认利好是否已经被市场提前定价。"
    elif primary_label == "negative":
        impact = "影响：短期可能压制股价和投资者情绪，并增加盈利预期或估值下修风险。"
    elif primary_label == "neutral":
        impact = "影响：短期方向性有限，更需要关注后续数据、管理层指引或市场反应是否提供新的边际信息。"

    content = "\n\n".join(part for part in [conclusion, reason, impact] if part)
    return AIMessage(content=content)


def _needs_pure_model_rewrite(messages: list[object]) -> bool:
    if _pure_model_rewrite_count(messages) >= 1:
        return False
    if not messages or not isinstance(messages[-1], AIMessage):
        return False
    return _contains_pseudo_tool_markup(_message_to_text(messages[-1].content))


def _build_pure_model_sanitized_update(messages: list[object]) -> AIMessage | None:
    if not messages or not isinstance(messages[-1], AIMessage):
        return None

    last_msg = messages[-1]
    raw_content = _message_to_text(last_msg.content)
    if not _contains_pseudo_tool_markup(raw_content):
        return None

    cleaned = _strip_pseudo_tool_markup(raw_content)
    latest = _latest_user_message(messages)
    user_text = _message_to_text(latest[1].content).strip() if latest else ""
    user_lowered = user_text.lower()
    needs_realtime = any(term in user_lowered or term in user_text for term in {"latest", "current", "today", "2026", "最新", "当前", "现在", "实时", "市值", "排名"})

    if needs_realtime:
        content = "我当前不能直接获取实时数据。基于已知信息，全球市值前列公司通常包括 Apple、Microsoft、NVIDIA、Alphabet、Amazon 等，但具体排名会随股价和汇率变化，正式展示或投资分析应以实时行情数据为准。"
    elif cleaned and len(cleaned) >= 12 and "<" not in cleaned and ">" not in cleaned:
        content = cleaned
    else:
        content = "当前是纯 GLM 模式，不能调用外部工具获取实时数据。请提供需要分析的数据或允许使用 Financial Agent 进行金融分析。"

    update = {"content": content, "tool_calls": []}
    additional_kwargs = dict(getattr(last_msg, "additional_kwargs", {}) or {})
    additional_kwargs.pop("tool_calls", None)
    additional_kwargs.pop("function_call", None)
    update["additional_kwargs"] = additional_kwargs
    return last_msg.model_copy(update=update)


def _is_label_only_response(content: str) -> bool:
    stripped = content.strip()
    lowered = stripped.lower()
    if lowered in _LABEL_ONLY_RESPONSES:
        return True
    if len(stripped.split()) <= 3 and len(stripped) <= 24 and lowered.replace(" ", "") in {item.replace(" ", "") for item in _LABEL_ONLY_RESPONSES}:
        return True
    return False


def _needs_rewrite_after_finma(messages: list[object]) -> bool:
    last_finma_index = _latest_finma_tool_index(messages)
    if last_finma_index == -1:
        return False

    last_ai: AIMessage | None = None
    for msg in messages[last_finma_index + 1 :]:
        if isinstance(msg, AIMessage):
            last_ai = msg

    if last_ai is None:
        return False

    tool_calls = getattr(last_ai, "tool_calls", None) or []
    if tool_calls:
        return False

    return _is_label_only_response(_message_to_text(last_ai.content))


class FinancialRoutingMiddleware(AgentMiddleware[AgentState]):
    """Force Financial Agent conversations through the FinMA tool before synthesis."""

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        model_name = (request.runtime.context or {}).get("model_name")
        if _is_pure_model(model_name):
            if request.tools:
                logger.debug("Filtered all tools for pure model '%s'", model_name)
                return request.override(tools=[])
            return request

        if _is_financial_agent_model(model_name):
            active_tools = [t for t in request.tools if getattr(t, "name", None) == "financial_analysis"]
            if len(active_tools) < len(request.tools):
                logger.debug("Filtered Financial Agent tools to financial_analysis only")
                return request.override(tools=active_tools)
            return request

        active_tools = [t for t in request.tools if getattr(t, "name", None) != "financial_analysis"]
        if len(active_tools) < len(request.tools):
            logger.debug("Filtered financial_analysis tool for non-Financial-Agent model '%s'", model_name)
            return request.override(tools=active_tools)
        return request

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return None

    def _build_forced_tool_call(self, request: ModelRequest) -> AIMessage | None:
        should_route, text, task, ticker, model_strategy = _should_force_route(request)
        if not should_route:
            return None

        tool_call_id = f"financial-analysis-{uuid.uuid4().hex[:12]}"
        logger.info(
            "FinancialRoutingMiddleware forcing financial_analysis tool call: task=%s ticker=%s strategy=%s",
            task,
            ticker or "-",
            model_strategy,
        )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": "financial_analysis",
                    "args": {
                        "text": text[:5000],
                        "task": task,
                        "ticker": ticker,
                        "model_strategy": model_strategy,
                    },
                    "type": "tool_call",
                }
            ],
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request = self._filter_tools(request)
        forced = self._build_forced_tool_call(request)
        if forced is not None:
            return forced
        direct_answer = _build_direct_financial_answer(request.messages)
        if direct_answer is not None and _is_financial_agent_model((request.runtime.context or {}).get("model_name")):
            logger.info("FinancialRoutingMiddleware returning direct FinMA answer without GLM synthesis")
            return direct_answer
        return handler(request)

    @override
    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        model_name = (runtime.context or {}).get("model_name")
        messages = state.get("messages") or []
        sanitized = _build_pure_model_sanitized_update(messages) if _is_pure_model(model_name) else None
        if sanitized is not None:
            logger.info("FinancialRoutingMiddleware sanitized pseudo tool markup from pure GLM response")
            return {"messages": [sanitized]}

        if not _is_financial_agent_model(model_name):
            return None

        if _rewrite_reminder_count(messages) >= 1:
            return None
        if not _needs_rewrite_after_finma(messages):
            return None

        return {
            "jump_to": "model",
            "messages": [
                HumanMessage(
                    name="financial_analysis_rewrite",
                    content=_FINANCIAL_ANALYSIS_REWRITE_REMINDER,
                )
            ],
        }

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        runtime = request.runtime
        model_name = (runtime.context or {}).get("model_name") if runtime else None
        tool_name = str(request.tool_call.get("name") or "")

        if _is_pure_model(model_name):
            return ToolMessage(
                content=f"Error: tool calls are disabled for pure model mode '{model_name}'.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "unknown_tool",
                status="error",
            )

        if tool_name == "financial_analysis" and not _is_financial_agent_model(model_name):
            return ToolMessage(
                content="Error: `financial_analysis` is reserved for Financial Agent mode.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "financial_analysis",
                status="error",
            )

        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request = self._filter_tools(request)
        forced = self._build_forced_tool_call(request)
        if forced is not None:
            return forced
        direct_answer = _build_direct_financial_answer(request.messages)
        if direct_answer is not None and _is_financial_agent_model((request.runtime.context or {}).get("model_name")):
            logger.info("FinancialRoutingMiddleware returning direct FinMA answer without GLM synthesis")
            return direct_answer
        return await handler(request)

    @override
    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self.after_model(state, runtime)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        runtime = request.runtime
        model_name = (runtime.context or {}).get("model_name") if runtime else None
        tool_name = str(request.tool_call.get("name") or "")

        if _is_pure_model(model_name):
            return ToolMessage(
                content=f"Error: tool calls are disabled for pure model mode '{model_name}'.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "unknown_tool",
                status="error",
            )

        if tool_name == "financial_analysis" and not _is_financial_agent_model(model_name):
            return ToolMessage(
                content="Error: `financial_analysis` is reserved for Financial Agent mode.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "financial_analysis",
                status="error",
            )

        return await handler(request)
