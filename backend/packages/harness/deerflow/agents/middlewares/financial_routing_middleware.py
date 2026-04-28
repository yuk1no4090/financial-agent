"""Middleware that forces Financial Agent requests through the FinMA expert tool first."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

_FINANCIAL_AGENT_MODEL_NAMES = {"financial-agent"}
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
_UPPER_TICKER_RE = re.compile(r"\$?([A-Z]{1,5})(?:\b|\))")


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
    return "sentiment"


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


def _should_force_route(request: ModelRequest) -> tuple[bool, str, str, str]:
    runtime_context = request.runtime.context or {}
    model_name = runtime_context.get("model_name")
    if model_name not in _FINANCIAL_AGENT_MODEL_NAMES:
        return False, "", "", ""

    latest = _latest_user_message(request.messages)
    if latest is None:
        return False, "", "", ""

    latest_idx, latest_msg = latest
    if _already_routed_since(request.messages, latest_idx):
        return False, "", "", ""

    text = _message_to_text(latest_msg.content).strip()
    if not text or not _is_finance_query(text):
        return False, "", "", ""

    return True, text, _choose_task(text), _extract_ticker(text)


class FinancialRoutingMiddleware(AgentMiddleware[AgentState]):
    """Force Financial Agent conversations through the FinMA tool before synthesis."""

    def _build_forced_tool_call(self, request: ModelRequest) -> AIMessage | None:
        should_route, text, task, ticker = _should_force_route(request)
        if not should_route:
            return None

        tool_call_id = f"financial-analysis-{uuid.uuid4().hex[:12]}"
        logger.info(
            "FinancialRoutingMiddleware forcing financial_analysis tool call: task=%s ticker=%s",
            task,
            ticker or "-",
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
        forced = self._build_forced_tool_call(request)
        if forced is not None:
            return forced
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        forced = self._build_forced_tool_call(request)
        if forced is not None:
            return forced
        return await handler(request)
