"""Middleware that forces Financial Agent requests through the FinMA expert tool first."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from deerflow.agents.skills import ReportSkillInput, ResearchReportSkill
from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

_FINANCIAL_AGENT_MODEL_NAMES = {"financial-agent", "agent-router"}
_PURE_MODEL_NAMES = {"glm"}
_ROUTE_FINANCIAL_FINMA = "financial_finma"
_ROUTE_FINANCIAL_GLM = "financial_glm"
_ROUTE_GENERAL_GLM = "general_glm"
_ROUTE_CONTEXT_MEMORY_GLM = "context_memory_glm"
_ROUTE_REPORT_SKILL_GLM = "report_skill_glm"
_REPORT_SKILL_NAME = "research-report-skill"
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
_DIRECT_QUERY_PREFIXES = {
    "告诉我",
    "请问",
    "列出",
    "介绍",
    "说明",
    "解释",
    "比较",
    "如何",
    "为什么",
    "哪些",
    "什么",
    "谁是",
    "tell me",
    "list",
    "show me",
    "what",
    "which",
    "who",
    "how",
    "why",
    "compare",
    "explain",
}
_ANALYSIS_PREFIXES = {
    "分析",
    "解读",
    "判断",
    "评估",
    "评价",
    "analyze",
    "assess",
    "evaluate",
    "review",
}
_FACT_OR_RANKING_TERMS = {
    "市值前",
    "前五",
    "前十",
    "排名",
    "排行",
    "top 5",
    "top5",
    "top 10",
    "top10",
    "largest companies",
    "biggest companies",
    "market cap",
    "market capitalization",
    "哪些公司",
    "哪几家公司",
    "公司是哪些",
    "公司有哪些",
    "最新",
    "当前",
    "目前",
    "now",
    "current",
    "latest",
}
_FINMA_SNIPPET_CUES = {
    "增长",
    "下滑",
    "上涨",
    "下跌",
    "超预期",
    "低于预期",
    "财报",
    "业绩",
    "利润",
    "收入",
    "指引",
    "亏损",
    "扭亏",
    "上调",
    "下调",
    "宣布",
    "显示",
    "预计",
    "收购",
    "裁员",
    "制裁",
    "禁运",
    "停产",
    "遇袭",
    "beat",
    "miss",
    "growth",
    "grew",
    "rose",
    "fell",
    "revenue",
    "earnings",
    "guidance",
    "margin",
    "profit",
    "loss",
    "decline",
    "surge",
    "drop",
    "announced",
    "reported",
    "forecast",
    "acquisition",
    "sanction",
    "embargo",
    "attack",
}
_REPORT_REQUEST_TERMS = {
    "研究报告",
    "分析报告",
    "写一份报告",
    "写一个报告",
    "生成报告",
    "生成一个报告",
    "写个报告",
    "做一份报告",
    "整理成报告",
    "整理成一份报告",
    "输出一份报告",
    "深度报告",
    "研报",
    "investment memo",
    "memo",
    "briefing note",
    "report",
    "research report",
    "deep dive",
    "white paper",
}
_REPORT_EXPLICIT_ACTION_TERMS = {
    "为我生成",
    "给我生成",
    "帮我生成",
    "请生成",
    "生成一份",
    "生成一个",
    "写一份",
    "写一个",
    "写个",
    "做一份",
    "做一个",
    "整理成报告",
    "整理成一份报告",
    "输出一份",
    "起草一份",
    "prepare a",
    "write a",
    "generate a",
    "create a",
    "draft a",
    "turn this into a report",
}
_REPORT_REFERENCE_TERMS = {
    "报告中的",
    "报告中",
    "报告里",
    "报告里面",
    "这份报告",
    "这个报告",
    "上面的报告",
    "前面的报告",
    "你生成的报告",
    "你刚刚生成的报告",
    "你刚才生成的报告",
    "你写的报告",
    "生成的报告",
    "写的报告",
    "做的报告",
    "the report",
    "your report",
    "in the report",
    "from the report",
}
_FOLLOW_UP_TERMS = {
    "继续",
    "展开",
    "详细说说",
    "具体说说",
    "具体讲讲",
    "那",
    "那么",
    "这个呢",
    "这个怎么看",
    "这意味着什么",
    "上面这个",
    "刚才这个",
    "刚刚的问题",
    "刚才的问题",
    "上面的问题",
    "前面的问题",
    "上一个问题",
    "基于上面",
    "基于上一个问题",
    "基于刚刚的问题",
    "基于刚才的问题",
    "结合上文",
    "针对刚刚的问题",
    "针对刚才的问题",
    "围绕刚刚的问题",
    "上面提到的",
    "刚刚提到的",
    "刚才提到的",
    "前面提到的",
    "你上面提到的",
    "你刚刚提到的",
    "你刚才提到的",
    "你前面提到的",
    "你上面说的",
    "你刚刚说的",
    "你刚才说的",
    "报告中的",
    "报告里",
    "这份报告",
    "这个报告",
    "你生成的报告",
    "它呢",
    "它们呢",
    "第二家公司",
    "继续往下",
    "what about",
    "and what about",
    "then what",
    "go deeper",
    "expand on that",
    "continue",
    "based on that",
}
_EXPLICIT_CONTEXT_REFERENCE_TERMS = {
    "刚刚的问题",
    "刚才的问题",
    "上面的问题",
    "前面的问题",
    "上一个问题",
    "基于上面",
    "基于上一个问题",
    "基于刚刚的问题",
    "基于刚才的问题",
    "结合上文",
    "针对刚刚的问题",
    "针对刚才的问题",
    "围绕刚刚的问题",
    "上面提到的",
    "刚刚提到的",
    "刚才提到的",
    "前面提到的",
    "你上面提到的",
    "你刚刚提到的",
    "你刚才提到的",
    "你前面提到的",
    "你上面说的",
    "你刚刚说的",
    "你刚才说的",
    "报告中的",
    "报告中",
    "报告里",
    "报告里面",
    "这份报告",
    "这个报告",
    "你生成的报告",
}
_BRIEF_REPORT_TERMS = {
    "简单",
    "简短",
    "简要",
    "简洁",
    "简单一点",
    "简短一点",
    "简要一点",
    "简单些",
    "短一点",
    "brief",
    "short",
    "concise",
    "summary",
}
_REPORT_REQUEST_RES = (
    re.compile(r"(生成|写|做|整理|输出|起草|准备).{0,8}(研究报告|分析报告|报告|研报)"),
    re.compile(r"(write|generate|create|draft|prepare|turn).{0,16}(research report|analysis report|report|memo|briefing|white paper|deep dive)", re.IGNORECASE),
)
_CONTEXT_REFERENCE_RES = (
    re.compile(r"(你|上面|刚刚|刚才|前面).{0,8}(提到|说到|说过|提及|生成|写过|列过)"),
    re.compile(r"(报告|回答|内容).{0,4}(中|里|里面)"),
    re.compile(r"(上面|刚刚|刚才|前面).{0,8}(问题|内容|回答|报告)"),
)
_UPPER_TICKER_RE = re.compile(r"\$?([A-Z]{1,5})(?:\b|\))")
_FINANCIAL_AGENT_SYNTHESIS_GUIDE = (
    "<system_reminder>\n"
    "The financial_analysis tool has already finished.\n"
    "Now answer the user directly in the user's language.\n"
    "- Do not mention tool names, JSON, model_strategy, v3, base model, or internal routing.\n"
    "- If both a sentiment model and a broader base model were used, merge them into one natural financial explanation.\n"
    "- Treat the tool synthesis as the source of truth. Do not ignore it and do not simply re-answer from the raw user sentence.\n"
    "- For short headline-like inputs, keep the answer compact and grounded in what this specific text implies, instead of drifting into generic market boilerplate.\n"
    "- Prefer natural prose over rigid headings. Start with the core judgment, then explain the driver, likely implication, and what to watch next.\n"
    "- Only mention data limitations if the user explicitly asks for live or latest data that is not present in the tool output.\n"
    "</system_reminder>"
)
_FINANCIAL_AGENT_DIRECT_GUIDE = (
    "<system_reminder>\n"
    "Financial Agent direct mode is active for this turn.\n"
    "- This user message is not a short financial text snippet that should be sent to financial_analysis.\n"
    "- Answer directly in the user's language using general financial reasoning.\n"
    "- Do not mention tool routing, hidden prompts, or internal model selection.\n"
    "- If the user asks for rankings, lists, company overviews, or broad mechanism explanations, answer naturally instead of forcing sentiment-style analysis.\n"
    "</system_reminder>"
)
_FINANCIAL_GLM_ROUTE_GUIDE = (
    "<system_reminder>\n"
    "Router route financial_glm is active for this turn.\n"
    "- This is a financial question, but it is not a strong fit for the short-text financial_analysis tool.\n"
    "- Answer directly in the user's language using normal financial reasoning.\n"
    "- Prefer mechanisms, comparisons, rankings, scenario analysis, and caveats over sentiment labels.\n"
    "- Do not force the answer into a short-snippet sentiment template.\n"
    "</system_reminder>"
)
_GENERAL_GLM_ROUTE_GUIDE = (
    "<system_reminder>\n"
    "Router route general_glm is active for this turn.\n"
    "- This request is not primarily a finance-analysis request.\n"
    "- Answer directly in the user's language.\n"
    "- Do not mention hidden routing or internal tools.\n"
    "</system_reminder>"
)
_MEMORY_GLM_ROUTE_GUIDE = (
    "<system_reminder>\n"
    "Router route context_memory_glm is active for this turn.\n"
    "- This looks like a multi-turn follow-up that depends on earlier context.\n"
    "- Resolve references like 'that', 'it', '刚才', or '上面' against the latest relevant turn before answering.\n"
    "- A future memory retrieval hook may be attached here. For now, rely on visible thread history only.\n"
    "- If the user is asking to expand one subsection from a previous answer or report, continue that topic directly instead of restarting a brand-new full report.\n"
    "- Answer directly in the user's language and do not mention hidden routing.\n"
    "</system_reminder>"
)
_REPORT_SKILL_ROUTE_GUIDE = (
    "<system_reminder>\n"
    "Router route report_skill_glm is active for this turn.\n"
    "- This looks like a report-generation request.\n"
    "- The research report skill route is active. Follow a stable report structure instead of free-form answering.\n"
    "- If the user refers to earlier turns such as '刚刚的问题', '上面', or 'based on the above', resolve that context from visible thread history before drafting.\n"
    "- Include a clear title, executive summary, key drivers, scenarios or risks, and watchpoints when appropriate.\n"
    "- Answer in the user's language and do not mention hidden routing or unavailable skills.\n"
    "</system_reminder>"
)
_FINANCIAL_ANALYSIS_REWRITE_REMINDER = (
    "<system_reminder>\n"
    "上一条回答还是太像内部模板或工具输出。\n"
    "请直接重写给用户：\n"
    "- 不要提工具名、JSON、model_strategy、v3、base model。\n"
    "- 不要使用“结论 / 理由 / 影响”这种固定三段标题。\n"
    "- 写成自然、像分析师会说的话：先给核心判断，再解释驱动因素、潜在市场含义，以及后续还要看的变量。\n"
    "- 保持和用户同语言。\n"
    "</system_reminder>"
)
_PURE_MODEL_CALL_GUIDE = (
    "<system_reminder>\n"
    "Pure GLM mode is active for this turn.\n"
    "- Tools are unavailable, so answer directly from your own general knowledge and reasoning.\n"
    "- Do not mention tool limitations unless the user explicitly asks for live/current/latest data that you cannot verify.\n"
    "- For historical, conceptual, or comparative finance questions, give a normal analytical answer with mechanisms, examples, and caveats.\n"
    "- Never output XML, pseudo function calls, or internal tool syntax.\n"
    "</system_reminder>"
)
_PURE_MODEL_REWRITE_REMINDER = (
    "<system_reminder>\n"
    "上一条回答暴露了内部工具调用标签或伪函数调用格式，例如 <function=...>。\n"
    "请直接重写成正常回答：\n"
    "- 不要输出任何 XML、函数调用、参数标签或内部标记。\n"
    "- 如果问题是历史、概念、机制或经验比较，请直接回答，不要先谈工具限制。\n"
    "- 只有在用户明确要求实时/最新数据且你无法核实时，才简短说明限制，再给出基于通用知识的分析。\n"
    "- 保持和用户同语言。\n"
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
    "regulatory_risk": "监管风险",
    "operational_risk": "经营风险",
    "margin_risk": "利润率压力",
    "demand_risk": "需求压力",
    "liquidity_risk": "流动性风险",
    "confident": "自信",
    "cautiously_optimistic": "谨慎乐观",
    "cautious": "谨慎",
    "balanced": "平衡",
    "guidance_cut": "指引下修",
    "growth_signal": "增长信号",
    "mixed_signal": "多空交织信号",
    "shareholder_return_and_growth": "回购与增长共振",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    use_finma: bool = False
    financial_question: bool = False
    task: str = ""
    ticker: str = ""
    model_strategy: str = "lead_only"
    memory_enabled: bool = False
    skill_enabled: bool = False
    skill_name: str | None = None
    brief_report: bool = False


_PSEUDO_TOOL_RE = re.compile(
    r"<tool_call>[\s\S]*?(?:</tool_call>|$)"
    r"|<function=[\s\S]*?</function>"
    r"|<parameter=[\s\S]*?</parameter>"
    r"|<arg_key>[\s\S]*?</arg_key>"
    r"|<arg_value>[\s\S]*?</arg_value>"
    r"|</?(?:tool_call|function|parameter|arg_key|arg_value)[^>]*>",
    re.IGNORECASE,
)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_JSON_BLOB_RE = re.compile(r"^\s*[\[{]")
_DEBUG_ROUTE_PREFIX = "当前路由："
_DEBUG_MODEL_PREFIX = "当前调用模型："


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


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def _strip_analysis_prefix(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in _ANALYSIS_PREFIXES:
        if lowered.startswith(prefix.lower()):
            return stripped[len(prefix) :].lstrip("：:，, \t")
    return stripped


def _starts_with_direct_query_prefix(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    return any(lowered.startswith(prefix.lower()) or stripped.startswith(prefix) for prefix in _DIRECT_QUERY_PREFIXES)


def _is_fact_or_ranking_query(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered or term in text for term in _FACT_OR_RANKING_TERMS)


def _extract_finma_candidate_text(text: str) -> str:
    candidate = _normalized_text(_strip_analysis_prefix(text))
    if not candidate:
        return ""
    if len(candidate) > 280:
        return ""
    if not _is_finance_query(candidate):
        return ""
    if _starts_with_direct_query_prefix(candidate):
        return ""
    if _is_fact_or_ranking_query(candidate):
        return ""
    lowered = candidate.lower()
    if not any(cue in lowered or cue in candidate for cue in _FINMA_SNIPPET_CUES):
        return ""
    return candidate


def _current_turn_finma_candidate(messages: list[object]) -> str:
    latest = _latest_user_message(messages)
    if latest is None:
        return ""
    return _extract_finma_candidate_text(_message_to_text(latest[1].content))


def _latest_visible_ai_before_index(messages: list[object], stop_idx: int) -> AIMessage | None:
    for msg in reversed(messages[:stop_idx]):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        if _message_to_text(msg.content).strip():
            return msg
    return None


def _is_report_request(text: str) -> bool:
    normalized = _normalized_text(text)
    lowered = normalized.lower()
    explicit_action = any(term in lowered or term in normalized for term in _REPORT_EXPLICIT_ACTION_TERMS)
    report_reference = any(term in lowered or term in normalized for term in _REPORT_REFERENCE_TERMS)
    if report_reference and not explicit_action:
        return False
    if any(term in lowered or term in normalized for term in _REPORT_REQUEST_TERMS):
        return True
    for pattern in _REPORT_REQUEST_RES:
        match = pattern.search(normalized)
        if not match:
            continue
        snippet = match.group(0)
        if any(marker in snippet for marker in ("生成的报告", "写的报告", "做的报告")):
            continue
        return True
    return False


def _has_explicit_context_reference(text: str) -> bool:
    normalized = _normalized_text(text)
    lowered = normalized.lower()
    if any(term in lowered or term in normalized for term in _EXPLICIT_CONTEXT_REFERENCE_TERMS):
        return True
    return any(pattern.search(normalized) for pattern in _CONTEXT_REFERENCE_RES)


def _is_brief_report_request(text: str) -> bool:
    normalized = _normalized_text(text)
    lowered = normalized.lower()
    return any(term in lowered or term in normalized for term in _BRIEF_REPORT_TERMS)


def _is_context_follow_up(messages: list[object], latest_idx: int, text: str) -> bool:
    if latest_idx <= 0:
        return False
    if _latest_visible_ai_before_index(messages, latest_idx) is None:
        return False

    normalized = _normalized_text(text)
    lowered = normalized.lower()
    if any(term in lowered or term in normalized for term in _FOLLOW_UP_TERMS):
        return True
    if _has_explicit_context_reference(normalized):
        return True

    # Very short turns after an existing exchange are often elliptical follow-ups.
    if len(normalized) <= 24 and any(char in normalized for char in {"这", "那", "它", "他"}):
        return True
    return False


def _route_decision(messages: list[object], runtime_context: dict | None = None) -> RouteDecision:
    latest = _latest_user_message(messages)
    if latest is None:
        return RouteDecision(route=_ROUTE_GENERAL_GLM, reason="no_user_message")

    latest_idx, latest_msg = latest
    original_text = _message_to_text(latest_msg.content).strip()
    normalized = _normalized_text(original_text)
    if not normalized:
        return RouteDecision(route=_ROUTE_GENERAL_GLM, reason="empty_user_message")

    has_visible_history = _latest_visible_ai_before_index(messages, latest_idx) is not None
    explicit_context_reference = has_visible_history and _has_explicit_context_reference(normalized)
    context_follow_up = _is_context_follow_up(messages, latest_idx, normalized)
    brief_report = _is_brief_report_request(normalized)

    if _is_report_request(normalized):
        report_uses_context = context_follow_up or explicit_context_reference
        return RouteDecision(
            route=_ROUTE_REPORT_SKILL_GLM,
            reason="report_request_with_context" if report_uses_context else "report_request",
            financial_question=_is_finance_query(normalized),
            memory_enabled=report_uses_context,
            skill_enabled=True,
            skill_name=_REPORT_SKILL_NAME,
            brief_report=brief_report,
        )

    if context_follow_up:
        return RouteDecision(
            route=_ROUTE_CONTEXT_MEMORY_GLM,
            reason="context_follow_up",
            financial_question=_is_finance_query(normalized),
            memory_enabled=True,
        )

    financial_question = _is_finance_query(normalized)
    finma_candidate = _extract_finma_candidate_text(normalized)
    if finma_candidate:
        task = _choose_task(finma_candidate)
        model_strategy = "v3_and_base" if _can_use_v3(finma_candidate, task) else "base_only"
        return RouteDecision(
            route=_ROUTE_FINANCIAL_FINMA,
            reason="short_financial_snippet",
            use_finma=True,
            financial_question=True,
            task=task,
            ticker=_extract_ticker(finma_candidate),
            model_strategy=model_strategy,
        )

    if financial_question:
        return RouteDecision(
            route=_ROUTE_FINANCIAL_GLM,
            reason="financial_direct",
            financial_question=True,
        )

    return RouteDecision(route=_ROUTE_GENERAL_GLM, reason="general_direct")


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

    decision = _route_decision(request.messages, runtime_context)
    if decision.route != _ROUTE_FINANCIAL_FINMA or not decision.use_finma:
        return False, "", "", "", ""

    text = _extract_finma_candidate_text(_message_to_text(latest_msg.content).strip())
    if not text:
        return False, "", "", "", ""

    return True, text, decision.task, decision.ticker, decision.model_strategy


def _is_financial_agent_model(model_name: object) -> bool:
    return isinstance(model_name, str) and model_name in _FINANCIAL_AGENT_MODEL_NAMES


def _is_pure_model(model_name: object) -> bool:
    return isinstance(model_name, str) and model_name in _PURE_MODEL_NAMES


def _rewrite_reminder_count(messages: list[object]) -> int:
    return sum(1 for msg in messages if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "financial_analysis_rewrite")


def _latest_finma_tool_index(messages: list[object]) -> int:
    latest = _latest_user_message(messages)
    if latest is None:
        return -1

    last_index = -1
    for idx, msg in enumerate(messages[latest[0] + 1 :], start=latest[0] + 1):
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
    for key in ("explanation", "rationale", "analysis", "summary", "raw_output"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _result_market_implication(result: dict | None) -> str:
    if not result:
        return ""
    for key in ("market_implication", "trading_takeaway", "implication"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _result_watch_items(result: dict | None) -> list[str]:
    if not result:
        return []
    raw = result.get("watch_items") or result.get("affected_factors")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _base_model_result(payload: dict) -> dict | None:
    results = payload.get("model_results")
    if not isinstance(results, list):
        return None
    for result in results:
        if not isinstance(result, dict):
            continue
        model_used = str(result.get("model_used") or "").lower()
        if "sentiment" not in model_used:
            return result
    return None


def _payload_synthesis(payload: dict) -> dict:
    synthesis = payload.get("synthesis")
    return synthesis if isinstance(synthesis, dict) else {}


def _lead_model_target(logical_model_name: str) -> str:
    if not logical_model_name:
        return ""
    try:
        model_config = get_app_config().get_model_config(logical_model_name)
    except Exception:
        model_config = None
    if model_config is None:
        return logical_model_name
    return str(getattr(model_config, "model", None) or logical_model_name)


def _tool_model_names(payload: dict) -> list[str]:
    names: list[str] = []
    results = payload.get("model_results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            model_used = str(result.get("model_used") or "").strip()
            if model_used and model_used not in names:
                names.append(model_used)
    requested = payload.get("models_requested")
    if isinstance(requested, list):
        for model_name in requested:
            candidate = str(model_name).strip()
            if candidate and candidate not in names:
                names.append(candidate)
    return names


def _build_model_debug_header(runtime_context: dict | None, messages: list[object]) -> str:
    context = runtime_context or {}
    logical_model_name = str(context.get("model_name") or "default")
    route = _route_decision(messages, context)
    route_parts = [f"{_DEBUG_ROUTE_PREFIX}{route.route}"]
    route_parts.append(f"memory={'on' if route.memory_enabled else 'off'}")
    route_parts.append(f"skill={'on' if route.skill_enabled else 'off'}")
    if route.skill_name:
        route_parts.append(f"skill_name={route.skill_name}")
    parts = [f"{_DEBUG_MODEL_PREFIX}入口={logical_model_name}"]

    lead_model = _lead_model_target(logical_model_name)
    if lead_model:
        parts.append(f"lead={lead_model}")

    if _is_financial_agent_model(logical_model_name) or _is_pure_model(logical_model_name):
        payload = _latest_finma_payload(messages) or {}
        tool_models = _tool_model_names(payload)
        if tool_models:
            parts.append(f"financial_tool={'+'.join(tool_models)}")
            strategy = str(payload.get("model_strategy") or "")
            if strategy:
                parts.append(f"strategy={strategy}")
        else:
            parts.append("financial_tool=none")
            parts.append("strategy=lead_only")

    return "\n".join((" | ".join(route_parts), " | ".join(parts)))


def _has_model_debug_prefix(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(_DEBUG_MODEL_PREFIX) or stripped.startswith(_DEBUG_ROUTE_PREFIX)


def _strip_debug_prefix_lines(text: str) -> str:
    if not text:
        return ""
    kept = [line for line in text.splitlines() if not line.lstrip().startswith((_DEBUG_ROUTE_PREFIX, _DEBUG_MODEL_PREFIX))]
    return "\n".join(kept).strip()


def _clip_context_text(text: str, *, max_chars: int = 220) -> str:
    cleaned = _normalized_text(_strip_debug_prefix_lines(text))
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}..."


def _detect_user_language(text: str) -> str:
    return "zh" if _contains_chinese(text) else "en"


def _build_report_conversation_context(messages: list[object], latest_idx: int) -> str:
    entries: list[str] = []
    for msg in messages[:latest_idx]:
        if isinstance(msg, HumanMessage) and not getattr(msg, "name", None):
            content = _clip_context_text(_message_to_text(msg.content))
            if content:
                entries.append(f"[User] {content}")
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = _clip_context_text(_message_to_text(msg.content))
            if content:
                entries.append(f"[Assistant] {content}")
    return "\n".join(entries[-6:])


def _build_report_memory_context(decision: RouteDecision) -> str:
    if not decision.memory_enabled:
        return ""
    try:
        from deerflow.agents.memory import format_memory_for_injection, get_memory_data
        from deerflow.config.memory_config import get_memory_config

        config = get_memory_config()
        if not config.enabled or not config.injection_enabled:
            return ""
        memory_data = get_memory_data()
        return format_memory_for_injection(memory_data, max_tokens=min(config.max_injection_tokens, 600))
    except Exception:
        logger.exception("Failed to build report skill memory context")
        return ""


def _build_report_skill_input(messages: list[object], runtime_context: dict | None, decision: RouteDecision) -> ReportSkillInput:
    latest = _latest_user_message(messages)
    if latest is None:
        raise ValueError("No user message available for report skill")

    latest_idx, latest_msg = latest
    user_query = _message_to_text(latest_msg.content).strip()
    language = _detect_user_language(user_query)
    report_type = "financial_analysis" if decision.financial_question else "general_report"
    constraints = {
        "max_sections": 4 if decision.brief_report else 6,
        "include_risks": True,
        "include_conclusion": True,
        "include_suggestions": report_type == "financial_analysis",
        "avoid_internal_debug_info": True,
    }
    return ReportSkillInput(
        user_query=user_query,
        language=language,
        brief_report=decision.brief_report,
        memory_enabled=decision.memory_enabled,
        conversation_context=_build_report_conversation_context(messages, latest_idx),
        memory_context=_build_report_memory_context(decision),
        report_type=report_type,
        audience="student_project",
        constraints=constraints,
    )


def _run_report_skill_sync(messages: list[object], runtime_context: dict | None, decision: RouteDecision) -> AIMessage:
    logical_model_name = str((runtime_context or {}).get("model_name") or "financial-agent")
    skill = ResearchReportSkill(model_name=logical_model_name)
    result = skill.run_sync(_build_report_skill_input(messages, runtime_context, decision))
    return AIMessage(content=result.markdown)


async def _run_report_skill_async(messages: list[object], runtime_context: dict | None, decision: RouteDecision) -> AIMessage:
    logical_model_name = str((runtime_context or {}).get("model_name") or "financial-agent")
    skill = ResearchReportSkill(model_name=logical_model_name)
    result = await skill.run(_build_report_skill_input(messages, runtime_context, decision))
    return AIMessage(content=result.markdown)


def _label_to_zh(label: str) -> str:
    return _LABEL_ZH.get(label.lower(), label or "待判断")


def _generic_reason_from_label(label: str, source_text: str, *, use_zh: bool) -> str:
    if use_zh:
        if label == "positive":
            return "原文包含超预期、上涨、改善或增长类信号，通常说明市场对公司业绩或预期的反应偏正面。"
        if label == "negative":
            return "原文包含低于预期、下跌、压力、亏损或风险类信号，通常会压制投资者情绪或估值预期。"
        if label == "neutral":
            return "原文没有给出足够明确的利好或利空方向，当前更适合作为中性信息处理。"
        if source_text:
            return "原文信号并不完全单一，需要结合事件背景和基本面进一步确认。"
        return "当前信息不足，需要结合原始材料继续核验。"

    if label == "positive":
        return "The text contains clear improvement, growth, or upside signals, which usually points to a constructive market read."
    if label == "negative":
        return "The text contains downside, pressure, or risk signals, which usually weighs on sentiment or valuation expectations."
    if label == "neutral":
        return "The text is not directional enough on its own, so it reads as broadly neutral for now."
    if source_text:
        return "The signal is not completely one-sided, so it should be read together with context, fundamentals, and positioning."
    return "There is not enough validated detail here yet, so the takeaway should stay provisional."


def _generic_market_implication_from_label(label: str, *, use_zh: bool) -> str:
    if use_zh:
        if label == "positive":
            return "如果市场原本预期不高，这类信息通常会先支撑情绪、盈利预期或估值。"
        if label == "negative":
            return "如果市场此前没有充分计入这层风险，短线往往会先压制情绪、盈利预期或估值。"
        if label == "neutral":
            return "这条信息本身还不够强，市场怎么反应通常还要看后续指引、预期修正和资金仓位。"
        return "这类信号本身带有一定混合性，市场反应往往取决于投资者最终更看重哪一面。"
    if label == "positive":
        return "If expectations were not already high, this kind of update can support sentiment, earnings expectations, or valuation."
    if label == "negative":
        return "If this risk was not already priced in, it can weigh on sentiment, earnings expectations, or valuation."
    if label == "neutral":
        return "On its own, the signal is not strong enough to force a clear market reaction, so follow-up guidance and estimate revisions matter more."
    return "Because the signal is mixed, market reaction usually depends on which side investors end up weighting more heavily."


def _localize_watch_item(item: str, *, use_zh: bool) -> str:
    cleaned = item.strip()
    if not use_zh or not cleaned:
        return cleaned

    lowered = cleaned.lower()
    translations = {
        "whether the signal is already priced in": "这部分利好或利空是否已被提前定价",
        "follow-up guidance": "后续指引",
        "next data point that confirms or weakens the thesis": "下一组验证或证伪这条逻辑的数据",
        "short-term price reaction": "短线价格反应",
        "estimate revisions": "盈利预期修正",
        "management follow-up disclosures": "管理层后续披露",
        "whether the risk is temporary or structural": "这类风险是阶段性的还是结构性的",
        "management response": "管理层应对",
        "impact on guidance or margins": "对指引和利润率的影响",
        "whether tone is matched by data": "表态是否被后续数据验证",
        "analyst estimate revisions": "分析师预期修正",
        "whether the signal persists next quarter": "下一季度这类信号能否延续",
        "margin and cash-flow follow-through": "利润率和现金流能否继续跟上",
        "management commentary": "管理层后续表态",
    }
    return translations.get(lowered, cleaned)


def _is_generic_market_implication(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("if the market was not already") or lowered.startswith("the update is not strongly directional") or lowered.startswith("the update is mixed enough")


def _is_internal_placeholder(text: str) -> bool:
    lowered = text.lower()
    return "label-only classification" in lowered or "use the original text" in lowered or "finma returned" in lowered or "mock fallback" in lowered


def _contains_chinese(text: str) -> bool:
    return bool(_CHINESE_RE.search(text))


def _clean_explanation(text: str, label: str, source_text: str, *, use_zh: bool) -> str:
    cleaned = text.strip()
    if not cleaned or _is_internal_placeholder(cleaned):
        return _generic_reason_from_label(label, source_text, use_zh=use_zh)
    cleaned = re.sub(r"\bFinMA\b", "模型" if use_zh else "the model", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbase model\b", "分析结果" if use_zh else "the broader analysis", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bv3\b", "情绪判断" if use_zh else "the short-text sentiment signal", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_pseudo_tool_markup(text: str) -> str:
    return _PSEUDO_TOOL_RE.sub("", text).strip()


def _contains_pseudo_tool_markup(text: str) -> bool:
    return bool(_PSEUDO_TOOL_RE.search(text))


def _pure_model_rewrite_count(messages: list[object]) -> int:
    return sum(1 for msg in messages if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "pure_model_rewrite")


def _has_ai_after_index(messages: list[object], start_idx: int) -> bool:
    return any(isinstance(msg, AIMessage) for msg in messages[start_idx + 1 :])


def _latest_ai_after_index(messages: list[object], start_idx: int) -> AIMessage | None:
    last_ai: AIMessage | None = None
    for msg in messages[start_idx + 1 :]:
        if isinstance(msg, AIMessage):
            last_ai = msg
    return last_ai


def _needs_finma_synthesis(messages: list[object]) -> bool:
    last_finma_index = _latest_finma_tool_index(messages)
    if last_finma_index == -1:
        return False
    return not _has_ai_after_index(messages, last_finma_index)


def _with_instruction(request: ModelRequest, instruction: HumanMessage) -> ModelRequest:
    return request.override(messages=[*request.messages, instruction])


def _route_instruction(decision: RouteDecision) -> HumanMessage:
    if decision.route == _ROUTE_FINANCIAL_GLM:
        content = _FINANCIAL_GLM_ROUTE_GUIDE
        name = "router_financial_glm_guide"
    elif decision.route == _ROUTE_CONTEXT_MEMORY_GLM:
        content = _MEMORY_GLM_ROUTE_GUIDE
        name = "router_context_memory_glm_guide"
    elif decision.route == _ROUTE_REPORT_SKILL_GLM:
        content = _REPORT_SKILL_ROUTE_GUIDE
        if decision.memory_enabled:
            content += "\n<system_reminder>\nThis report request explicitly depends on earlier thread context.\n- Reuse the immediately relevant earlier answer as grounding material before drafting.\n</system_reminder>"
        if decision.brief_report:
            content += (
                "\n"
                "<system_reminder>\n"
                "The user asked for a simple or brief report.\n"
                "- Keep it short: one title and 3 to 4 concise sections.\n"
                "- Prefer roughly 180 to 320 Chinese characters unless the topic clearly requires a little more.\n"
                "- Avoid turning this into a long deep-dive.\n"
                "</system_reminder>"
            )
        name = "router_report_skill_glm_guide"
    else:
        content = _GENERAL_GLM_ROUTE_GUIDE
        name = "router_general_glm_guide"
    return HumanMessage(name=name, content=content)


def _augment_request_for_pure_model(request: ModelRequest) -> ModelRequest:
    return _with_instruction(
        request,
        HumanMessage(name="pure_glm_mode_guide", content=_PURE_MODEL_CALL_GUIDE),
    )


def _augment_request_for_financial_agent_direct(request: ModelRequest) -> ModelRequest:
    return _with_instruction(
        request,
        HumanMessage(name="financial_agent_direct_mode_guide", content=_FINANCIAL_AGENT_DIRECT_GUIDE),
    )


def _augment_request_for_router_direct(request: ModelRequest, decision: RouteDecision) -> ModelRequest:
    return _with_instruction(request, _route_instruction(decision))


def _build_finma_synthesis_instruction(messages: list[object]) -> HumanMessage:
    payload = _latest_finma_payload(messages) or {}
    strategy = str(payload.get("model_strategy") or "base_only")
    task = str(payload.get("task") or "general_financial_analysis")
    synthesis = _payload_synthesis(payload)
    label = str(synthesis.get("label") or "")
    agreement = str(synthesis.get("agreement") or "")

    hints: list[str] = []
    if strategy == "v3_and_base":
        hints.append("Both a short-text sentiment specialist and a broader financial base model were consulted.")
    else:
        hints.append("Only the broader financial base model was consulted.")
    if label:
        hints.append(f"Current overall direction: {label}.")
    if agreement and agreement not in {"", "single_model"}:
        hints.append(f"Cross-model agreement: {agreement}.")
    if task:
        hints.append(f"Task type: {task}.")

    extra = " ".join(hints)
    grounded_watch_items = synthesis.get("watch_items")
    grounded_watch_text = ""
    if isinstance(grounded_watch_items, list):
        grounded_watch_text = ", ".join(str(item).strip() for item in grounded_watch_items if str(item).strip())
    grounded_fields = [
        f"summary={str(synthesis.get('summary') or '').strip()}",
        f"explanation={str(synthesis.get('explanation') or '').strip()}",
        f"market_implication={str(synthesis.get('market_implication') or '').strip()}",
        f"watch_items={grounded_watch_text}",
    ]
    return HumanMessage(
        name="financial_analysis_synthesis_guide",
        content=(f"{_FINANCIAL_AGENT_SYNTHESIS_GUIDE}\nAdditional context: {extra}\nGrounding facts from the tool synthesis:\n" + "\n".join(f"- {field}" for field in grounded_fields if not field.endswith("="))),
    )


def _augment_request_for_finma_synthesis(request: ModelRequest) -> ModelRequest:
    if not _needs_finma_synthesis(request.messages):
        return request
    return _with_instruction(request, _build_finma_synthesis_instruction(request.messages))


def _build_direct_financial_answer(messages: list[object]) -> AIMessage | None:
    payload = _latest_finma_payload(messages)
    if not payload:
        return None

    latest = _latest_user_message(messages)
    source_text = _message_to_text(latest[1].content).strip() if latest else ""
    use_zh = _contains_chinese(source_text)

    synthesis = _payload_synthesis(payload)
    strategy = str(payload.get("model_strategy") or synthesis.get("model_strategy") or "")
    task = str(payload.get("task") or synthesis.get("task") or "")
    v3_result = _find_model_result(payload, "sentiment")
    base_result = _base_model_result(payload)
    primary_result = base_result or _first_model_result(payload)

    primary_label = (str(synthesis.get("label") or "") or _result_label(base_result) or _result_label(v3_result) or _result_label(primary_result)).lower()
    explanation = _clean_explanation(
        str(synthesis.get("explanation") or "") or _result_explanation(primary_result),
        primary_label,
        source_text,
        use_zh=use_zh,
    )
    if use_zh and explanation and not _contains_chinese(explanation):
        explanation = _generic_reason_from_label(primary_label, source_text, use_zh=True)
    market_implication = _clean_explanation(
        str(synthesis.get("market_implication") or "") or _result_market_implication(primary_result),
        primary_label,
        source_text,
        use_zh=use_zh,
    )
    if not market_implication or (use_zh and (not _contains_chinese(market_implication) or _is_generic_market_implication(market_implication))):
        market_implication = _generic_market_implication_from_label(primary_label, use_zh=use_zh)

    watch_items = [_localize_watch_item(str(item).strip(), use_zh=use_zh) for item in (synthesis.get("watch_items") if isinstance(synthesis.get("watch_items"), list) else _result_watch_items(primary_result)) if str(item).strip()]
    agreement = str(synthesis.get("agreement") or "")

    if use_zh:
        label_zh = _label_to_zh(primary_label)
        opener = f"这段信息整体偏{label_zh}。"
        if (
            source_text
            and len(source_text) <= 96
            and task
            in {
                "sentiment",
                "financial_signal_extraction",
                "event_impact",
                "general_financial_analysis",
            }
        ):
            opener = f"如果只基于“{_short_source_excerpt(source_text, use_zh=True)}”这条信息来看，整体偏{label_zh}。"
        if task == "risk_classification" and primary_label:
            opener = f"这段信息里更值得关注的是{label_zh}这一层风险。"
        elif task == "management_tone" and primary_label:
            opener = f"从管理层表述看，整体语气偏{label_zh}。"

        details: list[str] = []
        if strategy == "v3_and_base" and agreement in {"aligned", "mostly_aligned"}:
            details.append("短线情绪信号和更宽泛的基本面解读方向基本一致。")
        elif strategy == "v3_and_base" and agreement == "diverged":
            details.append("短线情绪和更宽泛的基本面解读并不完全一致，因此更适合把它理解成“有方向，但还需要确认强度”的信号。")
        if explanation:
            details.append(explanation)
        if market_implication:
            details.append(market_implication)
        if watch_items:
            details.append(f"后续更值得继续跟踪的是{'、'.join(watch_items[:3])}。")

        content = "\n\n".join(part for part in [opener, " ".join(details).strip()] if part.strip())
    else:
        opener = f"The overall read is {primary_label or 'mixed'}."
        if (
            source_text
            and len(source_text) <= 160
            and task
            in {
                "sentiment",
                "financial_signal_extraction",
                "event_impact",
                "general_financial_analysis",
            }
        ):
            opener = f'Based on "{_short_source_excerpt(source_text, use_zh=False)}" alone, the read is {primary_label or "mixed"}.'
        if task == "risk_classification" and primary_label:
            opener = f"The main takeaway here is {primary_label.replace('_', ' ')} risk."
        elif task == "management_tone" and primary_label:
            opener = f"Management tone comes across as {primary_label.replace('_', ' ')}."

        details = []
        if strategy == "v3_and_base" and agreement in {"aligned", "mostly_aligned"}:
            details.append("The short-text sentiment signal and the broader base-model read point in roughly the same direction.")
        elif strategy == "v3_and_base" and agreement == "diverged":
            details.append("The short-text sentiment signal and the broader base-model read are not perfectly aligned, so conviction should stay measured.")
        if explanation:
            details.append(explanation)
        if market_implication:
            details.append(market_implication)
        if watch_items:
            details.append(f"What matters next is {', '.join(watch_items[:3])}.")

        content = "\n\n".join(part for part in [opener, " ".join(details).strip()] if part.strip())

    return AIMessage(content=content.strip())


def _short_source_excerpt(text: str, *, use_zh: bool) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().strip("。.;；")
    if not cleaned:
        return ""
    limit = 42 if use_zh else 88
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}..."


def _is_short_financial_snippet(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return False
    if "\n" in text and len(cleaned) > 240:
        return False
    return len(cleaned) <= 220


def _should_prefer_direct_financial_answer(messages: list[object]) -> bool:
    payload = _latest_finma_payload(messages)
    if not payload:
        return False

    latest = _latest_user_message(messages)
    source_text = _message_to_text(latest[1].content).strip() if latest else ""
    if not _is_short_financial_snippet(source_text):
        return False

    synthesis = _payload_synthesis(payload)
    strategy = str(payload.get("model_strategy") or synthesis.get("model_strategy") or "")
    task = str(payload.get("task") or synthesis.get("task") or "")

    if strategy == "v3_and_base":
        return True
    return task in {
        "sentiment",
        "risk_classification",
        "management_tone",
        "financial_signal_extraction",
    }


def _replace_ai_message_content(message: AIMessage, content: str) -> AIMessage:
    update = {"content": content, "tool_calls": []}
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs.pop("tool_calls", None)
    additional_kwargs.pop("function_call", None)
    update["additional_kwargs"] = additional_kwargs
    return message.model_copy(update=update)


def _latest_visible_ai_message(messages: list[object]) -> AIMessage | None:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        if _message_to_text(msg.content).strip():
            return msg
    return None


def _with_debug_prefix(message: AIMessage, runtime_context: dict | None, messages: list[object]) -> AIMessage | None:
    content = _message_to_text(message.content).strip()
    if not content or _has_model_debug_prefix(content):
        return None
    header = _build_model_debug_header(runtime_context, messages)
    return _replace_ai_message_content(message, f"{header}\n\n{content}")


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
        content = "我现在没法替你核验实时数据，但可以先基于通用金融知识给你一个方向性判断。如果你需要精确到当前价格、排名或最新事件，请再配合实时行情或改用 Financial Agent 模式。"
    elif cleaned and len(cleaned) >= 12 and "<" not in cleaned and ">" not in cleaned:
        content = cleaned
    else:
        content = "我先按纯 GLM 模式直接回答：如果你希望我结合实时数据、新闻或专门的金融专家模型，再切换到 Financial Agent 会更合适。"

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


def _looks_like_json_blob(text: str) -> bool:
    return bool(_JSON_BLOB_RE.match(text.strip()))


def _has_rigid_financial_headings(text: str) -> bool:
    lowered = text.lower()
    return ("结论：" in text and "理由：" in text) or ("结论：" in text and "影响：" in text) or ("conclusion:" in lowered and "reason:" in lowered) or ("conclusion:" in lowered and "impact:" in lowered)


def _mentions_internal_finance_markup(text: str) -> bool:
    lowered = text.lower()
    markers = {
        "financial_analysis",
        "model_strategy",
        "model_results",
        "v3_and_base",
        "base_only",
        "使用 “financial_analysis” 工具",
        '使用 "financial_analysis" 工具',
    }
    return any(marker in lowered or marker in text for marker in markers)


def _needs_financial_answer_rewrite(messages: list[object]) -> bool:
    last_finma_index = _latest_finma_tool_index(messages)
    if last_finma_index == -1:
        return False

    last_ai = _latest_ai_after_index(messages, last_finma_index)
    if last_ai is None:
        return False

    tool_calls = getattr(last_ai, "tool_calls", None) or []
    if tool_calls:
        return False

    content = _message_to_text(last_ai.content)
    return _is_label_only_response(content) or _looks_like_json_blob(content) or _has_rigid_financial_headings(content) or _mentions_internal_finance_markup(content)


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
            decision = _route_decision(request.messages, request.runtime.context)
            has_current_turn_finma = _latest_finma_tool_index(request.messages) != -1
            if decision.route != _ROUTE_FINANCIAL_FINMA or has_current_turn_finma:
                if request.tools:
                    logger.debug(
                        "Disabled Financial Agent tools for this turn: route=%s has_current_turn_finma=%s",
                        decision.route,
                        has_current_turn_finma,
                    )
                    return request.override(tools=[])
                return request

            active_tools = [t for t in request.tools if getattr(t, "name", None) == "financial_analysis"]
            if len(active_tools) < len(request.tools):
                logger.debug("Filtered Financial Agent tools to financial_analysis only for route=%s", decision.route)
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

        model_name = (request.runtime.context or {}).get("model_name")
        if _is_pure_model(model_name):
            request = _augment_request_for_pure_model(request)
        elif _is_financial_agent_model(model_name):
            decision = _route_decision(request.messages, request.runtime.context)
            if decision.route == _ROUTE_REPORT_SKILL_GLM:
                try:
                    return _run_report_skill_sync(request.messages, request.runtime.context, decision)
                except Exception:
                    logger.exception("Research report skill failed in sync path; falling back to direct model route")
            if decision.route == _ROUTE_FINANCIAL_FINMA and _needs_finma_synthesis(request.messages):
                request = _augment_request_for_finma_synthesis(request)
            else:
                request = _augment_request_for_router_direct(request, decision)

        return handler(request)

    @override
    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        model_name = (runtime.context or {}).get("model_name")
        messages = state.get("messages") or []

        if _is_pure_model(model_name):
            if _needs_pure_model_rewrite(messages):
                logger.info("FinancialRoutingMiddleware requesting rewrite for pseudo tool markup in pure GLM mode")
                return {
                    "jump_to": "model",
                    "messages": [
                        HumanMessage(
                            name="pure_model_rewrite",
                            content=_PURE_MODEL_REWRITE_REMINDER,
                        )
                    ],
                }

            sanitized = _build_pure_model_sanitized_update(messages)
            if sanitized is not None:
                logger.info("FinancialRoutingMiddleware sanitized pseudo tool markup from pure GLM response")
                debug_prefixed = _with_debug_prefix(sanitized, runtime.context, messages)
                return {"messages": [debug_prefixed or sanitized]}

            last_ai = _latest_visible_ai_message(messages)
            if last_ai is not None:
                debug_prefixed = _with_debug_prefix(last_ai, runtime.context, messages)
                if debug_prefixed is not None:
                    return {"messages": [debug_prefixed]}
            return None

        if not _is_financial_agent_model(model_name):
            return None

        decision = _route_decision(messages, runtime.context)
        if decision.route != _ROUTE_FINANCIAL_FINMA:
            if _needs_pure_model_rewrite(messages):
                logger.info("FinancialRoutingMiddleware requesting rewrite for router direct route=%s", decision.route)
                return {
                    "jump_to": "model",
                    "messages": [
                        HumanMessage(
                            name="pure_model_rewrite",
                            content=_PURE_MODEL_REWRITE_REMINDER,
                        )
                    ],
                }

            sanitized = _build_pure_model_sanitized_update(messages)
            if sanitized is not None:
                logger.info("FinancialRoutingMiddleware sanitized pseudo tool markup for router direct route=%s", decision.route)
                debug_prefixed = _with_debug_prefix(sanitized, runtime.context, messages)
                return {"messages": [debug_prefixed or sanitized]}

            last_ai = _latest_visible_ai_message(messages)
            if last_ai is not None:
                debug_prefixed = _with_debug_prefix(last_ai, runtime.context, messages)
                if debug_prefixed is not None:
                    return {"messages": [debug_prefixed]}
            return None

        if _needs_financial_answer_rewrite(messages):
            if _rewrite_reminder_count(messages) >= 1:
                direct_answer = _build_direct_financial_answer(messages)
                if direct_answer is not None:
                    logger.info("FinancialRoutingMiddleware falling back to direct FinMA synthesis after rewrite")
                    last_finma_index = _latest_finma_tool_index(messages)
                    last_ai = _latest_ai_after_index(messages, last_finma_index) if last_finma_index != -1 else None
                    if last_ai is not None:
                        replaced = _replace_ai_message_content(last_ai, direct_answer.content)
                        debug_prefixed = _with_debug_prefix(replaced, runtime.context, messages)
                        return {"messages": [debug_prefixed or replaced]}
                    debug_prefixed = _with_debug_prefix(direct_answer, runtime.context, messages)
                    return {"messages": [debug_prefixed or direct_answer]}
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

        if _should_prefer_direct_financial_answer(messages):
            direct_answer = _build_direct_financial_answer(messages)
            if direct_answer is not None:
                logger.info("FinancialRoutingMiddleware using direct FinMA synthesis for short financial snippet")
                last_finma_index = _latest_finma_tool_index(messages)
                last_ai = _latest_ai_after_index(messages, last_finma_index) if last_finma_index != -1 else None
                if last_ai is not None:
                    replaced = _replace_ai_message_content(last_ai, direct_answer.content)
                    debug_prefixed = _with_debug_prefix(replaced, runtime.context, messages)
                    return {"messages": [debug_prefixed or replaced]}
                debug_prefixed = _with_debug_prefix(direct_answer, runtime.context, messages)
                return {"messages": [debug_prefixed or direct_answer]}

        last_ai = _latest_visible_ai_message(messages)
        if last_ai is not None:
            debug_prefixed = _with_debug_prefix(last_ai, runtime.context, messages)
            if debug_prefixed is not None:
                return {"messages": [debug_prefixed]}
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        runtime = request.runtime
        model_name = (runtime.context or {}).get("model_name") if runtime else None
        tool_name = str(request.tool_call.get("name") or "")
        state_messages = (getattr(request, "state", {}) or {}).get("messages") or []

        if _is_pure_model(model_name):
            return ToolMessage(
                content=f"Error: tool calls are disabled for pure model mode '{model_name}'.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "unknown_tool",
                status="error",
            )

        if tool_name == "financial_analysis" and (not _is_financial_agent_model(model_name) or _route_decision(state_messages, runtime.context if runtime else {}).route != _ROUTE_FINANCIAL_FINMA):
            return ToolMessage(
                content="Error: `financial_analysis` is reserved for the financial_finma router route.",
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

        model_name = (request.runtime.context or {}).get("model_name")
        if _is_pure_model(model_name):
            request = _augment_request_for_pure_model(request)
        elif _is_financial_agent_model(model_name):
            decision = _route_decision(request.messages, request.runtime.context)
            if decision.route == _ROUTE_REPORT_SKILL_GLM:
                try:
                    return await _run_report_skill_async(request.messages, request.runtime.context, decision)
                except Exception:
                    logger.exception("Research report skill failed in async path; falling back to direct model route")
            if decision.route == _ROUTE_FINANCIAL_FINMA and _needs_finma_synthesis(request.messages):
                request = _augment_request_for_finma_synthesis(request)
            else:
                request = _augment_request_for_router_direct(request, decision)

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
        state_messages = (getattr(request, "state", {}) or {}).get("messages") or []

        if _is_pure_model(model_name):
            return ToolMessage(
                content=f"Error: tool calls are disabled for pure model mode '{model_name}'.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "unknown_tool",
                status="error",
            )

        if tool_name == "financial_analysis" and (not _is_financial_agent_model(model_name) or _route_decision(state_messages, runtime.context if runtime else {}).route != _ROUTE_FINANCIAL_FINMA):
            return ToolMessage(
                content="Error: `financial_analysis` is reserved for the financial_finma router route.",
                tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
                name=tool_name or "financial_analysis",
                status="error",
            )

        return await handler(request)
