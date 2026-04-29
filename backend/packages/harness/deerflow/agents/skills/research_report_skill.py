from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import replace
from typing import Any

from deerflow.agents.rag import RagSearchRequest, RagService, get_rag_service
from deerflow.models import create_chat_model

from .research_report_prompt import (
    build_planner_prompt,
    build_rewrite_prompt,
    build_writer_prompt,
)
from .research_report_schema import (
    ReportPlan,
    ReportPlanSection,
    ReportReview,
    ReportSkillInput,
    ReportSkillOutput,
)

logger = logging.getLogger(__name__)

_REPORT_REQUEST_SPLIT_RE = re.compile(
    r"[，,。.]?\s*(?:请|帮我|给我|为我)?\s*(?:简单地|简单|简短地|简短|简要地|简要)?\s*(?:生成|写|做|整理|输出|起草).{0,12}(?:研究报告|分析报告|报告|研报)",
    re.IGNORECASE,
)
_FOLLOWUP_TOPIC_PATTERNS = (
    re.compile(r"(?:针对|对于|围绕|基于|就)(?:你)?(?:刚刚|刚才|上面|前面)?(?:提到|说的|提及的)?(?P<topic>.{1,24}?)(?:这个话题|这个部分|这部分|方面|内容|问题)?(?:，|,|。|$)"),
    re.compile(r"(?:你)(?:刚刚|刚才|上面|前面)(?:提到|说的)(?P<topic>.{1,24}?)(?:，|,|。|$)"),
)
_TOPIC_TRAILING_NOISE = (
    "这个话题",
    "这个部分",
    "这部分",
    "这个问题",
    "这个内容",
    "的内容",
    "的话题",
    "方面",
)
_FULL_FINANCIAL_SECTIONS = (
    ("执行摘要", "总结报告的核心判断和主要结论。"),
    ("背景与问题定义", "说明分析对象、背景和本报告的范围。"),
    ("核心分析", "围绕主题展开主要逻辑和关键判断。"),
    ("风险与不确定性", "说明哪些变量可能影响结论。"),
    ("后续关注指标", "列出后续值得跟踪的信号、数据或事件。"),
    ("结论", "给出综合判断和简短建议。"),
)
_FULL_GENERAL_SECTIONS = (
    ("执行摘要", "概括本报告最重要的结论。"),
    ("背景与问题定义", "说明主题背景和讨论范围。"),
    ("核心分析", "分点展开主要分析逻辑。"),
    ("关键注意点", "说明限制、风险或需要补充验证的地方。"),
    ("结论", "总结最终判断。"),
)
_BRIEF_SECTIONS = (
    ("核心结论", "先给出最直接的判断。"),
    ("关键分析", "用最重要的 2-3 点支撑结论。"),
    ("风险点", "说明主要不确定性或限制。"),
    ("简短总结", "做一个简洁收束。"),
)
_INTERNAL_LEAK_MARKERS = (
    "当前路由：",
    "当前调用模型：",
    "system_reminder",
    "router_",
    "skill_name=",
    "financial_tool=",
    "model_strategy",
    "tool call",
    "v3_and_base",
    "base_only",
)
_DEBUG_LINE_PREFIXES = ("当前路由：", "当前调用模型：")
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_INLINE_JSON_RE = re.compile(r"(\{[\s\S]*\})")
_SYSTEM_REMINDER_BLOCK_RE = re.compile(r"<system_reminder>[\s\S]*?</system_reminder>", re.IGNORECASE)
_UPPER_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_CITATION_RE = re.compile(r"\[E\d+\]")


def _contains_chinese(text: str) -> bool:
    return bool(_CHINESE_RE.search(text))


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_debug_prefix_lines(text: str) -> str:
    if not text:
        return ""
    cleaned_lines = [line for line in text.splitlines() if not line.lstrip().startswith(_DEBUG_LINE_PREFIXES)]
    return "\n".join(cleaned_lines).strip()


def _sanitize_internal_text(text: str) -> str:
    cleaned = _SYSTEM_REMINDER_BLOCK_RE.sub("", text or "")
    cleaned = _strip_debug_prefix_lines(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates: list[str] = []
    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        candidates.append(block_match.group(1))
    inline_match = _INLINE_JSON_RE.search(text)
    if inline_match:
        candidates.append(inline_match.group(1))
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _trim_topic_noise(text: str) -> str:
    cleaned = _normalized_text(text).strip("，,。.;；:： ")
    cleaned = re.sub(r"^(?:的|关于|有关)\s*", "", cleaned)
    for suffix in _TOPIC_TRAILING_NOISE:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("，,。.;；:： ")
    return cleaned


def _build_default_title(topic: str, *, use_zh: bool, financial: bool) -> str:
    base = _trim_topic_noise(topic) or ("该主题" if use_zh else "This Topic")
    if use_zh:
        suffix = "分析报告" if financial else "研究报告"
        return f"{base}{suffix}"
    suffix = "Analysis Report" if financial else "Research Report"
    return f"{base} {suffix}"


def _coerce_outline(
    payload: dict[str, Any] | None,
    *,
    topic: str,
    brief_report: bool,
    financial: bool,
    use_zh: bool,
) -> ReportPlan:
    default_sections = _BRIEF_SECTIONS if brief_report else (_FULL_FINANCIAL_SECTIONS if financial else _FULL_GENERAL_SECTIONS)
    fallback = ReportPlan(
        title=_build_default_title(topic, use_zh=use_zh, financial=financial),
        sections=[ReportPlanSection(heading=heading, goal=goal) for heading, goal in default_sections],
    )
    if not payload:
        return fallback

    raw_title = str(payload.get("title") or "").strip()
    raw_sections = payload.get("sections")
    sections: list[ReportPlanSection] = []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading") or "").strip()
            goal = str(item.get("goal") or "").strip()
            if heading:
                sections.append(ReportPlanSection(heading=heading, goal=goal or "围绕该章节主题展开分析。"))

    if not sections:
        sections = fallback.sections

    return ReportPlan(
        title=raw_title or fallback.title,
        sections=sections,
    )


class ResearchReportSkill:
    name = "research-report-skill"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        llm: Any | None = None,
        financial_signal_provider: Any | None = None,
        rag_service: RagService | None = None,
    ) -> None:
        self.model_name = model_name
        self._llm = llm
        self._financial_signal_provider = financial_signal_provider
        self._rag_service = rag_service

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = create_chat_model(name=self.model_name, thinking_enabled=False)
        return self._llm

    def _invoke_model(self, prompt: str, *, run_name: str) -> str:
        response = self._get_llm().invoke(prompt, config={"run_name": run_name})
        return _sanitize_internal_text(str(getattr(response, "content", "") or ""))

    async def _ainvoke_model(self, prompt: str, *, run_name: str) -> str:
        response = await self._get_llm().ainvoke(prompt, config={"run_name": run_name})
        return _sanitize_internal_text(str(getattr(response, "content", "") or ""))

    def _get_rag_service(self) -> RagService:
        if self._rag_service is None:
            self._rag_service = get_rag_service()
        return self._rag_service

    def _extract_topic_from_query(self, user_query: str) -> str:
        normalized = _normalized_text(user_query)
        for pattern in _FOLLOWUP_TOPIC_PATTERNS:
            match = pattern.search(normalized)
            if match:
                topic = _trim_topic_noise(match.group("topic"))
                if topic:
                    return topic

        split_match = _REPORT_REQUEST_SPLIT_RE.search(normalized)
        if split_match and split_match.start() > 0:
            topic = _trim_topic_noise(normalized[: split_match.start()])
            if topic:
                return topic

        return _trim_topic_noise(normalized)

    def extract_topic(self, skill_input: ReportSkillInput) -> str:
        if skill_input.topic:
            return _trim_topic_noise(skill_input.topic)

        topic = self._extract_topic_from_query(skill_input.user_query)
        if topic and topic not in {"请", "帮我", "给我", "为我"}:
            return topic

        if skill_input.conversation_context:
            lines = [line.strip() for line in skill_input.conversation_context.splitlines() if line.strip()]
            for line in reversed(lines):
                stripped = re.sub(r"^\[(?:User|Assistant)\]\s*", "", line, flags=re.IGNORECASE).strip()
                stripped = _trim_topic_noise(stripped)
                if stripped:
                    return stripped[:40]

        return _trim_topic_noise(skill_input.user_query) or ("当前主题" if skill_input.language.startswith("zh") else "Current Topic")

    def build_context(self, skill_input: ReportSkillInput, topic: str) -> dict[str, Any]:
        context_text = _sanitize_internal_text(skill_input.conversation_context)
        memory_text = _sanitize_internal_text(skill_input.memory_context)
        key_entities = _dedupe_preserve_order([topic] + _UPPER_TICKER_RE.findall(context_text) + [part.strip() for part in re.split(r"[、,，/|和与及]", topic) if part.strip()])[:6]

        previous_claims: list[str] = []
        if context_text:
            for raw in context_text.splitlines():
                stripped = raw.strip()
                if not stripped or stripped.startswith("[User]"):
                    continue
                stripped = re.sub(r"^\[Assistant\]\s*", "", stripped).strip()
                if stripped:
                    previous_claims.append(stripped[:160])

        missing_information: list[str] = []
        if not skill_input.retrieved_context:
            if skill_input.rag_enabled:
                missing_information.append("RAG 已启用但当前没有检索到可用证据，报告应明确说明证据不足。")
            else:
                missing_information.append("当前没有额外检索证据，报告应避免编造精确数字。")
        if not memory_text and skill_input.memory_enabled:
            missing_information.append("当前没有显式 memory 摘要，优先依赖当前线程上下文。")

        return {
            "topic": topic,
            "key_entities": key_entities,
            "previous_claims": previous_claims[:5],
            "available_evidence": skill_input.retrieved_context,
            "missing_information": missing_information,
        }

    def _resolve_rag_bundle_sync(self, skill_input: ReportSkillInput, topic: str) -> Any:
        if not skill_input.rag_enabled:
            return skill_input.rag_bundle
        if skill_input.rag_bundle is not None:
            return skill_input.rag_bundle
        request = RagSearchRequest(
            query=skill_input.rag_query or skill_input.user_query,
            route="report_skill_glm",
            source_type=skill_input.rag_source_type,
            top_k=skill_input.rag_top_k,
            memory_context=skill_input.memory_context,
            conversation_context=skill_input.conversation_context,
            report_topic=topic,
            require_citations=skill_input.require_citations,
        )
        return self._get_rag_service().search(request)

    async def _resolve_rag_bundle(self, skill_input: ReportSkillInput, topic: str) -> Any:
        if not skill_input.rag_enabled:
            return skill_input.rag_bundle
        if skill_input.rag_bundle is not None:
            return skill_input.rag_bundle
        request = RagSearchRequest(
            query=skill_input.rag_query or skill_input.user_query,
            route="report_skill_glm",
            source_type=skill_input.rag_source_type,
            top_k=skill_input.rag_top_k,
            memory_context=skill_input.memory_context,
            conversation_context=skill_input.conversation_context,
            report_topic=topic,
            require_citations=skill_input.require_citations,
        )
        return await asyncio.to_thread(self._get_rag_service().search, request)

    async def retrieve_evidence(self, skill_input: ReportSkillInput, topic: str) -> list[dict[str, Any]]:
        if skill_input.rag_enabled and not skill_input.retrieved_context:
            bundle = await self._resolve_rag_bundle(skill_input, topic)
            return bundle.to_context_records() if bundle and bundle.used else []
        return list(skill_input.retrieved_context)

    def retrieve_evidence_sync(self, skill_input: ReportSkillInput, topic: str) -> list[dict[str, Any]]:
        if skill_input.rag_enabled and not skill_input.retrieved_context:
            bundle = self._resolve_rag_bundle_sync(skill_input, topic)
            return bundle.to_context_records() if bundle and bundle.used else []
        return list(skill_input.retrieved_context)

    def _should_use_financial_signal(self, skill_input: ReportSkillInput, topic: str) -> bool:
        text = _normalized_text(skill_input.user_query)
        if skill_input.report_type != "financial_analysis":
            return False
        if len(text) > 220:
            return False
        lowered = text.lower()
        finance_cues = (
            "财报",
            "业绩",
            "利润",
            "收入",
            "指引",
            "上涨",
            "下跌",
            "增长",
            "下降",
            "earnings",
            "revenue",
            "guidance",
            "profit",
            "decline",
            "growth",
            "rose",
            "fell",
        )
        if any(cue in lowered or cue in text for cue in finance_cues):
            return True
        return False

    def _financial_signal_candidate(self, skill_input: ReportSkillInput, topic: str) -> str:
        topic_candidate = self._extract_topic_from_query(skill_input.user_query)
        return topic_candidate if len(topic_candidate) <= 220 else ""

    def _call_financial_signal_provider(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        provider = self._financial_signal_provider
        if provider is not None:
            result = provider(payload)
        else:
            from deerflow.community.finma.tools import financial_analysis_tool

            result = financial_analysis_tool.invoke(payload)

        if isinstance(result, str):
            parsed = _extract_json_blob(result)
            if parsed:
                return parsed
            return None
        if isinstance(result, dict):
            return result
        return None

    def analyze_financial_signal(self, skill_input: ReportSkillInput, topic: str) -> dict[str, Any] | None:
        if skill_input.financial_signal is not None:
            return skill_input.financial_signal
        if not self._should_use_financial_signal(skill_input, topic):
            return None
        candidate = self._financial_signal_candidate(skill_input, topic)
        if not candidate:
            return None
        try:
            return self._call_financial_signal_provider(
                {
                    "text": candidate,
                    "task": "event_impact",
                    "model_strategy": "v3_and_base",
                }
            )
        except Exception:
            logger.exception("ResearchReportSkill failed to collect financial signal")
            return None

    async def aanalyze_financial_signal(self, skill_input: ReportSkillInput, topic: str) -> dict[str, Any] | None:
        if skill_input.financial_signal is not None:
            return skill_input.financial_signal
        if not self._should_use_financial_signal(skill_input, topic):
            return None
        candidate = self._financial_signal_candidate(skill_input, topic)
        if not candidate:
            return None
        try:
            return await asyncio.to_thread(
                self._call_financial_signal_provider,
                {
                    "text": candidate,
                    "task": "event_impact",
                    "model_strategy": "v3_and_base",
                },
            )
        except Exception:
            logger.exception("ResearchReportSkill failed to collect financial signal")
            return None

    def plan_report(
        self,
        skill_input: ReportSkillInput,
        *,
        topic: str,
        context_summary: str,
        financial_signal: dict[str, Any] | None,
    ) -> ReportPlan:
        prompt = build_planner_prompt(
            skill_input,
            topic=topic,
            context_summary=context_summary,
            financial_signal=financial_signal,
        )
        try:
            raw = self._invoke_model(prompt, run_name="research_report_planner")
            payload = _extract_json_blob(raw)
        except Exception:
            logger.exception("ResearchReportSkill planner failed, using fallback outline")
            payload = None

        return _coerce_outline(
            payload,
            topic=topic,
            brief_report=skill_input.brief_report,
            financial=skill_input.report_type == "financial_analysis",
            use_zh=skill_input.language.startswith("zh"),
        )

    async def aplan_report(
        self,
        skill_input: ReportSkillInput,
        *,
        topic: str,
        context_summary: str,
        financial_signal: dict[str, Any] | None,
    ) -> ReportPlan:
        prompt = build_planner_prompt(
            skill_input,
            topic=topic,
            context_summary=context_summary,
            financial_signal=financial_signal,
        )
        try:
            raw = await self._ainvoke_model(prompt, run_name="research_report_planner")
            payload = _extract_json_blob(raw)
        except Exception:
            logger.exception("ResearchReportSkill planner failed, using fallback outline")
            payload = None

        return _coerce_outline(
            payload,
            topic=topic,
            brief_report=skill_input.brief_report,
            financial=skill_input.report_type == "financial_analysis",
            use_zh=skill_input.language.startswith("zh"),
        )

    def write_report(
        self,
        skill_input: ReportSkillInput,
        *,
        topic: str,
        outline: ReportPlan,
        context_summary: str,
        retrieved_context: list[dict[str, Any]],
        financial_signal: dict[str, Any] | None,
    ) -> str:
        prompt = build_writer_prompt(
            skill_input,
            topic=topic,
            outline=outline,
            context_summary=context_summary,
            retrieved_context=retrieved_context,
            financial_signal=financial_signal,
        )
        return self._invoke_model(prompt, run_name="research_report_writer")

    async def awrite_report(
        self,
        skill_input: ReportSkillInput,
        *,
        topic: str,
        outline: ReportPlan,
        context_summary: str,
        retrieved_context: list[dict[str, Any]],
        financial_signal: dict[str, Any] | None,
    ) -> str:
        prompt = build_writer_prompt(
            skill_input,
            topic=topic,
            outline=outline,
            context_summary=context_summary,
            retrieved_context=retrieved_context,
            financial_signal=financial_signal,
        )
        return await self._ainvoke_model(prompt, run_name="research_report_writer")

    def review_report(self, skill_input: ReportSkillInput, *, topic: str, outline: ReportPlan, report_markdown: str) -> ReportReview:
        raw_text = report_markdown or ""
        raw_lowered = raw_text.lower()
        cleaned = _sanitize_internal_text(raw_text)
        issues: list[str] = []
        missing_sections: list[str] = []

        for marker in _INTERNAL_LEAK_MARKERS:
            if marker.lower() in raw_lowered:
                issues.append(f"报告包含内部字段：{marker}")

        if not cleaned.startswith("#"):
            issues.append("缺少 Markdown 标题。")

        expected_sections = [section.heading for section in outline.sections]
        for heading in expected_sections:
            if heading not in cleaned:
                missing_sections.append(heading)

        if missing_sections:
            issues.append(f"缺少章节：{', '.join(missing_sections)}")

        char_count = len(re.sub(r"\s+", "", cleaned))
        too_long = False
        too_short = False
        if skill_input.brief_report:
            too_short = char_count < 250
            too_long = char_count > 1200
        else:
            too_short = char_count < 700
            too_long = char_count > 3200
        if too_short:
            issues.append("报告过短，信息密度不足。")
        if too_long:
            issues.append("报告过长，需要收敛。")

        context_consistent = True
        normalized_topic = _trim_topic_noise(topic)
        if skill_input.memory_enabled and normalized_topic and normalized_topic not in cleaned:
            context_consistent = False
            issues.append("报告没有明显承接当前主题或上文焦点。")

        if skill_input.require_citations and skill_input.retrieved_context:
            citation_count = len(_CITATION_RE.findall(cleaned))
            if citation_count < 2:
                issues.append("报告缺少足够的证据引用标记。")

        if skill_input.rag_enabled and not skill_input.retrieved_context:
            uncertainty_markers = (
                "证据不足",
                "资料不足",
                "需要进一步验证",
                "仍需补充资料",
                "insufficient evidence",
                "need more evidence",
                "requires further validation",
            )
            if not any(marker in cleaned.lower() or marker in cleaned for marker in uncertainty_markers):
                issues.append("证据不足时没有明确说明限制。")

        needs_rewrite = bool(issues)
        return ReportReview(
            format_complete=not missing_sections and cleaned.startswith("#"),
            context_consistent=context_consistent,
            internal_trace_leaked=any(marker.lower() in raw_lowered for marker in _INTERNAL_LEAK_MARKERS),
            too_long=too_long,
            too_short=too_short,
            missing_sections=missing_sections,
            issues=issues,
            needs_rewrite=needs_rewrite,
        )

    def rewrite_report(
        self,
        skill_input: ReportSkillInput,
        *,
        topic: str,
        outline: ReportPlan,
        report_markdown: str,
        review: ReportReview,
    ) -> str:
        prompt = build_rewrite_prompt(
            skill_input,
            topic=topic,
            outline=outline,
            report_markdown=report_markdown,
            issues=review.issues,
        )
        return self._invoke_model(prompt, run_name="research_report_rewriter")

    async def arewrite_report(
        self,
        skill_input: ReportSkillInput,
        *,
        topic: str,
        outline: ReportPlan,
        report_markdown: str,
        review: ReportReview,
    ) -> str:
        prompt = build_rewrite_prompt(
            skill_input,
            topic=topic,
            outline=outline,
            report_markdown=report_markdown,
            issues=review.issues,
        )
        return await self._ainvoke_model(prompt, run_name="research_report_rewriter")

    def _finalize_markdown(self, outline: ReportPlan, markdown: str) -> str:
        cleaned = _sanitize_internal_text(markdown)
        if not cleaned.startswith("#"):
            cleaned = f"# {outline.title}\n\n{cleaned}".strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def run_sync(self, skill_input: ReportSkillInput) -> ReportSkillOutput:
        topic = self.extract_topic(skill_input)
        prepared_input = replace(skill_input, topic=topic)
        rag_bundle = self._resolve_rag_bundle_sync(prepared_input, topic)
        retrieved_context = self.retrieve_evidence_sync(prepared_input, topic)
        prepared_input = replace(
            prepared_input,
            rag_bundle=rag_bundle,
            retrieved_context=retrieved_context,
        )
        context_bundle = self.build_context(prepared_input, topic)
        context_summary = json.dumps(context_bundle, ensure_ascii=False, indent=2)
        financial_signal = self.analyze_financial_signal(prepared_input, topic)
        outline = self.plan_report(
            prepared_input,
            topic=topic,
            context_summary=context_summary,
            financial_signal=financial_signal,
        )
        report_markdown = self.write_report(
            prepared_input,
            topic=topic,
            outline=outline,
            context_summary=context_summary,
            retrieved_context=retrieved_context,
            financial_signal=financial_signal,
        )
        review = self.review_report(prepared_input, topic=topic, outline=outline, report_markdown=report_markdown)
        if review.needs_rewrite:
            report_markdown = self.rewrite_report(
                prepared_input,
                topic=topic,
                outline=outline,
                report_markdown=report_markdown,
                review=review,
            )
            review = self.review_report(prepared_input, topic=topic, outline=outline, report_markdown=report_markdown)

        final_markdown = self._finalize_markdown(outline, report_markdown)
        review = self.review_report(prepared_input, topic=topic, outline=outline, report_markdown=final_markdown)
        return ReportSkillOutput(
            title=outline.title,
            markdown=final_markdown,
            used_memory=bool(prepared_input.memory_enabled and (prepared_input.conversation_context or prepared_input.memory_context)),
            used_rag=bool(retrieved_context),
            used_financial_signal=bool(financial_signal),
            review_passed=not review.needs_rewrite,
            outline=outline,
            review=review,
        )

    async def run(self, skill_input: ReportSkillInput) -> ReportSkillOutput:
        topic = self.extract_topic(skill_input)
        prepared_input = replace(skill_input, topic=topic)
        rag_bundle = await self._resolve_rag_bundle(prepared_input, topic)
        retrieved_context = await self.retrieve_evidence(prepared_input, topic)
        prepared_input = replace(
            prepared_input,
            rag_bundle=rag_bundle,
            retrieved_context=retrieved_context,
        )
        context_bundle = self.build_context(prepared_input, topic)
        context_summary = json.dumps(context_bundle, ensure_ascii=False, indent=2)
        financial_signal = await self.aanalyze_financial_signal(prepared_input, topic)
        outline = await self.aplan_report(
            prepared_input,
            topic=topic,
            context_summary=context_summary,
            financial_signal=financial_signal,
        )
        report_markdown = await self.awrite_report(
            prepared_input,
            topic=topic,
            outline=outline,
            context_summary=context_summary,
            retrieved_context=retrieved_context,
            financial_signal=financial_signal,
        )
        review = self.review_report(prepared_input, topic=topic, outline=outline, report_markdown=report_markdown)
        if review.needs_rewrite:
            report_markdown = await self.arewrite_report(
                prepared_input,
                topic=topic,
                outline=outline,
                report_markdown=report_markdown,
                review=review,
            )
            review = self.review_report(prepared_input, topic=topic, outline=outline, report_markdown=report_markdown)

        final_markdown = self._finalize_markdown(outline, report_markdown)
        review = self.review_report(prepared_input, topic=topic, outline=outline, report_markdown=final_markdown)
        return ReportSkillOutput(
            title=outline.title,
            markdown=final_markdown,
            used_memory=bool(prepared_input.memory_enabled and (prepared_input.conversation_context or prepared_input.memory_context)),
            used_rag=bool(retrieved_context),
            used_financial_signal=bool(financial_signal),
            review_passed=not review.needs_rewrite,
            outline=outline,
            review=review,
        )
