from __future__ import annotations

import re
import uuid

from deerflow.agents.memory.memory_schema import MemoryRecord, MemoryWriteCandidate
from deerflow.agents.memory.storage import utc_now_iso_z

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")
_EXPERIMENT_RESULT_RE = re.compile(
    r"(?:\b\d+\s+passed\b|\b\d+\s+failed\b|\b\d+\s+tests?\b|测试通过|测试失败|lint 通过|lint 失败|CI 通过|CI 失败)",
    re.IGNORECASE,
)
_OPEN_TASK_RE = re.compile(r"(?:下一步|待完成|todo|to do|needs? to|需要继续|还需要|后续需要|尚未实现)", re.IGNORECASE)
_CONSTRAINT_RE = re.compile(r"(?:不要|不能|避免|限制|约束|边界|先不要|not yet|must not|should not)", re.IGNORECASE)
_SAVE_INTENT_RE = re.compile(r"(?:记住|记下来|以后|remember|keep in mind)", re.IGNORECASE)
_DECISION_RE = re.compile(r"(?:决定|采用|用这个方案|主线|就按这个|we will use|we should use|design choice)", re.IGNORECASE)
_PROJECT_FACT_RE = re.compile(r"(?:当前|已经|现有|目前|now|currently|implemented|running|入口是|路由是|skill 是)", re.IGNORECASE)
_CASUAL_RE = re.compile(r"^(?:谢谢|thanks|ok|好的|收到|嗯|哈哈|hi|hello)[!！。.\s]*$", re.IGNORECASE)
_REPORT_TITLE_RE = re.compile(r"^\s*#\s*(.+)$", re.MULTILINE)


def _extract_keywords(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip()
        lowered = token.casefold()
        if len(lowered) < 2 or lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
    return result[:12]


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for token in _extract_keywords(text):
        if token.isupper() or any(marker in token for marker in ("公司", "项目", "报告", "方案", "风险", "市场")):
            entities.append(token)
    return entities[:8]


def _first_sentences(text: str, *, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "..."


def _markdown_to_summary(markdown: str) -> tuple[str | None, str]:
    title_match = _REPORT_TITLE_RE.search(markdown or "")
    title = title_match.group(1).strip() if title_match else None
    cleaned = re.sub(r"^#+\s*", "", markdown or "", flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, _first_sentences(cleaned, max_chars=260)


class MemoryWriter:
    def extract_candidates(
        self,
        *,
        query: str,
        answer: str,
        route: str,
        skill_name: str | None = None,
        topic: str | None = None,
    ) -> list[MemoryWriteCandidate]:
        normalized_query = query.strip()
        normalized_answer = answer.strip()
        if not normalized_query or not normalized_answer:
            return []
        if _CASUAL_RE.match(normalized_query) and len(normalized_answer) < 80:
            return []

        candidates: list[MemoryWriteCandidate] = []
        combined = f"{normalized_query}\n{normalized_answer}"
        topic_value = (topic or "").strip() or None

        if _SAVE_INTENT_RE.search(normalized_query):
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="user_requirement",
                    content=_first_sentences(normalized_query, max_chars=220),
                    summary=_first_sentences(normalized_query, max_chars=160),
                    topic=topic_value,
                    entities=_extract_entities(normalized_query),
                    keywords=_extract_keywords(normalized_query),
                    importance=0.95,
                    confidence=0.95,
                )
            )

        if _CONSTRAINT_RE.search(normalized_query):
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="constraint",
                    content=_first_sentences(normalized_query, max_chars=220),
                    summary=_first_sentences(normalized_query, max_chars=160),
                    topic=topic_value,
                    entities=_extract_entities(normalized_query),
                    keywords=_extract_keywords(normalized_query),
                    importance=0.9,
                    confidence=0.9,
                )
            )

        if _EXPERIMENT_RESULT_RE.search(combined):
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="experiment_result",
                    content=_first_sentences(normalized_answer, max_chars=240),
                    summary=_first_sentences(normalized_answer, max_chars=180),
                    topic=topic_value,
                    entities=_extract_entities(combined),
                    keywords=_extract_keywords(combined),
                    importance=0.88,
                    confidence=0.92,
                )
            )

        if _OPEN_TASK_RE.search(combined):
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="open_task",
                    content=_first_sentences(normalized_answer, max_chars=240),
                    summary=_first_sentences(normalized_answer, max_chars=180),
                    topic=topic_value,
                    entities=_extract_entities(combined),
                    keywords=_extract_keywords(combined),
                    importance=0.82,
                    confidence=0.82,
                )
            )

        if route == "report_skill_glm":
            report_title, report_summary = _markdown_to_summary(normalized_answer)
            metadata = {"title": report_title} if report_title else {}
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="report_summary",
                    content=report_summary,
                    summary=report_summary,
                    topic=topic_value or report_title,
                    entities=_extract_entities(f"{report_title or ''} {report_summary}"),
                    keywords=_extract_keywords(f"{report_title or ''} {report_summary}"),
                    importance=0.8,
                    confidence=0.86,
                    metadata=metadata,
                )
            )

        if route in {"context_memory_glm", "report_skill_glm"} and _DECISION_RE.search(combined):
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="design_decision",
                    content=_first_sentences(normalized_answer, max_chars=240),
                    summary=_first_sentences(normalized_answer, max_chars=180),
                    topic=topic_value,
                    entities=_extract_entities(combined),
                    keywords=_extract_keywords(combined),
                    importance=0.9,
                    confidence=0.84,
                )
            )

        if route in {"context_memory_glm", "report_skill_glm"} and _PROJECT_FACT_RE.search(normalized_answer):
            candidates.append(
                MemoryWriteCandidate(
                    memory_type="project_fact",
                    content=_first_sentences(normalized_answer, max_chars=240),
                    summary=_first_sentences(normalized_answer, max_chars=180),
                    topic=topic_value,
                    entities=_extract_entities(combined),
                    keywords=_extract_keywords(combined),
                    importance=0.78,
                    confidence=0.78,
                )
            )

        deduped: list[MemoryWriteCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate.memory_type, candidate.content.casefold())
            if not candidate.content.strip() or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def build_record(
        self,
        candidate: MemoryWriteCandidate,
        *,
        thread_id: str | None,
        user_id: str | None,
        route: str,
        skill_name: str | None,
        source_turn_ids: list[str] | None = None,
    ) -> MemoryRecord:
        now = utc_now_iso_z()
        return MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            thread_id=thread_id,
            user_id=user_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            summary=candidate.summary,
            topic=candidate.topic,
            entities=list(candidate.entities),
            keywords=list(candidate.keywords),
            source_route=route,
            source_skill=skill_name,
            source_turn_ids=list(source_turn_ids or []),
            importance=candidate.importance,
            confidence=candidate.confidence,
            created_at=now,
            updated_at=now,
            metadata=dict(candidate.metadata),
        )
