from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from deerflow.agents.memory.memory_schema import MemoryQuery, MemoryRecord

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "analysis",
    "and",
    "are",
    "based",
    "continue",
    "for",
    "from",
    "help",
    "into",
    "just",
    "next",
    "please",
    "report",
    "that",
    "the",
    "then",
    "this",
    "with",
    "继续",
    "刚刚",
    "刚才",
    "分析",
    "基于",
    "报告",
    "展开",
    "我们",
    "那个",
    "这个",
    "继续讲",
}


def _iso_to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _extract_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip().casefold()
        if len(token) < 2 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


class MemoryRetriever:
    def score_record(self, query: MemoryQuery, record: MemoryRecord) -> float:
        query_tokens = _extract_tokens(query.query)
        record_tokens = set(token.casefold() for token in record.keywords) | _extract_tokens(record.content) | _extract_tokens(record.summary)
        query_entities = {item.casefold() for item in query.entities if item.strip()}
        record_entities = {item.casefold() for item in record.entities if item.strip()}

        keyword_overlap = len(query_tokens & record_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
        entity_overlap = len(query_entities & record_entities) / max(1, len(query_entities)) if query_entities else 0.0

        route_match = 0.0
        if query.route and record.source_route == query.route:
            route_match = 1.0
        elif query.route in {"context_memory_glm", "report_skill_glm"} and record.source_route in {"context_memory_glm", "report_skill_glm"}:
            route_match = 0.65
        elif query.route and record.source_skill and query.skill_name and record.source_skill == query.skill_name:
            route_match = 0.8

        recency_score = 0.0
        updated_at = _iso_to_datetime(record.updated_at) or _iso_to_datetime(record.created_at)
        if updated_at is not None:
            age_days = max(0.0, (datetime.now(UTC) - updated_at).total_seconds() / 86400.0)
            recency_score = max(0.0, 1.0 - min(age_days, 30.0) / 30.0)

        thread_score = 1.0
        if query.thread_id and record.thread_id and record.thread_id != query.thread_id:
            thread_score = 0.45

        score = 0.35 * keyword_overlap + 0.25 * entity_overlap + 0.20 * route_match + 0.10 * recency_score + 0.10 * _bounded(record.importance)
        return _bounded(score * thread_score)

    def retrieve(self, query: MemoryQuery, records: list[MemoryRecord]) -> list[MemoryRecord]:
        ranked: list[tuple[float, MemoryRecord]] = []
        for record in records:
            if _bounded(record.confidence) < 0.5:
                continue
            score = self.score_record(query, record)
            if score <= 0:
                continue
            ranked.append((score, record))

        ranked.sort(
            key=lambda item: (
                item[0],
                _iso_to_datetime(item[1].updated_at) or datetime.fromtimestamp(0, tz=UTC),
            ),
            reverse=True,
        )
        return [record for _, record in ranked[: max(1, query.top_k)]]
