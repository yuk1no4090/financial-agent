from __future__ import annotations

import re

from deerflow.agents.memory.memory_context import MemoryContextAssembler
from deerflow.agents.memory.memory_retriever import MemoryRetriever
from deerflow.agents.memory.memory_schema import MemoryBundle, MemoryQuery, MemoryRecord
from deerflow.agents.memory.memory_store import MemoryRecordStore, get_memory_record_store
from deerflow.agents.memory.memory_writer import MemoryWriter

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")


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


def _extract_entities(text: str, topic: str | None = None) -> list[str]:
    entities: list[str] = []
    if topic and topic.strip():
        entities.append(topic.strip())
    for token in _extract_keywords(text):
        if token.isupper() or any(marker in token for marker in ("公司", "项目", "报告", "方案", "风险", "市场")):
            entities.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        lowered = entity.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(entity)
    return deduped[:8]


class MemoryManager:
    def __init__(
        self,
        *,
        store: MemoryRecordStore | None = None,
        retriever: MemoryRetriever | None = None,
        writer: MemoryWriter | None = None,
        assembler: MemoryContextAssembler | None = None,
    ) -> None:
        self._store = store or get_memory_record_store()
        self._retriever = retriever or MemoryRetriever()
        self._writer = writer or MemoryWriter()
        self._assembler = assembler or MemoryContextAssembler()

    def should_retrieve(self, route: str, *, memory_enabled: bool) -> bool:
        if route == "context_memory_glm":
            return True
        if route == "report_skill_glm" and memory_enabled:
            return True
        return False

    def retrieve_for_route(
        self,
        *,
        query: str,
        route: str,
        topic: str | None = None,
        entities: list[str] | None = None,
        skill_name: str | None = None,
        thread_id: str | None = None,
        memory_enabled: bool = False,
    ) -> MemoryBundle:
        if not self.should_retrieve(route, memory_enabled=memory_enabled):
            return MemoryBundle()

        query_entities = list(entities or _extract_entities(query, topic))
        query_obj = MemoryQuery(
            query=query,
            route=route,
            topic=topic,
            entities=query_entities,
            skill_name=skill_name,
            thread_id=thread_id,
            top_k=5 if route == "context_memory_glm" else 4,
        )
        records = self._store.list_records()
        retrieved = self._retriever.retrieve(query_obj, records)
        return self._assembler.build(retrieved)

    def write_after_response(
        self,
        *,
        query: str,
        answer: str,
        route: str,
        skill_name: str | None = None,
        topic: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        source_turn_ids: list[str] | None = None,
    ) -> list[MemoryRecord]:
        candidates = self._writer.extract_candidates(
            query=query,
            answer=answer,
            route=route,
            skill_name=skill_name,
            topic=topic,
        )
        if not candidates:
            return []

        persisted: list[MemoryRecord] = []
        existing_records = self._store.list_records()

        for candidate in candidates:
            duplicate = next(
                (record for record in existing_records if record.memory_type == candidate.memory_type and record.thread_id == thread_id and record.content.casefold() == candidate.content.casefold()),
                None,
            )
            if duplicate is not None:
                updated = self._store.update(
                    duplicate.id,
                    summary=candidate.summary,
                    topic=candidate.topic,
                    entities=list(candidate.entities),
                    keywords=list(candidate.keywords),
                    importance=max(duplicate.importance, candidate.importance),
                    confidence=max(duplicate.confidence, candidate.confidence),
                    last_accessed_at=duplicate.updated_at,
                    metadata={**duplicate.metadata, **candidate.metadata},
                )
                if updated is not None:
                    persisted.append(updated)
                continue

            record = self._writer.build_record(
                candidate,
                thread_id=thread_id,
                user_id=user_id,
                route=route,
                skill_name=skill_name,
                source_turn_ids=source_turn_ids,
            )
            self._store.add(record)
            existing_records.append(record)
            persisted.append(record)

        return persisted


_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
