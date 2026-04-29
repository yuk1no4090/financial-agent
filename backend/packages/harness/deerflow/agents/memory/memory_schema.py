from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deerflow.agents.memory.storage import utc_now_iso_z


@dataclass
class MemoryRecord:
    id: str
    thread_id: str | None
    user_id: str | None
    memory_type: str
    content: str
    summary: str
    topic: str | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_route: str | None = None
    source_skill: str | None = None
    source_turn_ids: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.7
    created_at: str = field(default_factory=utc_now_iso_z)
    updated_at: str = field(default_factory=utc_now_iso_z)
    last_accessed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "summary": self.summary,
            "topic": self.topic,
            "entities": list(self.entities),
            "keywords": list(self.keywords),
            "source_route": self.source_route,
            "source_skill": self.source_skill,
            "source_turn_ids": list(self.source_turn_ids),
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryRecord:
        return cls(
            id=str(payload.get("id") or ""),
            thread_id=payload.get("thread_id"),
            user_id=payload.get("user_id"),
            memory_type=str(payload.get("memory_type") or ""),
            content=str(payload.get("content") or ""),
            summary=str(payload.get("summary") or ""),
            topic=payload.get("topic"),
            entities=[str(item) for item in payload.get("entities", []) if str(item).strip()],
            keywords=[str(item) for item in payload.get("keywords", []) if str(item).strip()],
            source_route=payload.get("source_route"),
            source_skill=payload.get("source_skill"),
            source_turn_ids=[str(item) for item in payload.get("source_turn_ids", []) if str(item).strip()],
            importance=float(payload.get("importance", 0.5)),
            confidence=float(payload.get("confidence", 0.7)),
            created_at=str(payload.get("created_at") or utc_now_iso_z()),
            updated_at=str(payload.get("updated_at") or utc_now_iso_z()),
            last_accessed_at=payload.get("last_accessed_at"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class MemoryQuery:
    query: str
    route: str
    topic: str | None
    entities: list[str]
    skill_name: str | None
    thread_id: str | None
    top_k: int = 5


@dataclass
class MemoryBundle:
    summary: str = ""
    relevant_facts: list[str] = field(default_factory=list)
    prior_decisions: list[str] = field(default_factory=list)
    prior_results: list[str] = field(default_factory=list)
    open_tasks: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            (
                self.summary,
                self.relevant_facts,
                self.prior_decisions,
                self.prior_results,
                self.open_tasks,
                self.constraints,
            )
        )

    def to_prompt_text(self) -> str:
        if self.is_empty():
            return ""

        sections: list[str] = ["[Relevant Memory]"]
        if self.summary:
            sections.append(f"- 摘要：{self.summary}")
        for label, items in (
            ("已确认项目结论", self.prior_decisions),
            ("相关事实", self.relevant_facts),
            ("已有实验或输出", self.prior_results),
            ("当前约束", self.constraints),
            ("待完成任务", self.open_tasks),
        ):
            if not items:
                continue
            joined = "；".join(items)
            sections.append(f"- {label}：{joined}")
        for warning in self.warnings:
            sections.append(f"- 注意：{warning}")
        return "\n".join(sections).strip()


@dataclass
class MemoryWriteCandidate:
    memory_type: str
    content: str
    summary: str
    topic: str | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryWriteDecision:
    action: str
    reason: str
    existing_id: str | None = None
