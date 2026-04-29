from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    section: str | None
    text: str
    chunk_index: int
    source_type: str
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    doc_id: str
    title: str
    section: str | None
    source_path: str
    text: str
    score: float
    rank: int

    def to_prompt_block(self) -> str:
        section = self.section or "未标注章节"
        source = self.source_path.rsplit("/", 1)[-1] or self.source_path
        return f"[E{self.rank}]\nTitle: {self.title}\nSection: {section}\nSource: {source}\nContent:\n{self.text.strip()}"

    def to_context_record(self) -> dict[str, Any]:
        source = self.source_path.rsplit("/", 1)[-1] or self.source_path
        return {
            "citation": f"E{self.rank}",
            "title": self.title,
            "section": self.section or "",
            "source": source,
            "content": self.text.strip(),
        }


@dataclass(frozen=True)
class RagSearchRequest:
    query: str
    route: str
    source_type: str = "auto"
    top_k: int = 5
    memory_context: str | None = None
    conversation_context: str | None = None
    report_topic: str | None = None
    require_citations: bool = False


@dataclass(frozen=True)
class RagBundle:
    query: str = ""
    rewritten_query: str | None = None
    evidences: list[RetrievedEvidence] = field(default_factory=list)
    summary: str = ""
    missing: list[str] = field(default_factory=list)
    used: bool = False
    source_type: str = "auto"

    def to_context_records(self) -> list[dict[str, Any]]:
        return [evidence.to_context_record() for evidence in self.evidences]

    def to_prompt_text(self) -> str:
        if not self.used or not self.evidences:
            return "[Evidence Context]\nNo reliable evidence was retrieved for this turn.\n\n[Instruction]\nIf the user asks for exact sources, figures, or citations, say the current evidence is insufficient."

        blocks = "\n\n".join(evidence.to_prompt_block() for evidence in self.evidences)
        return (
            "[Evidence Context]\n\n"
            f"{blocks}\n\n"
            "[Instruction]\n"
            "Please answer using the evidence above whenever it is relevant.\n"
            "Do not invent years, rankings, figures, or citations that are not supported by the evidence.\n"
            "If the evidence is incomplete, explicitly say so."
        )
