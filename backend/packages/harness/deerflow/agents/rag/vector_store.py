from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embedding_provider import HashingEmbeddingProvider
from .rag_schema import RagChunk, RetrievedEvidence


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[5]


def _repo_root() -> Path:
    return _backend_dir().parent


def default_rag_raw_dir() -> Path:
    return _repo_root() / "data" / "rag" / "raw"


def default_rag_index_dir() -> Path:
    return _repo_root() / "data" / "rag" / "index"


class LocalVectorStore:
    def __init__(
        self,
        *,
        index_dir: str | Path | None = None,
        embedding_provider: HashingEmbeddingProvider | None = None,
    ) -> None:
        self.index_dir = Path(index_dir or default_rag_index_dir()).expanduser().resolve()
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self._chunks: list[RagChunk] = []
        self._vectors: list[list[float]] = []
        self._loaded = False
        self._load_from_disk()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def reload(self) -> None:
        self._loaded = False
        self._chunks = []
        self._vectors = []
        self._load_from_disk()

    def build(self, chunks: list[RagChunk]) -> dict[str, Any]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = self.embedding_provider.embed_texts([self._indexable_text(chunk) for chunk in chunks])

        chunks_path = self.index_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        vectors_path = self.index_dir / "vectors.json"
        vectors_path.write_text(
            json.dumps(vectors, ensure_ascii=False),
            encoding="utf-8",
        )

        metadata = {
            "built_at": datetime.now(UTC).isoformat(),
            "chunk_count": len(chunks),
            "dims": self.embedding_provider.dims,
        }
        metadata_path = self.index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self._chunks = list(chunks)
        self._vectors = vectors
        self._loaded = True
        return metadata

    def search(self, *, query: str, source_type: str = "auto", top_k: int = 5) -> list[RetrievedEvidence]:
        if not self._loaded or not self._chunks or not query.strip():
            return []

        query_vector = self.embedding_provider.embed_text(query)
        query_tokens = set(self.embedding_provider.tokenize(query))
        scored: list[tuple[float, RagChunk]] = []

        for chunk, vector in zip(self._chunks, self._vectors, strict=False):
            if source_type != "auto" and chunk.source_type != source_type:
                continue

            cosine = self._cosine(query_vector, vector)
            overlap = self._keyword_overlap(query_tokens, chunk)
            title_bonus = self._title_bonus(query_tokens, chunk)
            score = cosine * 0.72 + overlap * 0.22 + title_bonus
            if score <= 0.08:
                continue
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_hits = scored[: max(1, top_k)]
        if not top_hits:
            return []

        return [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                section=chunk.section,
                source_path=chunk.source_path,
                text=chunk.text,
                score=round(score, 4),
                rank=rank,
            )
            for rank, (score, chunk) in enumerate(top_hits, start=1)
        ]

    def _load_from_disk(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.index_dir / "chunks.jsonl"
        vectors_path = self.index_dir / "vectors.json"
        if not chunks_path.exists() or not vectors_path.exists():
            self._loaded = True
            return

        chunks: list[RagChunk] = []
        with chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                chunks.append(RagChunk(**payload))

        vectors_payload = json.loads(vectors_path.read_text(encoding="utf-8"))
        self._chunks = chunks
        self._vectors = [list(map(float, vector)) for vector in vectors_payload]
        self._loaded = True

    def _indexable_text(self, chunk: RagChunk) -> str:
        section = chunk.section or ""
        return "\n".join(part for part in (chunk.title, section, chunk.text) if part.strip())

    def _keyword_overlap(self, query_tokens: set[str], chunk: RagChunk) -> float:
        if not query_tokens:
            return 0.0
        chunk_tokens = set(self.embedding_provider.tokenize(self._indexable_text(chunk)))
        if not chunk_tokens:
            return 0.0
        return len(query_tokens & chunk_tokens) / len(query_tokens)

    def _title_bonus(self, query_tokens: set[str], chunk: RagChunk) -> float:
        if not query_tokens:
            return 0.0
        bonus_text = f"{chunk.title} {chunk.section or ''}"
        bonus_tokens = set(self.embedding_provider.tokenize(bonus_text))
        overlap = len(query_tokens & bonus_tokens)
        if overlap == 0:
            return 0.0
        return min(0.08, overlap * 0.02)

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        return sum(l_value * r_value for l_value, r_value in zip(left, right, strict=False))
