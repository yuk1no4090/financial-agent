from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .document_loader import LoadedDocument
from .rag_schema import RagChunk

_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")


@dataclass(frozen=True)
class _SectionBlock:
    heading: str | None
    text: str


def _normalize_whitespace(text: str) -> str:
    compact = re.sub(r"\r\n?", "\n", text or "")
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact.strip()


class MarkdownAwareTextSplitter:
    def __init__(self, *, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: LoadedDocument) -> list[RagChunk]:
        sections = list(self._iter_sections(document.text, fallback_heading=document.title))
        chunks: list[RagChunk] = []
        chunk_index = 0
        for block in sections:
            for text in self._chunk_text(block.text):
                chunks.append(
                    RagChunk(
                        chunk_id=f"{document.doc_id}-{chunk_index:04d}",
                        doc_id=document.doc_id,
                        source_path=document.source_path,
                        title=document.title,
                        section=block.heading,
                        text=text,
                        chunk_index=chunk_index,
                        source_type=document.source_type,
                        metadata=dict(document.metadata),
                    )
                )
                chunk_index += 1
        return chunks

    def _iter_sections(self, text: str, *, fallback_heading: str) -> Iterable[_SectionBlock]:
        lines = _normalize_whitespace(text).splitlines()
        current_heading = fallback_heading
        current_lines: list[str] = []

        for line in lines:
            heading_match = _HEADING_RE.match(line.strip())
            if heading_match:
                if current_lines:
                    yield _SectionBlock(
                        heading=current_heading,
                        text="\n".join(current_lines).strip(),
                    )
                    current_lines = []
                current_heading = heading_match.group(2).strip() or fallback_heading
                continue
            current_lines.append(line)

        if current_lines:
            yield _SectionBlock(
                heading=current_heading,
                text="\n".join(current_lines).strip(),
            )

    def _chunk_text(self, text: str) -> list[str]:
        normalized = _normalize_whitespace(text)
        if len(normalized) <= self.chunk_size:
            return [normalized] if normalized else []

        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalized) if paragraph.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                overlap = current[-self.chunk_overlap :].strip()
                current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
            else:
                chunks.extend(self._slice_long_paragraph(paragraph))
                current = ""

        if current.strip():
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    def _slice_long_paragraph(self, text: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return pieces
