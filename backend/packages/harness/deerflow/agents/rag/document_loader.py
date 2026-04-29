from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    doc_id: str
    source_path: str
    title: str
    text: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "document"


def _extract_markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _load_pdf_text(path: Path) -> str:
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - depends on optional runtime env
        raise RuntimeError("PDF support requires pdfplumber to be installed in the backend environment") from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
    return "\n\n".join(pages).strip()


class DocumentLoader:
    def load_inputs(self, inputs: list[str | Path], *, source_type: str | None = None) -> list[LoadedDocument]:
        documents: list[LoadedDocument] = []
        for raw_input in inputs:
            path = Path(raw_input).expanduser().resolve()
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and child.suffix.lower() in _SUPPORTED_SUFFIXES:
                        loaded = self.load_file(child, source_type=source_type)
                        if loaded is not None:
                            documents.append(loaded)
                continue

            loaded = self.load_file(path, source_type=source_type)
            if loaded is not None:
                documents.append(loaded)
        return documents

    def load_file(self, path: str | Path, *, source_type: str | None = None) -> LoadedDocument | None:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            return None
        if file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            return None

        if file_path.suffix.lower() == ".pdf":
            text = _load_pdf_text(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        text = text.strip()
        if not text:
            return None

        inferred_source_type = source_type or self._infer_source_type(file_path)
        title = _extract_markdown_title(text, file_path.stem) if file_path.suffix.lower() == ".md" else file_path.stem
        return LoadedDocument(
            doc_id=_slugify(file_path.stem),
            source_path=file_path.as_posix(),
            title=title,
            text=text,
            source_type=inferred_source_type,
            metadata={"suffix": file_path.suffix.lower()},
        )

    def _infer_source_type(self, path: Path) -> str:
        for parent in path.parents:
            if parent.name in {"project_docs", "finance_docs", "eval_docs", "user_upload"}:
                return parent.name
        return "project_docs"
