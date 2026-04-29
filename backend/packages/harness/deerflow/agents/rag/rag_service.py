from __future__ import annotations

from pathlib import Path
from typing import Any

from .document_loader import DocumentLoader
from .evidence_builder import RagEvidenceBuilder
from .query_builder import RagQueryBuilder
from .rag_schema import RagBundle, RagSearchRequest
from .text_splitter import MarkdownAwareTextSplitter
from .vector_store import LocalVectorStore, default_rag_index_dir, default_rag_raw_dir


class RagService:
    def __init__(
        self,
        *,
        loader: DocumentLoader | None = None,
        splitter: MarkdownAwareTextSplitter | None = None,
        store: LocalVectorStore | None = None,
        query_builder: RagQueryBuilder | None = None,
        evidence_builder: RagEvidenceBuilder | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.splitter = splitter or MarkdownAwareTextSplitter()
        self.store = store or LocalVectorStore(index_dir=default_rag_index_dir())
        self.query_builder = query_builder or RagQueryBuilder()
        self.evidence_builder = evidence_builder or RagEvidenceBuilder()

    def search(self, request: RagSearchRequest) -> RagBundle:
        if not request.query.strip():
            return RagBundle(
                query=request.query,
                rewritten_query=None,
                summary="空查询不触发检索。",
                missing=["empty_query"],
                used=False,
                source_type=request.source_type,
            )

        rewritten_query = self.query_builder.build(
            current_query=request.query,
            route=request.route,
            memory_context=request.memory_context,
            conversation_context=request.conversation_context,
            report_topic=request.report_topic,
        )
        evidences = self.store.search(
            query=rewritten_query,
            source_type=request.source_type,
            top_k=request.top_k,
        )
        return self.evidence_builder.build(
            original_query=request.query,
            rewritten_query=rewritten_query,
            evidences=evidences,
            source_type=request.source_type,
        )

    def rebuild_index(
        self,
        *,
        inputs: list[str | Path] | None = None,
        source_type: str | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
    ) -> dict[str, Any]:
        effective_inputs = inputs or [default_rag_raw_dir()]
        documents = self.loader.load_inputs(effective_inputs, source_type=source_type)
        splitter = MarkdownAwareTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = []
        for document in documents:
            chunks.extend(splitter.split_document(document))
        metadata = self.store.build(chunks)
        metadata["document_count"] = len(documents)
        metadata["inputs"] = [str(Path(item)) for item in effective_inputs]
        return metadata

    def reload(self) -> None:
        self.store.reload()

    @property
    def chunk_count(self) -> int:
        return self.store.chunk_count


_rag_service = RagService()


def get_rag_service() -> RagService:
    return _rag_service
