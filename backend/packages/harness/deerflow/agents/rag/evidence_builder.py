from __future__ import annotations

from .rag_schema import RagBundle, RetrievedEvidence


def _source_label(source_type: str) -> str:
    labels = {
        "auto": "综合",
        "project_docs": "项目文档",
        "finance_docs": "金融资料",
        "eval_docs": "实验资料",
        "user_upload": "用户上传资料",
    }
    return labels.get(source_type, source_type)


class RagEvidenceBuilder:
    def build(
        self,
        *,
        original_query: str,
        rewritten_query: str | None,
        evidences: list[RetrievedEvidence],
        source_type: str,
    ) -> RagBundle:
        if not evidences:
            return RagBundle(
                query=original_query,
                rewritten_query=rewritten_query,
                evidences=[],
                summary="当前没有检索到可用证据。",
                missing=["no_relevant_evidence"],
                used=False,
                source_type=source_type,
            )

        evidence_titles = []
        for evidence in evidences[:3]:
            label = evidence.section or evidence.title
            if label not in evidence_titles:
                evidence_titles.append(label)

        summary = f"检索到 {len(evidences)} 条与当前问题相关的{_source_label(source_type)}证据，重点覆盖：{'、'.join(evidence_titles)}。"
        return RagBundle(
            query=original_query,
            rewritten_query=rewritten_query,
            evidences=evidences,
            summary=summary,
            missing=[],
            used=True,
            source_type=source_type,
        )
