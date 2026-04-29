from __future__ import annotations

from deerflow.agents.rag import LocalVectorStore, RagSearchRequest, RagService


def test_build_rag_index_from_markdown(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "project_docs"
    raw_dir.mkdir(parents=True)
    (raw_dir / "router.md").write_text(
        "# Router 机制说明\n\n## 总体设计\n\nRouter 当前支持 financial_finma、financial_glm、general_glm、context_memory_glm 和 report_skill_glm。\n",
        encoding="utf-8",
    )

    service = RagService(store=LocalVectorStore(index_dir=tmp_path / "index"))
    metadata = service.rebuild_index(inputs=[raw_dir])

    assert metadata["document_count"] == 1
    assert metadata["chunk_count"] >= 1
    assert service.chunk_count >= 1


def test_retrieve_router_doc(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "project_docs"
    raw_dir.mkdir(parents=True)
    (raw_dir / "router.md").write_text(
        "# Router 机制说明\n\n## 总体设计\n\nRouter 当前支持 financial_finma、financial_glm、general_glm、context_memory_glm 和 report_skill_glm。\n",
        encoding="utf-8",
    )
    (raw_dir / "report.md").write_text(
        "# Report Skill 设计\n\n## 主流程\n\nReport Skill 包括 extract_topic、build_context、retrieve_evidence、plan_report、write_report、review 和 rewrite。\n",
        encoding="utf-8",
    )

    service = RagService(store=LocalVectorStore(index_dir=tmp_path / "index"))
    service.rebuild_index(inputs=[raw_dir])
    bundle = service.search(
        RagSearchRequest(
            query="Router 当前有哪些路由",
            route="general_glm",
            source_type="project_docs",
            top_k=3,
        )
    )

    assert bundle.used is True
    assert bundle.evidences
    assert bundle.evidences[0].title == "Router 机制说明"


def test_empty_result_fallback(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "project_docs"
    raw_dir.mkdir(parents=True)
    (raw_dir / "router.md").write_text("# Router 机制说明\n\n支持五种路线。", encoding="utf-8")

    service = RagService(store=LocalVectorStore(index_dir=tmp_path / "index"))
    service.rebuild_index(inputs=[raw_dir])
    bundle = service.search(
        RagSearchRequest(
            query="!!!",
            route="general_glm",
            source_type="project_docs",
        )
    )

    assert bundle.used is False
    assert bundle.missing == ["no_relevant_evidence"]


def test_evidence_context_format(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "project_docs"
    raw_dir.mkdir(parents=True)
    (raw_dir / "report.md").write_text(
        "# Report Skill 设计\n\n## 主流程\n\nReport Skill 使用结构化 planner 和 review 流程来提升报告稳定性。\n",
        encoding="utf-8",
    )

    service = RagService(store=LocalVectorStore(index_dir=tmp_path / "index"))
    service.rebuild_index(inputs=[raw_dir])
    bundle = service.search(
        RagSearchRequest(
            query="Report Skill 为什么更稳定",
            route="report_skill_glm",
            source_type="project_docs",
        )
    )

    prompt_text = bundle.to_prompt_text()

    assert bundle.used is True
    assert "[E1]" in prompt_text
    assert "chunk_id" not in prompt_text
    assert "score" not in prompt_text
