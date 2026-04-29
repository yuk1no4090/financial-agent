from __future__ import annotations

from deerflow.agents.memory import MemoryBundle, MemoryManager
from deerflow.agents.memory.memory_schema import MemoryRecord
from deerflow.agents.memory.memory_store import MemoryRecordStore


def test_retrieve_for_route_builds_structured_memory_bundle(tmp_path) -> None:
    store = MemoryRecordStore(tmp_path / "memory_records.jsonl")
    store.add(
        MemoryRecord(
            id="mem-1",
            thread_id="thread-1",
            user_id=None,
            memory_type="design_decision",
            content="项目主线采用 Router + Report Skill + Memory。",
            summary="项目主线采用 Router + Report Skill + Memory。",
            topic="memory 方案",
            entities=["Router", "Report Skill", "Memory"],
            keywords=["Router", "Report", "Skill", "Memory"],
            source_route="context_memory_glm",
            importance=0.95,
            confidence=0.95,
        )
    )
    store.add(
        MemoryRecord(
            id="mem-2",
            thread_id="thread-1",
            user_id=None,
            memory_type="open_task",
            content="下一步需要把 memory_context 接成定向检索。",
            summary="下一步需要把 memory_context 接成定向检索。",
            topic="memory 方案",
            entities=["memory_context"],
            keywords=["memory_context", "定向", "检索"],
            source_route="report_skill_glm",
            importance=0.88,
            confidence=0.9,
        )
    )
    manager = MemoryManager(store=store)

    bundle = manager.retrieve_for_route(
        query="把刚刚的 Router 和 memory 方案继续展开一下",
        route="context_memory_glm",
        thread_id="thread-1",
        memory_enabled=True,
    )

    assert isinstance(bundle, MemoryBundle)
    assert "Router + Report Skill + Memory" in bundle.to_prompt_text()
    assert bundle.prior_decisions == ["项目主线采用 Router + Report Skill + Memory。"]
    assert bundle.open_tasks == ["下一步需要把 memory_context 接成定向检索。"]
    assert bundle.source_memory_ids == ["mem-1", "mem-2"]


def test_write_after_response_skips_casual_chat(tmp_path) -> None:
    store = MemoryRecordStore(tmp_path / "memory_records.jsonl")
    manager = MemoryManager(store=store)

    persisted = manager.write_after_response(
        query="谢谢",
        answer="不客气。",
        route="general_glm",
        thread_id="thread-1",
    )

    assert persisted == []
    assert store.list_records() == []


def test_write_after_response_persists_report_summary(tmp_path) -> None:
    store = MemoryRecordStore(tmp_path / "memory_records.jsonl")
    manager = MemoryManager(store=store)

    persisted = manager.write_after_response(
        query="针对刚刚的 memory 方案生成一份报告",
        answer=("# Memory 方案报告\n\n## 核心结论\n\nRouter 与 Report Skill 后面补一层显式 Memory Retrieval，可以更稳定地承接多轮设计讨论。"),
        route="report_skill_glm",
        skill_name="research-report-skill",
        thread_id="thread-1",
    )

    assert len(persisted) == 1
    record = persisted[0]
    assert record.memory_type == "report_summary"
    assert record.source_skill == "research-report-skill"
    assert "Memory 方案报告" in record.metadata["title"]
    assert "显式 Memory Retrieval" in record.summary


def test_write_after_response_updates_duplicate_memory_instead_of_appending(tmp_path) -> None:
    store = MemoryRecordStore(tmp_path / "memory_records.jsonl")
    manager = MemoryManager(store=store)

    first = manager.write_after_response(
        query="记住：项目主线是 Router + Skill + Memory",
        answer="后续实现都围绕 Router + Skill + Memory 这条主线推进。",
        route="context_memory_glm",
        thread_id="thread-1",
    )
    second = manager.write_after_response(
        query="记住：项目主线是 Router + Skill + Memory",
        answer="后续实现都围绕 Router + Skill + Memory 这条主线推进。",
        route="context_memory_glm",
        thread_id="thread-1",
    )

    assert len(first) >= 1
    assert len(second) == len(first)
    assert len(store.list_records()) == len(first)
