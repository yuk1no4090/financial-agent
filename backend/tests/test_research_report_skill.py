from __future__ import annotations

from types import SimpleNamespace

from deerflow.agents.rag import RagBundle, RetrievedEvidence
from deerflow.agents.skills import ReportSkillInput, ResearchReportSkill


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str, config: dict | None = None) -> SimpleNamespace:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("No fake LLM response queued")
        return SimpleNamespace(content=self._responses.pop(0))

    async def ainvoke(self, prompt: str, config: dict | None = None) -> SimpleNamespace:
        return self.invoke(prompt, config=config)


def test_extract_topic_from_followup_report_query_prefers_contextual_topic() -> None:
    skill = ResearchReportSkill(llm=FakeLLM([]))
    skill_input = ReportSkillInput(
        user_query="对于你刚刚提到的复合碳水，给我简单生成一份报告",
        language="zh",
        brief_report=True,
        memory_enabled=True,
        conversation_context="[Assistant] 复合碳水化合物包括燕麦、藜麦、红薯和全麦面包。",
    )

    topic = skill.extract_topic(skill_input)

    assert topic == "复合碳水"


def test_plan_report_falls_back_to_default_outline_when_planner_json_is_invalid() -> None:
    skill = ResearchReportSkill(llm=FakeLLM(["not valid json"]))
    skill_input = ReportSkillInput(
        user_query="请基于中东战争对油价影响写一份研究报告",
        language="zh",
        brief_report=True,
        report_type="financial_analysis",
    )

    outline = skill.plan_report(
        skill_input,
        topic="中东战争对油价影响",
        context_summary="{}",
        financial_signal=None,
    )

    assert outline.title == "中东战争对油价影响分析报告"
    assert [section.heading for section in outline.sections] == ["核心结论", "关键分析", "风险点", "简短总结"]


def test_review_report_detects_internal_leakage_and_missing_sections() -> None:
    skill = ResearchReportSkill(
        llm=FakeLLM([('{"title":"减肥食物研究报告","sections":[{"heading":"核心结论","goal":"总结核心判断"},{"heading":"关键分析","goal":"展开分析"},{"heading":"风险点","goal":"说明限制"},{"heading":"简短总结","goal":"做收束"}]}')])
    )
    skill_input = ReportSkillInput(
        user_query="请写一份报告",
        language="zh",
        brief_report=True,
        memory_enabled=True,
    )
    outline = skill.plan_report(
        skill_input,
        topic="减肥食物",
        context_summary="{}",
        financial_signal=None,
    )

    review = skill.review_report(
        skill_input,
        topic="减肥食物",
        outline=outline,
        report_markdown="当前路由：report_skill_glm\n\n## 核心结论\n\n这是一份过短的报告。",
    )

    assert review.internal_trace_leaked is True
    assert review.needs_rewrite is True
    assert "关键分析" in review.missing_sections


def test_run_sync_rewrites_report_and_preserves_memory_context() -> None:
    fake_llm = FakeLLM(
        [
            "invalid json",
            "当前路由：report_skill_glm\n\n## 核心结论\n\n复合碳水可以帮助稳定能量。",
            (
                "# 复合碳水研究报告\n\n"
                "## 核心结论\n\n复合碳水更适合作为减脂阶段的主食基础，因为消化吸收速度相对平稳，能帮助控制饥饿感并维持训练与日常活动所需的稳定能量。\n\n"
                "## 关键分析\n\n像燕麦、藜麦、红薯和全麦面包这类食物通常同时带来碳水、纤维和一定的饱腹感，"
                "因此比精制甜点或高糖面包更适合放进长期饮食结构中。对正在控制热量的人来说，它们更容易支持稳定执行，而不是只带来短时间饱足后又迅速饥饿。\n\n"
                "## 风险点\n\n如果总摄入量控制不好，复合碳水同样可能造成热量过剩；另外，具体搭配方式、烹饪方式和个人活动量也会影响最终效果，因此不能把它理解成单独决定减脂结果的唯一因素。\n\n"
                "## 简短总结\n\n如果只做一个简短判断，复合碳水是更适合长期减脂饮食的基础选择，但仍需要结合份量、蛋白质搭配和整体热量管理来看。"
            ),
        ]
    )
    skill = ResearchReportSkill(llm=fake_llm)
    skill_input = ReportSkillInput(
        user_query="对于你刚刚提到的复合碳水，给我简单生成一份报告",
        language="zh",
        brief_report=True,
        memory_enabled=True,
        conversation_context=("[User] 减肥食物推荐\n[Assistant] 复合碳水化合物包括燕麦、藜麦、红薯和全麦面包。"),
    )

    output = skill.run_sync(skill_input)

    assert output.review_passed is True
    assert output.used_memory is True
    assert output.review is not None
    assert output.review.internal_trace_leaked is False
    assert "当前路由：" not in output.markdown
    assert output.markdown.startswith("# 复合碳水研究报告")


def test_analyze_financial_signal_uses_provider_for_short_financial_report_query() -> None:
    captured: list[dict] = []

    def _provider(payload: dict) -> dict:
        captured.append(payload)
        return {"provider": "mock", "synthesis": {"label": "positive"}}

    skill = ResearchReportSkill(llm=FakeLLM([]), financial_signal_provider=_provider)
    skill_input = ReportSkillInput(
        user_query="苹果公司财报增长5%，请写一份简短报告",
        language="zh",
        brief_report=True,
        report_type="financial_analysis",
    )

    signal = skill.analyze_financial_signal(skill_input, "苹果公司财报增长5%")

    assert signal == {"provider": "mock", "synthesis": {"label": "positive"}}
    assert captured == [
        {
            "text": "苹果公司财报增长5%",
            "task": "event_impact",
            "model_strategy": "v3_and_base",
        }
    ]


def test_run_sync_uses_rag_context_and_preserves_citations() -> None:
    fake_llm = FakeLLM(
        [
            '{"title":"Router 与 RAG 设计报告","sections":[{"heading":"核心结论","goal":"总结系统设计"},{"heading":"关键分析","goal":"解释模块关系"},{"heading":"风险点","goal":"说明限制"},{"heading":"简短总结","goal":"收束"}]}',
            (
                "# Router 与 RAG 设计报告\n\n"
                "## 核心结论\n\n"
                "当前系统不是把 RAG 当成独立 route，而是把它作为 Router、Memory 与 Skill 之间的证据增强层 [E1][E2]。\n\n"
                "## 关键分析\n\n"
                "Router 先判断任务类型，再决定是否打开 rag_enabled；Report Skill 则消费 evidence context，让报告生成从自由生成变成基于证据的结构化生成 [E1][E2]。\n\n"
                "## 风险点\n\n"
                "如果本地知识库覆盖不够，系统应明确说明证据不足，而不是补写不存在的年份和数据 [E2]。\n\n"
                "## 简短总结\n\n"
                "因此这版实现的价值在于把路线判断、长期记忆和外部证据清晰分层 [E1]。"
            ),
        ]
    )

    rag_bundle = RagBundle(
        query="根据项目文档解释 Router 和 RAG 的关系",
        rewritten_query="Router RAG 项目文档 关系",
        evidences=[
            RetrievedEvidence(
                chunk_id="router-1",
                doc_id="router",
                title="Router 机制说明",
                section="总体设计",
                source_path="/tmp/router.md",
                text="Router 负责判断任务类型，并通过 rag_enabled 决定是否打开证据检索。",
                score=0.92,
                rank=1,
            ),
            RetrievedEvidence(
                chunk_id="rag-1",
                doc_id="rag",
                title="RAG 设计方案",
                section="总体架构",
                source_path="/tmp/rag.md",
                text="RAG 被设计为 Router、Memory 与 Skill 之间的证据增强层，而不是独立主 route。",
                score=0.89,
                rank=2,
            ),
        ],
        summary="检索到 Router 与 RAG 的项目文档证据。",
        used=True,
        source_type="project_docs",
    )

    skill = ResearchReportSkill(llm=fake_llm)
    skill_input = ReportSkillInput(
        user_query="根据项目文档解释 Router 和 RAG 的关系，并生成一个简短报告",
        language="zh",
        brief_report=True,
        rag_enabled=True,
        rag_bundle=rag_bundle,
        require_citations=True,
    )

    output = skill.run_sync(skill_input)

    assert output.used_rag is True
    assert "[E1]" in output.markdown
    assert "[E2]" in output.markdown
    assert "chunk_id" not in output.markdown
    assert "score" not in output.markdown


def test_review_report_detects_missing_citations_when_required() -> None:
    skill = ResearchReportSkill(llm=FakeLLM([]))
    skill_input = ReportSkillInput(
        user_query="根据项目文档生成报告",
        language="zh",
        brief_report=True,
        rag_enabled=True,
        require_citations=True,
        retrieved_context=[
            {
                "citation": "E1",
                "title": "Router 机制说明",
                "section": "总体设计",
                "source": "router.md",
                "content": "Router 负责任务分流。",
            }
        ],
    )
    outline = skill.plan_report(
        ReportSkillInput(
            user_query=skill_input.user_query,
            language="zh",
            brief_report=True,
        ),
        topic="Router 与 RAG",
        context_summary="{}",
        financial_signal=None,
    )

    review = skill.review_report(
        skill_input,
        topic="Router 与 RAG",
        outline=outline,
        report_markdown=("# Router 与 RAG 报告\n\n## 核心结论\n\nRouter 会先判断任务路线。\n\n## 关键分析\n\nRAG 负责补充证据。\n\n## 风险点\n\n知识库覆盖不足会影响回答质量。\n\n## 简短总结\n\n两者结合能提升稳定性。"),
    )

    assert review.needs_rewrite is True
    assert "证据引用标记" in " ".join(review.issues)


def test_review_report_requires_uncertainty_notice_when_rag_has_no_evidence() -> None:
    skill = ResearchReportSkill(llm=FakeLLM([]))
    skill_input = ReportSkillInput(
        user_query="根据资料生成报告",
        language="zh",
        brief_report=True,
        rag_enabled=True,
    )
    outline = skill.plan_report(
        ReportSkillInput(
            user_query=skill_input.user_query,
            language="zh",
            brief_report=True,
        ),
        topic="RAG 测试",
        context_summary="{}",
        financial_signal=None,
    )

    review = skill.review_report(
        skill_input,
        topic="RAG 测试",
        outline=outline,
        report_markdown=("# RAG 测试\n\n## 核心结论\n\n系统可以回答问题。\n\n## 关键分析\n\n它会把资料组织进报告。\n\n## 风险点\n\n当前实现还在继续完善。\n\n## 简短总结\n\n整体方向是对的。"),
    )

    assert review.needs_rewrite is True
    assert "证据不足" in " ".join(review.issues)
