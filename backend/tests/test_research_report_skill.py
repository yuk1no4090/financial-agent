from __future__ import annotations

from types import SimpleNamespace

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
        llm=FakeLLM(
            [
                (
                    '{"title":"减肥食物研究报告","sections":['
                    '{"heading":"核心结论","goal":"总结核心判断"},'
                    '{"heading":"关键分析","goal":"展开分析"},'
                    '{"heading":"风险点","goal":"说明限制"},'
                    '{"heading":"简短总结","goal":"做收束"}'
                    "]}"
                )
            ]
        )
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
                "## 关键分析\n\n像燕麦、藜麦、红薯和全麦面包这类食物通常同时带来碳水、纤维和一定的饱腹感，因此比精制甜点或高糖面包更适合放进长期饮食结构中。对正在控制热量的人来说，它们更容易支持稳定执行，而不是只带来短时间饱足后又迅速饥饿。\n\n"
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
        conversation_context=(
            "[User] 减肥食物推荐\n"
            "[Assistant] 复合碳水化合物包括燕麦、藜麦、红薯和全麦面包。"
        ),
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
