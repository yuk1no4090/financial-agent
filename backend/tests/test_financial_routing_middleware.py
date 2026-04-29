import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import deerflow.agents.middlewares.financial_routing_middleware as frm
from deerflow.agents.memory import MemoryBundle
from deerflow.agents.middlewares.financial_routing_middleware import (
    FinancialRoutingMiddleware,
    _build_direct_financial_answer,
    _build_pure_model_sanitized_update,
    _build_report_skill_input,
    _current_turn_finma_candidate,
    _needs_financial_answer_rewrite,
    _route_decision,
)
from deerflow.agents.rag import RagBundle, RetrievedEvidence
from deerflow.community.finma.tools import financial_analysis_tool


def _finma_tool_message(payload: dict) -> ToolMessage:
    return ToolMessage(
        name="financial_analysis",
        tool_call_id="financial-analysis-test",
        content=json.dumps(payload, ensure_ascii=False),
    )


def test_direct_financial_answer_fallback_is_natural_prose() -> None:
    payload = {
        "provider": "finma_ensemble",
        "task": "sentiment",
        "model_strategy": "v3_and_base",
        "model_results": [
            {
                "model_used": "finma-sentiment-v3",
                "label": "positive",
            },
            {
                "model_used": "finma-7b-nlp",
                "label": "positive",
                "summary": "Apple's first-quarter growth is constructive.",
                "rationale": "Revenue growth suggests demand and execution held up better than feared.",
                "market_implication": "If expectations were lower going in, the update can support sentiment and estimates.",
                "watch_items": ["guidance", "margin", "services growth"],
            },
        ],
        "synthesis": {
            "label": "positive",
            "agreement": "aligned",
            "explanation": "Revenue growth suggests demand and execution held up better than feared.",
            "market_implication": "If expectations were lower going in, the update can support sentiment and estimates.",
            "watch_items": ["guidance", "margin", "services growth"],
        },
    }
    messages = [
        HumanMessage(content="苹果公司财报显示第一季度增长5%"),
        _finma_tool_message(payload),
    ]

    answer = _build_direct_financial_answer(messages)

    assert answer is not None
    assert "如果只基于“苹果公司财报显示第一季度增长5%”这条信息来看，整体偏积极。" in answer.content
    assert "结论：" not in answer.content
    assert "理由：" not in answer.content
    assert "影响：" not in answer.content
    assert "financial_analysis" not in answer.content
    assert "后续更值得继续跟踪的是guidance、margin、services growth" in answer.content
    assert "If expectations were lower going in" not in answer.content


def test_financial_answer_rewrite_detects_internal_template() -> None:
    payload = {
        "provider": "finma_ensemble",
        "task": "sentiment",
        "model_strategy": "v3_and_base",
        "model_results": [{"model_used": "finma-sentiment-v3", "label": "positive"}],
    }
    messages = [
        HumanMessage(content="苹果公司财报显示第一季度增长5%"),
        _finma_tool_message(payload),
        AIMessage(content='使用 "financial_analysis" 工具\n\n结论：这条信息整体偏积极。\n\n理由：增长说明基本面改善。\n\n影响：可能提振股价。'),
    ]

    assert _needs_financial_answer_rewrite(messages) is True


def test_pure_glm_after_model_requests_rewrite_before_hardcoded_fallback() -> None:
    middleware = FinancialRoutingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="历史上类似冲突对油价影响如何？"),
            AIMessage(content='<function=web_search>{"query":"oil conflict history"}</function>'),
        ]
    }
    runtime = SimpleNamespace(context={"model_name": "glm"})

    result = middleware.after_model(state, runtime)

    assert result is not None
    assert result["jump_to"] == "model"
    assert result["messages"][0].name == "pure_model_rewrite"


def test_pure_glm_sanitize_after_rewrite_prefers_cleaned_answer() -> None:
    messages = [
        HumanMessage(content="历史上类似冲突对油价影响如何？"),
        HumanMessage(name="pure_model_rewrite", content="rewrite"),
        AIMessage(content="<function=web_search></function>类似冲突通常会先通过供应中断预期和风险溢价推高油价，随后再看实际供给损失和库存缓冲。"),
    ]

    sanitized = _build_pure_model_sanitized_update(messages)

    assert sanitized is not None
    assert sanitized.content.startswith("类似冲突通常会先通过供应中断预期")
    assert "不能调用外部工具" not in sanitized.content


def test_financial_analysis_tool_mock_returns_synthesis_payload() -> None:
    payload = json.loads(
        financial_analysis_tool.invoke(
            {
                "text": "Apple reported first-quarter revenue growth of 5% and raised guidance.",
                "task": "sentiment",
                "model_strategy": "v3_and_base",
                "use_mock": True,
            }
        )
    )

    assert payload["provider"] == "mock"
    assert payload["model_strategy"] == "v3_and_base"
    assert payload["model_results"]
    assert "synthesis" in payload
    assert payload["synthesis"]["label"] in {"positive", "negative", "neutral", "mixed"}
    assert isinstance(payload["synthesis"]["watch_items"], list)


def test_current_turn_finma_candidate_skips_fact_ranking_query() -> None:
    messages = [HumanMessage(content="世界市值前五的公司是哪些")]

    candidate = _current_turn_finma_candidate(messages)

    assert candidate == ""


def test_current_turn_finma_candidate_accepts_short_financial_snippet() -> None:
    messages = [HumanMessage(content="苹果公司第一季度财报增长速度快")]

    candidate = _current_turn_finma_candidate(messages)

    assert candidate == "苹果公司第一季度财报增长速度快"


def test_route_decision_for_report_request_prefers_skill_glm() -> None:
    decision = _route_decision([HumanMessage(content="请基于中东战争对油价影响写一份研究报告")])

    assert decision.route == "report_skill_glm"
    assert decision.skill_enabled is True
    assert decision.skill_name == "research-report-skill"
    assert decision.memory_enabled is False
    assert decision.rag_enabled is True
    assert decision.rag_source_type == "finance_docs"


def test_route_decision_for_contextual_report_request_enables_skill_and_memory() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="分析中东战争对石油市场的影响"),
            AIMessage(content="这会先通过风险溢价和潜在供给扰动影响油价。"),
            HumanMessage(content="针对刚刚的问题为我生成一个报告"),
        ]
    )

    assert decision.route == "report_skill_glm"
    assert decision.skill_enabled is True
    assert decision.memory_enabled is True
    assert decision.skill_name == "research-report-skill"
    assert decision.rag_enabled is True


def test_route_decision_for_project_doc_question_enables_rag() -> None:
    decision = _route_decision([HumanMessage(content="Router 文档里当前有哪些路线")])

    assert decision.route == "general_glm"
    assert decision.rag_enabled is True
    assert decision.rag_source_type == "project_docs"


def test_route_decision_for_contextual_brief_report_request_enables_memory_and_brief_mode() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="减肥食物推荐"),
            AIMessage(content="复合碳水化合物包括燕麦、藜麦、红薯和全麦面包。"),
            HumanMessage(content="对于你刚刚提到的复合碳水，给我简单生成一份报告"),
        ]
    )

    assert decision.route == "report_skill_glm"
    assert decision.memory_enabled is True
    assert decision.skill_enabled is True
    assert decision.brief_report is True


def test_route_decision_for_follow_up_prefers_memory_glm() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="苹果公司第一季度财报增长速度快"),
            AIMessage(content="这说明短线情绪偏积极。"),
            HumanMessage(content="那第二家公司呢"),
        ]
    )

    assert decision.route == "context_memory_glm"
    assert decision.memory_enabled is True


def test_route_decision_for_natural_context_follow_up_prefers_memory_glm() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="减肥食物推荐"),
            AIMessage(content="可以优先考虑鸡胸肉和高纤维蔬菜。"),
            HumanMessage(content="你上面提到的鸡胸肉每100g含有多少蛋白质"),
        ]
    )

    assert decision.route == "context_memory_glm"
    assert decision.memory_enabled is True


def test_route_decision_for_company_reference_follow_up_prefers_memory_glm() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="高通公司市值增长速度放缓"),
            AIMessage(content="这可能意味着它面临来自联发科等竞争对手的压力。"),
            HumanMessage(content="你刚刚提到联发科公司，请为我简单介绍一下"),
        ]
    )

    assert decision.route == "context_memory_glm"
    assert decision.memory_enabled is True


def test_route_decision_for_supply_risk_follow_up_prefers_memory_glm() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="分析美以谈判破裂对石油市场的影响"),
            AIMessage(content="核心影响之一是中东供应风险可能推高风险溢价。"),
            HumanMessage(content="针对你刚刚提到的供应风险展开讲讲"),
        ]
    )

    assert decision.route == "context_memory_glm"
    assert decision.memory_enabled is True


def test_financial_agent_after_model_hides_debug_header(monkeypatch) -> None:
    monkeypatch.setattr(
        frm,
        "get_app_config",
        lambda: SimpleNamespace(get_model_config=lambda name: SimpleNamespace(model="glm-4.5")),
    )

    middleware = FinancialRoutingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="苹果公司显示第一季度财报增长5%"),
            _finma_tool_message(
                {
                    "provider": "finma_ensemble",
                    "task": "sentiment",
                    "model_strategy": "v3_and_base",
                    "models_requested": ["finma-sentiment-v3", "finma-7b-nlp"],
                    "model_results": [
                        {"model_used": "finma-sentiment-v3", "label": "positive"},
                        {"model_used": "finma-7b-nlp", "label": "positive"},
                    ],
                }
            ),
            AIMessage(content="这条信息整体偏积极，短线情绪和基本面解读方向一致。"),
        ]
    }
    runtime = SimpleNamespace(context={"model_name": "financial-agent"})

    result = middleware.after_model(state, runtime)

    assert result is not None
    content = result["messages"][0].content
    assert "当前路由：" not in content
    assert "当前调用模型：" not in content
    assert "如果只基于“苹果公司显示第一季度财报增长5%”这条信息来看，整体偏积极。" in content


def test_financial_agent_after_model_does_not_reuse_previous_turn_finma_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        frm,
        "get_app_config",
        lambda: SimpleNamespace(get_model_config=lambda name: SimpleNamespace(model="glm-4.5")),
    )

    middleware = FinancialRoutingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="苹果公司第一季度财报增长速度快"),
            _finma_tool_message(
                {
                    "provider": "finma_ensemble",
                    "task": "sentiment",
                    "model_strategy": "v3_and_base",
                    "models_requested": ["finma-sentiment-v3", "finma-7b-nlp"],
                    "model_results": [
                        {"model_used": "finma-sentiment-v3", "label": "positive"},
                        {"model_used": "finma-7b-nlp", "label": "positive"},
                    ],
                    "synthesis": {
                        "task": "sentiment",
                        "model_strategy": "v3_and_base",
                        "label": "positive",
                        "agreement": "aligned",
                    },
                }
            ),
            AIMessage(content="如果只基于这条信息来看，整体偏积极。"),
            HumanMessage(content="世界市值前五的公司是哪些"),
            AIMessage(content="通常会包括苹果、微软、Alphabet、亚马逊和英伟达。"),
        ]
    }
    runtime = SimpleNamespace(context={"model_name": "financial-agent"})

    result = middleware.after_model(state, runtime)

    assert result is not None
    content = result["messages"][0].content
    assert "当前路由：" not in content
    assert "当前调用模型：" not in content
    assert "finma-sentiment-v3" not in content
    assert "v3_and_base" not in content
    assert "通常会包括苹果、微软" in content


def test_financial_agent_prefers_direct_synthesis_for_short_v3_snippet(monkeypatch) -> None:
    monkeypatch.setattr(
        frm,
        "get_app_config",
        lambda: SimpleNamespace(get_model_config=lambda name: SimpleNamespace(model="glm-4.5")),
    )

    middleware = FinancialRoutingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="苹果公司显示第一季度财报增长5%"),
            _finma_tool_message(
                {
                    "provider": "finma_ensemble",
                    "task": "sentiment",
                    "model_strategy": "v3_and_base",
                    "models_requested": ["finma-sentiment-v3", "finma-7b-nlp"],
                    "model_results": [
                        {"model_used": "finma-sentiment-v3", "label": "positive"},
                        {
                            "model_used": "finma-7b-nlp",
                            "label": "positive",
                            "summary": "苹果第一季度增长信号偏积极。",
                            "rationale": "增长至少说明本季经营动能没有走弱，短线更容易被市场理解成偏正面的增量信息。",
                            "market_implication": "如果市场此前预期偏保守，这条信息有助于支撑情绪和盈利预期。",
                            "watch_items": ["后续指引", "利润率", "增长持续性"],
                        },
                    ],
                    "synthesis": {
                        "task": "sentiment",
                        "model_strategy": "v3_and_base",
                        "label": "positive",
                        "agreement": "aligned",
                        "summary": "苹果第一季度增长信号偏积极。",
                        "explanation": "增长至少说明本季经营动能没有走弱，短线更容易被市场理解成偏正面的增量信息。",
                        "market_implication": "如果市场此前预期偏保守，这条信息有助于支撑情绪和盈利预期。",
                        "watch_items": ["后续指引", "利润率", "增长持续性"],
                    },
                }
            ),
            AIMessage(content=("苹果公司第一季度财报显示5%的增长，这是一个积极的财务表现。从市场情绪分析来看，这一增长信号对苹果公司的股价和投资者情绪具有正面影响。")),
        ]
    }
    runtime = SimpleNamespace(context={"model_name": "financial-agent"})

    result = middleware.after_model(state, runtime)

    assert result is not None
    content = result["messages"][0].content
    assert "当前路由：" not in content
    assert "当前调用模型：" not in content
    assert "如果只基于“苹果公司显示第一季度财报增长5%”这条信息来看，整体偏积极。" in content
    assert "短线情绪信号和更宽泛的基本面解读方向基本一致。" in content
    assert "股价和投资者情绪具有正面影响" not in content
    assert "If the market was not already" not in content


def test_glm_after_model_hides_debug_header(monkeypatch) -> None:
    monkeypatch.setattr(
        frm,
        "get_app_config",
        lambda: SimpleNamespace(get_model_config=lambda name: SimpleNamespace(model="glm-4.5")),
    )

    middleware = FinancialRoutingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="告诉我全球市值前五的公司"),
            AIMessage(content="通常会包括 Apple、Microsoft、NVIDIA、Alphabet 和 Amazon。"),
        ]
    }
    runtime = SimpleNamespace(context={"model_name": "glm"})

    result = middleware.after_model(state, runtime)

    assert result is not None
    content = result["messages"][0].content
    assert "当前路由：" not in content
    assert "当前调用模型：" not in content
    assert "通常会包括 Apple" in content


def test_route_decision_for_report_reference_follow_up_prefers_memory_not_skill() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="针对你刚刚提到的减肥食物为我生成一份报告"),
            AIMessage(content="# 减肥食物推荐报告\n\n其中提到了低糖水果。"),
            HumanMessage(content="针对你生成的报告中的低糖水果展开讲讲"),
        ]
    )

    assert decision.route == "context_memory_glm"
    assert decision.memory_enabled is True
    assert decision.skill_enabled is False


def test_route_decision_for_contextual_report_generation_enables_memory_and_skill() -> None:
    decision = _route_decision(
        [
            HumanMessage(content="减肥食物推荐"),
            AIMessage(content="蔬菜类可以优先考虑西兰花、黄瓜和胡萝卜。"),
            HumanMessage(content="对你刚刚提到的蔬菜类，为我生成一个简单的报告"),
        ]
    )

    assert decision.route == "report_skill_glm"
    assert decision.memory_enabled is True
    assert decision.skill_enabled is True


def test_wrap_model_call_runs_report_skill_directly(monkeypatch) -> None:
    class FakeSkill:
        def __init__(self, model_name: str | None = None, **_: object) -> None:
            self.model_name = model_name

        def run_sync(self, skill_input):
            assert skill_input.memory_enabled is True
            assert skill_input.brief_report is True
            return SimpleNamespace(markdown="# 复合碳水报告\n\n## 核心结论\n\n适合减脂阶段作为稳定能量来源。")

    monkeypatch.setattr(frm, "ResearchReportSkill", FakeSkill)

    middleware = FinancialRoutingMiddleware()
    request = SimpleNamespace(
        messages=[
            HumanMessage(content="减肥食物推荐"),
            AIMessage(content="复合碳水化合物包括燕麦、藜麦、红薯和全麦面包。"),
            HumanMessage(content="对于你刚刚提到的复合碳水，给我简单生成一份报告"),
        ],
        runtime=SimpleNamespace(context={"model_name": "financial-agent"}),
        tools=[],
        override=lambda **kwargs: SimpleNamespace(
            messages=kwargs.get("messages", request.messages),
            runtime=request.runtime,
            tools=kwargs.get("tools", request.tools),
            override=request.override,
        ),
    )

    def _handler(_: object) -> AIMessage:
        raise AssertionError("handler should not be called for report skill route")

    result = middleware.wrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    assert "# 复合碳水报告" in result.content


@pytest.mark.asyncio
async def test_awrap_model_call_runs_report_skill_directly(monkeypatch) -> None:
    class FakeSkill:
        def __init__(self, model_name: str | None = None, **_: object) -> None:
            self.model_name = model_name

        async def run(self, skill_input):
            assert skill_input.memory_enabled is True
            assert skill_input.brief_report is False
            return SimpleNamespace(markdown="# 供应风险报告\n\n## 执行摘要\n\n中东供应风险会先推高风险溢价。")

    monkeypatch.setattr(frm, "ResearchReportSkill", FakeSkill)

    middleware = FinancialRoutingMiddleware()
    request = SimpleNamespace(
        messages=[
            HumanMessage(content="分析中东战争对石油市场的影响"),
            AIMessage(content="核心影响之一是中东供应风险可能推高风险溢价。"),
            HumanMessage(content="针对你刚刚提到的供应风险生成一份报告"),
        ],
        runtime=SimpleNamespace(context={"model_name": "financial-agent"}),
        tools=[],
        override=lambda **kwargs: SimpleNamespace(
            messages=kwargs.get("messages", request.messages),
            runtime=request.runtime,
            tools=kwargs.get("tools", request.tools),
            override=request.override,
        ),
    )

    async def _handler(_: object) -> AIMessage:
        raise AssertionError("handler should not be called for report skill route")

    result = await middleware.awrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    assert "# 供应风险报告" in result.content


def test_build_report_skill_input_includes_explicit_memory_context(monkeypatch) -> None:
    class FakeMemoryManager:
        def retrieve_for_route(self, **_: object) -> MemoryBundle:
            return MemoryBundle(
                prior_decisions=["已确认项目主线：Router + Report Skill + Memory。"],
                open_tasks=["下一步需要实现显式 Memory Retrieval。"],
                source_memory_ids=["mem-1", "mem-2"],
            )

    monkeypatch.setattr(frm, "get_memory_manager", lambda: FakeMemoryManager())

    messages = [
        HumanMessage(content="分析中东战争对石油市场的影响"),
        AIMessage(content="核心影响之一是中东供应风险可能推高风险溢价。"),
        HumanMessage(content="针对你刚刚提到的供应风险生成一份报告"),
    ]
    decision = _route_decision(messages)

    skill_input = _build_report_skill_input(messages, {"model_name": "financial-agent", "thread_id": "thread-1"}, decision)

    assert skill_input.memory_enabled is True
    assert skill_input.rag_enabled is True
    assert skill_input.require_citations is True
    assert "Router + Report Skill + Memory" in skill_input.memory_context
    assert "显式 Memory Retrieval" in skill_input.memory_context


def test_wrap_model_call_injects_explicit_memory_for_context_route(monkeypatch) -> None:
    class FakeMemoryManager:
        def retrieve_for_route(self, **_: object) -> MemoryBundle:
            return MemoryBundle(
                prior_decisions=["已确认项目主线：Router + Report Skill + Memory。"],
                source_memory_ids=["mem-1"],
            )

    monkeypatch.setattr(frm, "get_memory_manager", lambda: FakeMemoryManager())

    middleware = FinancialRoutingMiddleware()
    captured_messages = {}
    request = SimpleNamespace(
        messages=[
            HumanMessage(content="memory 方案怎么做"),
            AIMessage(content="建议用 Router + Report Skill + Memory 作为主线。"),
            HumanMessage(content="把刚刚那个方案继续展开一下"),
        ],
        runtime=SimpleNamespace(context={"model_name": "financial-agent", "thread_id": "thread-1"}),
        tools=[],
        override=lambda **kwargs: SimpleNamespace(
            messages=kwargs.get("messages", request.messages),
            runtime=request.runtime,
            tools=kwargs.get("tools", request.tools),
            override=request.override,
        ),
    )

    def _handler(updated_request):
        captured_messages["messages"] = updated_request.messages
        return AIMessage(content="继续展开。")

    result = middleware.wrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    names = [getattr(message, "name", "") for message in captured_messages["messages"]]
    assert "router_explicit_memory_context" in names
    injected = next(message for message in captured_messages["messages"] if getattr(message, "name", "") == "router_explicit_memory_context")
    assert "Router + Report Skill + Memory" in injected.content


def test_wrap_model_call_injects_rag_evidence_for_project_doc_route(monkeypatch) -> None:
    class FakeRagService:
        def search(self, request):
            assert request.source_type == "project_docs"
            return RagBundle(
                query=request.query,
                rewritten_query=request.query,
                evidences=[
                    RetrievedEvidence(
                        chunk_id="router-1",
                        doc_id="router",
                        title="Router 机制说明",
                        section="总体设计",
                        source_path="/tmp/router.md",
                        text="Router 当前支持 financial_finma、financial_glm、general_glm、context_memory_glm 和 report_skill_glm。",
                        score=0.91,
                        rank=1,
                    )
                ],
                summary="检索到 Router 证据。",
                used=True,
                source_type="project_docs",
            )

    monkeypatch.setattr(frm, "get_rag_service", lambda: FakeRagService())

    middleware = FinancialRoutingMiddleware()
    captured_messages = {}
    request = SimpleNamespace(
        messages=[HumanMessage(content="Router 文档里当前有哪些路线")],
        runtime=SimpleNamespace(context={"model_name": "financial-agent", "thread_id": "thread-1"}),
        tools=[],
        override=lambda **kwargs: SimpleNamespace(
            messages=kwargs.get("messages", request.messages),
            runtime=request.runtime,
            tools=kwargs.get("tools", request.tools),
            override=request.override,
        ),
    )

    def _handler(updated_request):
        captured_messages["messages"] = updated_request.messages
        return AIMessage(content="共有五条路线。")

    result = middleware.wrap_model_call(request, _handler)

    assert isinstance(result, AIMessage)
    names = [getattr(message, "name", "") for message in captured_messages["messages"]]
    assert "router_rag_evidence_context" in names
    injected = next(message for message in captured_messages["messages"] if getattr(message, "name", "") == "router_rag_evidence_context")
    assert "[E1]" in injected.content
    assert "Router 当前支持" in injected.content


def test_after_agent_writes_explicit_task_memory(monkeypatch) -> None:
    captured = {}

    def _fake_enqueue(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(frm, "_enqueue_explicit_task_memory_write", _fake_enqueue)

    middleware = FinancialRoutingMiddleware()
    runtime = SimpleNamespace(context={"model_name": "financial-agent", "thread_id": "thread-1", "user_id": "user-1"})
    state = {
        "messages": [
            HumanMessage(content="针对刚刚的 memory 方案生成一份报告"),
            AIMessage(content="# Memory 方案报告\n\n## 核心结论\n\n用显式检索来承接上文。"),
        ]
    }

    middleware.after_agent(state, runtime)

    assert captured["thread_id"] == "thread-1"
    assert captured["user_id"] == "user-1"
    assert captured["route"] == "report_skill_glm"
    assert "显式检索" in captured["answer"]
