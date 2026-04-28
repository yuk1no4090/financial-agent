from deerflow.runtime.runs.worker import _build_runtime_context


def test_build_runtime_context_copies_model_selection_from_configurable() -> None:
    context = _build_runtime_context(
        "thread-123",
        {
            "configurable": {
                "thread_id": "thread-123",
                "model_name": "financial-agent",
                "mode": "pro",
                "reasoning_effort": "medium",
                "__pregel_runtime": object(),
            }
        },
    )

    assert context["thread_id"] == "thread-123"
    assert context["model_name"] == "financial-agent"
    assert context["mode"] == "pro"
    assert context["reasoning_effort"] == "medium"
    assert "__pregel_runtime" not in context


def test_build_runtime_context_explicit_context_overrides_configurable() -> None:
    context = _build_runtime_context(
        "thread-456",
        {
            "configurable": {
                "thread_id": "thread-456",
                "model_name": "financial-agent",
                "mode": "pro",
            },
            "context": {
                "model_name": "glm",
                "mode": "flash",
                "subagent_enabled": False,
            },
        },
    )

    assert context["thread_id"] == "thread-456"
    assert context["model_name"] == "glm"
    assert context["mode"] == "flash"
    assert context["subagent_enabled"] is False
