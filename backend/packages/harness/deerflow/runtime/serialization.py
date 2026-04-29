"""Canonical serialization for LangChain / LangGraph objects.

Provides a single source of truth for converting LangChain message
objects, Pydantic models, and LangGraph state dicts into plain
JSON-serialisable Python structures.

Consumers: ``deerflow.runtime.runs.worker`` (SSE publishing) and
``app.gateway.routers.threads`` (REST responses).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - keeps serialization usable in minimal test envs
    BaseMessage = None  # type: ignore[assignment]


_OMIT = object()
_HIDDEN_MESSAGE_NAMES = {
    "financial_analysis_rewrite",
    "pure_model_rewrite",
    "router_financial_glm_guide",
    "router_context_memory_glm_guide",
    "router_report_skill_glm_guide",
    "router_general_glm_guide",
    "pure_glm_mode_guide",
    "financial_agent_direct_mode_guide",
    "financial_analysis_synthesis_guide",
}


def _should_hide_message_dict(obj: dict[str, Any]) -> bool:
    msg_type = str(obj.get("type") or "").lower()
    msg_name = str(obj.get("name") or "")
    return msg_type in {"human", "user"} and msg_name in _HIDDEN_MESSAGE_NAMES


def _should_hide_message_object(obj: Any) -> bool:
    msg_name = getattr(obj, "name", None)
    msg_type = str(getattr(obj, "type", "") or "").lower()
    return msg_type in {"human", "user"} and isinstance(msg_name, str) and msg_name in _HIDDEN_MESSAGE_NAMES


def serialize_lc_object(obj: Any) -> Any:
    """Recursively serialize a LangChain object to a JSON-serialisable dict."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        if _should_hide_message_dict(obj):
            return _OMIT
        return {k: value for k, raw in obj.items() if (value := serialize_lc_object(raw)) is not _OMIT}
    if isinstance(obj, (list, tuple)):
        return [value for item in obj if (value := serialize_lc_object(item)) is not _OMIT]
    if BaseMessage is not None and isinstance(obj, BaseMessage):
        if _should_hide_message_object(obj):
            return _OMIT
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 / older objects
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Last resort
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def serialize_channel_values(channel_values: dict[str, Any]) -> dict[str, Any]:
    """Serialize channel values, stripping internal LangGraph keys.

    Internal keys like ``__pregel_*`` and ``__interrupt__`` are removed
    to match what the LangGraph Platform API returns.
    """
    result: dict[str, Any] = {}
    for key, value in channel_values.items():
        if key.startswith("__pregel_") or key == "__interrupt__":
            continue
        result[key] = serialize_lc_object(value)
    return result


def serialize_messages_tuple(obj: Any) -> Any:
    """Serialize a messages-mode tuple ``(chunk, metadata)``."""
    if isinstance(obj, tuple) and len(obj) == 2:
        chunk, metadata = obj
        serialized_chunk = serialize_lc_object(chunk)
        if serialized_chunk is _OMIT:
            return None
        return [serialized_chunk, metadata if isinstance(metadata, dict) else {}]
    return serialize_lc_object(obj)


def serialize(obj: Any, *, mode: str = "") -> Any:
    """Serialize LangChain objects with mode-specific handling.

    * ``messages`` — obj is ``(message_chunk, metadata_dict)``
    * ``values`` — obj is the full state dict; ``__pregel_*`` keys stripped
    * everything else — recursive ``model_dump()`` / ``dict()`` fallback
    """
    if mode == "messages":
        return serialize_messages_tuple(obj)
    if mode == "values":
        return serialize_channel_values(obj) if isinstance(obj, dict) else serialize_lc_object(obj)
    return serialize_lc_object(obj)
