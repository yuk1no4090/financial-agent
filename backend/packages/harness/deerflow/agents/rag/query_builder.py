from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")


def _compact(text: str, *, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}..."


def _extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip()
        lowered = token.casefold()
        if len(lowered) < 2 or lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if len(result) >= limit:
            break
    return result


class RagQueryBuilder:
    def build(
        self,
        *,
        current_query: str,
        route: str,
        memory_context: str | None,
        conversation_context: str | None,
        report_topic: str | None = None,
    ) -> str:
        parts: list[str] = []
        if route == "report_skill_glm" and report_topic:
            parts.append(_compact(report_topic, limit=80))

        parts.append(_compact(current_query, limit=220))

        if route == "context_memory_glm" and conversation_context:
            lines = [line.strip() for line in conversation_context.splitlines() if line.strip()]
            if lines:
                parts.append(_compact(lines[-1], limit=140))

        memory_keywords = _extract_keywords(memory_context or "", limit=5)
        if memory_keywords:
            parts.append(" ".join(memory_keywords))

        return " | ".join(part for part in parts if part).strip()
