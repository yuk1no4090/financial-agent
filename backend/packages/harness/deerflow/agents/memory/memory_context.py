from __future__ import annotations

from deerflow.agents.memory.memory_schema import MemoryBundle, MemoryRecord


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


class MemoryContextAssembler:
    def build(self, records: list[MemoryRecord]) -> MemoryBundle:
        relevant_facts: list[str] = []
        prior_decisions: list[str] = []
        prior_results: list[str] = []
        open_tasks: list[str] = []
        constraints: list[str] = []
        source_memory_ids: list[str] = []

        for record in records:
            source_memory_ids.append(record.id)
            rendered = record.summary.strip() or record.content.strip()
            if not rendered:
                continue

            if record.memory_type == "design_decision":
                prior_decisions.append(rendered)
            elif record.memory_type in {"experiment_result", "report_summary"}:
                prior_results.append(rendered)
            elif record.memory_type == "open_task":
                open_tasks.append(rendered)
            elif record.memory_type in {"constraint", "user_requirement"}:
                constraints.append(rendered)
            else:
                relevant_facts.append(rendered)

        prior_decisions = _dedupe_keep_order(prior_decisions)[:3]
        relevant_facts = _dedupe_keep_order(relevant_facts)[:3]
        prior_results = _dedupe_keep_order(prior_results)[:3]
        open_tasks = _dedupe_keep_order(open_tasks)[:3]
        constraints = _dedupe_keep_order(constraints)[:3]

        summary_parts = prior_decisions[:1] + relevant_facts[:1] + prior_results[:1] + open_tasks[:1] + constraints[:1]
        summary = "；".join(summary_parts[:3])
        warnings = ["如果记忆与当前用户输入冲突，以当前用户输入为准。"] if source_memory_ids else []

        return MemoryBundle(
            summary=summary,
            relevant_facts=relevant_facts,
            prior_decisions=prior_decisions,
            prior_results=prior_results,
            open_tasks=open_tasks,
            constraints=constraints,
            warnings=warnings,
            source_memory_ids=source_memory_ids,
        )
