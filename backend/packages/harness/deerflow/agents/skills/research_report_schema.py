from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSkillInput:
    user_query: str
    topic: str | None = None
    language: str = "zh"
    brief_report: bool = False
    memory_enabled: bool = False
    conversation_context: str = ""
    memory_context: str = ""
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    financial_signal: dict[str, Any] | None = None
    report_type: str = "general_report"
    audience: str = "student_project"
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportPlanSection:
    heading: str
    goal: str


@dataclass
class ReportPlan:
    title: str
    sections: list[ReportPlanSection] = field(default_factory=list)


@dataclass
class ReportReview:
    format_complete: bool = True
    context_consistent: bool = True
    internal_trace_leaked: bool = False
    too_long: bool = False
    too_short: bool = False
    missing_sections: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    needs_rewrite: bool = False


@dataclass
class ReportSkillOutput:
    title: str
    markdown: str
    used_memory: bool = False
    used_rag: bool = False
    used_financial_signal: bool = False
    review_passed: bool = True
    outline: ReportPlan | None = None
    review: ReportReview | None = None
