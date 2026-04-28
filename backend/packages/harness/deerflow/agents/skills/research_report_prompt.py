from __future__ import annotations

import json

from .research_report_schema import ReportPlan, ReportSkillInput


def _pretty_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_planner_prompt(
    skill_input: ReportSkillInput,
    *,
    topic: str,
    context_summary: str,
    financial_signal: dict | None,
) -> str:
    brief_hint = "只生成 3-4 个 section。" if skill_input.brief_report else "生成 5-6 个 section。"
    return (
        "你是一个报告规划器。请根据用户问题、主题和上下文，输出一份报告大纲。\n"
        "必须返回 JSON，不要返回 Markdown，不要解释。\n"
        "JSON 格式：\n"
        "{\n"
        '  "title": "报告标题",\n'
        '  "sections": [{"heading": "章节名", "goal": "该章节要完成什么"}]\n'
        "}\n\n"
        f"用户问题：\n{skill_input.user_query}\n\n"
        f"报告主题：\n{topic}\n\n"
        f"上下文摘要：\n{context_summary or '无'}\n\n"
        f"Memory 摘要：\n{skill_input.memory_context or '无'}\n\n"
        f"金融信号：\n{_pretty_json(financial_signal) if financial_signal else '无'}\n\n"
        "要求：\n"
        f"- {brief_hint}\n"
        "- 如果主题是金融或市场分析，应包含风险或后续观察模块。\n"
        "- 标题要自然，不要包含内部系统字段。\n"
        "- 输出语言保持和用户一致。\n"
        "- 不要提到 Router、Skill、Memory、JSON 以外的说明文字。\n"
    )


def build_writer_prompt(
    skill_input: ReportSkillInput,
    *,
    topic: str,
    outline: ReportPlan,
    context_summary: str,
    retrieved_context: list[dict],
    financial_signal: dict | None,
) -> str:
    brief_hint = "控制在 400-800 字左右，结构紧凑。" if skill_input.brief_report else "控制在 1200-2000 字左右，保证结构完整。"
    return (
        "你是一个报告生成器。请根据以下输入生成结构化 Markdown 报告。\n"
        "不要输出任何内部调试信息，不要提到 Router、Skill、Memory、model_strategy、financial_tool、tool call 或 JSON。\n"
        "如果没有证据支持，不要编造精确数字；可以使用定性表达，如“需要进一步验证”。\n\n"
        f"用户问题：\n{skill_input.user_query}\n\n"
        f"报告主题：\n{topic}\n\n"
        f"报告大纲：\n{_pretty_json({'title': outline.title, 'sections': [section.__dict__ for section in outline.sections]})}\n\n"
        f"上下文摘要：\n{context_summary or '无'}\n\n"
        f"Memory 摘要：\n{skill_input.memory_context or '无'}\n\n"
        f"检索证据：\n{_pretty_json(retrieved_context) if retrieved_context else '暂无外部检索证据'}\n\n"
        f"金融信号：\n{_pretty_json(financial_signal) if financial_signal else '无'}\n\n"
        "写作要求：\n"
        "- 输出必须是 Markdown。\n"
        "- 必须覆盖大纲中的每一个 section。\n"
        "- 如果是依赖上文的报告，必须承接上下文中的主题和前提。\n"
        f"- {brief_hint}\n"
        "- 语言与用户保持一致。\n"
    )


def build_rewrite_prompt(
    skill_input: ReportSkillInput,
    *,
    topic: str,
    outline: ReportPlan,
    report_markdown: str,
    issues: list[str],
) -> str:
    return (
        "你需要重写下面这份报告，使其满足格式和质量要求。\n"
        "只返回最终 Markdown，不要解释。\n"
        "不要提到 Router、Skill、Memory、model_strategy、financial_tool、tool call 或任何内部字段。\n\n"
        f"用户问题：\n{skill_input.user_query}\n\n"
        f"报告主题：\n{topic}\n\n"
        f"目标大纲：\n{_pretty_json({'title': outline.title, 'sections': [section.__dict__ for section in outline.sections]})}\n\n"
        f"当前报告：\n{report_markdown}\n\n"
        f"需要修正的问题：\n{_pretty_json(issues)}\n\n"
        "重写要求：\n"
        "- 保留主题一致性。\n"
        "- 修复缺失章节或内部信息泄漏。\n"
        "- 保持 Markdown 结构。\n"
        "- 语言与用户保持一致。\n"
    )
