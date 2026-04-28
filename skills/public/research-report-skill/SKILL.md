---
name: research-report-skill
description: Use this skill when the user asks for a structured report, research brief, analysis memo, or summary report. It is optimized for report generation with topic extraction, context inheritance, outline planning, concise or full-length report writing, and final leakage/format review.
---

# Research Report Skill

## Purpose

Use this skill for report-generation tasks that need more structure than a single free-form model response.

Typical triggers include:

- "请生成一份报告"
- "帮我写一个分析报告"
- "针对上面的话题整理成报告"
- "生成一个简短报告"
- "write a research report"

## Workflow

This skill follows five stages:

1. Topic extraction
2. Context building
3. Report outline planning
4. Report writing
5. Format and leakage review

## Core Rules

- Keep the report in the same language as the user.
- Never expose internal system fields such as router names, tool call markers, model strategy, memory flags, or debug headers.
- If context from earlier turns is referenced, inherit that context instead of restarting from scratch.
- If the user asks for a short or simple report, keep the structure compact.
- If no evidence is provided, avoid fabricated precise numbers; use qualitative phrasing instead.

## Full Report Structure

Use a stable structure such as:

- Title
- Executive Summary
- Background and Problem Definition
- Core Analysis
- Risks and Uncertainties
- Watchpoints or Follow-up Indicators
- Conclusion

## Brief Report Structure

When the user asks for a short report, prefer:

- Title
- Core Conclusion
- Key Analysis
- Risks
- Short Wrap-up

## Financial Notes

- For short financial event snippets, a separate financial signal tool may be used before writing the report.
- The financial signal is supporting input, not the final report generator.
- If the topic is non-financial, skip financial signal logic.
