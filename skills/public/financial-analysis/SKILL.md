---
name: financial-analysis
description: Use this skill for financial research workflows such as earnings analysis, valuation memos, peer comparison, risk reviews, market briefs, and portfolio diagnosis. It defines source discipline, report structure, FinMA expert routing, and risk disclaimers for investment research outputs.
---

# Financial Analysis Skill

## Purpose

Use this skill to produce structured, source-grounded financial analysis. The lead agent owns planning, retrieval, synthesis, and final writing. FinMA is a short-context expert module, not the lead agent.

## Core Rules

- Separate facts, model inference, assumptions, and unknowns.
- Cite sources for factual claims gathered through search, fetch, uploaded files, or user-provided documents.
- Do not present output as personalized investment advice.
- Use `financial_analysis` for short financial text snippets when sentiment, event impact, risk type, management tone, or financial signal extraction is useful.
- Keep each `financial_analysis` input focused and short. Do not send full filings, full transcripts, or full chat history.
- If data is unavailable, state the gap explicitly instead of inventing metrics.

## FinMA Routing

Call `financial_analysis` when the task needs one of these short-context judgments:

- `sentiment`: classify financial news, headlines, or management commentary.
- `risk_classification`: identify risk type and likely impact.
- `event_impact`: assess whether an event is positive, negative, mixed, or uncertain.
- `management_tone`: assess management confidence, caution, or uncertainty.
- `financial_signal_extraction`: extract revenue, margin, cash flow, guidance, demand, liquidity, regulation, or valuation signals.

Recommended request shape:

```json
{
  "task": "event_impact",
  "ticker": "NVDA",
  "text": "NVIDIA reported strong data center growth...",
  "output_schema": "label,impact_direction,affected_factors,confidence,rationale"
}
```

Expected response shape:

```json
{
  "label": "positive",
  "impact_direction": "positive",
  "affected_factors": ["revenue_growth", "margin_expectation"],
  "confidence": 0.82,
  "rationale": "The passage indicates strong demand in data center revenue."
}
```

## Standard Workflow

1. Restate the task and identify ticker(s), period, geography, and requested output.
2. Gather sources using web search/fetch or user-uploaded files.
3. Extract key facts and metrics. Mark missing metrics clearly.
4. Call `financial_analysis` on relevant short snippets for sentiment, risk, tone, and event impact.
5. Synthesize a report using the relevant template below.
6. Add a source list and a short non-advice disclaimer.

## Earnings Analysis Template

Use these sections:

- Executive Summary
- Business and Revenue Drivers
- Financial Performance
- Management Tone
- Catalysts
- Risks
- Investment View
- Sources

## Valuation Memo Template

Use these sections:

- Executive Summary
- Business Quality
- Key Financials
- Valuation Framework
- Peer Multiples or Scenario Assumptions
- Catalysts
- Risks
- Sources

## Peer Comparison Template

Use these sections:

- Executive Summary
- Comparison Table
- Growth
- Margins and Profitability
- Valuation
- Competitive Position
- Catalysts and Risks
- Sources

## Risk Review Template

Use these sections:

- Executive Summary
- Ranked Risk Table
- Accounting and Reporting Risk
- Liquidity and Balance Sheet Risk
- Regulatory Risk
- Competitive Risk
- Macro Sensitivity
- Monitoring Checklist
- Sources

## Market Brief Template

Use these sections:

- Executive Summary
- Key Market Moves
- Event Drivers
- Financial Signal and Sentiment
- What to Watch Next
- Sources

## Output Requirements

- Use concise tables where they improve comparison.
- Include a `Sources` section with links or file names.
- Include `Assumptions and Unknowns` if important data was unavailable.
- End with: `This is research support, not personalized investment advice.`
