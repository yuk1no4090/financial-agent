# FinAgent Workbench PRD

## 1. Overview

FinAgent Workbench is an agentic financial analysis system built on DeerFlow and PIXIU. The system uses a long-context general LLM as the orchestration model and uses PIXIU/FinMA as a domain-tuned financial expert module for short-context financial reasoning tasks.

The product goal is not to make FinMA the global agent brain. FinMA v0.1 has a short context window and is not suitable for long-horizon planning, multi-source synthesis, or full financial report generation. Instead, FinAgent treats FinMA as a specialist module that can be trained, evaluated, and compared against other models inside a larger agent workflow.

## 2. Problem Statement

Financial analysis tasks require long-context orchestration and domain-specific judgment at the same time.

Long-context orchestration is needed for:

- Reading long filings, earnings transcripts, news collections, and tool outputs.
- Maintaining multi-step agent state.
- Coordinating retrieval, calculation, critique, and report generation.
- Producing structured investment memos with citations.

Domain-specific financial judgment is needed for:

- Financial sentiment classification.
- Risk factor identification.
- Event impact analysis.
- Management tone analysis.
- Financial terminology interpretation.
- Short-context financial question answering.

PIXIU/FinMA is valuable for the second category, but its context length makes it a poor fit for the first category. The system architecture must reflect this division.

## 3. Product Positioning

FinAgent Workbench is a hybrid agentic financial analysis platform:

- DeerFlow provides the agent runtime, UI, tool execution, skills, memory, artifacts, and sub-agent orchestration.
- GLM, DeepSeek, Qwen, or another long-context model acts as the lead agent.
- PIXIU/FinMA acts as a financial expert module.
- Financial tools provide filings, news, market data, and calculations.
- FinBen and custom tasks provide benchmark evaluation.

Recommended project description:

> Agentic Financial Analysis with Long-Context Orchestration and Domain-Tuned Financial Expert Modules.

Short version:

> Agentic Financial Analysis with Domain-Tuned LLM Modules.

## 4. Goals

### 4.1 Product Goals

- Provide a web-based financial analysis workspace.
- Let users request earnings analysis, valuation memos, risk reviews, peer comparison, market briefs, and portfolio diagnosis.
- Generate structured investment research outputs with clear sections and source references.
- Support model routing between a lead model and a financial expert model.
- Support future training and evaluation of the FinMA expert module.

### 4.2 Research Goals

- Demonstrate that a domain-tuned financial LLM is useful as an expert module even when it is not suitable as a lead agent.
- Compare general LLM-only workflows against hybrid workflows.
- Evaluate original FinMA against post-trained FinMA.
- Build a measurable workload around agent orchestration, tool integration, post-training, and evaluation.

### 4.3 Non-Goals

- Do not force FinMA to be the main DeerFlow model.
- Do not claim FinMA can handle long financial documents end to end.
- Do not build a production investment advisor.
- Do not provide investment recommendations without risk disclaimers and source constraints.
- Do not rely only on UI changes as the project contribution.

## 5. Target Users

### 5.1 Primary Users

- Students or researchers building an agentic financial analysis project.
- Developers evaluating domain-tuned LLMs in finance.
- Analysts who need structured summaries from financial texts.

### 5.2 Secondary Users

- Instructors or reviewers evaluating the system.
- Research teammates comparing model performance.
- Demo users testing end-to-end financial workflows.

## 6. Core User Scenarios

### Scenario 1: Earnings Analysis

User enters:

```text
Analyze Apple's latest earnings. Cover revenue drivers, margin trend, cash flow, guidance, catalysts, risks and source-backed conclusions.
```

System behavior:

- Lead agent decomposes the task.
- Retrieval tools collect filings, earnings news, and market context.
- Financial calculator extracts or computes key metrics.
- FinMA expert module classifies sentiment, risks, event impact, and financial signals on short chunks.
- Lead agent synthesizes a final report.

Expected output:

- Executive Summary
- Business and Revenue Drivers
- Financial Performance
- Management Tone
- Catalysts
- Risks
- Investment View
- Sources

### Scenario 2: Peer Comparison

User enters:

```text
Compare NVIDIA and AMD across growth, margins, valuation, competitive position, catalysts and risks.
```

System behavior:

- Lead agent collects data for both companies.
- Tools normalize financial metrics.
- FinMA expert module evaluates short text snippets from news, transcripts, and filings.
- Lead agent produces a comparison table and narrative conclusion.

### Scenario 3: Risk Review

User enters:

```text
Review Tesla's main investment risks, including accounting, liquidity, regulation, competition and macro sensitivity.
```

System behavior:

- Lead agent collects risk factor disclosures and news.
- FinMA expert module classifies risk types and impact direction.
- Lead agent produces a ranked risk memo.

### Scenario 4: Model Comparison Experiment

Researcher selects a benchmark task and runs:

- GLM only.
- DeepSeek only.
- GLM + original FinMA module.
- GLM + post-trained FinMA module.
- DeepSeek + post-trained FinMA module.

Expected output:

- Accuracy or scoring metrics.
- Latency and cost.
- Error categories.
- Qualitative examples.

## 7. System Architecture

```text
User
  -> FinAgent Workbench UI
  -> DeerFlow Lead Agent
       -> Long-context model: GLM / DeepSeek / Qwen
       -> Tools:
            - Web search
            - Web fetch
            - Filings/news/market data retrieval
            - Financial calculator
            - FinMA expert module
       -> Skills:
            - financial-analysis
            - earnings-analysis
            - valuation
            - risk-review
       -> Artifacts:
            - investment memo
            - metric tables
            - citations
            - evaluation reports
```

## 8. Model Strategy

### 8.1 Lead Model

The lead model handles:

- Long context.
- Tool planning.
- Agent orchestration.
- Multi-step reasoning.
- Report synthesis.
- User interaction.

Candidate lead models:

- GLM long-context model.
- DeepSeek model.
- Qwen long-context model.
- Other OpenAI-compatible long-context models.

Selection criteria:

- Context length.
- Tool calling reliability.
- Cost.
- Latency.
- Chinese/English support.
- Report quality.

### 8.2 FinMA Expert Module

FinMA handles short-context financial tasks:

- News sentiment classification.
- Risk type classification.
- Event impact direction.
- Management tone analysis.
- Financial entity and relation extraction.
- Financial terminology explanation.
- Short financial QA.

Input should be chunked and bounded. Recommended input size:

- 512 to 1500 tokens per call.
- Include only task-specific context.
- Avoid sending full filings or full conversations.

Output should be structured JSON when possible:

```json
{
  "task": "risk_classification",
  "label": "regulatory_risk",
  "impact_direction": "negative",
  "confidence": 0.78,
  "rationale": "The passage discusses export restrictions that may limit revenue growth."
}
```

### 8.3 Model Router

The model router decides which model should handle each subtask.

Use lead model for:

- Planning.
- Retrieval.
- Long-document synthesis.
- Final report writing.
- Multi-source comparison.

Use FinMA for:

- Short financial classification.
- Financial signal extraction.
- Domain-specific interpretation.
- Post-trained financial expert evaluation.

## 9. Functional Requirements

### 9.1 Financial Workspace UI

Status: partially implemented.

Requirements:

- Show FinAgent branding.
- Provide financial task presets.
- Provide starter prompts.
- Route starter prompts into the chat input.
- Keep compatibility with DeerFlow workspace.

### 9.2 Financial Analysis Skill

Status: MVP implemented.

Requirements:

- Define standard financial report structure.
- Define evidence and citation expectations.
- Define when to use FinMA expert module.
- Define risk and uncertainty handling.
- Define output templates for:
  - Earnings analysis.
  - Valuation memo.
  - Peer comparison.
  - Risk review.
  - Market brief.

### 9.3 FinMA Expert Tool

Status: MVP implemented.

Requirements:

- Expose FinMA through a tool callable by DeerFlow.
- Start with a mock implementation if GPU is not ready.
- Later connect to a real FinMA API.
- Accept a task type and short context.
- Return structured JSON.

Implementation:

- Skill: `skills/public/financial-analysis/SKILL.md`
- Tool: `backend/packages/harness/deerflow/community/finma/tools.py`
- Samples: `evals/finma_expert_samples.jsonl`
- Evaluation runner: `scripts/evaluate_finma_expert.py`

Example tool request:

```json
{
  "task": "event_impact",
  "ticker": "NVDA",
  "text": "NVIDIA reported strong data center growth...",
  "output_schema": "impact_direction,rationale,confidence"
}
```

Example tool response:

```json
{
  "impact_direction": "positive",
  "affected_factors": ["revenue_growth", "margin_expectation"],
  "confidence": 0.82,
  "rationale": "The passage indicates strong demand in data center revenue."
}
```

### 9.4 Financial Data Tools

Status: planned.

MVP tools:

- Web search and web fetch through existing DeerFlow tools.
- Simple market data retrieval.
- Filing retrieval or user-provided filing upload.
- Python financial calculator.

Later tools:

- SEC EDGAR integration.
- Yahoo Finance or Polygon integration.
- News API integration.
- Earnings call transcript retrieval.
- FinBen evaluation runner.

### 9.5 Report Generation

Status: planned.

Reports should include:

- Executive Summary.
- Company Overview.
- Financial Performance.
- Valuation.
- Catalysts.
- Risks.
- Investment View.
- Sources.

Reports must distinguish:

- Facts from retrieved sources.
- Model inference.
- Assumptions.
- Unknown or missing data.

## 10. Post-Training Plan

### 10.1 Training Objective

Do not train FinMA to become a long-context lead agent. Train it to become a stronger short-context financial expert.

Target capabilities:

- Financial sentiment classification.
- Risk factor classification.
- Event impact analysis.
- Management tone analysis.
- Financial QA over short passages.
- Structured financial signal extraction.

### 10.2 Training Method

Recommended method:

- LoRA or QLoRA.
- Base: FinMA-7B-full or FinMA-7B-NLP.
- Training framework: Hugging Face Transformers + PEFT.
- Hardware: single GPU if quantized, multi-GPU if full precision.

### 10.3 Training Data

Data sources:

- PIXIU FIT instruction data.
- FinBen tasks converted to instruction format.
- Custom financial analysis data.
- Public filings and earnings call snippets.
- Synthetic but verified financial classification examples.

Recommended instruction format:

```json
{
  "instruction": "Classify the financial risk type and explain the impact.",
  "input": "The company warned that export controls may reduce revenue in China...",
  "output": {
    "risk_type": "regulatory_risk",
    "impact_direction": "negative",
    "confidence": 0.86,
    "rationale": "Export controls can restrict sales in a key market."
  }
}
```

### 10.4 Training Variants

Run at least two training variants:

- FinMA-original: no additional training.
- FinMA-LoRA-financial-signals: trained on short-context financial classification and extraction.

Optional:

- FinMA-LoRA-report-style: trained to produce structured analyst-style outputs from short context.

## 11. Evaluation Plan

### 11.1 Baselines

Evaluate:

- GLM only.
- DeepSeek only.
- Original FinMA only on supported short tasks.
- GLM + original FinMA module.
- GLM + post-trained FinMA module.
- DeepSeek + post-trained FinMA module.

### 11.2 Benchmarks

Use two benchmark groups.

Group A: Standard financial NLP tasks:

- FinBen classification tasks.
- Sentiment tasks.
- Headline classification.
- Named entity recognition.
- Short QA.

Group B: Agentic financial analysis tasks:

- Earnings analysis.
- Risk review.
- Peer comparison.
- Market brief.
- Investment memo generation.

### 11.3 Metrics

For classification/extraction:

- Accuracy.
- F1.
- Macro-F1.
- JSON validity.
- Calibration or confidence quality.

For report generation:

- Factual accuracy.
- Citation correctness.
- Coverage of required sections.
- Risk identification quality.
- Financial reasoning quality.
- Human evaluator score.

For system performance:

- Latency.
- Cost.
- Number of tool calls.
- FinMA calls per report.
- Failure rate.

### 11.4 Expected Hypothesis

Expected result:

- Long-context general models perform better as lead agents.
- Original FinMA performs competitively on short financial NLP tasks.
- Post-trained FinMA improves short-context financial expert outputs.
- Hybrid workflows improve domain-specific signal quality compared with lead-model-only workflows.

## 12. MVP Scope

### 12.1 MVP Must Have

- FinAgent-branded DeerFlow UI.
- Working local DeerFlow service.
- Financial prompt presets.
- `financial-analysis` skill.
- Mock `finma_expert` tool returning structured JSON.
- At least one complete workflow:
  - user asks for earnings analysis,
  - lead agent gathers context,
  - calls FinMA expert module,
  - produces structured report.

### 12.2 MVP Should Have

- Real FinMA API integration if GPU is available.
- Simple market data retrieval.
- Report export as Markdown.
- Evaluation script for a small benchmark set.

### 12.3 MVP Could Have

- SEC filing retrieval.
- News API integration.
- Dashboard of model comparison results.
- Post-trained LoRA model endpoint.

## 13. Milestones

### Milestone 1: Product Shell

Deliverables:

- FinAgent UI.
- Local DeerFlow service on port 2026.
- Financial starter prompts.
- Documentation of architecture.

Status:

- Mostly complete.

### Milestone 2: Financial Workflow

Deliverables:

- `financial-analysis` skill.
- Report templates.
- Prompting rules.
- Example generated report.

### Milestone 3: FinMA Expert Module

Deliverables:

- Mock FinMA expert tool.
- Tool schema.
- Router rule for when to call FinMA.
- Example workflow using mock FinMA.

### Milestone 4: Real Model Integration

Deliverables:

- FinMA served through vLLM or a compatible API.
- DeerFlow tool calls real FinMA endpoint.
- Latency and failure logging.

### Milestone 5: Post-Training

Deliverables:

- Training dataset.
- LoRA/QLoRA script.
- Trained adapter.
- Model card and training notes.

### Milestone 6: Evaluation

Deliverables:

- FinBen task evaluation.
- Custom agentic benchmark.
- Baseline comparison table.
- Error analysis.

## 14. Technical Risks

### Risk 1: FinMA Context Length

Issue:

- FinMA is not suitable for long-context lead-agent use.

Mitigation:

- Use FinMA only as short-context expert module.
- Use chunking and structured task-specific prompts.

### Risk 2: GPU Availability

Issue:

- FinMA-7B requires GPU resources.

Mitigation:

- Start with mock tool.
- Deploy real FinMA later on GPU.
- Keep DeerFlow local and model remote.

### Risk 3: Weak FinMA Generation Quality

Issue:

- FinMA v0.1 is older and may underperform newer general models.

Mitigation:

- Evaluate it on short financial tasks where domain tuning matters.
- Post-train using LoRA.
- Compare original and post-trained versions.

### Risk 4: Overclaiming Financial Capability

Issue:

- Financial analysis output can be mistaken for investment advice.

Mitigation:

- Add disclaimers.
- Require sources.
- Distinguish facts, assumptions, and model inference.

### Risk 5: Benchmark Misalignment

Issue:

- FinBen tasks may not measure end-to-end agent quality.

Mitigation:

- Use FinBen for module-level evaluation.
- Use custom tasks for workflow-level evaluation.

## 15. Success Criteria

The project is successful if:

- DeerFlow can run a financial analysis workflow end to end.
- The system clearly demonstrates long-context orchestration plus domain expert routing.
- FinMA is integrated as an expert module.
- A post-trained FinMA variant can be compared against the original.
- Experiments show measurable differences across GLM, DeepSeek, original FinMA module, and post-trained FinMA module.
- Final reports are structured, source-aware, and auditable.

## 16. Immediate Next Steps

1. Create `financial-analysis` skill.
2. Define the `finma_expert` tool schema.
3. Implement a mock `finma_expert` endpoint or Python tool.
4. Configure DeerFlow to call the mock expert during financial workflows.
5. Prepare a small evaluation set with 20 to 50 examples.
6. Decide GPU deployment path for real FinMA.
7. Draft LoRA/QLoRA training plan and data schema.
