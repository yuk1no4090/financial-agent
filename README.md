# FinAgent Workbench

English | [中文](./README_zh.md)

FinAgent Workbench is an agentic financial analysis project built on [DeerFlow](https://github.com/bytedance/deer-flow) and [PIXIU](https://github.com/The-FinAI/PIXIU). It uses DeerFlow as the agent orchestration and workspace layer, and uses PIXIU/FinMA as a domain-tuned financial expert module for short-context financial reasoning tasks.

The central design choice is deliberate: PIXIU/FinMA is not used as the lead agent model. FinMA v0.1 is based on early LLaMA-style models and has a short context window, which makes it unsuitable for long-horizon agent orchestration, multi-source synthesis, full filing analysis, and final report generation. FinAgent instead uses a long-context general model such as GLM, DeepSeek, or Qwen as the lead model, while FinMA is routed as a specialist module for finance-specific subtasks.

## Project Summary

```text
User
  -> FinAgent Workbench UI
  -> DeerFlow lead agent
       -> Long-context model: GLM / DeepSeek / Qwen
       -> Tools:
            - Web search and fetch
            - Financial data retrieval
            - Financial calculator
            - FinMA expert module
       -> Skills:
            - financial analysis
            - earnings analysis
            - valuation
            - risk review
       -> Outputs:
            - investment memo
            - earnings review
            - peer comparison
            - risk memo
            - benchmark report
```

## Why This Architecture

Financial analysis needs two different capabilities:

- Long-context orchestration: reading filings, collecting news, coordinating tools, maintaining state, and writing final reports.
- Financial domain judgment: classifying sentiment, identifying risks, interpreting events, extracting financial signals, and answering short financial questions.

Modern long-context models are better suited for the first capability. PIXIU/FinMA is useful for the second capability. FinAgent combines both instead of forcing one model to do everything.

## Core Workload

The project workload is intentionally spread across system design, model training, and evaluation.

### 1. Agent System

- Rebrand DeerFlow into a financial analysis workspace.
- Add financial task presets and report flows.
- Define finance-specific skills.
- Add model routing between a lead model and financial expert module.
- Add tools for filings, news, market data, and calculations.

### 2. FinMA Expert Module

- Wrap PIXIU/FinMA as a callable financial expert.
- Keep inputs short and task-specific.
- Return structured outputs such as labels, rationales, confidence scores, and extracted signals.
- Start with a mock tool if the GPU endpoint is not ready.
- Replace the mock with a real vLLM or OpenAI-compatible FinMA service later.

### 3. Post-Training

- Post-train FinMA with LoRA or QLoRA.
- Focus training on short-context financial expert tasks instead of long-context agent planning.
- Candidate tasks:
  - financial sentiment classification
  - risk factor classification
  - event impact analysis
  - management tone analysis
  - financial QA over short passages
  - structured financial signal extraction

### 4. Evaluation

Compare:

- GLM only
- DeepSeek only
- original FinMA on supported short tasks
- GLM + original FinMA module
- GLM + post-trained FinMA module
- DeepSeek + post-trained FinMA module

Evaluate with:

- FinBen tasks from PIXIU
- custom end-to-end financial analysis tasks
- report quality scoring
- citation correctness
- latency and cost
- failure analysis

## Current Status

Completed:

- DeerFlow repository cloned and running locally.
- Local frontend entry unified to `http://localhost:2026/`.
- Frontend rebranded to FinAgent Workbench.
- Financial landing page added.
- Financial chat presets added.
- Initial PRD drafted.

Planned:

- Add `financial-analysis` skill.
- Add `finma_expert` mock tool.
- Connect real FinMA endpoint through vLLM or an OpenAI-compatible API.
- Build a small evaluation set.
- Prepare post-training data and LoRA/QLoRA scripts.

## Local Development

### Requirements

- Python 3.12+
- Node.js 22+
- pnpm
- uv
- nginx

On the current local setup, Node.js 22 is installed through Homebrew as `node@22`, so commands should be run with this PATH:

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH
```

### Start the App

From the repository root:

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH make stop
screen -dmS deerflow zsh -lc 'cd /Users/yuk1no/6052/deer-flow && PATH=/opt/homebrew/opt/node@22/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin ./scripts/serve.sh --dev --skip-install'
```

Open:

```text
http://localhost:2026/
```

### Stop the App

```bash
screen -S deerflow -X quit
PATH=/opt/homebrew/opt/node@22/bin:$PATH make stop
```

### Verify

```bash
curl -I http://localhost:2026/
```

Expected result:

```text
HTTP/1.1 200 OK
```

## Model Integration Plan

### Lead Model

Use a long-context model for orchestration:

```yaml
models:
  - name: glm-lead
    display_name: GLM Lead Model
    use: langchain_openai:ChatOpenAI
    model: your-glm-model
    api_key: $GLM_API_KEY
    base_url: https://your-glm-compatible-endpoint/v1
    request_timeout: 600.0
    max_retries: 2
```

### FinMA Expert Endpoint

Expose FinMA as a separate expert service:

```text
DeerFlow tool call
  -> finma_expert endpoint
  -> FinMA model
  -> structured financial judgment
```

The expert endpoint should accept short inputs:

```json
{
  "task": "risk_classification",
  "ticker": "NVDA",
  "text": "NVIDIA reported strong data center growth but faces export control restrictions.",
  "schema": "risk_type,impact_direction,confidence,rationale"
}
```

Expected output:

```json
{
  "risk_type": "regulatory_risk",
  "impact_direction": "negative",
  "confidence": 0.78,
  "rationale": "Export controls can limit sales in specific markets."
}
```

## Documentation

- [English PRD](./docs/FINAGENT_PRD.md)
- [中文 PRD](./docs/FINAGENT_PRD_zh.md)
- [Local setup notes](./docs/FINANCIAL_AGENT_SETUP.md)

## Built On

This project is built on top of the following open-source projects:

- [DeerFlow](https://github.com/bytedance/deer-flow): agent orchestration, workspace UI, tools, skills, memory, sandbox, and sub-agent runtime.
- [PIXIU](https://github.com/The-FinAI/PIXIU): financial LLMs, instruction data, and FinBen evaluation benchmark.

## Citation

If you use PIXIU/FinMA, cite the PIXIU paper:

```bibtex
@misc{xie2023pixiu,
  title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance},
  author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
  year={2023},
  eprint={2306.05443},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

If you use DeerFlow, cite or reference the upstream project:

```text
https://github.com/bytedance/deer-flow
```

## License

This repository inherits the original DeerFlow MIT license unless otherwise changed by the project maintainers. See [LICENSE](./LICENSE).

