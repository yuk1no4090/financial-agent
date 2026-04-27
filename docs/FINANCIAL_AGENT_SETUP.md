# Financial Agent Setup

This repository is configured as a financial-agent workspace based on DeerFlow.

## Architecture

- Main agent model: Zhipu GLM via OpenAI-compatible API.
- Financial specialist model: FinMA served by vLLM on AutoDL.
- DeerFlow calls FinMA through the `financial_analysis` tool when focused financial NLP analysis is needed.
- Financial workflow instructions live in `skills/public/financial-analysis/SKILL.md`.

## Required Environment Variables

Create a local `.env` file from `.env.example` and set:

```bash
ZHIPUAI_API_KEY=your-zhipuai-api-key
FINMA_API_KEY=your-finma-vllm-api-key
```

Do not commit `.env`.

## FinMA vLLM Endpoint

The default local configuration expects an SSH tunnel from the Mac to AutoDL:

```text
http://127.0.0.1:8000/v1
```

The FinMA service should expose:

```text
model: finma-7b-nlp
api key: FINMA_API_KEY
```

Example AutoDL vLLM command:

```bash
vllm serve /root/autodl-tmp/models/finma-7b-nlp \
  --served-model-name finma-7b-nlp \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --api-key "$FINMA_API_KEY"
```

If the FinMA API is unavailable, the `financial_analysis` tool can still return deterministic mock JSON when called with `use_mock=true`. This is useful for validating the DeerFlow workflow without GPU access.

## Financial Analysis Tool

The tool is registered in `config.yaml`:

```text
financial_analysis
```

Recommended payload shape:

```json
{
  "task": "event_impact",
  "ticker": "NVDA",
  "text": "NVIDIA reported strong data center growth...",
  "output_schema": "label,impact_direction,affected_factors,confidence,rationale"
}
```

The tool returns JSON. If FinMA returns non-JSON text, the wrapper returns a JSON object containing `raw_output` and a schema warning.

## Evaluation

Run the small FinMA expert sample set in mock mode:

```bash
cd backend
uv run python ../scripts/evaluate_finma_expert.py
```

Run against the real FinMA endpoint:

```bash
cd backend
uv run python ../scripts/evaluate_finma_expert.py --real
```

## Local Start

```bash
PATH=/opt/homebrew/opt/node@22/bin:$PATH make doctor
PATH=/opt/homebrew/opt/node@22/bin:$PATH make dev
```

Open:

```text
http://localhost:2026
```
