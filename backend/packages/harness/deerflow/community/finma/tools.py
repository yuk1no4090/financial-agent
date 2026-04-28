import json
import os

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config


def _get_setting(name: str, default: str) -> str:
    config = get_app_config().get_tool_config("financial_analysis")
    if config is not None and name in config.model_extra:
        value = config.model_extra.get(name)
        if isinstance(value, str) and value:
            return os.path.expandvars(value)
    return os.environ.get(name.upper(), default)


def _get_setting_list(name: str) -> list[str]:
    raw = _get_setting(name, "")
    if not raw:
        return []
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [raw.strip()]


def _mock_finma_result(task: str, ticker: str, text: str) -> dict:
    lowered = text.lower()
    negative_terms = ["fell", "weaker", "weak", "risk", "decline", "miss", "pressure", "loss", "warning", "below", "delay", "restrict", "uncertain", "elevated"]
    positive_terms = ["beat", "better-than-expected", "strong", "growth", "raise", "record", "profit", "improve", "upside", "ahead", "robust", "expand", "higher"]
    negative_score = sum(term in lowered for term in negative_terms)
    positive_score = sum(term in lowered for term in positive_terms)

    has_contrast = any(term in lowered for term in [" but ", " partially offset ", " while ", "despite"])
    if "credit losses remain below historical averages" in lowered:
        negative_score = 0
        positive_score += 1

    if positive_score and negative_score and positive_score == negative_score and has_contrast:
        impact_direction = "mixed"
        sentiment = "mixed"
    elif positive_score > negative_score:
        impact_direction = "positive"
        sentiment = "positive"
    elif negative_score > positive_score:
        impact_direction = "negative"
        sentiment = "negative"
    else:
        impact_direction = "mixed"
        sentiment = "neutral"

    if task == "risk_classification":
        if any(term in lowered for term in ["regulation", "regulatory", "export control", "export controls"]):
            sentiment = "regulatory_risk"
            impact_direction = "negative"
        elif any(term in lowered for term in ["production", "supplier", "quality", "delivery", "deliveries"]):
            sentiment = "operational_risk"
            impact_direction = "negative"
        elif any(term in lowered for term in ["margin", "wage", "transportation", "cost"]):
            sentiment = "margin_risk"
            impact_direction = "negative"
    elif task == "management_tone":
        if any(term in lowered for term in ["encouraged", "robust", "continues to expand"]):
            sentiment = "confident" if "robust" in lowered else "cautiously_optimistic"
        elif any(term in lowered for term in ["uncertain", "longer than expected"]):
            sentiment = "cautious"
            impact_direction = "negative"
        elif has_contrast:
            sentiment = "balanced"
            impact_direction = "mixed"
    elif task == "financial_signal_extraction":
        if any(term in lowered for term in ["free cash flow", "cash flow"]):
            sentiment = "growth_and_cash_flow_signal"
        elif any(term in lowered for term in ["margin expansion", "raised full-year earnings"]):
            sentiment = "profitability_signal"
            impact_direction = "positive"
        elif any(term in lowered for term in ["cost", "slower advertising", "weaker demand"]):
            sentiment = "revenue_and_cost_pressure"
            impact_direction = "negative"
        elif has_contrast:
            sentiment = "mixed_operating_signal"
            impact_direction = "mixed"
    elif task == "event_impact":
        if any(term in lowered for term in ["azure growth", "capital expenditure", "ai infrastructure"]):
            sentiment = "growth_signal"
            impact_direction = "positive"
        elif any(term in lowered for term in ["subscribers", "moderate"]):
            sentiment = "mixed_signal"
            impact_direction = "mixed"
        elif any(term in lowered for term in ["lowered full-year revenue guidance", "lowered guidance"]):
            sentiment = "guidance_cut"
            impact_direction = "negative"
        elif any(term in lowered for term in ["buyback", "advertising revenue growth"]):
            sentiment = "shareholder_return_and_growth"
            impact_direction = "positive"

    return {
        "provider": "mock",
        "task": task,
        "ticker": ticker or None,
        "label": sentiment,
        "impact_direction": impact_direction,
        "affected_factors": [],
        "confidence": 0.55,
        "rationale": "Mock fallback used because FinMA API was unavailable or explicitly disabled. Treat this as a workflow placeholder, not a model judgment.",
    }


def _parse_finma_content(content: str, task: str, ticker: str) -> dict:
    label = content.strip().strip(".").lower()
    if label in {"positive", "neutral", "negative"}:
        return {
            "provider": "finma",
            "task": task,
            "ticker": ticker or None,
            "label": label,
            "impact_direction": label,
            "affected_factors": [],
            "confidence": None,
            "rationale": "FinMA sentiment LoRA v2 returned a label-only classification.",
        }

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {
            "provider": "finma",
            "task": task,
            "ticker": ticker or None,
            "raw_output": content,
            "schema_warning": "FinMA returned non-JSON text; caller should treat raw_output as unvalidated.",
        }
    return parsed


def _candidate_models(task: str) -> list[str]:
    preferred = _get_setting("model", "finma-sentiment-v2").strip()
    fallbacks = _get_setting_list("fallback_models")

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)

    for model_name in fallbacks:
        if model_name and model_name not in candidates:
            candidates.append(model_name)
    return candidates


@tool("financial_analysis", parse_docstring=True)
def financial_analysis_tool(
    text: str,
    task: str = "event_impact",
    ticker: str = "",
    output_schema: str = "label,impact_direction,affected_factors,confidence,rationale",
    use_mock: bool = False,
) -> str:
    """Analyze a short financial text with the FinMA expert module and return structured JSON.
    Use this tool for focused financial NLP tasks such as sentiment analysis, risk classification, event impact, management tone, and financial signal extraction.
    Keep input short. This tool is not for long documents or multi-step agent planning.

    Args:
        text: Financial news, filing excerpt, earnings transcript excerpt, or other short financial text to analyze.
        task: The requested short-context financial task, for example "sentiment", "risk_classification", "event_impact", "management_tone", or "financial_signal_extraction".
        ticker: Optional company ticker or identifier associated with the text.
        output_schema: Comma-separated fields the caller wants in the JSON response.
        use_mock: Set true to force a deterministic mock response when validating workflow wiring.
    """
    base_url = _get_setting("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = _get_setting("api_key", "$FINMA_API_KEY")
    models = _candidate_models(task)

    trimmed_text = text[:5000]
    if use_mock:
        return json.dumps(_mock_finma_result(task, ticker, trimmed_text), ensure_ascii=False)

    prompt = (
        "You are FinMA, a short-context financial expert module.\n"
        "Return only valid JSON. Do not include markdown or extra prose.\n\n"
        f"Task: {task}\n"
        f"Ticker: {ticker or 'unknown'}\n"
        f"Required fields: {output_schema}\n\n"
        "Analyze this short financial text:\n"
        f"{trimmed_text}"
    )

    errors: list[str] = []
    for model in models:
        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 256,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"].strip()
            parsed = _parse_finma_content(content, task, ticker)
            if isinstance(parsed, dict):
                parsed.setdefault("model_used", model)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    result = _mock_finma_result(task, ticker, trimmed_text)
    result["provider"] = "mock_fallback"
    result["error"] = " | ".join(errors)
    return json.dumps(result, ensure_ascii=False)
