import json
import os
import re

import httpx
from langchain.tools import tool

from deerflow.config import get_app_config

_LABEL_DIRECTIONS = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "mixed": "mixed",
    "bullish": "positive",
    "bearish": "negative",
}


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


def _clean_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().strip(".").lower().replace(" ", "_")


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _to_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,\n;/|、；]+", value)
        return [part.strip() for part in parts if part.strip()]
    return []


def _coerce_confidence(value: object) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return max(0.0, min(1.0, numeric))


def _generic_summary(task: str, label: str) -> str:
    if task == "risk_classification":
        return f"The dominant takeaway is {label or 'a relevant'} risk factor."
    if task == "management_tone":
        return f"Management tone reads as {label or 'mixed'}."
    if task == "financial_signal_extraction":
        return "The text contains actionable operating or financial signals."
    if task == "event_impact":
        return f"The event impact looks {label or 'mixed'}."
    return f"The overall read is {label or 'mixed'}."


def _generic_market_implication(label: str) -> str:
    if label == "positive":
        return "If the market was not already fully pricing this in, the update can support sentiment, earnings expectations, or valuation in the near term."
    if label == "negative":
        return "If the market was not already discounting this risk, the update can pressure sentiment, earnings expectations, or valuation in the near term."
    if label == "neutral":
        return "The update is not strongly directional on its own, so market reaction usually depends on follow-up guidance, estimates, and positioning."
    return "The update is mixed enough that the market reaction will depend on which signal investors weight more heavily."


def _default_watch_items(task: str) -> list[str]:
    if task == "risk_classification":
        return ["whether the risk is temporary or structural", "management response", "impact on guidance or margins"]
    if task == "management_tone":
        return ["forward guidance", "whether tone is matched by data", "analyst estimate revisions"]
    if task == "financial_signal_extraction":
        return ["whether the signal persists next quarter", "margin and cash-flow follow-through", "management commentary"]
    if task == "event_impact":
        return ["short-term price reaction", "estimate revisions", "management follow-up disclosures"]
    return ["whether the signal is already priced in", "follow-up guidance", "next data point that confirms or weakens the thesis"]


def _normalize_model_result(result: dict, task: str, ticker: str) -> dict:
    normalized = dict(result)
    normalized["task"] = task
    normalized["ticker"] = ticker or None

    label = _clean_label(normalized.get("label") or normalized.get("stance") or normalized.get("sentiment"))
    if label:
        normalized["label"] = label

    impact_direction = _clean_label(normalized.get("impact_direction"))
    if impact_direction:
        normalized["impact_direction"] = impact_direction
    elif label in _LABEL_DIRECTIONS:
        normalized["impact_direction"] = _LABEL_DIRECTIONS[label]

    summary = _clean_text(normalized.get("summary"))
    if not summary:
        summary = _clean_text(normalized.get("analysis"))
    if not summary and label:
        summary = _generic_summary(task, label)
    if summary:
        normalized["summary"] = summary

    rationale = _clean_text(normalized.get("rationale"))
    if not rationale:
        rationale = _clean_text(normalized.get("analysis"))
    if rationale:
        normalized["rationale"] = rationale

    market_implication = _clean_text(normalized.get("market_implication") or normalized.get("trading_takeaway") or normalized.get("implication"))
    if not market_implication and label:
        market_implication = _generic_market_implication(label)
    if market_implication:
        normalized["market_implication"] = market_implication

    watch_items = _to_string_list(normalized.get("watch_items") or normalized.get("watch_list") or normalized.get("next_watch_items"))
    if watch_items:
        normalized["watch_items"] = watch_items[:4]

    affected_factors = _to_string_list(normalized.get("affected_factors"))
    if affected_factors:
        normalized["affected_factors"] = affected_factors[:4]

    confidence = _coerce_confidence(normalized.get("confidence"))
    if confidence is not None:
        normalized["confidence"] = confidence

    return normalized


def _mock_finma_result(task: str, ticker: str, text: str) -> dict:
    lowered = text.lower()
    negative_terms = [
        "fell",
        "weaker",
        "weak",
        "risk",
        "decline",
        "miss",
        "pressure",
        "loss",
        "warning",
        "below",
        "delay",
        "restrict",
        "uncertain",
        "elevated",
    ]
    positive_terms = [
        "beat",
        "better-than-expected",
        "strong",
        "growth",
        "raise",
        "record",
        "profit",
        "improve",
        "upside",
        "ahead",
        "robust",
        "expand",
        "higher",
    ]
    negative_score = sum(term in lowered for term in negative_terms)
    positive_score = sum(term in lowered for term in positive_terms)

    has_contrast = any(term in lowered for term in [" but ", " partially offset ", " while ", "despite"])
    if "credit losses remain below historical averages" in lowered:
        negative_score = 0
        positive_score += 1

    if positive_score and negative_score and positive_score == negative_score and has_contrast:
        impact_direction = "mixed"
        label = "mixed"
    elif positive_score > negative_score:
        impact_direction = "positive"
        label = "positive"
    elif negative_score > positive_score:
        impact_direction = "negative"
        label = "negative"
    else:
        impact_direction = "mixed"
        label = "neutral"

    if task == "risk_classification":
        if any(term in lowered for term in ["regulation", "regulatory", "export control", "export controls"]):
            label = "regulatory_risk"
            impact_direction = "negative"
        elif any(term in lowered for term in ["production", "supplier", "quality", "delivery", "deliveries"]):
            label = "operational_risk"
            impact_direction = "negative"
        elif any(term in lowered for term in ["margin", "wage", "transportation", "cost"]):
            label = "margin_risk"
            impact_direction = "negative"
    elif task == "management_tone":
        if any(term in lowered for term in ["encouraged", "robust", "continues to expand"]):
            label = "confident" if "robust" in lowered else "cautiously_optimistic"
        elif any(term in lowered for term in ["uncertain", "longer than expected"]):
            label = "cautious"
            impact_direction = "negative"
        elif has_contrast:
            label = "balanced"
            impact_direction = "mixed"
    elif task == "financial_signal_extraction":
        if any(term in lowered for term in ["free cash flow", "cash flow"]):
            label = "growth_and_cash_flow_signal"
        elif any(term in lowered for term in ["margin expansion", "raised full-year earnings"]):
            label = "profitability_signal"
            impact_direction = "positive"
        elif any(term in lowered for term in ["cost", "slower advertising", "weaker demand"]):
            label = "revenue_and_cost_pressure"
            impact_direction = "negative"
        elif has_contrast:
            label = "mixed_operating_signal"
            impact_direction = "mixed"
    elif task == "event_impact":
        if any(term in lowered for term in ["azure growth", "capital expenditure", "ai infrastructure"]):
            label = "growth_signal"
            impact_direction = "positive"
        elif any(term in lowered for term in ["subscribers", "moderate"]):
            label = "mixed_signal"
            impact_direction = "mixed"
        elif any(term in lowered for term in ["lowered full-year revenue guidance", "lowered guidance"]):
            label = "guidance_cut"
            impact_direction = "negative"
        elif any(term in lowered for term in ["buyback", "advertising revenue growth"]):
            label = "shareholder_return_and_growth"
            impact_direction = "positive"

    return _normalize_model_result(
        {
            "provider": "mock",
            "task": task,
            "ticker": ticker or None,
            "label": label,
            "impact_direction": impact_direction,
            "affected_factors": [],
            "confidence": 0.55,
            "summary": _generic_summary(task, label),
            "rationale": "Mock fallback used because the FinMA API was unavailable or explicitly disabled. Treat this as a workflow placeholder, not a production model judgment.",
            "market_implication": _generic_market_implication(impact_direction),
            "watch_items": _default_watch_items(task),
        },
        task,
        ticker,
    )


def _parse_finma_content(content: str, task: str, ticker: str) -> dict:
    label = _clean_label(content)
    if label in {"positive", "neutral", "negative", "mixed"}:
        return _normalize_model_result(
            {
                "provider": "finma",
                "task": task,
                "ticker": ticker or None,
                "label": label,
                "impact_direction": _LABEL_DIRECTIONS.get(label, label),
                "affected_factors": [],
                "confidence": None,
                "summary": _generic_summary(task, label),
                "rationale": "Label-only classification returned by the sentiment model.",
                "market_implication": _generic_market_implication(label),
                "watch_items": _default_watch_items(task),
            },
            task,
            ticker,
        )

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _normalize_model_result(
            {
                "provider": "finma",
                "task": task,
                "ticker": ticker or None,
                "raw_output": content,
                "schema_warning": "FinMA returned non-JSON text; caller should treat raw_output as unvalidated.",
            },
            task,
            ticker,
        )

    if not isinstance(parsed, dict):
        parsed = {"raw_output": content}
    parsed.setdefault("provider", "finma")
    return _normalize_model_result(parsed, task, ticker)


def _configured_models() -> tuple[str, str | None]:
    preferred = _get_setting("model", "finma-sentiment-v3").strip()
    fallbacks = _get_setting_list("fallback_models")
    base_model = fallbacks[0] if fallbacks else None
    return preferred, base_model


def _candidate_models(model_strategy: str) -> list[str]:
    v3_model, base_model = _configured_models()
    if model_strategy == "v3_and_base":
        return [model for model in (v3_model, base_model) if model]
    if base_model:
        return [base_model]
    return [v3_model] if v3_model else []


def _task_label_guide(task: str) -> str:
    if task == "sentiment":
        return "label must be one of positive, negative, neutral, mixed."
    if task == "risk_classification":
        return "label should be a concise risk type such as regulatory_risk, margin_risk, operational_risk, demand_risk, or liquidity_risk."
    if task == "management_tone":
        return "label should capture management tone such as confident, cautious, balanced, cautiously_optimistic, or defensive."
    if task == "financial_signal_extraction":
        return "label should summarize the dominant operating or financial signal."
    return "label should summarize the dominant financial takeaway."


def _prompt_for_model(model: str, task: str, ticker: str, output_schema: str, text: str) -> str:
    if "sentiment" in model.lower():
        return f"Classify the sentiment of this financial text. Reply with exactly one label and nothing else: Positive, Neutral, Negative, or Mixed.\n\nText:\n{text}"

    return (
        "You are FinMA, a financial analysis specialist.\n"
        "Return ONLY valid JSON and no markdown.\n"
        "Write all JSON string values in the same language as the financial text.\n"
        f"Task: {task}\n"
        f"Ticker: {ticker or 'unknown'}\n"
        f"Requested fields from caller: {output_schema}\n"
        f"Label guidance: {_task_label_guide(task)}\n"
        "Required JSON keys:\n"
        "- label\n"
        "- impact_direction\n"
        "- summary\n"
        "- rationale\n"
        "- market_implication\n"
        "- watch_items\n"
        "- affected_factors\n"
        "- confidence\n"
        "Rules:\n"
        "- summary: one concise sentence with the core judgment.\n"
        "- rationale: 2-4 sentences grounded only in the provided text.\n"
        "- market_implication: one concise sentence on why investors should care, if relevant.\n"
        "- watch_items: 1-3 short follow-up items.\n"
        "- affected_factors: short list of financial drivers or variables.\n"
        "- confidence: number between 0 and 1.\n\n"
        f"Financial text:\n{text}"
    )


def _find_model_result(model_results: list[dict], keyword: str) -> dict | None:
    for result in model_results:
        model_used = str(result.get("model_used") or "").lower()
        if keyword in model_used:
            return result
    return None


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _agreement(primary_label: str, v3_label: str) -> str:
    if not v3_label:
        return "single_model"
    if not primary_label:
        return "single_model"
    if primary_label == v3_label:
        return "aligned"
    if {primary_label, v3_label} <= {"positive", "mixed"} or {primary_label, v3_label} <= {"negative", "mixed"}:
        return "mostly_aligned"
    return "diverged"


def _build_synthesis(
    task: str,
    ticker: str,
    model_strategy: str,
    model_results: list[dict],
    errors: list[str],
) -> dict:
    v3_result = _find_model_result(model_results, "sentiment")
    base_result = next(
        (result for result in model_results if "sentiment" not in str(result.get("model_used") or "").lower()),
        None,
    )
    primary_result = base_result or v3_result or (model_results[0] if model_results else {})
    primary_label = _clean_label(primary_result.get("label") or primary_result.get("impact_direction"))
    v3_label = _clean_label(v3_result.get("label")) if isinstance(v3_result, dict) else ""
    impact_direction = _clean_label(primary_result.get("impact_direction") or _LABEL_DIRECTIONS.get(primary_label, primary_label))

    explanation = _first_nonempty(
        primary_result.get("rationale"),
        primary_result.get("summary"),
        v3_result.get("rationale") if isinstance(v3_result, dict) else "",
    )
    market_implication = _first_nonempty(
        primary_result.get("market_implication"),
        v3_result.get("market_implication") if isinstance(v3_result, dict) else "",
        _generic_market_implication(impact_direction or primary_label),
    )

    watch_items = []
    for source in (primary_result, v3_result):
        if not isinstance(source, dict):
            continue
        for item in _to_string_list(source.get("watch_items")) + _to_string_list(source.get("affected_factors")):
            if item not in watch_items:
                watch_items.append(item)
    if not watch_items:
        watch_items = _default_watch_items(task)

    confidences = [
        confidence
        for confidence in (
            _coerce_confidence(primary_result.get("confidence")),
            _coerce_confidence(v3_result.get("confidence")) if isinstance(v3_result, dict) else None,
        )
        if confidence is not None
    ]

    return {
        "task": task,
        "ticker": ticker or None,
        "model_strategy": model_strategy,
        "label": primary_label,
        "impact_direction": impact_direction or primary_label,
        "agreement": _agreement(primary_label, v3_label),
        "summary": _first_nonempty(primary_result.get("summary"), _generic_summary(task, primary_label)),
        "explanation": explanation,
        "market_implication": market_implication,
        "watch_items": watch_items[:4],
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "errors": errors,
    }


@tool("financial_analysis", parse_docstring=True)
def financial_analysis_tool(
    text: str,
    task: str = "event_impact",
    ticker: str = "",
    output_schema: str = "label,impact_direction,affected_factors,confidence,rationale",
    model_strategy: str = "base_only",
    use_mock: bool = False,
) -> str:
    """Analyze a short financial text with the FinMA expert module and return structured JSON.
    Use this tool for focused financial NLP tasks such as sentiment analysis, risk classification, event impact, management tone, and financial signal extraction.
    Use model_strategy="v3_and_base" for short financial news sentiment where both LoRA v3 and base should be consulted.
    Use model_strategy="base_only" for broader financial analysis where the sentiment LoRA is not applicable.
    Keep input short. This tool is not for long documents or multi-step agent planning.

    Args:
        text: Financial news, filing excerpt, earnings transcript excerpt, or other short financial text to analyze.
        task: The requested short-context financial task, for example "sentiment", "risk_classification", "event_impact", "management_tone", or "financial_signal_extraction".
        ticker: Optional company ticker or identifier associated with the text.
        output_schema: Comma-separated fields the caller wants in the JSON response.
        model_strategy: "v3_and_base" or "base_only".
        use_mock: Set true to force a deterministic mock response when validating workflow wiring.
    """
    base_url = _get_setting("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = _get_setting("api_key", "$FINMA_API_KEY")
    strategy = model_strategy if model_strategy in {"v3_and_base", "base_only"} else "base_only"
    models = _candidate_models(strategy)

    trimmed_text = text[:5000]
    if use_mock:
        mock_result = _mock_finma_result(task, ticker, trimmed_text)
        return json.dumps(
            {
                "provider": "mock",
                "task": task,
                "ticker": ticker or None,
                "model_strategy": strategy,
                "models_requested": models,
                "model_results": [mock_result],
                "synthesis": _build_synthesis(task, ticker, strategy, [mock_result], []),
            },
            ensure_ascii=False,
        )

    errors: list[str] = []
    model_results: list[dict] = []
    for model in models:
        try:
            prompt = _prompt_for_model(model, task, ticker, output_schema, trimmed_text)
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
                    "max_tokens": 384,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"].strip()
            parsed = _parse_finma_content(content, task, ticker)
            if isinstance(parsed, dict):
                parsed.setdefault("model_used", model)
                model_results.append(parsed)
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    if model_results:
        return json.dumps(
            {
                "provider": "finma_ensemble",
                "task": task,
                "ticker": ticker or None,
                "model_strategy": strategy,
                "models_requested": models,
                "model_results": model_results,
                "errors": errors,
                "synthesis": _build_synthesis(task, ticker, strategy, model_results, errors),
            },
            ensure_ascii=False,
        )

    result = _mock_finma_result(task, ticker, trimmed_text)
    return json.dumps(
        {
            "provider": "mock_fallback",
            "task": task,
            "ticker": ticker or None,
            "model_strategy": strategy,
            "models_requested": models,
            "model_results": [result],
            "errors": errors,
            "synthesis": _build_synthesis(task, ticker, strategy, [result], errors),
            "error": " | ".join(errors),
        },
        ensure_ascii=False,
    )
