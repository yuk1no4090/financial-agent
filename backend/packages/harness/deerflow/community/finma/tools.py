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


@tool("financial_analysis", parse_docstring=True)
def financial_analysis_tool(text: str, task: str = "analyze") -> str:
    """Analyze a short financial text with the FinMA model.
    Use this tool for focused financial NLP tasks such as sentiment analysis, risk extraction, financial event interpretation, and brief implications analysis.
    Keep input short. This tool is not for long documents or multi-step agent planning.

    Args:
        text: Financial news, filing excerpt, earnings transcript excerpt, or other short financial text to analyze.
        task: The requested financial analysis task, for example "sentiment", "risk factors", "earnings impact", or "summarize".
    """
    base_url = _get_setting("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = _get_setting("api_key", "$FINMA_API_KEY")
    model = _get_setting("model", "finma-7b-nlp")

    trimmed_text = text[:5000]
    prompt = f"Task: {task}\n\nAnalyze the following financial text. Return a concise, structured answer with the key financial signal, sentiment, risks, and likely implication.\n\nText:\n{trimmed_text}"

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
        return payload["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"FinMA financial_analysis failed: {exc}"
