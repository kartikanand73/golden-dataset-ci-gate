"""Provider for any OpenAI-compatible Chat Completions endpoint.

Environment:
  OPENAI_API_KEY   required
  OPENAI_BASE_URL  optional, defaults to https://api.openai.com/v1
                   (point it at Azure OpenAI, vLLM, Ollama, LiteLLM, etc.)
"""

from __future__ import annotations

import os

import httpx

from app.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResult,
    parse_model_output,
    post_json_with_retries,
    user_message,
)


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.client = client or httpx.Client()

    def classify(self, request: ProviderRequest) -> ProviderResult:
        payload = {
            "model": request.model,
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": request.prompt},
                {"role": "user", "content": user_message(request.ticket)},
            ],
        }
        outcome = post_json_with_retries(
            self.client,
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload,
            request.timeout_seconds,
            request.max_retries,
        )
        if outcome.body is None:
            return ProviderResult(None, None, 0, 0, outcome.latency_ms, outcome.retry_count,
                                  None, outcome.error_type, outcome.error_message)

        body = outcome.body
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = None
        usage = body.get("usage") or {}
        output, error_type, error_message = parse_model_output(text)
        return ProviderResult(
            output=output,
            raw_response=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=outcome.latency_ms,
            retry_count=outcome.retry_count,
            model_served=body.get("model"),
            error_type=error_type,
            error_message=error_message,
        )
