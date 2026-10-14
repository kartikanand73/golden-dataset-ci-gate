"""Provider for the Anthropic Messages API.

Environment:
  ANTHROPIC_API_KEY   required
  ANTHROPIC_BASE_URL  optional, defaults to https://api.anthropic.com
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

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 client: httpx.Client | None = None, max_tokens: int = 256) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL")
                         or "https://api.anthropic.com").rstrip("/")
        self.client = client or httpx.Client()
        self.max_tokens = max_tokens

    def classify(self, request: ProviderRequest) -> ProviderResult:
        payload = {
            "model": request.model,
            "max_tokens": self.max_tokens,
            "temperature": request.temperature,
            "system": request.prompt,
            "messages": [{"role": "user", "content": user_message(request.ticket)}],
        }
        outcome = post_json_with_retries(
            self.client,
            f"{self.base_url}/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            payload,
            request.timeout_seconds,
            request.max_retries,
        )
        if outcome.body is None:
            return ProviderResult(None, None, 0, 0, outcome.latency_ms, outcome.retry_count,
                                  None, outcome.error_type, outcome.error_message)

        body = outcome.body
        text_blocks = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
        text = "".join(text_blocks) or None
        usage = body.get("usage") or {}
        output, error_type, error_message = parse_model_output(text)
        return ProviderResult(
            output=output,
            raw_response=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=outcome.latency_ms,
            retry_count=outcome.retry_count,
            model_served=body.get("model"),
            error_type=error_type,
            error_message=error_message,
        )
