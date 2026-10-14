"""Provider interface plus the shared output parser and HTTP retry helper."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import httpx
from pydantic import ValidationError

from app.schemas import ModelOutput

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    prompt: str
    ticket: str
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    max_retries: int = 2


@dataclass
class ProviderResult:
    output: ModelOutput | None
    raw_response: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retry_count: int
    model_served: str | None
    error_type: str | None = None
    error_message: str | None = None


class Provider(ABC):
    name: str = "base"

    @abstractmethod
    def classify(self, request: ProviderRequest) -> ProviderResult:
        """Classify one ticket. Must return errors in the result rather than raise."""


def user_message(ticket: str) -> str:
    return f"Ticket:\n{ticket}\n\nReturn only the JSON object."


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_model_output(text: str | None) -> tuple[ModelOutput | None, str | None, str | None]:
    """Parse model text into ModelOutput. Returns (output, error_type, error_message)."""
    if text is None or not text.strip():
        return None, "invalid_json", "empty response"
    cleaned = _FENCE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, "invalid_json", f"could not parse JSON: {exc.msg}"
    try:
        return ModelOutput.model_validate(data), None, None
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "<root>" for e in exc.errors())
        return None, "schema_error", f"schema violation in: {fields}"


@dataclass
class HttpOutcome:
    body: dict | None
    retry_count: int
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


def post_json_with_retries(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: float,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> HttpOutcome:
    """POST with exponential backoff on timeouts, transport errors, 429 and 5xx."""
    retries = 0
    started = time.perf_counter()
    error_type: str | None = None
    error_message: str | None = None

    for attempt in range(max_retries + 1):
        if attempt:
            retries += 1
            sleep(min(0.5 * 2 ** (attempt - 1), 8.0))
        try:
            response = client.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        except httpx.TimeoutException:
            error_type, error_message = "timeout", f"no response within {timeout_seconds}s"
            continue
        except httpx.TransportError as exc:
            error_type, error_message = "transport_error", type(exc).__name__
            continue

        if response.status_code in RETRYABLE_STATUS:
            error_type, error_message = "http_error", f"HTTP {response.status_code}"
            continue
        if response.status_code >= 400:
            # Deliberately omit headers and request body so credentials never reach logs.
            return HttpOutcome(None, retries, _elapsed(started), "http_error",
                               f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            return HttpOutcome(response.json(), retries, _elapsed(started))
        except ValueError:
            return HttpOutcome(None, retries, _elapsed(started), "invalid_provider_response",
                               "provider returned non-JSON body")

    return HttpOutcome(None, retries, _elapsed(started), error_type, error_message)


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
