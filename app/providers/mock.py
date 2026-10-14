"""Deterministic mock "models" for offline demonstrations and tests.

Each profile is a small keyword router. The profiles differ the way real model or
prompt versions differ: rule precedence and vocabulary. Nothing here looks at case
IDs or expected labels, so results come only from the ticket text.

  mock-v1-baseline   security first, broad vocabulary. Passes every gate.
  mock-v2-regression checks password keywords before security keywords (a
                     "simplified prompt"). Security tickets that mention passwords
                     get automated. Slower, pricier and overconfident too.
  mock-v3-fixed      restores security-first precedence, recognises departed-staff
                     incidents, and stops treating a bare "reset" as a password reset.
  mock-invalid-json  always returns malformed JSON (for tests).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from app.providers.base import Provider, ProviderRequest, ProviderResult, parse_model_output
from app.schemas import Category

SECURITY_V1 = [
    r"phishing", r"suspicious", r"malware", r"ransomware", r"\bhack(ed|er|ers)?\b",
    r"breach", r"unauthori[sz]ed", r"recogni[sz]e", r"leaked", r"\bstolen\b",
    r"without my knowledge", r"another country", r"pretending", r"impersonat",
    r"unknown usb", r"compromised", r"\bvirus\b",
]
SECURITY_V3 = SECURITY_V1 + [r"former (employee|contractor)", r"ex-employee"]

ACCESS = [
    r"\baccess\b", r"permission", r"add me to", r"admin rights", r"licen[cs]e",
    r"provision", r"distribution list", r"\brole\b", r"\bgrant\b",
]

PASSWORD_V1 = [r"password", r"forgot", r"locked out", r"\breset\b", r"\bunlock", r"\blocked\b"]
PASSWORD_V3 = [r"password", r"forgot", r"locked out", r"\bunlock", r"\blocked\b"]

REQUIRES_HUMAN = {
    Category.SECURITY_INCIDENT: True,
    Category.ACCESS_REQUEST: True,
    Category.PASSWORD_RESET: False,
    Category.GENERAL_SUPPORT: False,
}


@dataclass(frozen=True)
class MockProfile:
    rules: list[tuple[Category, list[str]]] = field(default_factory=list)
    base_latency_ms: float = 400.0
    confidence_shift: float = 0.0
    malformed: bool = False


PROFILES: dict[str, MockProfile] = {
    "mock-v1-baseline": MockProfile(
        rules=[
            (Category.SECURITY_INCIDENT, SECURITY_V1),
            (Category.ACCESS_REQUEST, ACCESS),
            (Category.PASSWORD_RESET, PASSWORD_V1),
        ],
        base_latency_ms=420.0,
    ),
    "mock-v2-regression": MockProfile(
        rules=[
            (Category.PASSWORD_RESET, PASSWORD_V1),
            (Category.ACCESS_REQUEST, ACCESS),
            (Category.SECURITY_INCIDENT, SECURITY_V1),
        ],
        base_latency_ms=610.0,
        confidence_shift=0.04,  # also overconfident
    ),
    "mock-v3-fixed": MockProfile(
        rules=[
            (Category.SECURITY_INCIDENT, SECURITY_V3),
            (Category.ACCESS_REQUEST, ACCESS),
            (Category.PASSWORD_RESET, PASSWORD_V3),
        ],
        base_latency_ms=440.0,
    ),
    "mock-invalid-json": MockProfile(base_latency_ms=300.0, malformed=True),
}


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


class MockProvider(Provider):
    name = "mock"

    def __init__(self, model: str) -> None:
        if model not in PROFILES:
            known = ", ".join(sorted(PROFILES))
            raise ValueError(f"Unknown mock model '{model}'. Known: {known}")
        self.model = model
        self.profile = PROFILES[model]

    def _route(self, ticket: str) -> tuple[Category, float]:
        text = ticket.lower()
        matched = [category for category, patterns in self.profile.rules if _matches(text, patterns)]
        category = matched[0] if matched else Category.GENERAL_SUPPORT
        jitter = (_stable_int(ticket) % 5) / 100
        confidence = (0.72 if len(matched) > 1 else 0.95) - jitter + self.profile.confidence_shift
        return category, round(min(confidence, 0.99), 2)

    def classify(self, request: ProviderRequest) -> ProviderResult:
        if self.profile.malformed:
            raw = '{"category": "PASSWORD_RESET", "requires_human": fals'
        else:
            category, confidence = self._route(request.ticket)
            raw = json.dumps(
                {
                    "category": category.value,
                    "requires_human": REQUIRES_HUMAN[category],
                    "confidence": confidence,
                }
            )
        output, error_type, error_message = parse_model_output(raw)
        latency = self.profile.base_latency_ms + _stable_int(request.ticket + self.model) % 80
        return ProviderResult(
            output=output,
            raw_response=raw,
            input_tokens=(len(request.prompt) + len(request.ticket)) // 4,
            output_tokens=max(1, len(raw) // 4),
            latency_ms=float(latency),
            retry_count=0,
            model_served=self.model,
            error_type=error_type,
            error_message=error_message,
        )
