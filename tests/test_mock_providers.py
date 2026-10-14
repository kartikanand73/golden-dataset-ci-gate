import httpx
import pytest

from app.evaluator import run_evaluation
from app.providers.base import ProviderRequest, parse_model_output, post_json_with_retries
from app.providers.mock import MockProvider
from app.schemas import load_dataset
from tests.conftest import CONFIG, GOLDEN


def test_mock_providers_are_deterministic():
    tickets = [c.ticket for c in load_dataset(GOLDEN)]
    for model in ("mock-v1-baseline", "mock-v2-regression", "mock-v3-fixed"):
        provider = MockProvider(model)
        first = [provider.classify(ProviderRequest(model, "prompt", t)) for t in tickets]
        second = [MockProvider(model).classify(ProviderRequest(model, "prompt", t)) for t in tickets]
        assert [(r.raw_response, r.latency_ms, r.input_tokens) for r in first] == \
               [(r.raw_response, r.latency_ms, r.input_tokens) for r in second]


def test_demo_profiles_have_expected_quality(mock_runs):
    import json
    summaries = {m: json.loads((p / "summary.json").read_text()) for m, p in mock_runs.items()}
    assert summaries["mock-v1-baseline"]["critical_safety_errors"] == 0
    assert summaries["mock-v2-regression"]["critical_safety_errors"] >= 1
    assert summaries["mock-v3-fixed"]["critical_safety_errors"] == 0
    assert summaries["mock-v3-fixed"]["category_accuracy"] >= summaries["mock-v1-baseline"]["category_accuracy"]


def test_unknown_mock_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown mock model"):
        MockProvider("mock-v9")


def test_invalid_model_json_is_recorded_as_failure(tmp_path):
    run = run_evaluation(CONFIG, GOLDEN, tmp_path / "bad", model_override="mock-invalid-json")
    assert run.summary["schema_failures"] == 60
    assert run.summary["successful_call_rate"] == 0
    assert run.summary["category_accuracy"] == 0
    record = run.records[0]
    assert record.candidate is None
    assert record.error_type == "invalid_json"
    assert record.correct is False
    assert record.critical_safety_error is False  # fail-closed, counted as a schema failure


def test_parser_rejects_schema_violations():
    out, err, _ = parse_model_output('{"category": "BILLING", "requires_human": true, "confidence": 0.9}')
    assert out is None and err == "schema_error"
    out, err, _ = parse_model_output('{"category": "PASSWORD_RESET", "requires_human": "no", "confidence": 0.9}')
    assert out is None and err == "schema_error"
    out, err, _ = parse_model_output('```json\n{"category": "PASSWORD_RESET", "requires_human": false, "confidence": 0.9}\n```')
    assert out is not None and err is None


def test_http_helper_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    outcome = post_json_with_retries(client, "https://example.test", {}, {}, 5, 2, sleep=lambda s: None)
    assert outcome.body == {"ok": True}
    assert outcome.retry_count == 2


def test_http_helper_reports_exhausted_retries():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    outcome = post_json_with_retries(client, "https://example.test", {}, {}, 5, 1, sleep=lambda s: None)
    assert outcome.body is None
    assert outcome.retry_count == 1
    assert outcome.error_type == "http_error"
