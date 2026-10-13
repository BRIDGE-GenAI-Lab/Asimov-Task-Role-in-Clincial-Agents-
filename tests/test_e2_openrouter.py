# tests/test_e2_openrouter.py
import json
import re

from scarcity_ward.models.openrouter import OpenRouterAdapter

REPLY = {"model": "openai/gpt-5.1-20251113", "provider": "OpenAI",
         "choices": [{"message": {"content": '{"action":{"verb":"wait"}}'}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 420, "completion_tokens": 180, "cost": 0.0021,
                   "completion_tokens_details": {"reasoning_tokens": 64}}}


class _Resp:
    def __init__(self, d):
        self._b = json.dumps(d).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, captured, reply=REPLY):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def fake(req, timeout=None):
        captured.append(json.loads(req.data.decode()))
        return _Resp(reply)
    monkeypatch.setattr("urllib.request.urlopen", fake)


def test_payload_carries_provider_pin_usage_and_max_tokens(monkeypatch):
    cap = []
    _patch(monkeypatch, cap)
    a = OpenRouterAdapter("openai/gpt-5.1", provider={"order": ["openai"], "allow_fallbacks": False},
                          max_tokens=4096, record_usage=True)
    a.generate("hi")
    p = cap[0]
    assert p["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert p["max_tokens"] == 4096 and p["usage"] == {"include": True} and p["temperature"] == 0.7


def test_temperature_none_omits_the_field(monkeypatch):
    cap = []
    _patch(monkeypatch, cap)
    OpenRouterAdapter("x/y", temperature=None, record_usage=True).generate("hi")
    assert "temperature" not in cap[0]


def test_call_log_records_provider_and_returned_model(monkeypatch):
    cap = []
    _patch(monkeypatch, cap)
    a = OpenRouterAdapter("openai/gpt-5.1", record_usage=True)
    assert a.generate("hi") == '{"action":{"verb":"wait"}}'
    log = a.call_log[0]
    assert log["requested_model"] == "openai/gpt-5.1"
    assert log["returned_model"] == "openai/gpt-5.1-20251113" and log["provider"] == "OpenAI"
    assert log["prompt_tokens"] == 420 and log["completion_tokens"] == 180
    assert log["reasoning_tokens"] == 64 and log["cost"] == 0.0021 and log["finish_reason"] == "stop"
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", log["ts"])


def test_legacy_payload_is_unchanged(monkeypatch):
    cap = []
    _patch(monkeypatch, cap)
    a = OpenRouterAdapter("x/y")
    a.generate("hi")
    assert set(cap[0]) == {"model", "temperature", "messages"} and a.call_log == []


def test_rev_roster_key_builds_pinned_adapter(tmp_path, monkeypatch):
    roster = {"locked_at": "2026-10-03", "models": [
        {"key": "e2_dir-test", "slug": "x/y", "status": "locked",
         "provider": {"order": ["p"], "allow_fallbacks": False}, "extra_body": None},
        {"key": "e2_dir-cand", "slug": "x/z", "status": "candidate", "provider": None, "extra_body": None},
        {"key": "e2_dir-notemp", "slug": "x/r", "status": "locked", "provider": None, "extra_body": None,
         "supports_temperature": False}]}
    f = tmp_path / "roster.json"
    f.write_text(json.dumps(roster))
    monkeypatch.setenv("WARD_REV_ROSTER", str(f))
    monkeypatch.setenv("WARD_TEMPERATURE", "0")
    from scarcity_ward.models import registry
    a = registry.build_adapter("e2_dir-test")
    assert a.model == "x/y" and a.temperature == 0.0 and a.provider == {"order": ["p"], "allow_fallbacks": False}
    assert a.record_usage is True and a.max_tokens == 4096
    assert registry.build_adapter("e2_dir-notemp").temperature is None   # e.g. OpenAI GPT-6.x reject temperature
    try:
        registry.build_adapter("e2_dir-cand")
        raise AssertionError("candidate (unlocked) roster entries must not build")
    except KeyError:
        pass


import urllib.error

import pytest


def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)


def _scripted_urlopen(monkeypatch, steps):
    """steps: list of dicts (returned as 200 bodies) or Exceptions (raised). Returns the request counter."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    seen = []

    def fake(req, timeout=None):
        step = steps[len(seen)]
        seen.append(json.loads(req.data.decode()))
        if isinstance(step, Exception):
            raise step
        return _Resp(step)
    monkeypatch.setattr("urllib.request.urlopen", fake)
    return seen


def test_http_402_raises_after_exactly_one_request(monkeypatch):
    _no_sleep(monkeypatch)
    err = urllib.error.HTTPError("https://openrouter.ai/api/v1/chat/completions", 402,
                                 "Payment Required", {}, None)
    seen = _scripted_urlopen(monkeypatch, [err] * 8)
    a = OpenRouterAdapter("x/y", record_usage=True, max_retries=8)
    with pytest.raises(urllib.error.HTTPError) as ei:
        a.generate("hi")
    assert ei.value.code == 402 and len(seen) == 1 and a.call_log == []


def test_body_embedded_402_raises_after_one_request_and_logs_nothing(monkeypatch):
    _no_sleep(monkeypatch)
    body = {"error": {"code": 402, "message": "Insufficient credits"}}
    seen = _scripted_urlopen(monkeypatch, [body] * 8)
    a = OpenRouterAdapter("x/y", record_usage=True, max_retries=8)
    with pytest.raises(urllib.error.HTTPError) as ei:
        a.generate("hi")
    assert ei.value.code == 402 and len(seen) == 1 and a.call_log == []


@pytest.mark.parametrize("code", [401, 403])
def test_body_embedded_auth_errors_are_not_retried(monkeypatch, code):
    _no_sleep(monkeypatch)
    seen = _scripted_urlopen(monkeypatch, [{"error": {"code": code, "message": "nope"}}] * 8)
    a = OpenRouterAdapter("x/y", record_usage=True, max_retries=8)
    with pytest.raises(urllib.error.HTTPError) as ei:
        a.generate("hi")
    assert ei.value.code == code and len(seen) == 1 and a.call_log == []


def test_body_embedded_502_is_retried_and_only_the_success_is_logged(monkeypatch):
    _no_sleep(monkeypatch)
    bad = {"error": {"code": 502, "message": "Provider returned error"}}
    seen = _scripted_urlopen(monkeypatch, [bad, bad, REPLY])
    a = OpenRouterAdapter("openai/gpt-5.1", record_usage=True, max_retries=5)
    assert a.generate("hi") == '{"action":{"verb":"wait"}}'
    assert len(seen) == 3 and len(a.call_log) == 1
    assert a.call_log[0]["returned_model"] == "openai/gpt-5.1-20251113" and a.call_log[0]["cost"] == 0.0021


def test_persistent_body_error_exhausts_retries_without_logging(monkeypatch):
    _no_sleep(monkeypatch)
    bad = {"error": {"code": 502, "message": "Provider returned error"}}
    seen = _scripted_urlopen(monkeypatch, [bad] * 3)
    a = OpenRouterAdapter("x/y", record_usage=True, max_retries=3)
    with pytest.raises(Exception):
        a.generate("hi")
    assert len(seen) == 3 and a.call_log == []


def test_paid_reply_that_fails_to_parse_is_still_logged(monkeypatch):
    # a reply with choices but no usable message is a billed call: it must reach call_log before parsing fails
    _no_sleep(monkeypatch)
    odd = dict(REPLY, choices=[{"finish_reason": "length"}])
    _scripted_urlopen(monkeypatch, [odd] * 2)
    a = OpenRouterAdapter("x/y", record_usage=True, max_retries=2)
    with pytest.raises(Exception):
        a.generate("hi")
    assert len(a.call_log) == 2 and a.call_log[0]["finish_reason"] == "length"
