from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from .base import ModelAdapter


def parse_openai_chat(d: dict) -> str:
    """Extract text from an OpenAI-compatible chat completion (OpenRouter)."""
    msg = d["choices"][0]["message"]
    c = msg.get("content")
    if c:
        return c
    for k in ("reasoning", "reasoning_content"):  # some reasoning models route text here
        if msg.get(k):
            return msg[k]
    return c or ""


class OpenRouterAdapter(ModelAdapter):
    """OpenRouter model via its OpenAI-compatible endpoint (stdlib HTTP, no SDK dep)."""

    def __init__(self, model: str, key_env: str = "OPENROUTER_API_KEY",
                 base: str = "https://openrouter.ai/api/v1", temperature=0.7,
                 max_retries: int = 8, timeout: int = 180, extra_body: dict = None,
                 provider: dict = None, max_tokens: int = None, record_usage: bool = False):
        self.name = model
        self.model = model
        self.key_env = key_env
        self.base = base.rstrip("/")
        self.temperature = temperature          # None = omit (provider default; some reasoning models reject it)
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_body = extra_body or {}  # extra payload fields, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        self.provider = provider                # OpenRouter provider routing, e.g. {"order": [...], "allow_fallbacks": False}
        self.max_tokens = max_tokens
        self.record_usage = record_usage
        self.call_log = []

    def _headers(self) -> dict:
        return {"Authorization": "Bearer " + os.environ[self.key_env],
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/scarcity-ward",
                "X-Title": "Scarcity Ward"}

    def build_payload(self, prompt: str) -> dict:
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.provider:
            payload["provider"] = self.provider
        if self.record_usage:
            payload["usage"] = {"include": True}
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def _record(self, d: dict) -> None:
        u = d.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        ch = (d.get("choices") or [{}])[0]
        self.call_log.append({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "requested_model": self.model, "returned_model": d.get("model"),
            "provider": d.get("provider"), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": det.get("reasoning_tokens"), "cost": u.get("cost"),
            "finish_reason": ch.get("finish_reason")})

    def _check_error_body(self, d: dict) -> None:
        """OpenRouter can return HTTP 200 with {"error": {...}} and no choices. Such a reply is never logged.
        402/401/403 inside the body are fatal (raised as the HTTPError the header form would give, so the
        handler below re-raises without retry); any other embedded error is retried like a transient failure."""
        if not (isinstance(d, dict) and "error" in d and not d.get("choices")):
            return
        err = d["error"]
        raw = err.get("code") if isinstance(err, dict) else None
        try:
            code = int(raw)
        except (TypeError, ValueError):
            code = None
        msg = str(err.get("message") if isinstance(err, dict) else err)[:200]
        if code in (401, 402, 403):
            raise urllib.error.HTTPError(self.base + "/chat/completions", code, msg, {}, None)
        raise RuntimeError("OpenRouter error body (code %s): %s" % (code, msg))

    def generate(self, prompt: str) -> str:
        data = json.dumps(self.build_payload(prompt)).encode()
        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(self.base + "/chat/completions",
                                             data=data, headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read().decode())
                    self._check_error_body(d)
                    if self.record_usage:
                        self._record(d)
                    return parse_openai_chat(d)
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 429:
                    body = ""
                    try:
                        body = e.read().decode()[:200].lower()
                    except Exception:
                        pass
                    if "per-day" in body or "daily" in body:
                        raise                            # daily cap: retrying is futile -> fail fast
                    # upstream provider throttle (temporary) -> honor Retry-After, keep retrying
                    try:
                        ra = float(e.headers.get("Retry-After", 0) or 0)
                    except Exception:
                        ra = 0
                    time.sleep(min(max(ra, 2 ** attempt * 3) + 1, 40))  # upstream/per-min limit -> back off
                elif e.code == 402:
                    raise                                   # out of credit: never retry
                elif 500 <= e.code < 600:
                    time.sleep(min(2 ** attempt, 20))
                else:
                    raise
            except Exception as e:
                last = e
                time.sleep(min(2 ** attempt, 20))
        raise last
