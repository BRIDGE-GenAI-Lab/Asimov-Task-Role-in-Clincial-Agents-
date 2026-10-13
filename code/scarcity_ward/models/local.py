from __future__ import annotations
import json
import time
import urllib.request
from typing import Optional
from .base import ModelAdapter


def parse_ollama_chat(d: dict) -> str:
    """Extract assistant text from an Ollama /api/chat or /api/generate response."""
    msg = d.get("message") or {}
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    if "response" in d:  # /api/generate shape
        return d["response"]
    raise ValueError(f"unexpected ollama response keys: {list(d)}")


class OllamaAdapter(ModelAdapter):
    """Local model via a running Ollama server (stdlib HTTP, no extra deps)."""

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 temperature: float = 0.7, num_ctx: int = 8192, num_predict: Optional[int] = None,
                 seed: Optional[int] = None, max_retries: int = 4, timeout: int = 600,
                 think: Optional[bool] = None):
        self.name = model
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict   # cap generated tokens (bounds verbose/reasoning local models)
        self.seed = seed
        self.max_retries = max_retries
        self.timeout = timeout
        self.think = think   # False disables the reasoning channel on thinking models (qwen3.5),
                             # return empty `content`. None omits the field (gemma etc. unaffected).

    def _options(self) -> dict:
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.num_predict is not None:
            opts["num_predict"] = self.num_predict
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": self._options()}
        if self.think is not None:
            payload["think"] = self.think
        last = None
        for attempt in range(self.max_retries):
            try:
                return parse_ollama_chat(self._post("/api/chat", payload))
            except Exception as e:  # network / 500 / transient
                last = e
                time.sleep(min(2 ** attempt, 15))
        raise last
