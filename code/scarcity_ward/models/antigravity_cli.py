"""Antigravity CLI adapter — routes calls through the authenticated `agy` binary (Google
Antigravity, the successor to the Gemini CLI as of 2026-06-18). Uses the user's Pro/Ultra
subscription quota, no API key. `agy --model <id> -p <prompt>` runs one prompt non-interactively
and prints the completion to stdout; the ward parser extracts the JSON action downstream.
Exposes a richer panel than the old gemini CLI: Gemini 3.5 Flash, Gemini 3.1 Pro, Claude
Opus/Sonnet 4.6, GPT-OSS 120B. Pure stdlib."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

from .base import ModelAdapter

# agy resolves models from workspace-scoped settings; this repo carries a `.antigravitycli`
# config that hides the subscription model slugs (e.g. claude-opus-4.6, gemini-3.1-pro-high)
# so `agy --model <slug>` fails with "not recognized" when run from the repo. Run agy from a
# neutral empty dir so it uses the global model registry (verified: clean JSON completions).
_AGY_CWD = os.path.join(tempfile.gettempdir(), "agy_neutral_cwd")
os.makedirs(_AGY_CWD, exist_ok=True)


class AntigravityCLIAdapter(ModelAdapter):
    def __init__(self, model: str, timeout: int = 600, attempts: int = 2):
        self.name = model
        self._timeout = timeout
        self._attempts = attempts

    def generate(self, prompt: str) -> str:
        last = ""
        for i in range(self._attempts):
            try:
                # --print-timeout bounds the agent's single-turn wait; -p prints and exits.
                r = subprocess.run(
                    ["agy", "--model", self.name, "--print-timeout", f"{self._timeout}s", "-p", prompt],
                    capture_output=True, text=True, timeout=self._timeout + 30, cwd=_AGY_CWD)
                out = (r.stdout or "").strip()
                if r.returncode == 0 and out:
                    return out
                last = (r.stderr or out or f"exit {r.returncode}").strip()[-300:]
            except subprocess.TimeoutExpired:
                last = f"timeout after {self._timeout}s"
            time.sleep(2 ** (i + 1))
        raise RuntimeError(f"agy CLI failed after {self._attempts} attempts: {last}")
