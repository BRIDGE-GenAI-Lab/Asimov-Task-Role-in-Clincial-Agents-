from __future__ import annotations

import subprocess
import time

from .base import ModelAdapter


class GeminiCLIAdapter(ModelAdapter):
    def __init__(self, model: str, timeout: int = 180, attempts: int = 3):
        self.name = model
        self._timeout = timeout
        self._attempts = attempts

    def generate(self, prompt: str) -> str:
        last = ""
        for i in range(self._attempts):
            try:
                r = subprocess.run(["gemini", "-m", self.name, "-p", prompt],
                                   capture_output=True, text=True, timeout=self._timeout)
                out = (r.stdout or "").strip()
                if r.returncode == 0 and out:
                    return out
                last = (r.stderr or out or f"exit {r.returncode}").strip()[-300:]  # tail: quota errors print last
            except subprocess.TimeoutExpired:
                last = f"timeout after {self._timeout}s"
            time.sleep(2 ** (i + 1))                 # quota hiccups: brief backoff
        raise RuntimeError(f"gemini CLI failed after {self._attempts} attempts: {last}")
