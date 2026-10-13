from __future__ import annotations
from typing import List, Tuple
from .base import ModelAdapter


class MockAdapter(ModelAdapter):
    """Deterministic adapter: returns the value of the first rule whose substring is in the prompt."""

    def __init__(self, name: str, rules: List[Tuple[str, str]], default: str = "{}"):
        self.name = name
        self.rules = rules
        self.default = default
        self.calls: List[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        for needle, reply in self.rules:
            if needle in prompt:
                return reply
        return self.default
