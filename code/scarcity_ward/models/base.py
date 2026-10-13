from __future__ import annotations
import abc


class ModelAdapter(abc.ABC):
    name: str

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        ...
