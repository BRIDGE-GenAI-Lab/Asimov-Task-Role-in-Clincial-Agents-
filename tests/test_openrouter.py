import pytest
from scarcity_ward.models.openrouter import parse_openai_chat, OpenRouterAdapter


def test_parse_openai_chat_content():
    assert parse_openai_chat({"choices": [{"message": {"content": "hi"}}]}) == "hi"


def test_parse_openai_chat_reasoning_fallback():
    assert parse_openai_chat({"choices": [{"message": {"content": "", "reasoning": "r"}}]}) == "r"


def test_adapter_headers_and_payload(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    a = OpenRouterAdapter("nvidia/nemotron-3-ultra-550b-a55b:free")
    h = a._headers()
    assert h["Authorization"] == "Bearer sk-test" and a.model.endswith(":free")
