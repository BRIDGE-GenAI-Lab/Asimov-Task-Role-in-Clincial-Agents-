import pytest
from scarcity_ward.models.local import parse_ollama_chat, OllamaAdapter


def test_parse_chat_shape():
    assert parse_ollama_chat({"message": {"role": "assistant", "content": "hi"}}) == "hi"


def test_parse_generate_shape():
    assert parse_ollama_chat({"response": "yo"}) == "yo"


def test_parse_raises_on_unexpected():
    with pytest.raises(ValueError):
        parse_ollama_chat({"weird": 1})


def test_adapter_builds_options_with_seed():
    a = OllamaAdapter("gemma3", temperature=0.0, seed=7)
    opts = a._options()
    assert opts["temperature"] == 0.0 and opts["seed"] == 7 and "num_ctx" in opts


def test_adapter_options_include_num_predict_when_set():
    a = OllamaAdapter("gemma3", num_predict=256)
    assert a._options()["num_predict"] == 256
    assert "num_predict" not in OllamaAdapter("gemma3")._options()   # default: unbounded


def test_adapter_generate_uses_post(monkeypatch):
    a = OllamaAdapter("gemma3")
    captured = {}
    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"message": {"content": "ok"}}
    monkeypatch.setattr(a, "_post", fake_post)
    assert a.generate("hello") == "ok"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["messages"][0]["content"] == "hello"
