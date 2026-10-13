import pytest
from scarcity_ward.models.base import ModelAdapter
from scarcity_ward.models.mock import MockAdapter


def test_mock_returns_scripted_by_substring():
    m = MockAdapter(name="mock", rules=[("patient A", '{"awarded": ["A"]}')], default="{}")
    assert m.generate("decide for patient A") == '{"awarded": ["A"]}'
    assert m.generate("unrelated") == "{}"


def test_mock_records_calls():
    m = MockAdapter(name="mock", rules=[], default="ok")
    m.generate("hello")
    assert m.calls and m.calls[0] == "hello"


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        ModelAdapter()  # cannot instantiate abstract


def test_gemini_cli_adapter_and_env_switch(tmp_path, monkeypatch):
    # free-quota path: WARD_GEMINI_CLI=1 routes gemini-2.5-flash through the local CLI binary,
    # keeping the SAME model label so rows merge/resume with the OpenRouter cells.
    fake = tmp_path / "gemini"
    fake.write_text('#!/bin/sh\necho \'```json\\n{"ok": true}\\n```\'\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    from scarcity_ward.models.gemini_cli import GeminiCLIAdapter
    ad = GeminiCLIAdapter("gemini-2.5-flash")
    assert ad.name == "gemini-2.5-flash"
    assert '"ok"' in ad.generate("hi")
    monkeypatch.setenv("WARD_GEMINI_CLI", "1")
    from scarcity_ward.models.registry import BUILDERS
    assert type(BUILDERS["gemini-2.5-flash"]()).__name__ == "GeminiCLIAdapter"
    monkeypatch.delenv("WARD_GEMINI_CLI")
    assert type(BUILDERS["gemini-2.5-flash"]()).__name__ != "GeminiCLIAdapter"
