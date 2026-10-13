"""Claude Code CLI adapter - routes ward prompts through the local `claude` binary in headless
print mode (`claude -p <prompt> --model <alias>`), using the user's Claude subscription (no API
key, no per-token cost). Successor-agnostic: `--model opus` resolves to the current Opus.

Run from a NEUTRAL cwd: invoked inside the study repo, `claude -p` becomes repo-aware and refuses
to role-play a study subject ("I'm the coding assistant for this repo, not a ward endpoint");
from an empty dir with no repo context it returns the ward-contract JSON directly (verified).
The ward parser (parsing.extract_json) is lenient to any preamble, so no output-format flag is
needed. Pure stdlib."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

from .base import ModelAdapter

# neutral, repo-free working directory so `claude -p` does not detect the study repo and refuse
_CLAUDE_CWD = os.path.join(tempfile.gettempdir(), "claude_cli_neutral_cwd")
os.makedirs(_CLAUDE_CWD, exist_ok=True)


class ClaudeCLIAdapter(ModelAdapter):
    def __init__(self, model: str = "opus", timeout: int = 240, attempts: int = 2):
        self.name = f"claude-cc-{model}"
        self._model = model
        self._timeout = timeout
        self._attempts = attempts

    def generate(self, prompt: str) -> str:
        last = ""
        for i in range(self._attempts):
            try:
                # Prompt piped via stdin (not argv): avoids arg-length limits and the
                # "no stdin data received" failure seen under sustained concurrent load.
                # --strict-mcp-config: skip loading the user's MCP servers on every call
                # (faster cold start; the ward prompt needs no tools/MCP).
                r = subprocess.run(
                    ["claude", "-p", "--model", self._model, "--strict-mcp-config"],
                    input=prompt, capture_output=True, text=True,
                    timeout=self._timeout + 30, cwd=_CLAUDE_CWD)
                out = (r.stdout or "").strip()
                if r.returncode == 0 and out:
                    return out
                last = (r.stderr or out or f"exit {r.returncode}").strip()[-300:]
            except subprocess.TimeoutExpired:
                last = f"timeout after {self._timeout}s"
            time.sleep(2 ** (i + 1))
        raise RuntimeError(f"claude CLI failed after {self._attempts} attempts: {last}")
