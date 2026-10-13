from __future__ import annotations

import os
import subprocess
import tempfile
import time

from .base import ModelAdapter

# neutral, repo-free working directory so `codex exec` does not detect the study repo and refuse
_CODEX_CWD = os.path.join(tempfile.gettempdir(), "codex_cli_neutral_cwd")
os.makedirs(_CODEX_CWD, exist_ok=True)


class CodexCLIAdapter(ModelAdapter):
    def __init__(self, model: str = "gpt-5.6-luna", effort: str = "medium",
                 timeout: int = 300, attempts: int = 2):
        # env overrides let a collaborator retarget without touching the registry
        self._model = os.environ.get("CODEX_WARD_MODEL", model)
        self._effort = os.environ.get("CODEX_WARD_EFFORT", effort)
        self.name = f"codex-{self._model}-{self._effort}"
        self._timeout = timeout
        self._attempts = attempts

    def generate(self, prompt: str) -> str:
        last = ""
        for i in range(self._attempts):
            out_path = os.path.join(_CODEX_CWD, f"last_{os.getpid()}_{i}.txt")
            try:
                cmd = [
                    "codex", "exec",
                    "-m", self._model,
                    "-c", f'model_reasoning_effort="{self._effort}"',
                    "-c", 'approval_policy="never"',
                    "--sandbox", "read-only",
                    "--skip-git-repo-check",
                    "--ephemeral",                 # do not persist session files
                    "-o", out_path,                # write ONLY the final message here
                    "-",                           # read the prompt from stdin
                ]
                r = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True,
                    timeout=self._timeout + 30, cwd=_CODEX_CWD)
                msg = ""
                if os.path.exists(out_path):
                    with open(out_path) as fh:
                        msg = fh.read().strip()
                    os.remove(out_path)
                # prefer the clean last-message file; fall back to stdout if empty
                out = msg or (r.stdout or "").strip()
                if out:
                    return out
                last = (r.stderr or f"exit {r.returncode}").strip()[-300:]
            except subprocess.TimeoutExpired:
                last = f"timeout after {self._timeout}s"
            finally:
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
            time.sleep(2 ** (i + 1))
        raise RuntimeError(f"codex CLI failed after {self._attempts} attempts: {last}")
