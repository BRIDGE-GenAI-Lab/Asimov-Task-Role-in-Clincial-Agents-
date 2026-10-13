from __future__ import annotations
import os


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (no python-dotenv dependency). Does not override existing env vars."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v and k not in os.environ:
                        os.environ[k] = v
    except FileNotFoundError:
        pass
