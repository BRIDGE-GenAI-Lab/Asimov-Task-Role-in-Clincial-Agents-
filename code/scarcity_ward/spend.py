from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request


class SpendCapExceeded(RuntimeError):
    """The cap is reached, or the guard can no longer be trusted (see SpendBaselineError). Carries `.spent` when known."""

    def __init__(self, msg: str, spent: float | None = None):
        super().__init__(msg)
        self.spent = spent


class SpendBaselineError(SpendCapExceeded):
    """Key usage is below the recorded baseline (key rotated or recreated): spent cannot be measured, so halt."""


def _key_info(key_env: str = "OPENROUTER_API_KEY", base: str = "https://openrouter.ai/api/v1") -> dict:
    req = urllib.request.Request(base + "/key", headers={"Authorization": "Bearer " + os.environ[key_env]})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["data"]


def key_usage(fetch=_key_info) -> float:
    return float(fetch()["usage"])


def key_limit(fetch=_key_info):
    lim = fetch().get("limit")
    return None if lim is None else float(lim)


class SpendGuard:
    def __init__(self, cap_usd: float, baseline_usd: float, usage_fn):
        self.cap = float(cap_usd)
        self.baseline = float(baseline_usd)
        self.usage_fn = usage_fn

    def spent(self) -> float:
        usage = self.usage_fn()
        if usage < self.baseline:
            # A negative spend would silently grant extra headroom. Fail closed instead.
            raise SpendBaselineError(
                "key usage $%.2f is below the recorded baseline $%.2f (key rotated or recreated?); "
                "spend cannot be measured, so the run is halted. Re-baseline deliberately with "
                "`scripts/spend.py baseline --cap N --force`." % (usage, self.baseline))
        return usage - self.baseline

    def check(self) -> float:
        s = self.spent()
        if s >= self.cap:
            raise SpendCapExceeded("spent $%.2f >= cap $%.2f" % (s, self.cap), spent=s)
        return s


def guard_from_file(path: str, usage_fn=None) -> SpendGuard:
    with open(path) as fh:
        d = json.load(fh)
    return SpendGuard(d["cap_usd"], d["baseline_usd"], usage_fn or key_usage)


class Halter:

    def __init__(self, guard: SpendGuard, every: int = 20, retries: int = 3, backoff: float = 0.5,
                 sleep=None):
        self.guard, self.every = guard, every
        self.retries, self.backoff = max(1, retries), backoff
        self._sleep = sleep if sleep is not None else time.sleep     # resolved here so tests can patch it
        self._n = 0
        self.last_spent = None
        self.reason = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._probe()

    def _halt(self, reason: str, spent=None) -> None:
        self.reason = reason
        if spent is not None:
            self.last_spent = spent
        print("  [SPEND HALT] " + reason)
        self._stop.set()

    def _probe(self):
        last = None
        for i in range(self.retries):
            try:
                self.last_spent = self.guard.check()
                return
            except SpendCapExceeded as e:       # definitive answer: do not retry
                self._halt(str(e), e.spent)
                return
            except Exception as e:              # failed read of the usage endpoint
                last = e
                if i < self.retries - 1:
                    self._sleep(self.backoff * (2 ** i))
        # Name the error type only, never its text.
        self._halt("usage probe failed %d times (%s); cannot verify the spend cap" % (self.retries, type(last).__name__))
        raise last

    def ok(self) -> bool:
        return not self._stop.is_set()

    def after_attempt(self) -> None:
        with self._lock:
            self._n += 1
            due = self._n % self.every == 0
        if due:
            self._probe()

    after_write = after_attempt     # the name used by the brief; same behaviour

    def final_spent(self):
        """Best available final figure: a fresh read if possible, otherwise the last probe."""
        try:
            return self.guard.spent()
        except Exception:
            return self.last_spent


def exit_if_halted(halter) -> None:
    """No-op when there is no guard or the guard has not halted. Otherwise report the final spent figure and
    exit with status 3, so a `set -e` launcher stops too and a halted run never looks complete."""
    if halter is None or halter.ok():
        return
    sp = halter.final_spent()
    print("SPEND GUARD HALT: final spent %s of cap $%.2f; exiting with status 3"
          % ("$%.2f" % sp if sp is not None else "unknown", halter.guard.cap))
    sys.exit(3)


def halter_from_path(path, every: int, env: str = "WARD_SPEND_FILE"):
    """Build the run's Halter from a spend-baseline file path. An unset or empty path means no guard: warn on
    stderr and return None. Construction fails closed (Halter re-raises when the usage read cannot be verified).
    If the cap is already reached at start, nothing may launch: report and exit with status 3."""
    if not path:
        print("NO SPEND GUARD (%s unset)" % env, file=sys.stderr)
        return None
    halter = Halter(guard_from_file(path), every=every)
    if not halter.ok():
        print("spend cap already reached; nothing launched")
        exit_if_halted(halter)
    return halter


def halter_from_env(every: int, env: str = "WARD_SPEND_FILE"):
    """halter_from_path for the path named by environment variable `env`."""
    return halter_from_path(os.environ.get(env), every, env)
