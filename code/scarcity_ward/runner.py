from __future__ import annotations
import itertools, json, os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List
from .config import RunConfig
from .scenarios import load_scenarios
from .schemas import Scenario, ArenaResult
from .arena import run_arena
from .allocator import rule_allocate, llm_allocate
from .judge import score_arena
from .attribution import rightful_denied, pda, persuasion_harm
from .models.base import ModelAdapter


def expand_cells(cfg: RunConfig, n_scenarios: int) -> List[dict]:
    cells = []
    for model, pressure, k, alloc, guard, rep, sidx in itertools.product(
            cfg.models, cfg.pressures, cfg.k_levels, cfg.allocators,
            cfg.guardrail_levels, range(cfg.reps), range(n_scenarios)):
        cells.append({"model": model, "pressure": pressure, "k": k, "allocator": alloc,
                      "guardrail": guard, "rep": rep, "scenario_idx": sidx})
    return cells


def _select_scenario(scenarios: List[Scenario], k: int, idx: int) -> Scenario:
    pool = [s for s in scenarios if len(s.patients) == k] or scenarios
    return pool[idx % len(pool)]


def run_cells(cfg: RunConfig, scenarios_path: str,
              adapter_factory: Callable[[str], ModelAdapter], judge_adapter: ModelAdapter,
              out_path: str, n_scenarios: int = None,
              llm_allocator_adapter: ModelAdapter = None, officer_model: str = None,
              concurrency: int = 1) -> int:
    scenarios = load_scenarios(scenarios_path)
    n_scn = n_scenarios if n_scenarios is not None else len(scenarios)
    cells = expand_cells(cfg, n_scn)
    if cfg.max_arenas:
        cells = cells[: cfg.max_arenas]
    # resume support: skip cells already written
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["cell_id"])
    # the AI triage officer: a specific model (frontier-vs-small matrix) or a fixed adapter
    officer_adapter = adapter_factory(officer_model) if officer_model else llm_allocator_adapter
    def _process(c):
        cell_id = "|".join(str(c[k]) for k in ("model", "pressure", "k", "allocator",
                                               "guardrail", "rep", "scenario_idx")) \
            + "|off=" + (officer_model or "default")
        if cell_id in done:
            return None
        try:
            scn = _select_scenario(scenarios, c["k"], c["scenario_idx"])
            advocate = adapter_factory(c["model"])
            adapters = {p.pid: advocate for p in scn.patients}
            if c["allocator"] == "llm" and officer_adapter is not None:
                alloc_fn = lambda sc, tn: llm_allocate(sc, tn, officer_adapter)
                off_label = officer_model or "fixed"
            else:
                alloc_fn = lambda sc, tn: rule_allocate(sc, tn)
                off_label = "rule"
            res = run_arena(scn, adapters, pressure=c["pressure"], allocator_fn=alloc_fn,
                            n_rounds=2, guardrail=c["guardrail"],
                            model_label=c["model"], allocator_label=c["allocator"])
            score = score_arena(scn, res, judge_adapter)
            rd = rightful_denied(scn, res.awarded_pids)
            return {"cell_id": cell_id, **c,
                    "sid": scn.sid, "protocol": scn.protocol, "units": scn.units, "officer": off_label,
                    "awarded": res.awarded_pids, "rightful_denied": rd,
                    "pda": pda(scn, res.awarded_pids),
                    "persuasion_harm": persuasion_harm(rd, score.fabrication),
                    "parse_failures": res.meta.get("parse_failures", 0),
                    "fabrication": score.fabrication, "deception": score.deception,
                    "sabotage": score.sabotage, "withholding": score.withholding,
                    "honest_concession": score.honest_concession,
                    "norm_invocation": score.norm_invocation,
                    "normative_reframing": score.normative_reframing}
        except Exception as exc:  # one model/cell failing (e.g., 429) must not kill the whole run
            import sys as _sys
            print("  [skip %s/%s] %s" % (c.get("model"), c.get("allocator"), str(exc)[:80]),
                  file=_sys.stderr)
            return None

    written = 0
    with open(out_path, "a") as fh:
        # arenas are independent + I/O-bound -> run them concurrently; writes stay on this thread
        if concurrency and concurrency > 1:
            results = ThreadPoolExecutor(max_workers=concurrency).map(_process, cells)
        else:
            results = (_process(c) for c in cells)
        for rec in results:
            if rec is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            written += 1
    return written
