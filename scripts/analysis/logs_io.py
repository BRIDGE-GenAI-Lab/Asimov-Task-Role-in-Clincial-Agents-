"""Log helpers shared by the analysis scripts: load_jsonl reads a JSON-lines file and skips malformed lines, and primary_files returns the (model, log file) list of the primary ladder from scripts/analysis/ladder_pooled.py. It writes nothing."""
import ast
import json
import pathlib


def primary_files(src="scripts/analysis/ladder_pooled.py"):
    tree = ast.parse(pathlib.Path(src).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FILES" for t in node.targets):
            return [tuple(x) for x in ast.literal_eval(node.value)]
    raise ValueError("FILES list not found in " + src)


def load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows
