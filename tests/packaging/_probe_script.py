"""Probe run INSIDE a freshly installed venv (never imported by the test suite).

Usage: ``python -I _probe_script.py SPEC_JSON``. Prints one line
``PROBE-RESULT <json>`` describing the import, the health call, one trace, one
save and load round trip, and the outcome of every dependency gate in SPEC.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any


def _resolve(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _gate_outcome(gate: dict[str, Any], missing_error: type[BaseException]) -> dict[str, Any]:
    try:
        if gate.get("check") == "import":
            importlib.import_module(gate["module"])
        elif "access" in gate:
            _resolve(gate["access"])
        else:
            _resolve(gate["call"])(*gate.get("call_args", []))
            if "cleanup" in gate:
                _resolve(gate["cleanup"])()
    except missing_error as exc:
        fields = getattr(exc, "fields", {})
        return {
            "outcome": "refused",
            "class": type(exc).__name__,
            "code": fields.get("code"),
            "dependency": fields.get("dependency"),
        }
    except Exception as exc:  # reported to the test, which fails on it
        return {"outcome": "error", "class": type(exc).__name__, "message": str(exc)[:400]}
    return {"outcome": "resolved"}


def main() -> None:
    spec = json.loads(sys.argv[1])
    import torch
    from torch import nn

    import torchlens as tl

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchlens_file": tl.__file__,
    }
    report = tl.report.doctor()
    result["doctor"] = type(report).__name__
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    trace = tl.trace(model, torch.randn(2, 4))
    trace.save(spec["save_path"])
    loaded = tl.load(spec["save_path"])
    result["num_ops"] = [trace.num_ops, loaded.num_ops]
    missing_error = tl.errors.MissingDependencyError
    result["gates"] = {
        name: _gate_outcome(gate, missing_error) for name, gate in spec["gates"].items()
    }
    print("PROBE-RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
