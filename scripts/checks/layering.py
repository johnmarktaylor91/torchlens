"""Layering ratchet: import-cycle size and deferred-import count only shrink.

``check`` builds the package import graph with ``tools/layering/graph.py``
(module-top and in-function imports both count; ``TYPE_CHECKING`` imports do
not) and compares two numbers with ``scripts/checks/baselines.json``: the
size of the largest strongly connected component and the number of
in-function internal import statements. Either rising above its baseline
fails.

``check_contracts`` runs import-linter over ``.importlinter`` when it is
installed and reports a skip when it is not.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FAIL, OK, SKIP, Outcome, load_baselines, ratchet  # noqa: E402

PACKAGE = "torchlens"
GRAPH_SOURCE = Path(__file__).resolve().parents[2] / "tools" / "layering" / "graph.py"


def _load_graph_module() -> ModuleType:
    """Load ``tools/layering/graph.py`` by path (this module shares its package name)."""

    spec = importlib.util.spec_from_file_location("_tl_layering_graph", GRAPH_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GRAPH_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure(repo: Path) -> dict[str, int]:
    """Return the ratcheted layering numbers for the package under ``repo``."""

    graph = _load_graph_module()
    summary = graph.summarize(graph.build_graph(repo / PACKAGE))
    return {
        "largest_scc": summary["largest_scc"],
        "in_function_internal_imports": summary["in_function_internal_imports"],
    }


def check(repo: Path) -> Outcome:
    """Compare the measured layering numbers with their baselines."""

    return ratchet("layering", measure(repo), load_baselines(repo)["layering"])


def check_contracts(repo: Path) -> Outcome:
    """Run ``lint-imports`` over ``.importlinter``; skip when it is not installed."""

    config = repo / ".importlinter"
    if not config.is_file():
        return Outcome(SKIP, "import-contracts: no .importlinter config")
    executable = shutil.which("lint-imports")
    if executable is None or importlib.util.find_spec("importlinter") is None:
        return Outcome(SKIP, "import-contracts: import-linter not installed")
    proc = subprocess.run(
        [executable, "--config", str(config), "--no-cache"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = tuple(line for line in proc.stdout.splitlines() if line.strip())
    kept = next((line for line in reversed(lines) if "kept" in line), "no summary line")
    if proc.returncode != 0:
        return Outcome(FAIL, f"import-contracts: {kept}", lines[-40:])
    return Outcome(OK, f"import-contracts: {kept}")


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    outcomes = (check(repo), check_contracts(repo))
    for outcome in outcomes:
        print(f"[{outcome.status.upper():4}] {outcome.message}")
        for line in outcome.details:
            print(f"       {line}")
    return 1 if any(o.failed for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
