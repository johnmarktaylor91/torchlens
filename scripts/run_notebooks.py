"""Execute every user notebook with nbclient and report each failure by notebook and cell.

The user notebooks are ``notebooks/*.ipynb``, the learning path ``notebooks/learn/*.ipynb``
and every ``examples/**/*.ipynb`` (the developer audit notebooks under ``notebooks/audit/``
are not part of this set). Each notebook runs in its own child process with a per-cell
timeout and a whole-notebook timeout, so a hung kernel cannot stall the run. The nightly
CI leg runs this script;
locally it needs the ``test`` extra (nbclient, ipykernel)::

    python scripts/run_notebooks.py -j 2
    python scripts/run_notebooks.py notebooks/torchlens_in_10_minutes.ipynb
    python scripts/run_notebooks.py --write-outputs   # regenerate stored outputs
    python scripts/run_notebooks.py --write-outputs notebooks/learn/*.ipynb

Only the notebooks in ``STORED_OUTPUT_NOTEBOOKS`` keep outputs in git (the ten-minute
tutorial, the learning path, the multi-call bundle and activation patching notebooks the
README links, the Jacobian lens recipe, whose GPT-2 small fit takes minutes to rerun, the
Mamba-130m Jacobian lens Colab notebook, whose stored figures show before Colab runs anything, and
the logit lens recipe the README links;
GitHub renders them as the first impression).
``--write-outputs`` executes them and writes their outputs back in place with machine paths, timings,
widget state and other run-to-run noise scrubbed (``_scrub_outputs``); every other notebook is
committed without outputs (``tests/test_notebook_output_policy.py``).

Notebooks in ``HEAVY_NOTEBOOKS`` each run for minutes (batched backward passes through
GPT-2 small or Mamba-130m, or a long concept-erasure sweep). Two of them sharing a 4-CPU runner each slow
several-fold (past the cell timeout), so the runner executes the light notebooks in the ``-j`` pool first and then each heavy
notebook alone, one at a time. The per-cell and whole-notebook timeouts are the same for
both, so a heavy notebook that slows down still fails.

Exit status: 0 when every selected notebook ran clean, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STORED_OUTPUT_NOTEBOOKS = (
    "notebooks/torchlens_in_10_minutes.ipynb",
    "notebooks/learn/00_quickstart.ipynb",
    "notebooks/learn/01_capture_and_look.ipynb",
    "notebooks/learn/02_address_and_read.ipynb",
    "notebooks/learn/03_see_it.ipynb",
    "notebooks/learn/04_keep_what_you_need.ipynb",
    "notebooks/learn/05_change_and_rerun.ipynb",
    "notebooks/learn/06_gradients.ipynb",
    "notebooks/learn/07_compare_runs.ipynb",
    "notebooks/learn/08_semantic_shortcuts.ipynb",
    "notebooks/learn/09_scale.ipynb",
    "notebooks/learn/10_beyond_one_forward.ipynb",
    "notebooks/learn/11_dynamic_graphs.ipynb",
    "examples/workflows/multi_call_bundle.ipynb",
    "examples/workflows/paired_prompt_patching.ipynb",
    "examples/methods/jacobian_lens_recipe.ipynb",
    "examples/methods/jacobian_lens_mamba_colab.ipynb",
    "examples/methods/logit_lens_recipe.ipynb",
)
# Run alone, after the parallel pool: each runs for minutes, and sharing a 4-CPU runner with
# another of them pushed each past the 600 s cell timeout (the LEACE recipe also took about
# 450 s inside the pool, three quarters of the cap).
HEAVY_NOTEBOOKS = frozenset(
    {
        "examples/methods/attribution_graph_transcoders_recipe.ipynb",
        "examples/methods/eap_ig_circuit_recipe.ipynb",
        "examples/methods/jacobian_lens_mamba_colab.ipynb",
        "examples/methods/jacobian_lens_recipe.ipynb",
        "examples/methods/leace_concept_erasure_recipe.ipynb",
    }
)
DEFAULT_CELL_TIMEOUT = 600
DEFAULT_NOTEBOOK_TIMEOUT = 1200
_RESULT_PREFIX = "NOTEBOOK_RESULT "
# Every directory level under /tmp: workers run with TMPDIR set to a per-job subdirectory.
_TMP_DIRS = re.compile(r"/tmp/(?:[^/\s'\"]+/)+")
_GRAD_FN_ID = re.compile(r"\bgrad_fn_(\d{6,})(?!\d)")
_SCRUBBED_MIMES = ("application/json", "image/svg+xml")


@dataclass(frozen=True)
class NotebookResult:
    """Outcome of one notebook execution."""

    path: str
    status: str
    seconds: float
    cell: int | None = None
    detail: str = ""


def user_notebooks(root: Path = REPO_ROOT) -> list[Path]:
    """Return the user notebooks in a stable order."""

    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    notebooks += sorted((root / "notebooks" / "learn").glob("*.ipynb"))
    notebooks += sorted((root / "examples").rglob("*.ipynb"))
    return notebooks


def _scrub(text: str) -> str:
    """Replace machine-specific absolute paths with repo-relative or elided ones."""

    paths = sysconfig.get_paths()
    for prefix, replacement in (
        (str(REPO_ROOT) + "/", ""),
        (paths["purelib"], ".../site-packages"),
        (paths["stdlib"], f".../lib/python{sysconfig.get_python_version()}"),
    ):
        text = text.replace(prefix, replacement)
    text = _TMP_DIRS.sub("/tmp/", text)
    return text.replace(str(Path.home()), "~")


def _stable_grad_fn_ids(text: str, ids: dict[str, str]) -> str:
    """Replace ``grad_fn_<memory address>`` node ids with ids numbered by first appearance."""

    return _GRAD_FN_ID.sub(lambda m: ids.setdefault(m.group(1), f"grad_fn_{len(ids) + 1}"), text)


def _scrub_value(value: object, scrub: Callable[[str], str]) -> object:
    """Scrub a string or list-of-strings notebook field; leave other values alone."""

    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [scrub(item) if isinstance(item, str) else item for item in value]
    return value


def _coalesce_streams(outputs: list[Any]) -> list[Any]:
    """Merge adjacent stream outputs with the same name; the kernel splits them arbitrarily."""

    merged: list[Any] = []
    for output in outputs:
        last = merged[-1] if merged else None
        if (
            last is not None
            and output.get("output_type") == "stream" == last.get("output_type")
            and output.get("name") == last.get("name")
        ):
            last["text"] = "".join(last.get("text", "")) + "".join(output.get("text", ""))
        else:
            merged.append(output)
    return merged


def _scrub_outputs(notebook: Any) -> None:
    """Make an executed notebook's outputs machine-independent, in place.

    Drops machine paths, per-cell execution timings and the ipywidgets state (random model
    ids), merges the kernel's arbitrary stdout chunks and numbers ``grad_fn`` node ids.
    """

    ids: dict[str, str] = {}

    def scrub(text: str) -> str:
        return _stable_grad_fn_ids(_scrub(text), ids)

    notebook.get("metadata", {}).pop("widgets", None)
    for cell in notebook.cells:
        cell.get("metadata", {}).pop("execution", None)
        if "outputs" not in cell:
            continue
        cell["outputs"] = _coalesce_streams(cell["outputs"])
        for output in cell["outputs"]:
            for key in ("text", "evalue", "traceback"):
                if key in output:
                    output[key] = _scrub_value(output[key], scrub)
            data = output.get("data", {})
            for mime in list(data):
                if mime.startswith("text/") or mime in _SCRUBBED_MIMES:
                    data[mime] = _scrub_value(data[mime], scrub)


def _first_error(notebook: Any) -> tuple[int | None, str]:
    """Return the index and a short description of the first error output."""

    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                source = "".join(cell.get("source", "")).strip().splitlines()
                head = source[0][:120] if source else ""
                trace = "\n".join(output.get("traceback", [])[-6:])
                trace = re.sub(r"\x1b\[[0-9;]*m", "", trace)
                detail = f"{output.get('ename')}: {str(output.get('evalue'))[:400]}"
                return index, f"{detail}\n  first line: {head}\n{trace[-2000:]}"
    return None, ""


def execute_one(path: Path, cell_timeout: int, write_outputs: bool) -> NotebookResult:
    """Execute one notebook in this process and return its result."""

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=cell_timeout,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    start = time.monotonic()
    status, cell, detail = "ok", None, ""
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError, DeadKernelError) as exc:
        status = "timeout" if isinstance(exc, CellTimeoutError) else "error"
        cell, detail = _first_error(notebook)
        detail = detail or str(exc)[-2000:]
    seconds = round(time.monotonic() - start, 1)
    if write_outputs and status == "ok" and rel in STORED_OUTPUT_NOTEBOOKS:
        _scrub_outputs(notebook)
        nbformat.write(notebook, path)
    return NotebookResult(rel, status, seconds, cell, detail)


def _run_child(path: Path, args: argparse.Namespace) -> NotebookResult:
    """Run one notebook in a child process bounded by the whole-notebook timeout."""

    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    command = [sys.executable, __file__, "--child", "--cell-timeout", str(args.cell_timeout)]
    if args.write_outputs:
        command.append("--write-outputs")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [*command, str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=args.notebook_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        seconds = round(time.monotonic() - start, 1)
        detail = f"whole-notebook timeout of {args.notebook_timeout}s"
        return NotebookResult(rel, "timeout", seconds, None, detail)
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return NotebookResult(**json.loads(line[len(_RESULT_PREFIX) :]))
    seconds = round(time.monotonic() - start, 1)
    detail = f"child exited {proc.returncode}\n{proc.stderr[-2000:]}"
    return NotebookResult(rel, "crash", seconds, None, detail)


def _split_heavy(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split the selected notebooks into the parallel pool and the heavy ones run alone.

    Parameters
    ----------
    paths:
        Selected notebooks, in run order.

    Returns
    -------
    tuple[list[Path], list[Path]]
        The light notebooks and the ``HEAVY_NOTEBOOKS`` members, each in the given order.
    """

    light: list[Path] = []
    heavy: list[Path] = []
    for path in paths:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        (heavy if rel in HEAVY_NOTEBOOKS else light).append(path)
    return light, heavy


def _print_result(result: NotebookResult) -> None:
    """Print the one-line status of a finished notebook."""

    print(f"NB {result.status:7s} {result.seconds:7.1f}s {result.path}", flush=True)


def _report(results: list[NotebookResult]) -> int:
    """Print one line per notebook plus the failure details; return the exit status."""

    failed = [result for result in results if result.status != "ok"]
    for result in failed:
        where = f"cell {result.cell}" if result.cell is not None else "no cell recorded"
        print(f"\nFAILED {result.path} ({result.status}, {where})\n{result.detail}")
    total = sum(result.seconds for result in results)
    print(f"\n{len(results) - len(failed)} of {len(results)} notebooks ran clean ({total:.0f}s)")
    return 1 if failed else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks (default: all user)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="notebooks run at once")
    parser.add_argument("--cell-timeout", type=int, default=DEFAULT_CELL_TIMEOUT)
    parser.add_argument("--notebook-timeout", type=int, default=DEFAULT_NOTEBOOK_TIMEOUT)
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="write outputs back into the stored-output notebooks (all of them by default)",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected notebooks and return the exit status."""

    args = _parse_args(argv)
    if args.child:
        result = execute_one(args.paths[0], args.cell_timeout, args.write_outputs)
        print(_RESULT_PREFIX + json.dumps(result.__dict__), flush=True)
        return 0
    os.environ.setdefault("MPLBACKEND", "Agg")
    default = [REPO_ROOT / rel for rel in STORED_OUTPUT_NOTEBOOKS] if args.write_outputs else None
    paths = args.paths or default or user_notebooks()
    light, heavy = _split_heavy(list(paths))
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for result in pool.map(lambda path: _run_child(path, args), light):
            _print_result(result)
            results.append(result)
    for path in heavy:  # one at a time, with the pool drained
        result = _run_child(path, args)
        _print_result(result)
        results.append(result)
    return _report(results)


if __name__ == "__main__":
    sys.exit(main())
