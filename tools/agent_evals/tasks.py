"""Task definitions for the agent-from-docs harness.

Each task is a short prompt, the shared fixture model, and a grader. Graders compare the
script's outputs against oracles computed independently with plain torch (forward hooks and
direct calls), or against stable codes and schema ids the docs state. They never take the
expected value from the script under test.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

#: Fixture model written into every task directory as ``model.py``.
MODEL_SOURCE = '''"""Fixture model for one agent-eval task."""

import torch
from torch import nn


class TinyNet(nn.Module):
    """Two hidden ReLU layers and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 8)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.act2 = nn.ReLU()
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act2(self.fc2(self.act1(self.fc1(x)))))


def build_model() -> nn.Module:
    """Return the seeded fixture model in eval mode."""

    torch.manual_seed(0)
    return TinyNet().eval()


def make_input() -> torch.Tensor:
    """Return the fixed (4, 6) input batch."""

    generator = torch.Generator().manual_seed(1)
    return torch.randn(4, 6, generator=generator)
'''

#: Prompt preamble shared by every task. ``{docs_dir}`` is filled per run.
PREAMBLE = """You are writing one Python script that uses the TorchLens library (import torchlens as tl).
TorchLens and torch are installed. TorchLens's documentation is in the directory {docs_dir}; read it
to learn the API. Do not read TorchLens's source code or its installed package files, and do not
run Python: the documentation is all you get.

The working directory holds model.py, which defines build_model() (returns a small eval-mode torch
model with submodules fc1, act1, fc2, act2, head) and make_input() (returns its (4, 6) input
tensor). Import them with: from model import build_model, make_input

Write your script to solution.py in the working directory. It is run as `python solution.py` from
that directory with no arguments, must exit 0, and must finish within two minutes.

Task:
"""

Check = tuple[str, bool, str]


@dataclass(frozen=True)
class Task:
    """One harness task.

    Attributes
    ----------
    task_id:
        Stable identifier; also the reference solution's file stem.
    surface:
        The part of the documented surface the task exercises.
    prompt:
        Task text appended to :data:`PREAMBLE`.
    grade:
        Callable taking the task directory and returning ``(name, passed, detail)`` checks.
    """

    task_id: str
    surface: str
    prompt: str
    grade: Callable[[Path], list[Check]]

    def full_prompt(self, docs_dir: str) -> str:
        """Return the complete prompt for one run."""

        return PREAMBLE.format(docs_dir=docs_dir) + self.prompt.strip() + "\n"


def load_fixture(task_dir: Path) -> ModuleType:
    """Import the task directory's ``model.py`` under a unique module name."""

    path = task_dir / "model.py"
    spec = importlib.util.spec_from_file_location(f"agent_eval_model_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle(task_dir: Path, *, zero_act1: bool = False) -> dict[str, Any]:
    """Run the fixture with plain torch hooks and return independent reference facts."""

    fixture = load_fixture(task_dir)
    model = fixture.build_model()
    x = fixture.make_input()
    relu_outs: list[torch.Tensor] = []
    leaf_calls: list[str] = []
    handles = []
    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(
                module.register_forward_hook(lambda _m, _i, _o, name=name: leaf_calls.append(name))
            )
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(lambda _m, _i, out: relu_outs.append(out)))
    if zero_act1:
        handles.append(model.act1.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out)))
    with torch.no_grad():
        out = model(x)
    for handle in handles:
        handle.remove()
    return {"out": out, "relu_outs": relu_outs, "leaf_calls": leaf_calls}


def _read_json(task_dir: Path, checks: list[Check]) -> dict[str, Any] | None:
    """Read ``result.json``, recording a failed check when it is missing or malformed."""

    path = task_dir / "result.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        checks.append(("result.json readable", False, f"{type(exc).__name__}: {exc}"))
        return None
    if not isinstance(data, dict):
        checks.append(("result.json readable", False, "top level is not an object"))
        return None
    checks.append(("result.json readable", True, ""))
    return data


def _close(got: Any, want: float, *, tol: float = 1e-4) -> bool:
    """Return whether ``got`` is a finite number within ``tol`` (absolute and relative)."""

    if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got):
        return False
    return math.isclose(float(got), want, rel_tol=tol, abs_tol=tol)


def _check(checks: list[Check], name: str, passed: bool, detail: Any = "") -> None:
    checks.append((name, bool(passed), "" if passed else str(detail)))


def grade_capture_inspect(task_dir: Path) -> list[Check]:
    """Relu count, relu output shapes and output sum against hook-recorded values."""

    checks: list[Check] = []
    data = _read_json(task_dir, checks)
    if data is None:
        return checks
    oracle = _oracle(task_dir)
    want_shapes = [list(out.shape) for out in oracle["relu_outs"]]
    _check(checks, "relu_count", data.get("relu_count") == len(want_shapes), data.get("relu_count"))
    _check(checks, "relu_shapes", data.get("relu_shapes") == want_shapes, data.get("relu_shapes"))
    want_sum = float(oracle["out"].sum())
    _check(checks, "output_sum", _close(data.get("output_sum"), want_sum), data.get("output_sum"))
    return checks


def grade_save_load(task_dir: Path) -> list[Check]:
    """A saved artifact exists, and loaded-trace facts match the hook oracle."""

    import torchlens as tl

    checks: list[Check] = []
    artifact = task_dir / "run.tlspec"
    _check(checks, "run.tlspec written", artifact.exists(), "missing")
    data = _read_json(task_dir, checks)
    if data is None:
        return checks
    oracle = _oracle(task_dir)
    n_ops = len(oracle["leaf_calls"])
    _check(checks, "num_ops", data.get("num_ops") == n_ops, data.get("num_ops"))
    want = float(oracle["relu_outs"][0].sum())
    _check(
        checks,
        "first_relu_sum",
        _close(data.get("first_relu_sum"), want),
        data.get("first_relu_sum"),
    )
    if artifact.exists():
        try:
            loaded = tl.load(str(artifact))
            _check(checks, "artifact reloads", loaded.num_ops == n_ops, loaded.num_ops)
        except Exception as exc:  # noqa: BLE001 - the grader reports any load failure as a check
            _check(checks, "artifact reloads", False, f"{type(exc).__name__}: {exc}")
    return checks


def grade_intervention(task_dir: Path) -> list[Check]:
    """The edited output equals a forward hook that zeroes ``act1``'s output."""

    checks: list[Check] = []
    data = _read_json(task_dir, checks)
    if data is None:
        return checks
    want = _oracle(task_dir, zero_act1=True)["out"]
    clean = _oracle(task_dir)["out"]
    try:
        got = torch.tensor(data.get("output"), dtype=want.dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        _check(checks, "output shape", False, f"{type(exc).__name__}: {exc}")
        return checks
    _check(checks, "output shape", tuple(got.shape) == tuple(want.shape), list(got.shape))
    if tuple(got.shape) == tuple(want.shape):
        _check(
            checks,
            "output matches hook oracle",
            torch.allclose(got, want, rtol=1e-4, atol=1e-5),
            (got - want).abs().max().item(),
        )
        _check(
            checks,
            "output differs from clean run",
            not torch.allclose(got, clean, rtol=1e-4, atol=1e-5),
            "edit had no effect",
        )
    return checks


def grade_report_export(task_dir: Path) -> list[Check]:
    """Budgeted report, HTML viewer and per-op CSV are present and well formed."""

    checks: list[Check] = []
    report = task_dir / "report.txt"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    _check(checks, "report.txt written", bool(text), "missing or empty")
    _check(checks, "report keeps Capture status", "Capture status" in text, "section absent")
    approx_tokens = -(-len(text) // 4)
    _check(checks, "report within 300 tokens", approx_tokens <= 300, approx_tokens)
    html = task_dir / "trace.html"
    head = html.read_text(encoding="utf-8", errors="replace")[:4096] if html.exists() else ""
    _check(checks, "trace.html is HTML", "<html" in head.lower(), "missing or not HTML")
    n_ops = len(_oracle(task_dir)["leaf_calls"])
    rows: list[list[str]] = []
    ops_csv = task_dir / "ops.csv"
    if ops_csv.exists():
        with ops_csv.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.reader(handle) if row]
    _check(checks, "ops.csv has a header", len(rows) >= 1, "missing or empty")
    _check(checks, "ops.csv has one row per op", len(rows) - 1 == n_ops, len(rows) - 1)
    return checks


def grade_refusal(task_dir: Path) -> list[Check]:
    """The documented stable code and a remedy are recorded."""

    checks: list[Check] = []
    data = _read_json(task_dir, checks)
    if data is None:
        return checks
    _check(checks, "code", data.get("code") == "input_rung_conflict", data.get("code"))
    remedy = data.get("remedy")
    _check(checks, "remedy recorded", isinstance(remedy, str) and bool(remedy.strip()), remedy)
    _check(
        checks,
        "exception_type recorded",
        isinstance(data.get("exception_type"), str) and bool(data.get("exception_type")),
        data.get("exception_type"),
    )
    return checks


def grade_agent_tools(task_dir: Path) -> list[Check]:
    """Manifest preflight schema id and a complete, duplicate-free paged dump."""

    from torchlens.agent import call_tool

    checks: list[Check] = []
    data = _read_json(task_dir, checks)
    if data is None:
        return checks
    _check(
        checks,
        "manifest_schema",
        data.get("manifest_schema") == "torchlens.agent.overview_manifest.v1",
        data.get("manifest_schema"),
    )
    _check(checks, "first_page_rows", data.get("first_page_rows") == 2, data.get("first_page_rows"))
    labels = data.get("row_labels")
    if not isinstance(labels, list):
        _check(checks, "row_labels", False, labels)
        return checks
    _check(checks, "row_labels unique", len(labels) == len(set(labels)), "duplicates")
    artifact = task_dir / "run.tlspec"
    if not artifact.exists():
        _check(checks, "run.tlspec written", False, "missing")
        return checks
    whole = call_tool("torchlens_dump", {"path": str(artifact), "view": "full"})
    want = [row["layer_label"] for row in whole["data"]["ops"]]
    _check(checks, "row_labels complete", labels == want, f"{len(labels)} of {len(want)}")
    _check(checks, "pages", data.get("pages") == -(-len(want) // 2), data.get("pages"))
    return checks


TASKS: tuple[Task, ...] = (
    Task(
        "capture_inspect",
        "capture and inspect",
        """Capture the model's forward pass on make_input() with TorchLens, keeping every
activation. Write result.json with keys:
- relu_count: int, the number of relu operations TorchLens recorded;
- relu_shapes: the output shape of each of those relu operations in execution order, each a list
  of ints;
- output_sum: float, the sum of the model's output as read from the trace's output layer (do not
  call the model yourself).""",
        grade_capture_inspect,
    ),
    Task(
        "save_load",
        "save and load",
        """Capture the model's forward pass on make_input() keeping every activation, save the trace
to run.tlspec in the working directory, then load it back from disk. From the LOADED trace, write
result.json with keys:
- num_ops: int, the trace's count of operations (input and output rows excluded);
- first_relu_sum: float, the sum of the first relu operation's saved output.""",
        grade_save_load,
    ),
    Task(
        "intervention",
        "intervention",
        """Using a TorchLens capture-time intervention, replace the output of the act1 module (the
first ReLU) with zeros during one forward pass on make_input(). Write result.json with key
output: the model's final output under that intervention, read from the trace, as a nested list
of floats (4 rows of 3).""",
        grade_intervention,
    ),
    Task(
        "report_export",
        "report and export",
        """Capture the model's forward pass on make_input(), then:
1. write TorchLens's plain-language report of the trace to report.txt, budgeted to at most 300
   tokens;
2. export the trace as a self-contained HTML viewer to trace.html;
3. write the per-operation table (one row per counted operation; input and output rows excluded)
   to ops.csv with a header row.""",
        grade_report_export,
    ),
    Task(
        "refusal",
        "a documented refusal",
        """TorchLens refuses to trace when it is given both a real input and a declared input shape.
Do that on purpose: pass make_input() and also input_shape=(4, 6). Catch the refusal and write
result.json with keys:
- code: the refusal's stable machine-readable code;
- remedy: the refusal's remedy text;
- exception_type: the exception's class name.
The script must exit 0.""",
        grade_refusal,
    ),
    Task(
        "agent_tools",
        "agent tool surface",
        """Capture the model's forward pass on make_input() and save the trace to run.tlspec. Then
use TorchLens's agent tool surface through its Python transport (not the CLI and not MCP):
1. run the manifest-only overview preflight on the artifact and record the result's schema id;
2. dump the full view's op rows at most 2 rows per page, following the continuation until no
   pages remain.
Write result.json with keys: manifest_schema (the schema id from step 1), pages (the number of
pages fetched), first_page_rows (the row count on the first page) and row_labels (every row's
layer label across all pages, in order).""",
        grade_agent_tools,
    ),
)

TASKS_BY_ID: dict[str, Task] = {task.task_id: task for task in TASKS}
