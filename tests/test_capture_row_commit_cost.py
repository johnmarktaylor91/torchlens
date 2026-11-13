"""Deterministic per-op cost counters for the row-direct exhaustive commit.

The row-direct series (capture C2a, C2b, C2c) commits each exhaustive op as a
store row plus a row handle instead of freezing an ``OpRecord`` and scattering
it back into cells at ingest. These tests pin, on a tiny GPT-2 ``generate``,
what one function-call op costs in Python calls into TorchLens code and in
container builds (``BUILD_*`` opcodes executed in TorchLens code plus Python
``__init__`` runs TorchLens code starts): once for the whole trace (capture and
postprocess; dropping the trace is excluded) and once for the commit alone
(``ExhaustiveOpDraft.freeze``). The pins are the values measured on CI's pinned
environment after the commit cost cut. The trace pins are below the eager
freeze path the series replaced, measured the same way on the same subject.
The commit pins are above that path's freeze alone (33.79 calls and 28.21
builds per op) because the commit now writes the op's cells, which the eager
path wrote at ingest; they are below the row commit before the cut (68.41
calls and 56.47 builds per op).

Source tensors and module-boundary outputs have their own builders (no function
call: ``func_call_id`` is None) and are not counted as ops.
"""

from __future__ import annotations

import dis
import importlib
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc

import torchlens as tl
import torchlens.backends.torch.ops as torch_ops
from torchlens.backends.torch import _thin_emit

_TORCHLENS_ROOT = os.path.dirname(tl.__file__)
_BUILD_OPCODES = frozenset(
    dis.opmap[name]
    for name in (
        "BUILD_TUPLE",
        "BUILD_LIST",
        "BUILD_MAP",
        "BUILD_SET",
        "BUILD_CONST_KEY_MAP",
        "BUILD_STRING",
        "BUILD_SLICE",
    )
    if name in dis.opmap
)
# Per function-call op on the tiny GPT-2 generate (416 ops), CI's pinned
# environment (torch 2.13.0+cpu). Trace = capture plus postprocess. Each pin is
# the printed two-decimal value plus 0.01 (the print rounds); a pin measured on
# several workers takes the highest.
# The container-builds pins are measured in a fresh interpreter with a fixed
# hash seed, so earlier tests in the same worker cannot move them: a worker that
# has imported the tensor-parallel or pipelining namespaces (the capability
# snapshot's forced probes do) runs per-module checks of the capture's
# distributed guard that an ordinary process skips. The second pin imports
# those namespaces first, so a regression in that state still fails. The count
# is the same with or without the nightly env's ``test`` extra installed: the
# session tensor walk no longer follows a reachable logger into the process
# logging registry, whose size grew with ``import accelerate``. Measured on
# 70543d2222 in both envs: 943.39 plain, 945.03 with the namespaces.
_MAX_TRACE_CALLS_PER_OP = 3337.07
_MAX_TRACE_BUILDS_PER_OP = 943.40
_MAX_TRACE_BUILDS_PER_OP_DISTRIBUTED = 945.04
_MAX_COMMIT_CALLS_PER_OP = 35.72
_MAX_COMMIT_BUILDS_PER_OP = 30.91
#: The distributed namespaces the capture's distributed guard checks, as a worker
#: holds them after the capability snapshot's forced probes (``tl.report.doctor()``).
_DISTRIBUTED_NAMESPACES = (
    "torch.distributed._shard",
    "torch.distributed.device_mesh",
    "torch.distributed.tensor",
    "torch.distributed.tensor.parallel",
    "torch.distributed.pipelining",
)
_FRESH_MARK = "[row-commit-fresh]"


def _tiny_gpt2_generate() -> tuple[Callable[..., Any], torch.Tensor, dict[str, Any]]:
    """A random-weight two-layer GPT-2 and its bound ``generate`` (three new tokens)."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2, pad_token_id=0
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    kwargs = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0}
    return model.generate, torch.tensor([[3, 7, 1, 9]]), kwargs


class _Window:
    """Counts for one measured trace: ops, and per-scope tallies (trace, commit)."""

    def __init__(self) -> None:
        """Start every tally at zero on the calling thread."""

        self.owner = threading.get_ident()
        self.ops = 0
        self.commit_depth = 0
        self.trace = 0
        self.commit = 0

    def tally(self) -> None:
        """Count one event in the trace scope, and in the commit scope while inside it."""

        self.trace += 1
        if self.commit_depth:
            self.commit += 1


def _measure(
    monkeypatch: pytest.MonkeyPatch, install: Callable[[_Window], Callable[[], None]]
) -> _Window:
    """Trace the tiny GPT-2 generate once with ``install``'s hook on; return the window.

    ``install`` sets the hook and returns its remover. The trace is held until
    the hook is off, so its teardown is not counted. Today's emit is pinned: the
    window counts ops and the commit scope at its freeze, which the thin emit's
    rows skip (they are checked against today's in ``tests/test_thin_emit_soundness.py``).
    """

    previous_thin = _thin_emit.set_thin_emit_enabled(False)
    try:
        return _measure_today_emit(monkeypatch, install)
    finally:
        _thin_emit.set_thin_emit_enabled(previous_thin)


def _measure_today_emit(
    monkeypatch: pytest.MonkeyPatch, install: Callable[[_Window], Callable[[], None]]
) -> _Window:
    """The body of :func:`_measure`, with today's emit already pinned."""

    fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    window = _Window()
    real_freeze = torch_ops.ExhaustiveOpDraft.freeze

    def freeze(draft: Any) -> Any:
        """Count one function-call op and mark the commit scope around the real commit."""

        if draft.fields_dict.get("func_call_id") is not None:
            window.ops += 1
        window.commit_depth += 1
        try:
            return real_freeze(draft)
        finally:
            window.commit_depth -= 1

    monkeypatch.setattr(torch_ops.ExhaustiveOpDraft, "freeze", freeze)
    with quiet_gc():
        remove = install(window)
        try:
            traced = tl.trace(fn, ids, input_kwargs=kw)
        finally:
            remove()
    del traced
    monkeypatch.setattr(torch_ops.ExhaustiveOpDraft, "freeze", real_freeze)
    assert window.ops > 100
    return window


def _install_call_counter(window: _Window) -> Callable[[], None]:
    """Count Python-level calls whose code lives in the torchlens package."""

    def profile(frame: FrameType, event: str, _arg: Any) -> None:
        """Tally one call event of TorchLens code on the measuring thread."""

        if (
            event == "call"
            and frame.f_code.co_filename.startswith(_TORCHLENS_ROOT)
            and threading.get_ident() == window.owner
        ):
            window.tally()

    previous = sys.getprofile()
    sys.setprofile(profile)
    return lambda: sys.setprofile(previous)


def _install_build_counter(window: _Window) -> Callable[[], None]:
    """Count container builds made by TorchLens code (``BUILD_*`` and started ``__init__``)."""

    def local(frame: FrameType, event: str, _arg: Any) -> Any:
        """Tally one ``BUILD_*`` opcode executed in a TorchLens frame."""

        if event == "opcode" and frame.f_code.co_code[frame.f_lasti] in _BUILD_OPCODES:
            window.tally()
        return local

    def global_trace(frame: FrameType, event: str, _arg: Any) -> Any:
        """Tally ``__init__`` runs TorchLens code starts; trace opcodes of TorchLens frames."""

        if event != "call" or threading.get_ident() != window.owner:
            return None
        code = frame.f_code
        if code.co_name == "__init__":
            caller = frame.f_back
            if caller is not None and caller.f_code.co_filename.startswith(_TORCHLENS_ROOT):
                window.tally()
        if not code.co_filename.startswith(_TORCHLENS_ROOT):
            return None
        frame.f_trace_opcodes = True
        return local

    previous = sys.gettrace()
    sys.settrace(global_trace)
    return lambda: sys.settrace(previous)


def test_gpt2_generate_python_calls_per_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python calls into TorchLens per op, whole trace and commit alone, stay at the pins."""

    window = _measure(monkeypatch, _install_call_counter)
    trace_per_op = window.trace / window.ops
    commit_per_op = window.commit / window.ops
    print(
        f"[row-commit] ops={window.ops} trace_calls_per_op={trace_per_op:.2f} "
        f"commit_calls_per_op={commit_per_op:.2f}"
    )
    assert trace_per_op <= _MAX_TRACE_CALLS_PER_OP
    assert commit_per_op <= _MAX_COMMIT_CALLS_PER_OP


def _fresh_container_builds_per_op(*preimport: str) -> tuple[float, float]:
    """Run the container-builds measurement in a fresh interpreter; return trace and commit per op.

    ``preimport`` names modules imported before the measurement. The child runs
    this file as a script with ``PYTHONHASHSEED`` fixed, so the count does not
    depend on what earlier tests left in the worker or on set iteration order.
    """

    env = {**os.environ, "PYTHONHASHSEED": "0"}
    proc = subprocess.run(
        [sys.executable, __file__, *preimport],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith(_FRESH_MARK)]
    assert len(lines) == 1, proc.stdout[-4000:]
    trace_text, commit_text = lines[0].split()[1:3]
    return float(trace_text), float(commit_text)


def _print_builds(label: str, trace_per_op: float, commit_per_op: float) -> None:
    """Print one measured pair in the two-decimal form the pins are read from."""

    print(
        f"[row-commit] {label} trace_builds_per_op={trace_per_op:.2f} "
        f"commit_builds_per_op={commit_per_op:.2f}"
    )


@pytest.mark.slow
def test_gpt2_generate_container_builds_per_op() -> None:
    """Container builds by TorchLens code per op, whole trace and commit alone, stay at the pins."""

    trace_per_op, commit_per_op = _fresh_container_builds_per_op()
    _print_builds("fresh", trace_per_op, commit_per_op)
    assert trace_per_op <= _MAX_TRACE_BUILDS_PER_OP
    assert commit_per_op <= _MAX_COMMIT_BUILDS_PER_OP


@pytest.mark.slow
def test_gpt2_generate_container_builds_per_op_with_distributed_imported() -> None:
    """With the ``torch.distributed`` namespaces imported, trace builds per op stay at their pin."""

    trace_per_op, commit_per_op = _fresh_container_builds_per_op(*_DISTRIBUTED_NAMESPACES)
    _print_builds("distributed-imported", trace_per_op, commit_per_op)
    assert trace_per_op <= _MAX_TRACE_BUILDS_PER_OP_DISTRIBUTED
    assert commit_per_op <= _MAX_COMMIT_BUILDS_PER_OP


def test_gpt2_generate_whole_trace_builds_no_public_group_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-trace capture reads Op and Layer storage directly: no public group view is built.

    Internal readers in capture and postprocess read a moved member's storage
    attribute (``op._graph_is_buffer``) or go through ``record_member``, which
    reads plain stored leaves without a view. Public group views (``op.graph``,
    ``op.call``, ...) are for users and are rebuilt on every access.
    """

    from torchlens._data_classes._record_group_base import _RecordGroup

    fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    built: list[str] = []
    real_init = _RecordGroup.__init__

    def counting_init(self: Any, record: Any) -> None:
        """Record the view class, then bind it as usual."""

        built.append(type(self).__name__)
        real_init(self, record)

    monkeypatch.setattr(_RecordGroup, "__init__", counting_init)
    with quiet_gc():
        traced = tl.trace(fn, ids, input_kwargs=kw)
    monkeypatch.setattr(_RecordGroup, "__init__", real_init)
    del traced
    assert built == [], f"{len(built)} group views built: {sorted(set(built))}"


def _measure_in_this_process(preimport: list[str]) -> None:
    """Script entry of :func:`_fresh_container_builds_per_op`: import, measure, print one line."""

    for name in preimport:
        importlib.import_module(name)
    with pytest.MonkeyPatch.context() as monkeypatch:
        window = _measure(monkeypatch, _install_build_counter)
    print(f"{_FRESH_MARK} {window.trace / window.ops!r} {window.commit / window.ops!r}")


if __name__ == "__main__":
    _measure_in_this_process(sys.argv[1:])
