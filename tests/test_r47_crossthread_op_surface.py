"""Round-47 hon2_1 immunizer -- cross-thread captured-operand consumption over the FULL op surface.

r45 hon2_1 ceilinged a runnable capture to UNVERIFIABLE when a NON-owner thread consumed a captured
tensor as an operand -- but ONLY through the global torch-FUNCTION wrapper. The entire
``torch.ops.*`` (aten / higher-order / TorchBind) surface bypasses that wrapper, and its aten
dispatch census is thread-LOCAL (a ``TorchDispatchMode`` cannot see a non-owner thread), so a worker
deriving-from / reading a captured tensor via ``torch.ops.aten.*`` never ceilinged -> false
``VERIFIED`` on a diverging changed input (the r46 hon2_1 finding, re-opening the r44 class).

r47 installs a PROCESS-WIDE, armed-lifecycle-scoped class-level observer on every ``torch._ops``
class defining its own ``__call__`` (feature-detected structurally, version-robust), which routes a
non-owner thread's captured-operand consumption through the SAME storage-identity membership test as
the wrapper belt. This immunizer pins:

* the behavioral matrix (aten packet / overload / ``_local_scalar_dense`` + wrapped controls) x
  (in-window thread / pre-existing worker) -> every captured-operand consumer ceilings to
  UNVERIFIABLE, and a changed-input run is never a false VERIFIED;
* the arm/disarm LIFECYCLE (every discovered class is patched while armed, ALL restored after, the
  scan re-runs so a future torch class is auto-required) -- forbids a silent surface drop or a
  LEAKED process-wide patch;
* the over-trigger controls (benign own-tensor / own-aten-derived / idle worker stay VERIFIED; a
  disarmed ``torch.ops.aten.*`` call is a pure passthrough) and the owner-thread perf premise
  (an eager owner forward hits the Python ``torch.ops.*.__call__`` path zero times);
* the documented residuals (private-C ``torch._C._VariableFunctions.<op>``; HOP-subclass C++-only
  ``__call__``) as auditable xfail cells.
"""

from __future__ import annotations

import functools
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import completeness_witness as cw
from torchlens.backends.torch.completeness_witness import (
    _torch_ops_call_classes,
    host_escape_has_cross_thread_captured_tensor,
    host_escape_observer_install_failed,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)


def _gate_pos() -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 1.0, 1.0])  # captured -> flag True -> path a


def _gate_neg() -> torch.Tensor:
    return torch.tensor([-1.0, -1.0, -1.0, -1.0])  # run -> flag False -> path b (diverges)


# --- captured-operand consumers over distinct op surfaces (worker body: gate -> bool) ----------
def _consume_aten_packet(gate: torch.Tensor) -> bool:
    derived = torch.ops.aten.mul(gate, torch.tensor(2.0))
    return bool(torch.ops.aten._local_scalar_dense(torch.ops.aten.sum.default(derived)) > 0)


def _consume_aten_overload(gate: torch.Tensor) -> bool:
    derived = torch.ops.aten.mul.Tensor(gate, torch.tensor(2.0))
    return bool(torch.ops.aten._local_scalar_dense(torch.ops.aten.sum.default(derived)) > 0)


def _consume_aten_direct(gate: torch.Tensor) -> bool:
    # pure-aten value read of the captured tensor itself (the aten spelling of ``.item()``)
    return bool(torch.ops.aten._local_scalar_dense(torch.ops.aten.sum.default(gate)) > 0)


def _consume_torch_fn(gate: torch.Tensor) -> bool:
    return bool(torch.mul(gate, 2.0).sum().item() > 0)  # wrapped-surface control


def _consume_tensor_method(gate: torch.Tensor) -> bool:
    return bool((gate * 2.0).sum().item() > 0)  # wrapped-surface control


_CONSUMERS: dict[str, Callable[[torch.Tensor], bool]] = {
    "aten_packet": _consume_aten_packet,
    "aten_overload": _consume_aten_overload,
    "aten_direct": _consume_aten_direct,
    "torch_fn": _consume_torch_fn,
    "tensor_method": _consume_tensor_method,
}


class _InWindowRunner:
    """Runs a job on a fresh in-window ``threading.Thread``."""

    def run(self, job: Callable[[], Any]) -> Any:
        box: dict[str, Any] = {}

        def _target() -> None:
            box["v"] = job()

        t = threading.Thread(target=_target)
        t.start()
        t.join()
        return box.get("v")

    def stop(self) -> None:  # pragma: no cover - symmetry with the pre-existing worker
        pass


class _PreexistingWorker:
    """A worker thread started BEFORE any capture window (non-hooked, foreign)."""

    def __init__(self) -> None:
        import queue

        self._jobs: "queue.Queue[Any]" = queue.Queue()
        self._results: "queue.Queue[Any]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self._results.put(("ok", job()))
            except BaseException as exc:  # noqa: BLE001 - relayed to test thread
                self._results.put(("err", exc))

    def run(self, job: Callable[[], Any]) -> Any:
        self._jobs.put(job)
        kind, value = self._results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5)


class _CrossThreadModel(nn.Module):
    def __init__(self, consume: Callable[[torch.Tensor], bool], runner: Any) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self._consume = consume
        self._runner = runner

    def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        flag = self._runner.run(lambda: self._consume(gate))
        return self.a(data) if flag else self.b(data)


def _make_runner(thread_mode: str) -> Any:
    return _InWindowRunner() if thread_mode == "in_window" else _PreexistingWorker()


@pytest.mark.parametrize("surface", list(_CONSUMERS))
@pytest.mark.parametrize("thread_mode", ["in_window", "preexisting"])
def test_crossthread_captured_operand_consumption_ceilings(
    surface: str, thread_mode: str, tmp_path: Path
) -> None:
    """A non-owner thread consuming a captured operand on ANY op surface ceilings the capture, and a
    changed-input run is UNVERIFIABLE -- never a false VERIFIED."""

    runner = _make_runner(thread_mode)
    try:
        torch.manual_seed(0)
        data = torch.randn(1, 4)
        model = _CrossThreadModel(_CONSUMERS[surface], runner)
        trace = tl.trace(model, (_gate_pos(), data), capture=CaptureOptions(random_seed=1, **_CAP))
        assert host_escape_has_cross_thread_captured_tensor(trace), (
            f"{surface}/{thread_mode}: captured-operand consumption did not ceiling"
        )
        path = tmp_path / f"{surface}_{thread_mode}.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        result = tl.load(path).run(inputs=(_gate_neg(), data))
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    finally:
        runner.stop()


# --- over-trigger controls -------------------------------------------------------------------
class _BenignOwnTensorModel(nn.Module):
    """A worker that operates ONLY on tensors it created independently of the capture."""

    def __init__(self, runner: Any, *, use_aten: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self._runner = runner
        self._use_aten = use_aten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _job() -> float:
            own = torch.tensor([1.0, 2.0, 3.0])
            if self._use_aten:
                own = torch.ops.aten.mul.Tensor(own, torch.tensor(2.0))
            return float(own.sum().item())

        self._runner.run(_job)
        return self.lin(x).relu()


@pytest.mark.parametrize("thread_mode", ["in_window", "preexisting"])
@pytest.mark.parametrize("use_aten", [False, True])
def test_benign_own_tensor_worker_stays_verified(
    thread_mode: str, use_aten: bool, tmp_path: Path
) -> None:
    """Over-trigger pin: a worker touching only its OWN tensors (incl. via aten) consumes no
    captured operand and stays VERIFIED, on both an in-window and a pre-existing thread."""

    runner = _make_runner(thread_mode)
    try:
        model = _BenignOwnTensorModel(runner, use_aten=use_aten)
        x = torch.randn(2, 4)
        trace = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
        assert not host_escape_has_cross_thread_captured_tensor(trace)
        path = tmp_path / f"benign_{thread_mode}_{use_aten}.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        result = tl.load(path).run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    finally:
        runner.stop()


def test_idle_worker_stays_verified(tmp_path: Path) -> None:
    """Over-trigger pin: a worker that runs no torch op at all stays VERIFIED."""

    runner = _PreexistingWorker()
    try:

        class _IdleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                runner.run(lambda: 42)
                return self.lin(x).relu()

        x = torch.randn(2, 4)
        trace = tl.trace(_IdleModel(), x, capture=CaptureOptions(random_seed=1, **_CAP))
        assert not host_escape_has_cross_thread_captured_tensor(trace)
        path = tmp_path / "idle.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        assert tl.load(path).run(inputs=x).report.path_faithfulness is PathFaithfulness.VERIFIED
    finally:
        runner.stop()


# --- structural arm/disarm lifecycle ---------------------------------------------------------
def test_ops_call_classes_scan_is_nonvacuous() -> None:
    """The structural scan discovers the ``torch._ops`` call-class surface (non-vacuous)."""

    classes = _torch_ops_call_classes()
    assert len(classes) >= 2
    for cls in classes:
        assert callable(cls.__dict__.get("__call__"))


def test_patches_installed_while_armed_and_restored_after() -> None:
    """LIFECYCLE: EVERY discovered class is patched during the armed capture window, and ALL are
    restored (original identity, no marker) after -- a leaked process-wide patch or a silent
    surface drop fails this."""

    originals = {cls: cls.__dict__["__call__"] for cls in _torch_ops_call_classes()}

    class _ProbeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.during: dict[str, Any] = {}

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # runs INSIDE the armed window; re-scan so a future torch class is auto-required
            classes = _torch_ops_call_classes()
            self.during["count"] = len(classes)
            self.during["all_patched"] = all(
                getattr(c.__dict__.get("__call__"), "__tl_nonowner_ops_observer__", False)
                for c in classes
            )
            return self.lin(x).relu()

    model = _ProbeModel()
    tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(random_seed=1, **_CAP))

    assert model.during["count"] == len(originals)
    assert model.during["all_patched"] is True, "not every torch._ops call class was patched"
    # restored: original identity, no lingering wrapper marker
    for cls, original in originals.items():
        current = cls.__dict__.get("__call__")
        assert current is original, f"{cls.__name__}.__call__ was not restored"
        assert not getattr(current, "__tl_nonowner_ops_observer__", False)


def test_disarmed_torch_ops_call_is_pure_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over-trigger pin: with the belt disarmed, a ``torch.ops.aten.*`` call runs the original and
    never invokes the operand observer."""

    calls = {"n": 0}
    real = cw.observe_nonowner_operands

    def _spy(args: Any, kwargs: Any) -> None:
        calls["n"] += 1
        real(args, kwargs)

    monkeypatch.setattr(cw, "observe_nonowner_operands", _spy)
    assert _state._nonowner_belt_armed is False
    out = torch.ops.aten.mul.Tensor(torch.ones(2), torch.tensor(3.0))
    assert out.tolist() == [3.0, 3.0]
    assert calls["n"] == 0


def test_eager_forward_hits_torch_ops_call_zero_times() -> None:
    """Perf premise (probe P6): a plain eager forward hits the Python ``torch.ops.*.__call__`` path
    zero times (dispatch is C++), so class-patching those ``__call__`` methods for the armed forward
    window adds ~no overhead on the OWNER thread. Measured on a RAW forward (no capture) with a
    counting wrapper, so it isolates the model's eager ops from TorchLens's own capture machinery."""

    classes = _torch_ops_call_classes()
    hits = {"n": 0}
    saved: dict[type, Any] = {}

    def _make_counter(original: Any) -> Any:
        @functools.wraps(original)
        def _counter(self: Any, *args: Any, **kwargs: Any) -> Any:
            hits["n"] += 1
            return original(self, *args, **kwargs)

        return _counter

    for cls in classes:
        saved[cls] = cls.__dict__["__call__"]
        setattr(cls, "__call__", _make_counter(saved[cls]))
    try:
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).eval()
        with torch.no_grad():
            model(torch.randn(2, 8))
    finally:
        for cls, original in saved.items():
            setattr(cls, "__call__", original)
    assert hits["n"] == 0


def test_install_failure_marks_observer_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty scan (install failure surrogate) fails CLOSED: the observer-install-failed flag is
    set so the descriptor is INCOMPLETE, never a silent 'no non-owner op touch'."""

    monkeypatch.setattr(cw, "_torch_ops_call_classes", lambda: ())

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    trace = tl.trace(_M(), torch.randn(2, 4), capture=CaptureOptions(random_seed=1, **_CAP))
    assert host_escape_observer_install_failed(trace)


# --- documented residuals (auditable) --------------------------------------------------------
@pytest.mark.xfail(
    reason="documented residual: torch._C._VariableFunctions.<op> is a read-only, non-Python-"
    "patchable private-C free-function surface; its public alias torch.<op> IS wrapped",
    strict=False,
)
def test_variable_functions_private_spelling_residual(tmp_path: Path) -> None:
    """A worker consuming a captured operand through the PRIVATE ``torch._C._VariableFunctions.<op>``
    spelling is the accepted residual (not Python-patchable)."""

    runner = _PreexistingWorker()
    try:

        class _VFModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)

            def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
                def _job() -> bool:
                    vf = torch._C._VariableFunctions
                    return bool(vf.sum(gate).item() > 0)

                flag = runner.run(_job)
                return self.a(data) if flag else self.b(data)

        torch.manual_seed(0)
        data = torch.randn(1, 4)
        trace = tl.trace(
            _VFModel(), (_gate_pos(), data), capture=CaptureOptions(random_seed=1, **_CAP)
        )
        # If this ever ceilings, the residual is CLOSED and the xfail flips to XPASS.
        assert host_escape_has_cross_thread_captured_tensor(trace)
    finally:
        runner.stop()
