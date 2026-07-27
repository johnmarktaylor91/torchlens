"""Round-49 hon2_1 immunizer -- cross-thread captured-operand consumption over the PATCHABLE
private-C free-function modules (`torch._C._{nn,special,fft,linalg,sparse,nested}`).

r47 closed the `torch.ops.*` class surface. But a private-C FREE function
(`torch._C._nn.gelu(gate)`) is a THIRD op surface: it bypasses BOTH the global torch-FUNCTION
wrapper (no `__torch_function__`) AND the `torch._ops.*` class patch (it dispatches its inner aten
op in C++), so a non-owner worker consuming a captured operand through one went unwitnessed -> false
`VERIFIED` on a diverging changed input (the r48 hon2_1 finding). r49 extends the armed-lifecycle
observer to every module-level callable of the patchable private-C modules, structurally enumerated
from the SAME curated forward-op module authority (`_ALLOWED_FORWARD_OP_MODULES` via
`private_c_forward_op_module_names`), so the class -- not a known-alias subset -- is closed and a
future private-C op module is auto-covered.

This immunizer pins: the behavioral ceiling matrix (private-C consumers x thread-mode); driver
non-vacuity + the `_sparse` enumeration-completeness / gate-neutral pin; the arm/disarm lifecycle
(every discovered callable sentinel-wrapped while armed, ALL restored by identity, no leak);
over-trigger controls (benign own-tensor private-C op / disarmed passthrough stay VERIFIED); and
fail-closed install failure -> INCOMPLETE. The read-only `_VariableFunctions` CLASS spelling stays
the accepted xfail residual (covered in `test_r47_crossthread_op_surface.py`).
"""

from __future__ import annotations

import shutil
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import completeness_witness as cw
from torchlens.backends.torch.completeness_witness import (
    _private_c_forward_op_modules,
    _private_c_module_callables,
    host_escape_has_cross_thread_captured_tensor,
    host_escape_observer_install_failed,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils._callable_safety import (
    _ALLOWED_FORWARD_OP_MODULES,
    is_pure_forward_callable,
    private_c_forward_op_module_names,
)

_CAP = dict(intervention_ready=True, capture_container_structure=True, cache=False)
_EXPECTED_MODULES = {
    "torch._C._nn",
    "torch._C._special",
    "torch._C._fft",
    "torch._C._linalg",
    "torch._C._sparse",
    "torch._C._nested",
}


def _gate_pos() -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 1.0, 1.0])  # captured -> flag True -> path a


def _gate_neg() -> torch.Tensor:
    return torch.tensor([-1.0, -1.0, -1.0, -1.0])  # run -> flag False -> path b (diverges)


# --- private-C captured-operand consumers (worker body: captured gate -> bool) -----------------
def _consume_nn_gelu(gate: torch.Tensor) -> bool:
    return bool(float(torch._C._nn.gelu(gate).sum()) > 0)


def _consume_special_erf(gate: torch.Tensor) -> bool:
    return bool(float(torch._C._special.special_erf(gate).sum()) > 0)


def _consume_fft(gate: torch.Tensor) -> bool:
    return bool(float(torch._C._fft.fft_fft(gate).real.sum()) > 0)


def _consume_linalg(gate: torch.Tensor) -> bool:
    # vector-norm is sign-insensitive, but the CEILING fires on OPERAND CONSUMPTION (op-agnostic),
    # so a worker reading the captured gate through a private-C linalg op still ceilings.
    return bool(float(torch._C._linalg.linalg_vector_norm(gate)) > 1.0)


_CONSUMERS: dict[str, Callable[[torch.Tensor], bool]] = {
    "nn_gelu": _consume_nn_gelu,
    "special_erf": _consume_special_erf,
    "fft_fft": _consume_fft,
    "linalg_vector_norm": _consume_linalg,
}


class _InWindowRunner:
    """Runs a job on a fresh in-window ``threading.Thread``."""

    def run(self, job: Callable[[], Any]) -> Any:
        box: dict[str, Any] = {}

        def _target() -> None:
            box["v"] = job()

        thread = threading.Thread(target=_target)
        thread.start()
        thread.join()
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


# --- behavioral ceiling matrix ----------------------------------------------------------------
@pytest.mark.parametrize("surface", list(_CONSUMERS))
@pytest.mark.parametrize("thread_mode", ["in_window", "preexisting"])
def test_private_c_captured_operand_consumption_ceilings(
    surface: str, thread_mode: str, tmp_path: Path
) -> None:
    """A non-owner thread consuming a captured operand through a PRIVATE-C free function ceilings
    the capture, and a changed-input run is UNVERIFIABLE -- never a false VERIFIED (r48 hon2_1)."""

    runner = _make_runner(thread_mode)
    try:
        torch.manual_seed(0)
        data = torch.randn(1, 4)
        model = _CrossThreadModel(_CONSUMERS[surface], runner)
        trace = tl.trace(model, (_gate_pos(), data), capture=CaptureOptions(random_seed=1, **_CAP))
        assert host_escape_has_cross_thread_captured_tensor(trace), (
            f"{surface}/{thread_mode}: private-C captured-operand consumption did not ceiling"
        )
        path = tmp_path / f"{surface}_{thread_mode}.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        result = tl.load(path).run(inputs=(_gate_neg(), data))
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    finally:
        runner.stop()


# --- driver non-vacuity + _sparse enumeration / gate-neutral pin ------------------------------
def test_private_c_module_driver_is_the_six_modules() -> None:
    """The structural driver resolves EXACTLY the six patchable private-C forward-op modules
    (incl. ``_sparse``, ``_nested``) and NONE of the class-typed unpatchable holders."""

    resolved = {mod.__name__ for mod in _private_c_forward_op_modules()}
    assert resolved == _EXPECTED_MODULES, resolved
    for mod in _private_c_forward_op_modules():
        assert isinstance(mod, types.ModuleType)
    # class-typed holders (read-only / non-Python-patchable) are excluded (accepted residual).
    assert "torch._C._VariableFunctions" not in resolved
    assert "torch._C._TensorBase" not in resolved


def test_private_c_module_callables_nonvacuous() -> None:
    """Every discovered private-C module exposes >=1 module-level callable (non-vacuous scan)."""

    counts: dict[str, int] = {}
    for module, _attr, original in _private_c_module_callables():
        assert callable(original)
        counts[module.__name__] = counts.get(module.__name__, 0) + 1
    for name in _EXPECTED_MODULES:
        assert counts.get(name, 0) >= 1, f"{name} contributed no patch targets"


def test_sparse_enumeration_completeness_is_gate_neutral() -> None:
    """``torch._C._sparse`` is present for ENUMERATION-COMPLETENESS only: it was already prefix-
    admitted via the ``torch._C`` entry, so its gate outcome for a sparse op is UNCHANGED by the
    addition (pins the 'enumeration-only, gate-neutral' property)."""

    assert "torch._C._sparse" in _ALLOWED_FORWARD_OP_MODULES
    assert "torch._C._sparse" in private_c_forward_op_module_names()
    # A sparse forward op is admitted with OR without the explicit ``_sparse`` entry, because the
    # bare ``torch._C`` prefix already covers it -- the addition changes no admission decision.
    sparse_op = torch._C._sparse.sparse_sampled_addmm
    admitted_with = is_pure_forward_callable(sparse_op)
    reduced = frozenset(m for m in _ALLOWED_FORWARD_OP_MODULES if m != "torch._C._sparse")
    module = str(getattr(sparse_op, "__module__", "") or "")
    admitted_without = any(module == p or module.startswith(p + ".") for p in reduced)
    assert admitted_with is True
    assert admitted_without is True, "sparse op should already be prefix-admitted via torch._C"


# --- arm/disarm lifecycle ---------------------------------------------------------------------
def test_private_c_patched_while_armed_and_restored_after() -> None:
    """LIFECYCLE: EVERY discovered private-C callable is sentinel-wrapped during the armed window
    and restored (original identity, no marker) after -- a leaked patch or a silent surface drop
    fails this."""

    originals = {
        (id(module), attr): original for module, attr, original in _private_c_module_callables()
    }

    class _ProbeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.during: dict[str, Any] = {}

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            targets = _private_c_module_callables()
            self.during["count"] = len(targets)
            self.during["all_patched"] = all(
                getattr(getattr(module, attr, None), "__tl_nonowner_ops_observer__", False)
                for module, attr, _original in targets
            )
            return self.lin(x).relu()

    model = _ProbeModel()
    tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(random_seed=1, **_CAP))

    assert model.during["count"] == len(originals)
    assert model.during["all_patched"] is True, "not every private-C callable was patched"
    for module in _private_c_forward_op_modules():
        for attr in dir(module):
            if attr.startswith("__"):
                continue
            current = getattr(module, attr, None)
            if not callable(current):
                continue
            assert originals.get((id(module), attr)) is current, (
                f"{module.__name__}.{attr} was not restored to its original identity"
            )
            assert not getattr(current, "__tl_nonowner_ops_observer__", False)


def test_install_failure_marks_observer_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty private-C scan (install-failure surrogate) fails CLOSED: the observer-install-failed
    flag is set so the descriptor is INCOMPLETE, never a silent 'no non-owner op touch'."""

    monkeypatch.setattr(cw, "_private_c_module_callables", lambda: ())

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x).relu()

    trace = tl.trace(_M(), torch.randn(2, 4), capture=CaptureOptions(random_seed=1, **_CAP))
    assert host_escape_observer_install_failed(trace)


# --- over-trigger controls --------------------------------------------------------------------
class _BenignOwnTensorModel(nn.Module):
    """A worker that runs a private-C op ONLY on a tensor it created independently of the capture."""

    def __init__(self, runner: Any) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self._runner = runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._runner.run(lambda: float(torch._C._nn.gelu(torch.tensor([1.0, 2.0, 3.0, 4.0])).sum()))
        return self.lin(x).relu()


@pytest.mark.parametrize("thread_mode", ["in_window", "preexisting"])
def test_benign_own_tensor_private_c_stays_verified(thread_mode: str, tmp_path: Path) -> None:
    """Over-trigger pin: a worker calling a private-C op on its OWN tensor consumes no captured
    operand and stays VERIFIED, on both an in-window and a pre-existing thread."""

    runner = _make_runner(thread_mode)
    try:
        model = _BenignOwnTensorModel(runner)
        x = torch.randn(2, 4)
        trace = tl.trace(model, x, capture=CaptureOptions(random_seed=1, **_CAP))
        assert not host_escape_has_cross_thread_captured_tensor(trace)
        path = tmp_path / f"benign_{thread_mode}.tlspec"
        shutil.rmtree(path, ignore_errors=True)
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
        result = tl.load(path).run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    finally:
        runner.stop()


def test_disarmed_private_c_call_is_pure_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over-trigger pin: with the belt disarmed, a private-C op call runs the original and never
    invokes the operand observer."""

    from torchlens import _state

    calls = {"n": 0}
    real = cw.observe_nonowner_operands

    def _spy(args: Any, kwargs: Any) -> None:
        calls["n"] += 1
        real(args, kwargs)

    monkeypatch.setattr(cw, "observe_nonowner_operands", _spy)
    assert _state._nonowner_belt_armed is False
    out = torch._C._nn.gelu(torch.zeros(4))
    assert isinstance(out, torch.Tensor)
    assert calls["n"] == 0
