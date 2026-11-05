"""Round-45 hon2_1 immunizer -- cross-thread captured-OPERAND consumption ceiling.

r43 closed the DIRECT / owner-derived cross-thread escape rings: a non-owner thread's
tensor->host escape ceilings only when the escaped tensor's OBJECT IDENTITY is captured or
owner-registered. r44 hon2_1 found the WORKER-DERIVED sibling still open: a tensor derived on
the worker from a captured input (``(gate*2).sum()``, ``gate.clone()``, ``gate+0``, ``gate@w``,
``torch.cat([gate],0)`` ...) has fresh worker-side storage the OWNER-thread-only census never
registered, so its value escape was unwitnessed -> false ``VERIFIED`` on a changed input a fresh
live run would branch differently on.

r45 closes the class at the process-wide chokepoint: EVERY Python-visible torch op flows through
the global torch-function wrapper, so the FIRST op a non-owner thread runs that CONSUMES a
captured tensor as an operand permanently ceilings the artifact to ``unverifiable`` /
``not_applicable`` -- op-agnostic, derivation-depth-independent, escape-spelling-independent.

This immunizer is a CROSS PRODUCT of {thread API} x {derivation} (every cell must ceiling) plus
an escape sweep, a deep-derivation chain, structural op-agnostic / flag-lifetime assertions, and
benign over-trigger pins (own-tensor worker, idle worker) that MUST stay ``VERIFIED``. A future
op-shape or thread API that slips past the operand observer turns a cell RED.
"""

from __future__ import annotations

import _thread
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._completeness_escape_state import (
    host_escape_has_cross_thread_captured_tensor,
)
from torchlens.options import CaptureOptions

_CAP = {"replayable": True, "capture_container_structure": True, "cache": False}


def _capture(model: nn.Module, x: Any, *, seed: int = 1) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""

    return tl.trace(model, x, capture=CaptureOptions(seed=seed, **_CAP))


# --------------------------------------------------------------------------------------
# Thread-API runners: a non-owner thread executes ``job`` and the forward blocks on it.
# --------------------------------------------------------------------------------------


def _run_thread(job: Callable[[], None]) -> None:
    t = threading.Thread(target=job)
    t.start()
    t.join()


def _run_daemon(job: Callable[[], None]) -> None:
    t = threading.Thread(target=job, daemon=True)
    t.start()
    t.join()


def _run_raw_thread(job: Callable[[], None]) -> None:
    done = threading.Event()

    def _wrapped() -> None:
        try:
            job()
        finally:
            done.set()

    _thread.start_new_thread(_wrapped, ())
    done.wait(5.0)


def _run_pool(job: Callable[[], None]) -> None:
    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(job).result()


class _PreexistingWorker:
    """A worker thread started BEFORE any capture window (a foreign, non-hooked thread)."""

    def __init__(self) -> None:
        import queue

        self.jobs: queue.Queue[Any] = queue.Queue()
        self.results: queue.Queue[Any] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.results.put(("ok", job()))
            except BaseException as exc:  # noqa: BLE001 - relayed to the test thread
                self.results.put(("err", exc))

    def run(self, job: Callable[[], Any]) -> Any:
        self.jobs.put(job)
        kind, value = self.results.get()
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        """Stop the worker and join it, so its exit cannot land in a later capture window."""

        self.jobs.put(None)
        self.thread.join()


@pytest.fixture
def preexisting_worker() -> Any:
    worker = _PreexistingWorker()
    try:
        yield worker
    finally:
        worker.stop()


# The five thread-API spellings. The pre-existing worker is injected per-test.
_THREAD_APIS: dict[str, Callable[[Callable[[], None]], None]] = {
    "thread": _run_thread,
    "daemon": _run_daemon,
    "raw_thread": _run_raw_thread,
    "pool": _run_pool,
}


# --------------------------------------------------------------------------------------
# Derivations: each CONSUMES ``gate`` (a captured input leaf) as an operand and returns a
# FRESH worker-side tensor whose storage the owner census never registered.
# --------------------------------------------------------------------------------------

_DERIVATIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mul": lambda g, w: g * 2.0,
    "add": lambda g, w: g + 0.0,
    "clone": lambda g, w: g.clone(),
    "sum": lambda g, w: g.sum(),
    "matmul": lambda g, w: g @ w,
    "view": lambda g, w: g.view(-1).contiguous(),
    "cat": lambda g, w: torch.cat([g], 0),
    "slice": lambda g, w: g[0:1],
    "to": lambda g, w: g.to(torch.float64),
    "stack": lambda g, w: torch.stack([g]),
}

# Escape spellings: extract a python-host value from a derived tensor.
_ESCAPES: dict[str, Callable[[torch.Tensor], bool]] = {
    "item": lambda d: bool(d.reshape(-1).sum().item() > 0),
    "tolist": lambda d: bool(float(d.reshape(-1).sum().tolist()) > 0),
    "numpy": lambda d: bool(d.detach().reshape(-1).sum().numpy() > 0),
    "float": lambda d: bool(float(d.reshape(-1).sum()) > 0),
    "bool_": lambda d: bool(d.reshape(-1).sum() > 0),
}


class _WorkerDerivedGuard(nn.Module):
    """A model whose forward hands a captured input to a non-owner worker that DERIVES a fresh
    tensor from it, escapes that tensor's value, and branches on the result."""

    def __init__(
        self,
        derivation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        escape: Callable[[torch.Tensor], bool],
        runner: Callable[[Callable[[], None]], None],
    ) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.w = nn.Parameter(torch.randn(4, 4))
        self._derivation = derivation
        self._escape = escape
        self._runner = runner

    def forward(self, gate: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        box: dict[str, bool] = {}

        def _worker() -> None:
            derived = self._derivation(gate, self.w)
            box["flag"] = self._escape(derived)

        self._runner(_worker)
        return self.a(data) if box.get("flag", True) else self.b(data)


def _gate_pos() -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 1.0, 1.0])


def _gate_neg() -> torch.Tensor:
    return torch.tensor([-1.0, -1.0, -1.0, -1.0])


def _data() -> torch.Tensor:
    return torch.randn(1, 4)


# Kept: shares a parametrize list that is not trimmable per id with kept params.
@pytest.mark.parametrize("api_name", list(_THREAD_APIS))
@pytest.mark.parametrize("deriv_name", list(_DERIVATIONS))
def test_worker_derived_consumption_ceilings(api_name: str, deriv_name: str) -> None:
    """Cross product {thread API} x {derivation}: every worker-derived consumption of a captured
    operand ceilings the capture (``host_escape_has_cross_thread_captured_tensor``)."""

    model = _WorkerDerivedGuard(_DERIVATIONS[deriv_name], _ESCAPES["item"], _THREAD_APIS[api_name])
    trace = _capture(model, (_gate_pos(), _data()))
    assert host_escape_has_cross_thread_captured_tensor(trace), (
        f"{api_name}/{deriv_name}: worker-derived captured-operand consumption did not ceiling"
    )


# --------------------------------------------------------------------------------------
# Over-trigger pins -- MUST stay VERIFIED (no captured operand consumed).
# --------------------------------------------------------------------------------------


# Kept: shares a parametrize list that is not trimmable per id with kept params.
@pytest.mark.parametrize("api_name", list(_THREAD_APIS))
def test_own_tensor_worker_stays_clean(api_name: str) -> None:
    """A non-owner worker that derives + escapes only from a tensor IT created independently of
    the capture consumes no captured operand and does NOT ceiling."""

    class _OwnTensorWorker(nn.Module):
        def __init__(self, runner: Callable[[Callable[[], None]], None]) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self._runner = runner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            def _worker() -> None:
                own = torch.tensor([1.0, 2.0, 3.0, 4.0])
                _ = bool((own * 2.0).clone().sum().item() > 0)

            self._runner(_worker)
            return self.lin(x)

    trace = _capture(_OwnTensorWorker(_THREAD_APIS[api_name]), torch.randn(1, 4))
    assert not host_escape_has_cross_thread_captured_tensor(trace)
