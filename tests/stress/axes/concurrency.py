"""Concurrency-misuse axis: captures started while another capture is open.

TorchLens does not support concurrent captures across Python threads or nested
captures; these rows provoke exactly that misuse and expect a typed refusal
(judged by ``typed_refusal_names_condition``) and a clean process afterwards
(judged by ``torch_clean_after_cell``). The misuse fires only while a capture
is logging, so the cell's plain eager forward stays a plain forward and its
own result is never hidden as not applicable.

A side thread is always joined with a bound, so a deadlocked capture fails the
cell instead of hanging the worker.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens import _state

from ..generator import Inputs, ModelSpec

WIDTH = 8
JOIN_SECONDS = 60.0


class StressUserError(RuntimeError):
    """A plain, non-TorchLens exception raised on purpose by a stress row.

    One class for every deliberate user-side raise, so an oracle carve-out
    for "the user's own exception passed through unchanged" can stay narrow.
    """


def capture_is_logging() -> bool:
    """Return whether a TorchLens capture is logging ops right now (process-global)."""

    return bool(_state._logging_enabled)


def side_model() -> nn.Module:
    """A tiny model the misuse rows capture on the side."""

    return nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU())


def side_capture(x: torch.Tensor) -> None:
    """Capture ``side_model()`` on ``x`` and discard the trace."""

    trace = tl.trace(side_model(), [x.detach().clone()])
    del trace


def run_in_thread(target: Any, *, started: threading.Event | None = None) -> threading.Thread:
    """Start ``target`` in a daemon thread that stores its exception on the thread."""

    def body() -> None:
        try:
            target()
        except BaseException as error:  # stored and re-raised by join_and_raise
            thread.error = error  # type: ignore[attr-defined]
        finally:
            if started is not None:
                started.set()

    thread = threading.Thread(target=body, daemon=True, name="stress-side-capture")
    thread.error = None  # type: ignore[attr-defined]
    thread.start()
    if started is not None:
        started.wait(JOIN_SECONDS)
    return thread


def join_and_raise(thread: threading.Thread) -> None:
    """Join ``thread`` with a bound; re-raise its exception in this thread."""

    thread.join(JOIN_SECONDS)
    if thread.is_alive():
        raise StressUserError(f"side-thread capture still running after {JOIN_SECONDS:.0f} s")
    error = getattr(thread, "error", None)
    if error is not None:
        raise error


class _Body(nn.Module):
    """Two linear layers every concurrency row runs around its misuse."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)


class NestedCaptureInForward(_Body):
    """Start a second ``tl.trace`` from inside the forward of a model being captured."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        if capture_is_logging():
            side_capture(x)
        return self.fc2(h)


class ThreadCaptureInForward(_Body):
    """Start a capture from a second thread while this model's capture is open, then join."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        if capture_is_logging():
            side_x = x.detach().clone()
            join_and_raise(run_in_thread(lambda: side_capture(side_x)))
        return self.fc2(h)


class _GatedSide(nn.Module):
    """Side model that signals it started, then waits for the main thread's midpoint."""

    def __init__(self, started: threading.Event, midway: threading.Event) -> None:
        super().__init__()
        self.fc = nn.Linear(WIDTH, WIDTH)
        self._started = started
        self._midway = midway

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        self._started.set()
        self._midway.wait(JOIN_SECONDS)
        return torch.relu(h) * 2.0


class InterleavedThreadCapture(_Body):
    """Interleave two captures: a side thread's capture runs while this forward's ops run."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not capture_is_logging():
            return self.fc2(torch.relu(self.fc1(x)))
        started, midway = threading.Event(), threading.Event()
        side = _GatedSide(started, midway)
        side_x = x.detach().clone()
        thread = run_in_thread(lambda: tl.trace(side, [side_x]), started=started)
        h = torch.relu(self.fc1(x))
        midway.set()
        out = self.fc2(h)
        join_and_raise(thread)
        return out


class RaiseDuringCapture(_Body):
    """Raise a plain user exception mid-forward, only while a capture is logging."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        if capture_is_logging():
            raise StressUserError("stress row: the model raises during capture")
        return self.fc2(h)


def _input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, WIDTH),), {})


def _model(row_id: str, factory: type[nn.Module], reason: str) -> ModelSpec:
    return ModelSpec(
        id=row_id,
        build=factory,
        example_input=_input,
        reason=reason,
        tags=frozenset({"concurrency"}),
    )


MODELS: tuple[ModelSpec, ...] = (
    _model(
        "cc_nested_capture",
        NestedCaptureInForward,
        "a second capture started from inside a forward being captured: typed refusal expected",
    ),
    _model(
        "cc_thread_capture",
        ThreadCaptureInForward,
        "a capture started from a second thread while one is open: typed refusal expected",
    ),
    _model(
        "cc_interleaved",
        InterleavedThreadCapture,
        "two models captured at once, their ops interleaved across two threads",
    ),
    _model(
        "cc_raise_mid_capture",
        RaiseDuringCapture,
        "a capture interrupted by a user exception; the process must be clean for the next",
    ),
)
