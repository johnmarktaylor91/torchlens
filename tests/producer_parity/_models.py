"""Deterministic scenario set for the producer parity harness.

Each scenario returns a fresh model + input + the capture callable so a run is
reproducible in-process and cross-process (fixed seeds, no external data).
The P0 skeleton carries the scenarios the design-of-record names for the
harness proofs; the P5 campaign widens the model set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

import torchlens as tl

_SEED = 20260812


class SmallCNN(nn.Module):
    """Conv -> BN -> ReLU -> pool -> linear; params, buffers, multi-type ops."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED)
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(self.conv(x)))
        h = torch.nn.functional.max_pool2d(h, 2)
        return self.fc(h.flatten(1))


class TinyRecurrent(nn.Module):
    """One linear cell applied three times (recurrence/equivalence classes)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 1)
        self.cell = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(3):
            h = torch.tanh(self.cell(h))
        return h


class BufferOutputModel(nn.Module):
    """Returns a registered buffer alongside the computation (late-buffer path)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 2)
        self.fc = nn.Linear(4, 4)
        self.register_buffer("gauge", torch.arange(4.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc(x), self.gauge


def _cnn_input() -> torch.Tensor:
    torch.manual_seed(_SEED + 10)
    return torch.randn(2, 1, 8, 8)


def _vec_input() -> torch.Tensor:
    torch.manual_seed(_SEED + 11)
    return torch.randn(2, 6)


def _small_vec_input() -> torch.Tensor:
    torch.manual_seed(_SEED + 12)
    return torch.randn(2, 4)


@dataclass(frozen=True)
class Scenario:
    """One reproducible capture scenario."""

    name: str
    build: Callable[[], tuple[nn.Module, Any]]
    capture: Callable[[nn.Module, Any], Any]  # returns the Trace/Recording
    kind: str = "trace"  # trace | record


def _plain_trace(model: nn.Module, x: Any) -> Any:
    return tl.trace(model, x)


def _reference_trace(model: nn.Module, x: Any) -> Any:
    # reference-mode save: journal payloads are the live output tensors
    return tl.trace(model, x, save_mode="reference")


def _save_arg_values_trace(model: nn.Module, x: Any) -> Any:
    # The save_arg_values-only capture the DoR names as a parity case
    # (templates facet gate: saved_args without intervention_ready).
    return tl.trace(model, x, capture=tl.options.CaptureOptions(save_arg_values=True))


def _predicate_trace(model: nn.Module, x: Any) -> Any:
    return tl.trace(model, x, save=tl.func("relu"))


def _lookback_trace(model: nn.Module, x: Any) -> Any:
    # linear ops whose graph successor is tanh: exercises followed_by +
    # lookback retention (the lookback_retention amendment family's writer).
    return tl.trace(
        model,
        x,
        save=tl.func("linear") & tl.followed_by(tl.func("tanh")),
        lookback=4,
        lookback_payload_policy="detached_raw",
    )


def _intervene_trace(model: nn.Module, x: Any) -> Any:
    return tl.trace(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )


def _backward_trace(model: nn.Module, x: Any) -> Any:
    trace = tl.trace(
        model,
        x.requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    output = trace.output_ops[0].out if trace.output_ops else None
    if output is not None and output.requires_grad:
        output.sum().backward()
    return trace


def _record_capture(model: nn.Module, x: Any) -> Any:
    return tl.record(model, x, save=tl.func("relu"))


class ManualEdgeModel(nn.Module):
    """Calls the public ``register_tensor_connection`` amendment mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 3)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.fc1(x)
        b = self.fc2(x)
        # Public amendment channel: graph_edge_insertion family.
        tl.register_tensor_connection(a, b)
        return a + b


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("cnn_exhaustive", lambda: (SmallCNN(), _cnn_input()), _plain_trace),
    Scenario("cnn_reference", lambda: (SmallCNN(), _cnn_input()), _reference_trace),
    Scenario("cnn_save_arg_values", lambda: (SmallCNN(), _cnn_input()), _save_arg_values_trace),
    Scenario("cnn_predicate", lambda: (SmallCNN(), _cnn_input()), _predicate_trace),
    Scenario("recurrent_lookback", lambda: (TinyRecurrent(), _vec_input()), _lookback_trace),
    Scenario("cnn_intervene", lambda: (SmallCNN(), _cnn_input()), _intervene_trace),
    Scenario("cnn_backward", lambda: (SmallCNN(), _cnn_input()), _backward_trace),
    Scenario("recurrent_exhaustive", lambda: (TinyRecurrent(), _vec_input()), _plain_trace),
    Scenario("cnn_record", lambda: (SmallCNN(), _cnn_input()), _record_capture, kind="record"),
    Scenario(
        "buffer_output_exhaustive",
        lambda: (BufferOutputModel(), _small_vec_input()),
        _plain_trace,
    ),
    Scenario(
        "manual_edge_exhaustive",
        lambda: (ManualEdgeModel(), _small_vec_input()),
        _plain_trace,
    ),
)


def scenario_by_name(name: str) -> Scenario:
    """Return one scenario by name."""

    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)
