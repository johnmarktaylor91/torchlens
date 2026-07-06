"""S2 capture-spine golden projections across public capture surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext


class S2Mlp(nn.Module):
    """Small feed-forward model for capture-spine golden coverage."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.net(x)


class S2CnnInplaceBuffer(nn.Module):
    """Convolutional model with an in-place op and registered buffer."""

    def __init__(self) -> None:
        """Initialize convolution and buffer."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.register_buffer("bias", torch.full((1, 2, 1, 1), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolution, in-place relu, and buffer add."""

        y = self.conv(x)
        y.relu_()
        return y + self.bias


class S2Recurrent(nn.Module):
    """Tiny recurrent model with repeated calls through one module."""

    def __init__(self) -> None:
        """Initialize recurrent cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps."""

        state = x
        for _ in range(3):
            state = torch.tanh(self.cell(state))
        return state


class S2MultiOutput(nn.Module):
    """Model returning a nested multi-output container."""

    def __init__(self) -> None:
        """Initialize projection layer."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Return nested tensor outputs."""

        y = self.proj(x)
        return {"main": y, "parts": (torch.relu(y), y.mean(dim=-1))}


class S2Intervention(nn.Module):
    """Model used for intervention trace projection coverage."""

    def __init__(self) -> None:
        """Initialize linear layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by relu."""

        return torch.relu(self.linear(x))


@dataclass(frozen=True, slots=True)
class ModelCase:
    """Deterministic model/input case for S2 golden projection tests."""

    name: str
    factory: Callable[[], nn.Module]
    input_factory: Callable[[], torch.Tensor]


def _save_all_ops(ctx: RecordContext) -> bool:
    """Return True for operation contexts and False for module contexts."""

    return ctx.kind in {"op", "source", "input", "buffer"}


def _projection(trace: tl.Trace) -> tuple[dict[str, Any], ...]:
    """Return a stable comparable trace projection.

    Parameters
    ----------
    trace
        Trace to project.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Per-op projection ordered by final trace order.
    """

    rows: list[dict[str, Any]] = []
    for op in trace.layer_list:
        rows.append(
            {
                "label": op.layer_label,
                "raw": op._label_raw,
                "type": op.layer_type,
                "func": "none" if op.func_name is None else op.func_name,
                "parents": tuple(op.parents),
                "shape": tuple(op.shape) if op.shape is not None else None,
                "saved": bool(op.has_saved_activation),
                "output": op.layer_label in trace.output_layers,
            }
        )
    return tuple(rows)


def _recording_projection(
    recording: tl.Recording,
    raw_to_final: dict[str, str],
) -> tuple[dict[str, Any], ...]:
    """Return a stable comparable projection for a ``Recording`` event stream.

    Parameters
    ----------
    recording
        Fastlog recording to project.
    raw_to_final
        Mapping from exhaustive raw labels to final labels.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Per-event projection ordered by raw capture order.
    """

    events = (
        recording._capture_events.op_events if recording._capture_events is not None else []  # noqa: SLF001
    )
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.kind not in {"op", "source"}:
            continue
        rows.append(
            {
                "label": raw_to_final.get(event.label_raw, event.label_raw),
                "raw": event.label_raw,
                "type": event.layer_type,
                "func": "none" if event.function.func_name is None else event.function.func_name,
                "parents": tuple(
                    raw_to_final.get(edge.parent_label_raw, edge.parent_label_raw)
                    for edge in event.parents
                ),
                "shape": tuple(event.output.tensor.shape)
                if event.output.tensor.shape is not None
                else None,
                "saved": bool(event.output.has_saved_activation),
                "output": bool(event.is_output_parent),
            }
        )
    return tuple(rows)


def _topology(projection: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Return projection fields that should be identical across save surfaces."""

    return tuple({key: value for key, value in row.items() if key != "saved"} for row in projection)


def _op_rows(projection: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Return operation rows, excluding source-only input and buffer nodes."""

    return tuple(row for row in projection if row["type"] not in {"input", "buffer", "output"})


def _recording_operation_identity(
    projection: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Return fields that Recording currently projects with Trace parity."""

    comparable_keys = ("label", "raw", "type", "func", "shape")
    return tuple({key: row[key] for key in comparable_keys} for row in _op_rows(projection))


def _recording_raw_operation_identity(
    projection: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Return raw operation fields that do not require recurrence equivalence classes."""

    comparable_keys = ("raw", "type", "func", "shape")
    return tuple({key: row[key] for key in comparable_keys} for row in _op_rows(projection))


def _saved_labels(projection: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Return labels with retained activation payloads."""

    return tuple(str(row["label"]) for row in projection if row["saved"])


@pytest.mark.parametrize(
    "case",
    (
        ModelCase("mlp", S2Mlp, lambda: torch.randn(2, 4)),
        ModelCase("cnn_inplace_buffer", S2CnnInplaceBuffer, lambda: torch.randn(1, 1, 4, 4)),
        ModelCase("recurrent", S2Recurrent, lambda: torch.randn(2, 4)),
        ModelCase("multi_output", S2MultiOutput, lambda: torch.randn(2, 4)),
    ),
    ids=lambda case: case.name,
)
def test_s2_trace_surfaces_match_exhaustive_projection(case: ModelCase) -> None:
    """Capture the fixed S2 zoo through every non-intervention surface."""

    torch.manual_seed(20260705)
    model = case.factory().eval()
    x = case.input_factory()

    exhaustive = tl.trace(model, x.clone(), layers_to_save="all", random_seed=7)
    exhaustive_projection = _projection(exhaustive)
    target_label = next(row["label"] for row in _op_rows(exhaustive_projection))

    predicate = tl.trace(model, x.clone(), save=_save_all_ops, random_seed=7)
    two_pass = tl.trace(model, x.clone(), layers_to_save=[target_label], random_seed=7)
    recording = tl.record(
        model,
        x.clone(),
        save=_save_all_ops,
        include_source_events=True,
        random_seed=7,
    )

    assert _topology(_projection(predicate)) == _topology(exhaustive_projection)
    assert _topology(_projection(two_pass)) == _topology(exhaustive_projection)
    recording_trace = recording.to_trace()
    recording_projection = _projection(recording_trace)
    if case.name == "recurrent":
        assert _recording_raw_operation_identity(
            recording_projection
        ) == _recording_raw_operation_identity(exhaustive_projection)
    else:
        assert recording_projection == exhaustive_projection
    assert _saved_labels(exhaustive_projection)
    assert _saved_labels(_projection(predicate)) == _saved_labels(exhaustive_projection)
    two_pass_op_saved = set(_saved_labels(_op_rows(_projection(two_pass))))
    assert target_label in two_pass_op_saved
    assert two_pass_op_saved.issubset(set(_saved_labels(_op_rows(exhaustive_projection))))


def test_s2_intervention_trace_preserves_projection_identity() -> None:
    """Capture intervention trace topology in the S2 golden harness."""

    torch.manual_seed(20260705)
    model = S2Intervention().eval()
    x = torch.randn(2, 4)
    baseline = tl.trace(model, x.clone(), layers_to_save="all", random_seed=11)
    intervened = tl.trace(
        model,
        x.clone(),
        layers_to_save="all",
        intervene=tl.when(tl.func("relu"), tl.add(0.0)),
        random_seed=11,
    )

    assert _topology(_projection(intervened)) == _topology(_projection(baseline))
    assert _saved_labels(_projection(intervened)) == _saved_labels(_projection(baseline))
