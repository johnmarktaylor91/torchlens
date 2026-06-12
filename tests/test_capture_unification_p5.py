"""Phase 5 capture-unification predicate intervention tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext, Recording


class LinearRelu(nn.Module):
    """Tiny model with an intervention site and downstream consumer."""

    def __init__(self) -> None:
        """Initialize deterministic weights."""

        super().__init__()
        self.attn = nn.Linear(4, 4)
        with torch.no_grad():
            self.attn.weight.copy_(torch.eye(4))
            self.attn.bias.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a named linear site followed by a relu consumer."""

        hidden = self.attn(x)
        return torch.relu(hidden)


class Layer4Predecessor(nn.Module):
    """Model whose relu is preceded by an op inside ``layer4``."""

    def __init__(self) -> None:
        """Initialize deterministic modules."""

        super().__init__()
        self.layer4 = nn.Linear(4, 4)
        with torch.no_grad():
            self.layer4.weight.copy_(torch.eye(4))
            self.layer4.bias.fill_(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a layer4 parent followed by an outside relu."""

        hidden = self.layer4(x)
        return torch.relu(hidden)


class RecordingAliasMutation(nn.Module):
    """Model whose sparse recording lacks parent-version replay data."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a view of an unsaved parent before a saved downstream child."""

        parent = x + 1
        view = parent.view_as(parent)
        view.add_(2)
        return parent * 3


def _linear_op(log: tl.Trace) -> tl.Op:
    """Return the first linear op in a trace."""

    return next(op for op in log.layer_list if op.layer_type == "linear")


def _relu_op(log: tl.Trace) -> tl.Op:
    """Return the first relu op in a trace."""

    return next(op for op in log.layer_list if op.layer_type == "relu")


def _save_only_mul(ctx: RecordContext) -> bool:
    """Select only the downstream multiplication op."""

    return ctx.func_name in {"__mul__", "mul"}


def test_intervene_save_captures_post_hook_tensor_and_is_deterministic() -> None:
    """Saved payload at an intervened op equals the hook-returned tensor."""

    captured: list[torch.Tensor] = []

    def add_marker(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Return a marked tensor and retain the exact expected payload."""

        del hook
        result = out + 7
        captured.append(result.detach().clone())
        return result

    x = torch.ones(1, 4)
    log = tl.trace(
        LinearRelu(),
        x,
        intervene=tl.when(tl.func("linear"), add_marker),
        save=tl.func("linear"),
    )
    rerun = tl.trace(
        LinearRelu(),
        x,
        intervene=tl.when(tl.func("linear"), add_marker),
        save=tl.func("linear"),
    )

    assert torch.equal(_linear_op(log).out, captured[0])
    assert torch.equal(_linear_op(log).out, _linear_op(rerun).out)
    assert _linear_op(log).intervention_replaced


def test_intervene_and_save_compose_with_downstream_saved_child() -> None:
    """Downstream saved ops consume the mutated parent output."""

    log = tl.trace(
        LinearRelu(),
        torch.ones(1, 4),
        intervene=tl.when(tl.func("linear"), tl.scale(0.5)),
        save=tl.func("linear") | tl.func("relu"),
    )

    linear = _linear_op(log)
    relu = _relu_op(log)
    assert torch.equal(relu.out, torch.relu(linear.out))
    assert linear.intervention_replaced


def test_intervene_only_mutates_forward_output_without_saving_site() -> None:
    """Intervention without save still mutates the model output."""

    log = tl.trace(
        LinearRelu(),
        torch.ones(1, 4),
        layers_to_save="none",
        intervene=tl.when(tl.func("linear"), tl.zero_ablate()),
        output_transform=lambda output: output,
    )
    output = log.raw_output

    assert isinstance(output, torch.Tensor)
    assert torch.equal(output, torch.zeros_like(output))
    assert not _linear_op(log).has_saved_activation
    assert _linear_op(log).intervention_replaced


def test_conditional_intervention_uses_live_index_lookback() -> None:
    """``preceded_by(in_module(...))`` fires using forward-time predicate history."""

    log = tl.trace(
        Layer4Predecessor(),
        torch.ones(1, 4),
        intervene=tl.when(tl.func("relu") & tl.preceded_by(tl.in_module("layer4")), tl.scale(0.5)),
        save=tl.func("linear") | tl.func("relu"),
        lookback=4,
    )

    linear = _linear_op(log)
    relu = _relu_op(log)
    assert torch.equal(relu.out, torch.relu(linear.out) * 0.5)
    assert relu.intervention_replaced
    assert not linear.intervention_replaced


def test_recording_to_trace_refuses_missing_child_version_replay_data() -> None:
    """Sparse ``to_trace`` validation loudly refuses absent arg-version data."""

    model = RecordingAliasMutation()
    x = torch.ones(2, 4)
    recording = cast(Recording, tl.record(model, x, save=_save_only_mul))
    trace = recording.to_trace()

    assert not trace._replay_arg_version_data_complete
    assert all(not op.out_versions_by_child for op in trace.layer_list)
    with pytest.raises(ValueError, match="child-version snapshots"):
        trace.validate_forward_pass([model(x).detach().clone()], validate_metadata=False)


def test_fastlog_record_intervene_save_captures_post_hook_payload() -> None:
    """``fastlog.record`` composes intervention and save in one predicate pass."""

    output, recording = cast(
        tuple[torch.Tensor, Recording],
        tl.fastlog.record(
            LinearRelu(),
            torch.ones(1, 4),
            keep_op=tl.func("linear"),
            intervene=tl.when(tl.func("linear"), tl.zero_ablate()),
            return_output=True,
        ),
    )

    assert torch.equal(output, torch.zeros_like(output))
    assert len(recording.records) == 1
    payload = recording.records[0].ram_payload
    assert isinstance(payload, torch.Tensor)
    assert torch.equal(
        payload,
        torch.zeros_like(payload),
    )
