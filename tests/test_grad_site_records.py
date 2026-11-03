"""Gradient-site records: FireRecord fields for gradient fires and the signal lockstep.

Packet-P2 criteria of the gradient-sites spec: one fire per output slot of a
multi-output op with forward labels, ``Edit.signal`` replacing ``kind`` and
``direction``, ``FireRecord.signal`` and the audit ``signal`` key, ``Rule`` with no
``direction`` field, and the removed gradient-only helpers refusing typed on load.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import _helpers as helper_module
from torchlens.intervention._audit import rule_signal, rules_payload
from torchlens.intervention._types import Edit, FireRecord

_CAPTURE = tl.options.CaptureOptions(backward_ready=True, save_grads="all")


class _Split(nn.Module):
    """Two outputs of one ``torch.split`` call, both reaching the loss."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, 1, dim=1)
        return (a * 2 + b * 3).sum()


def test_multi_output_op_fires_once_per_output_slot_with_forward_labels() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 2, requires_grad=True)
    base = tl.trace(_Split(), x, capture=_CAPTURE)
    split_labels = [label for label in base.layer_labels if label.startswith("split")]
    assert len(split_labels) == 2
    trace = tl.trace(
        _Split(), x, capture=_CAPTURE, intervene=tl.when(tl.grad(tl.func("split")), tl.scale(0.5))
    )
    trace.backward(trace[trace.output_layers[0]].out)
    records = [record for record in trace._intervention_spec.records if record.signal == "backward"]
    assert len(records) == 2
    assert {record.grad_kind for record in records} == {"grad_output"}
    assert sorted(record.tuple_index for record in records) == [0, 1]
    for record in records:
        assert record.site_label in split_labels
        assert record.target_label == record.site_label
        assert record.call_label == record.site_label
    # d loss / d a = 2 and d loss / d b = 3, both halved by the edit.
    assert torch.allclose(x.grad, torch.tensor([[1.0, 1.5]] * 3))


def test_edit_signal_replaces_kind_and_direction() -> None:
    field_names = {field.name for field in dataclasses.fields(Edit)}
    assert "signal" in field_names
    assert not {"kind", "direction"} & field_names
    assert "signal" in Edit._PORTABLE_STATE_SPEC
    assert not {"kind", "direction"} & set(Edit._PORTABLE_STATE_SPEC)
    for neutral in (
        tl.scale(0.5),
        tl.zero_ablate(),
        tl.clamp(low=-1.0, high=1.0),
        tl.add_noise(0.1, seed=0),
        tl.clip_norm(1.0),
        helper_module.swap_with(torch.zeros(2)),
    ):
        assert neutral.signal is None, neutral.name
    model = nn.Linear(2, 2)
    donor = tl.trace(model, torch.randn(1, 2))
    for forward_only in (
        tl.mean_ablate(dim=None),
        tl.splice_module(nn.Identity()),
        tl.patch_from(donor),
        helper_module.swap_with(donor[donor.layer_labels[1]]),
    ):
        assert forward_only.signal == "forward", forward_only.name


def test_fire_record_and_audit_payload_carry_signal() -> None:
    field_names = {field.name for field in dataclasses.fields(FireRecord)}
    assert "signal" in field_names and "direction" not in field_names
    assert "signal" in FireRecord._PORTABLE_STATE_SPEC
    assert "direction" not in FireRecord._PORTABLE_STATE_SPEC
    recipe = tl.when(tl.func("relu"), tl.scale(0.5))
    (row,) = rules_payload(recipe)
    assert row["signal"] == "forward" and "direction" not in row
    assert rule_signal(tl.grad("relu_1")) == "backward"
    assert rule_signal(tl.grad("relu_1") | "relu_1") == "both"


def test_rule_has_no_direction_field_and_the_site_decides_the_pass() -> None:
    rule_cls = tl.intervention.Rule
    assert "direction" not in {field.name for field in dataclasses.fields(rule_cls)}
    rule = rule_cls(tl.func("relu"), tl.scale(0.5))
    assert rule.signal == "forward" and "direction" not in repr(rule)
    assert rule_cls(tl.grad(tl.func("relu")), tl.scale(0.5)).signal == "backward"
    # The rule id keeps the digest it had while the field was always None.
    tail = f"{rule.where_repr}||{rule.action_repr}||None"
    assert rule.rule_id == f"r-{hashlib.sha256(tail.encode()).hexdigest()[:10]}"
    stale_spellings = (
        lambda: rule_cls(tl.func("relu"), tl.scale(0.5), direction="backward"),
        lambda: rule_cls(tl.func("relu"), tl.scale(0.5), direction=None),
        lambda: rule_cls(tl.func("relu"), tl.scale(0.5), "both"),
    )
    for build in stale_spellings:
        with pytest.raises(InvalidArgumentError) as exc_info:
            build()
        assert exc_info.value.fields["code"] == "intervention_direction_invalid"
        assert "tl.grad(site)" in exc_info.value.fields["remedy"]


@pytest.mark.parametrize(
    "removed", ["zero_grad", "scale_grad", "clamp_grad", "add_grad_noise", "clip_grad", "hook_grad"]
)
def test_removed_gradient_helpers_are_gone_and_refuse_on_load(removed: str) -> None:
    assert removed not in helper_module.__all__
    assert not hasattr(helper_module, removed)
    with pytest.raises(InvalidArgumentError) as exc_info:
        helper_module.rebuild_builtin_helper(removed, (), {})
    assert exc_info.value.fields["code"] == "intervention_helper_renamed"
    assert "tl.grad(site)" in exc_info.value.fields["remedy"]


def test_clip_norm_round_trips_through_the_builtin_registry() -> None:
    rebuilt = helper_module.rebuild_builtin_helper("clip_norm", (2.0,), {"norm_type": 1.0})
    assert rebuilt == tl.clip_norm(2.0, norm_type=1.0)
    assert repr(rebuilt).endswith("clip_norm(2.0, norm_type=1.0)")
