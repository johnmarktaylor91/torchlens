"""Intervention honesty: a depth-2 store key cannot reach the three depth-1
splice sites without refusing (edits memo row 0d).

The v1 splice sites substitute at ``arg_path[0]`` directly; a nested key
reaching one of them would silently replace the WHOLE top-level argument at
the wrong address -- well-formed, audited, wrong. The entry gate already
refuses nested addresses at construction, so these tripwires defend against
foreign, forged, or future-schema store rows at:

1. the edge re-execution splice (``edge_substitution``),
2. the param-substitution replay splice (``replay``),
3. the validation-side splice (``validation/_edge_boundary``), which must
   FAIL validation, never guess.
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.edge_substitution import (
    _reexecute_child_with_substitution,
    require_depth1_arg_path,
)
from torchlens.intervention.replay import _splice_param_substitutions
from torchlens.selection import SelectionError
from torchlens.validation.core import _check_edge_intervention_boundary


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


@pytest.fixture(scope="module")
def capture():
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(1, 1, 12, 12)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _identity_edge_fork(trace: tl.Trace) -> tl.Trace:
    edge = next(e for e in trace.edges if e.parent_label == "relu_1_2")
    fork = trace.fork()
    fork.do(edge.__selection__(), tl.scale(1.0))
    return fork


def test_guard_accepts_depth1_and_refuses_deeper() -> None:
    require_depth1_arg_path((0,), where="unit", site="s")
    for nested in [(0, 1), (0, "k"), ()]:
        with pytest.raises(SelectionError) as excinfo:
            require_depth1_arg_path(nested, where="unit", site="s")
        assert excinfo.value.fields["code"] == "selection_apply_invalid"
        assert "nested argument path" in str(excinfo.value) or "()" in str(excinfo.value)


def test_edge_reexecution_splice_refuses_nested_address(capture) -> None:
    """Site 1: the edge splice refuses a forged depth-2 occurrence address."""

    model, x, trace = capture
    child = trace["conv2d_2_3"].ops[0]
    address = (child.func_call_id, "positional", (0, 1))
    with pytest.raises(SelectionError) as excinfo:
        _reexecute_child_with_substitution(trace, child, address, torch.zeros(1), strict=False)
    assert excinfo.value.fields["code"] == "selection_apply_invalid"
    assert "nested argument path" in str(excinfo.value)


def test_param_replay_splice_refuses_nested_store_key(capture) -> None:
    """Site 2: the param-kind replay splice refuses a forged depth-2 key."""

    member = SimpleNamespace(
        label="conv2d_2_3:1",
        edge_substitutions={
            ("positional", (0, 1)): {
                "substitution_kind": "param",
                "value": torch.zeros(1),
            }
        },
    )
    with pytest.raises(SelectionError) as excinfo:
        _splice_param_substitutions([member], (torch.ones(1),), {})
    assert excinfo.value.fields["code"] == "selection_apply_invalid"
    assert "param-substitution replay splice" in str(excinfo.value)


def test_validation_splice_fails_nested_key_even_when_corroborated(capture) -> None:
    """Site 3: a COHERENT forgery (store key + FireRecord address + stamp all
    nested) passes the corroboration gate and must still FAIL validation at
    the splice -- never re-execute from a guessed address."""

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]

    old_key = next(iter(child.edge_substitutions))
    arg_kind, arg_path = old_key
    nested_key = (arg_kind, tuple(arg_path) + (1,))

    store = dict(child.edge_substitutions)
    store[nested_key] = store.pop(old_key)
    child._internal_set("edge_substitutions", store)

    stamps = dict(child.edge_replacement_stamps)
    stamps[nested_key] = stamps.pop(old_key)
    child._internal_set("edge_replacement_stamps", stamps)

    records = []
    for record in child.interventions:
        if record.edge_address:
            records.append(
                dataclasses.replace(
                    record,
                    edge_address=(child.func_call_id, arg_kind, tuple(arg_path) + (1,)),
                )
            )
        else:
            records.append(record)
    child._internal_set("interventions", records)

    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_nested_path"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False
