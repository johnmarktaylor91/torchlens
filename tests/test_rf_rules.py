"""Golden receptive-field tests for the built-in T5 rule pack."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as functional
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._types import ReceptiveFieldStatus


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the pack only for this test, preserving other registry tests' isolation."""

    with preserved_rf_registry(install_builtin=True):
        yield


def _op(trace: object, name: str) -> object:
    """Return the first captured operation with the requested raw function name."""

    return next(item for item in trace._layer_list if item.func_name == name)  # type: ignore[union-attr]


def _descriptor(op: object) -> object:
    """Return the sole input descriptor for a one-input captured operation."""

    return next(iter(_engine.lookup(op.source_trace, op).values()))  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "training,track_running_stats,status,batch_coupled",
    [
        (True, True, ReceptiveFieldStatus.WHOLE_INPUT, True),
    ],
)
def test_batch_norm_mode_matrix(
    training: bool,
    track_running_stats: bool,
    status: ReceptiveFieldStatus,
    batch_coupled: bool,
) -> None:
    """Use the captured functional statistics switch, never a transparent training rule."""

    model = nn.BatchNorm2d(
        2,
        affine=False,
        track_running_stats=track_running_stats,
    ).train(training)
    trace = tl.trace(model, torch.randn(3, 2, 4, 4))
    descriptor = _descriptor(_op(trace, "batch_norm"))

    assert descriptor.status is status
    assert descriptor.batch_coupled is batch_coupled


def test_training_instance_norm_projective_maps_running_stat_parents() -> None:
    """Map InstanceNorm running-stat vectors onto the channel axis."""

    trace = tl.trace(
        nn.InstanceNorm1d(3, track_running_stats=True).train(),
        torch.randn(2, 3, 4),
    )
    instance_norm = _op(trace, "instance_norm")
    solution = solve_projective(trace, trace.output_ops)

    for parent_reference in instance_norm.parents:
        parent = trace._layer_dict_all_keys[parent_reference]
        state = solution.states[(parent.label, trace.output_ops[0].io_role)]
        assert state.axes is not None


def test_group_and_current_stat_instance_norm_globalize_normalized_axes() -> None:
    """Keep InstanceNorm exact while bounding GroupNorm to its group envelope."""

    class Norms(nn.Module):
        """Small current-stat normalization fixture."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply instance then group normalization using input statistics."""

            value = functional.instance_norm(value, use_input_stats=True)
            return functional.group_norm(value, 2)

    trace = tl.trace(Norms(), torch.randn(2, 4, 3, 3))
    instance = _descriptor(_op(trace, "instance_norm"))
    assert instance.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert all(axis.kind == "full" for axis in instance.axes[2:])

    group = _descriptor(_op(trace, "group_norm"))
    assert group.status is ReceptiveFieldStatus.UPPER_BOUND
    assert all(axis.kind == "full" for axis in group.axes[1:])
    assert group.axes[1].exact is False
