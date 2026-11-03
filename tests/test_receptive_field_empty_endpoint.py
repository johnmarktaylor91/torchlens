"""An empty tensor at a model input or output has an empty receptive field.

A model returning an empty tensor (a zero-size ``split`` chunk) made the
always-on receptive-field metadata invariant raise ``ValueError: input_shape must
contain only positive integer extents`` while seeding the projective solve at
that output. An empty endpoint has no units, so nothing projects into it (and an
empty input influences nothing): no geometry is seeded from it and no descriptor
names it.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._engine import solve
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._validation import check_geometric_metadata_invariants
from torchlens.validation import last_validation_failure


class _EmptyOutput(nn.Module):
    """Returns a used chunk and an empty chunk of one split."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        used, empty = torch.split(x * 2, [4, 0], dim=1)
        return used + 1, empty


class _EmptyInput(nn.Module):
    """Reads a non-empty input and an empty one."""

    def forward(self, x: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
        return x * 2 + empty.sum()


def test_empty_output_validates() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_EmptyOutput().eval(), torch.randn(2, 4), scope="forward") is True, (
            last_validation_failure()
        )


def test_empty_output_has_no_projective_descriptor() -> None:
    trace = tl.trace(_EmptyOutput().eval(), torch.randn(2, 4))
    empty_output = next(op for op in trace.output_ops if 0 in tuple(op.shape))
    used_output = next(op for op in trace.output_ops if 0 not in tuple(op.shape))
    solution = solve_projective(trace, trace.output_ops)
    roles = {role for _label, role in solution.descriptors}
    assert (used_output.io_role or used_output.label) in roles
    assert (empty_output.io_role or empty_output.label) not in roles
    assert check_geometric_metadata_invariants(trace)


def test_empty_input_has_no_receptive_descriptor() -> None:
    trace = tl.trace(_EmptyInput().eval(), (torch.randn(2, 4), torch.zeros(2, 0)))
    empty_input = next(op for op in trace.input_ops if 0 in tuple(op.shape))
    full_input = next(op for op in trace.input_ops if 0 not in tuple(op.shape))
    roles = {role for _label, role in solve(trace).descriptors}
    assert (full_input.io_role or full_input.label) in roles
    assert (empty_input.io_role or empty_input.label) not in roles
    assert check_geometric_metadata_invariants(trace)
