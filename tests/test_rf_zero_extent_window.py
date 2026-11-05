"""An intermediate tensor with a zero-length axis relays no projective geometry.

RF-DETR and WaveFormer-sEMG (Menagerie m2416, m21458) failed validation with
``ValueError: A transposed window map requires a positive output extent``: the
always-on receptive-field metadata invariant solves the projective field from the
outputs back to the inputs, and transposing a window rule onto an empty child (an
empty slice later concatenated with real data) asked for a forward map into an
axis with no coordinates. A tensor with no units carries no value onward, so no
unit of its parent reaches a target through it: the projective solve and the
per-unit forward walk skip that edge, and the other paths keep their geometry.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._forward_query import box_for_source_unit
from torchlens.receptive_field._validation import check_geometric_metadata_invariants
from torchlens.validation import last_validation_failure


class _EmptySliceConcat(nn.Module):
    """Concatenate a shifted slice with an empty slice of the same input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``2 * x[..., 1:]`` with an empty slice concatenated after it."""

        kept = x[:, :, 1:]
        empty = x[:, :, 3:3]
        return torch.cat([kept, empty], dim=2) * 2


def _windowed_bounds(box: object) -> tuple[int | None, int | None]:
    """Return the sole windowed axis bounds of a one-dimensional box."""

    spatial = next(axis for axis in getattr(box, "axes") if axis.kind == "windowed")
    return spatial.clipped_start, spatial.clipped_stop


def test_empty_window_child_validates() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_EmptySliceConcat().eval(), torch.randn(1, 1, 8), scope="forward"), (
            last_validation_failure()
        )


def test_empty_window_child_relays_nothing_and_the_dense_path_keeps_its_geometry() -> None:
    trace = tl.trace(_EmptySliceConcat().eval(), torch.randn(1, 1, 8))
    assert check_geometric_metadata_invariants(trace)
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    solution = solve_projective(trace, [target])
    empty_ops = [op.label for op in trace._layer_list if 0 in tuple(op.shape or ())]
    assert empty_ops
    named = {label for label, _role in solution.descriptors}
    assert source.label in named
    assert not named.intersection(empty_ops)
    box = box_for_source_unit(solution, source, (4,), target=target)
    assert _windowed_bounds(box) == (3, 4)
