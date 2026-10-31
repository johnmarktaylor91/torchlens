"""Token-position editing, phase 2 (U42.1): edit chosen positions, padding-aware.

Each row edits the middle layer's ``(batch, tokens, hidden)`` output through
``torchlens.positions`` and compares the model output with the
same edit written by hand on the same positions.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.errors import HookValueError
from torchlens.transforms._errors import TransformContractError

_IDS = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
#: Right-padded batch: row 0 has 3 real tokens, row 1 has 5.
_RIGHT_MASK = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
#: Left-padded batch: row 0 has 2 real tokens at the end.
_LEFT_MASK = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])


def _model() -> nn.Sequential:
    """Embedding, then the edited linear site, then a readout."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Embedding(12, 4), nn.Linear(4, 4), nn.Linear(4, 2)).eval()


def _run(model: nn.Sequential, edit: Any) -> torch.Tensor:
    """Run ``edit`` at the middle layer through TorchLens and return the output."""

    trace = tl.trace(model, _IDS, intervene=tl.when(tl.module("1"), edit))
    return trace[trace.output_layers[0]].out.detach()


def _by_hand(model: nn.Sequential, rows: list[int], cols: list[int], fn: Any) -> torch.Tensor:
    """Apply ``fn`` to the middle layer at the (row, token) pairs, by hand."""

    with torch.no_grad():
        hidden = model[1](model[0](_IDS))
        edited = hidden.clone()
        for row, col in zip(rows, cols, strict=True):
            edited[row, col] = fn(hidden[row, col])
        return model[2](edited)


def _zero(value: torch.Tensor) -> torch.Tensor:
    """Zero one position's hidden vector."""

    return torch.zeros_like(value)


@pytest.mark.parametrize(
    ("positions", "rows", "cols"),
    [
        (-1, [0, 1], [4, 4]),
        ([0, 2], [0, 0, 1, 1], [0, 2, 0, 2]),
        (slice(3, None), [0, 0, 1, 1], [3, 4, 3, 4]),
        (torch.tensor([1, 3]), [0, 1], [1, 3]),
    ],
)
def test_physical_positions_match_by_hand(positions: Any, rows: list[int], cols: list[int]) -> None:
    """Int, list, slice and per-row positions edit exactly those tokens."""

    model = _model()
    got = _run(model, tl.positions(tl.zero_ablate(), positions))
    torch.testing.assert_close(got, _by_hand(model, rows, cols, _zero), rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("mask", "which", "cols"),
    [
        (_RIGHT_MASK, "last", [2, 4]),
        (_LEFT_MASK, "last", [4, 4]),
        (_LEFT_MASK, "first", [3, 0]),
    ],
)
def test_last_real_token_in_a_padded_batch(mask: torch.Tensor, which: str, cols: list[int]) -> None:
    """``"last"`` / ``"first"`` resolve each row's real token from the attention mask."""

    model = _model()
    direction = torch.ones(4)
    edit = tl.positions(tl.steer(direction, 3.0, dim=-1), which, mask=mask)
    got = _run(model, edit)
    want = _by_hand(model, [0, 1], cols, lambda value: value + 3.0 * direction)
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)


def test_capture_free_bind_edits_the_last_real_token() -> None:
    """The same position edit runs on the capture-free bind lane."""

    model = _model()
    edit = tl.positions(tl.zero_ablate(), "last", mask=_RIGHT_MASK)
    bound = tl.when(tl.module("1"), edit).bind(model)
    with torch.no_grad():
        got = bound(_IDS)
    assert bound.last_report is not None and bound.last_report.fire_count == 1
    torch.testing.assert_close(got, _by_hand(model, [0, 1], [2, 4], _zero), rtol=0.0, atol=1e-6)


def test_assume_no_padding_licenses_physical_last() -> None:
    """Without a mask, the recorded no-padding assertion makes ``"last"`` physical."""

    model = _model()
    got = _run(model, tl.positions(tl.zero_ablate(), "last", assume_no_padding=True))
    torch.testing.assert_close(got, _by_hand(model, [0, 1], [4, 4], _zero), rtol=0.0, atol=1e-6)


def test_last_without_mask_refuses_typed() -> None:
    """``"last"`` with neither a mask nor the assertion refuses (the pooling mask policy)."""

    with pytest.raises(TransformContractError) as info:
        _run(_model(), tl.positions(tl.zero_ablate(), "last"))
    assert info.value.fields["code"] == "transform_mask_missing"


def test_mask_contradicting_the_site_refuses_typed() -> None:
    """A mask whose shape does not describe the site value refuses."""

    with pytest.raises(TransformContractError) as info:
        _run(_model(), tl.positions(tl.zero_ablate(), "last", mask=torch.ones(2, 3)))
    assert info.value.fields["code"] == "transform_mask_incompatible"


@pytest.mark.parametrize("positions", ["middle", [], [1.5], 2.0, torch.tensor([0.5])])
def test_invalid_positions_refuse_at_construction(positions: Any) -> None:
    """Unsupported position spellings refuse when the edit is built."""

    with pytest.raises(InvalidArgumentError) as info:
        tl.positions(tl.zero_ablate(), positions)
    assert info.value.fields["code"] == "position_edit_invalid"


def test_out_of_range_position_refuses_typed() -> None:
    """A position past the sequence end refuses at fire time."""

    with pytest.raises(HookValueError) as info:
        _run(_model(), tl.positions(tl.zero_ablate(), 7))
    assert info.value.fields["code"] == "position_edit_out_of_range"


def test_shape_changing_edit_refuses_typed() -> None:
    """Only a same-shape edit can be kept at some positions: ``position_edit_shape_mismatch``."""

    def _narrow(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        """Return a value of another shape."""

        del hook
        return out[..., :2]

    with pytest.raises(HookValueError) as info:
        _run(_model(), tl.positions(_narrow, [0]))
    assert info.value.fields["code"] == "position_edit_shape_mismatch"
