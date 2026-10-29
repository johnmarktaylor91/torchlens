"""Exact value proofs for replay gaps that are value facts, each with an armed negative.

Mirrors the Menagerie call shapes behind the batch-2 replay gaps:

- ``scaled_dot_product_attention`` with an all-zero value (MaskFormer, DETR, Kandinsky) and
  with a single key (ARTrack);
- ``torch.einsum('bij,bj->bi', (S, k))`` with a fresh all-zero state ``S``;
- ``new_zeros(...).scatter_reduce_(0, index, src, 'amax', include_self=False)`` whose index
  addresses every destination row;
- an empty tensor times a scalar (no detections left);
- ``grid_sample`` of an all-zero image and ``conv1d`` of an all-zero input;
- ``a | b`` with ``b`` all True, and ``pow(1.0, x)``.

Each positive case is exempted under its decision name. Each negative keeps the call shape but
breaks one precondition (an operand near zero but not zero, a masked-out row, a second key, an
unaddressed row, a non-empty output), freezes the replay so the edge is really dead, and must
still fail validation.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation.core import _check_whether_func_on_saved_parents_yields_saved_tensor

#: Near zero but not zero: a normal float32, far below anything the proofs could miss.
TINY = 1e-30


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    torch.manual_seed(0)
    return tl.trace(
        model, x, capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=0)
    )


def _op(trace: Any, *func_names: str) -> Any:
    return next(op for op in trace.layer_list if op.func_name in func_names)


def _edge(trace: Any, op: Any, slot: Any) -> Any:
    parent = op.parent_arg_positions["args"][slot]
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent]
    )


def _freeze(op: Any) -> None:
    """Simulate a dead edge: the replay returns the saved out whatever its arguments."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


def _assert_exempted(result: Any, code: str) -> None:
    assert (result.decision, result.reason) == ("exempted", code), (
        result.decision,
        result.reason,
    )


def _assert_dead_edge_fails(trace: Any, op: Any, slot: Any) -> None:
    _freeze(op)
    result = _edge(trace, op, slot)
    assert (result.decision, result.reason) == ("failed", "perturbation_insensitive"), (
        result.decision,
        result.reason,
    )


def _validates(model: nn.Module, x: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, x, random_seed=0))


# ------------------------------------------------------------------ proofs.1: sdpa, value zero


class _ZeroValueAttention(nn.Module):
    """``sdpa(q, k, v)`` with ``v`` built from ``zeros_like`` plus ``fill``."""

    def __init__(self, fill: float = 0.0, mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self.fill = fill
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x * 2.0
        k = x + 1.0
        v = torch.zeros_like(k) + self.fill
        return F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)


def _attention_input() -> torch.Tensor:
    return torch.randn(1, 2, 4, 8)


def _row_zero_masked() -> torch.Tensor:
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[0] = False
    return mask


def test_sdpa_zero_value_validates() -> None:
    assert _validates(_ZeroValueAttention(), _attention_input())


@pytest.mark.parametrize("slot", [0, 1], ids=["query", "key"])
def test_sdpa_zero_value_edge_is_exempted(slot: int) -> None:
    trace = _capture(_ZeroValueAttention(), _attention_input())
    result = _edge(trace, _op(trace, "scaled_dot_product_attention"), slot)
    _assert_exempted(result, "sdpa_zero_value_output")


def test_dead_sdpa_query_edge_with_near_zero_value_still_fails() -> None:
    trace = _capture(_ZeroValueAttention(fill=TINY), _attention_input())
    _assert_dead_edge_fails(trace, _op(trace, "scaled_dot_product_attention"), 0)


def test_dead_sdpa_query_edge_with_a_fully_masked_row_still_fails() -> None:
    trace = _capture(_ZeroValueAttention(mask=_row_zero_masked()), _attention_input())
    _assert_dead_edge_fails(trace, _op(trace, "scaled_dot_product_attention"), 0)


# ------------------------------------------------------------------ proofs.2: sdpa, one key


class _FewKeyAttention(nn.Module):
    """``sdpa(q, k, v)`` over the first ``keys`` positions of the input."""

    def __init__(self, keys: int) -> None:
        super().__init__()
        self.keys = keys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x * 2.0
        k = x[:, :, : self.keys] + 1.0
        v = x[:, :, : self.keys] * 3.0
        return F.scaled_dot_product_attention(q, k, v)


def test_sdpa_single_key_query_edge_is_exempted() -> None:
    trace = _capture(_FewKeyAttention(keys=1), _attention_input())
    result = _edge(trace, _op(trace, "scaled_dot_product_attention"), 0)
    _assert_exempted(result, "sdpa_single_key_identity")


def test_dead_sdpa_query_edge_over_two_keys_still_fails() -> None:
    trace = _capture(_FewKeyAttention(keys=2), _attention_input())
    _assert_dead_edge_fails(trace, _op(trace, "scaled_dot_product_attention"), 0)


def test_dead_sdpa_key_edge_over_one_key_still_fails() -> None:
    """Only the query is covered: a dead key edge of the same call still fails."""

    trace = _capture(_FewKeyAttention(keys=1), _attention_input())
    _assert_dead_edge_fails(trace, _op(trace, "scaled_dot_product_attention"), 1)


# ------------------------------------------------------------------ proofs.4: einsum


class _ZeroStateEinsum(nn.Module):
    """``einsum('bij,bj->bi', (S, k))`` with a fresh state ``S`` of ``zeros`` plus ``fill``."""

    def __init__(self, fill: float = 0.0) -> None:
        super().__init__()
        self.fill = fill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = x / (x.norm(dim=-1, keepdim=True) + 1.0)
        state = torch.zeros(x.shape[0], x.shape[1], x.shape[1]) + self.fill
        return torch.einsum("bij,bj->bi", (state, k)) + 1


def test_einsum_with_zero_state_validates() -> None:
    assert _validates(_ZeroStateEinsum(), torch.randn(2, 3))


def test_einsum_nested_operand_edge_is_exempted() -> None:
    trace = _capture(_ZeroStateEinsum(), torch.randn(2, 3))
    _assert_exempted(_edge(trace, _op(trace, "einsum"), (1, 1)), "einsum_zero_operand")


def test_dead_einsum_edge_with_near_zero_state_still_fails() -> None:
    trace = _capture(_ZeroStateEinsum(fill=TINY), torch.randn(2, 3))
    _assert_dead_edge_fails(trace, _op(trace, "einsum"), (1, 1))


# ------------------------------------------------------------------ proofs.5: scatter_reduce_


class _ScatterAmax(nn.Module):
    """GAT-style ``new_zeros(...).scatter_reduce_(0, index, src, 'amax', include_self=False)``."""

    def __init__(self, index: list[int], include_self: bool = False) -> None:
        super().__init__()
        self.register_buffer("index", torch.tensor(index).view(-1, 1))
        self.include_self = include_self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = torch.nn.functional.leaky_relu(x * 2).detach()
        index = self.index.expand_as(src)
        out = src.new_zeros(4, 1)
        out.scatter_reduce_(0, index, src, reduce="amax", include_self=self.include_self)
        return out + 1


ALL_ROWS = [0, 1, 2, 3, 0, 1]


def test_scatter_reduce_full_coverage_validates() -> None:
    assert _validates(_ScatterAmax(ALL_ROWS), torch.randn(6, 1))


def test_scatter_reduce_destination_edge_is_exempted() -> None:
    trace = _capture(_ScatterAmax(ALL_ROWS), torch.randn(6, 1))
    result = _edge(trace, _op(trace, "scatter_reduce_"), 0)
    _assert_exempted(result, "scatter_reduce_destination_overwritten")


@pytest.mark.parametrize(
    ("index", "include_self", "slot"),
    [
        pytest.param([0, 1, 2, 0, 1, 2], False, 0, id="row-3-unaddressed"),
        pytest.param(ALL_ROWS, True, 0, id="include-self"),
        pytest.param(ALL_ROWS, False, 3, id="src-edge"),
    ],
)
def test_dead_scatter_reduce_edge_still_fails(
    index: list[int], include_self: bool, slot: int
) -> None:
    trace = _capture(_ScatterAmax(index, include_self=include_self), torch.randn(6, 1))
    _assert_dead_edge_fails(trace, _op(trace, "scatter_reduce_"), slot)


# ------------------------------------------------------------------ proofs.7: empty output


class _KeptTimesTotal(nn.Module):
    """The elements above ``threshold`` (none, for a high threshold) times the input's sum."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kept = x[x > self.threshold]
        return kept * x.sum()


SCORES = torch.tensor([1.0, 2.0, 300.0])


def test_empty_output_scalar_edge_is_exempted() -> None:
    trace = _capture(_KeptTimesTotal(threshold=1000.0), SCORES)
    op = _op(trace, "__mul__")
    assert op.out.numel() == 0
    _assert_exempted(_edge(trace, op, 1), "empty_output_no_elements")


def test_dead_scalar_edge_with_one_output_element_still_fails() -> None:
    trace = _capture(_KeptTimesTotal(threshold=100.0), SCORES)
    op = _op(trace, "__mul__")
    assert op.out.numel() == 1
    _assert_dead_edge_fails(trace, op, 1)


# ------------------------------------------------------------------ proofs.8: zero input


class _ZeroImageGrid(nn.Module):
    """``grid_sample`` of an image built from ``zeros_like`` plus ``fill``."""

    def __init__(self, fill: float = 0.0) -> None:
        super().__init__()
        self.fill = fill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = torch.zeros_like(x) + self.fill
        grid = torch.tanh(x[:, :2].permute(0, 2, 3, 1))
        return F.grid_sample(image, grid, align_corners=False)


class _ZeroInputConv(nn.Module):
    """``conv1d`` of an input built from ``zeros_like`` plus ``fill``, with a traced weight."""

    def __init__(self, fill: float = 0.0) -> None:
        super().__init__()
        self.fill = fill
        self.weight = nn.Parameter(torch.randn(3, 2, 3))
        self.bias = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = torch.zeros_like(x) + self.fill
        weight = self.weight * 1.0
        return F.conv1d(signal, weight, self.bias)


GRID_SAMPLER = ("grid_sampler", "grid_sample")


def test_grid_sampler_grid_edge_on_a_zero_image_is_exempted() -> None:
    trace = _capture(_ZeroImageGrid(), torch.randn(1, 2, 4, 4))
    result = _edge(trace, _op(trace, *GRID_SAMPLER), 1)
    _assert_exempted(result, "grid_sampler_zero_input")


def test_dead_grid_edge_on_a_near_zero_image_still_fails() -> None:
    trace = _capture(_ZeroImageGrid(fill=TINY), torch.randn(1, 2, 4, 4))
    _assert_dead_edge_fails(trace, _op(trace, *GRID_SAMPLER), 1)


def test_conv_weight_edge_on_a_zero_input_is_exempted() -> None:
    torch.manual_seed(0)
    trace = _capture(_ZeroInputConv(), torch.randn(1, 2, 8))
    result = _edge(trace, _op(trace, "conv1d"), 1)
    _assert_exempted(result, "conv_zero_input_output_is_bias")


def test_dead_conv_weight_edge_on_a_near_zero_input_still_fails() -> None:
    """The near-zero input moves the output by less than the bias resolution: still refused."""

    torch.manual_seed(0)
    trace = _capture(_ZeroInputConv(fill=TINY), torch.randn(1, 2, 8))
    op = _op(trace, "conv1d")
    assert torch.equal(op.out, op.saved_args[2].view(1, -1, 1).expand_as(op.out))
    _assert_dead_edge_fails(trace, op, 1)


# ------------------------------------------------------------------ proofs.9: absorbing operand


class _OrWithMask(nn.Module):
    """``(x > 0) | mask`` for a fixed bool ``mask``."""

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0) | self.mask


class _PowOfBase(nn.Module):
    """``torch.pow(base, x)`` for a Python float ``base``."""

    def __init__(self, base: float) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.base, x * 2.0)


def test_or_edge_against_an_all_true_operand_is_exempted() -> None:
    trace = _capture(_OrWithMask(torch.ones(6, dtype=torch.bool)), torch.randn(6))
    _assert_exempted(_edge(trace, _op(trace, "__or__"), 0), "or_absorbing_true_operand")


def test_dead_or_edge_against_an_operand_with_one_false_still_fails() -> None:
    """The input is True where the mask is False, so the saved output is still all True."""

    mask = torch.ones(6, dtype=torch.bool)
    mask[2] = False
    x = torch.randn(6).abs() + 0.1
    trace = _capture(_OrWithMask(mask), x)
    op = _op(trace, "__or__")
    assert bool(op.out.all())
    _assert_dead_edge_fails(trace, op, 0)


def test_pow_exponent_edge_with_unit_base_is_exempted() -> None:
    trace = _capture(_PowOfBase(1.0), torch.randn(5))
    _assert_exempted(_edge(trace, _op(trace, "pow"), 1), "pow_unit_base")


def test_dead_pow_exponent_edge_with_a_base_just_above_one_still_fails() -> None:
    trace = _capture(_PowOfBase(1.0 + 2.0**-20), torch.randn(5))
    _assert_dead_edge_fails(trace, _op(trace, "pow"), 1)
