"""Degenerate-domain exemptions: arguments whose saved call admits one valid value.

An argument whose valid domain, given the saved call, holds one value cannot be
perturbed into a different valid call, so the perturbation check could only
raise or leave the output unchanged. Each case is exempt only for its named op,
argument slot and saved condition; the forward replay still runs. Every case
has a planted-bug twin of the same shape outside the condition (a live domain
with a frozen replay, or a predicate refusal), which still fails.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from _validation_capture import _capture

from torchlens.validation._core import _check_whether_func_on_saved_parents_yields_saved_tensor
from torchlens.validation._degenerate_domains import (
    _check_index_domain_degenerate,
    check_argsort_singleton_dim_exempt,
    check_conv_transpose_unit_output_padding_exempt,
    check_ctc_single_target_length_exempt,
    check_masked_scatter_empty_source_exempt,
)
from torchlens.validation._saved_value_proofs import setitem_empty_mask_value_decision


def _op(trace: Any, func_name: str) -> Any:
    """Return the first op in ``trace`` whose ``func_name`` matches."""

    return next(op for op in trace._layer_list if op.func_name == func_name)


def _perturb(trace: Any, op: Any, label: str) -> Any:
    """Run the perturbation check for one parent of ``op``."""

    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[label]
    )


def _freeze(op: Any) -> None:
    """Freeze ``op``'s replay to return its saved output (a dead recorded edge)."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


def _assert_frozen_live_edge_fails(trace: Any, op: Any, label: str) -> None:
    """A live edge is verified, then fails ``perturbation_insensitive`` once frozen."""

    verified = _perturb(trace, op, label)
    assert verified.decision == "validated", (verified.decision, verified.reason)
    _freeze(op)
    frozen = _perturb(trace, op, label)
    assert frozen.decision == "failed", (frozen.decision, frozen.reason)
    assert frozen.reason == "perturbation_insensitive"


def _arg_parent(op: Any, position: int) -> str:
    """Return the parent label recorded at positional slot ``position``."""

    return op.call.parent_arg_positions["args"][position]


# ---------------------------------------------------------------------------
# index ops joining the existing degenerate-index-domain row
# ---------------------------------------------------------------------------


class _PoolIntoRows(nn.Module):
    """Graph pooling of node features into per-graph rows (CHGNet m291 shape)."""

    def __init__(self, graphs: int) -> None:
        super().__init__()
        self.graphs = graphs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool every node into its graph row; one graph means one valid index."""

        batch = (x[:, 0] > 0).long() if self.graphs > 1 else (x[:, 0] * 0).long()
        pooled = x.new_zeros(self.graphs, x.shape[1]).index_add_(0, batch, x)
        return pooled * 2.0


_NODES = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [2.0, -1.0], [-2.0, 0.5]])


def test_index_add_into_one_row_is_exempt() -> None:
    """A batch of one graph: index 0 is the only valid index, so the edge is exempt."""

    trace, _ = _capture(_PoolIntoRows(graphs=1), _NODES)
    op = _op(trace, "index_add_")
    assert _perturb(trace, op, _arg_parent(op, 2)).decision == "exempted"


def test_index_add_into_two_rows_still_fails_when_frozen() -> None:
    """Two graph rows admit an alternate index, so a dead index edge still fails."""

    trace, _ = _capture(_PoolIntoRows(graphs=2), _NODES)
    op = _op(trace, "index_add_")
    _assert_frozen_live_edge_fails(trace, op, _arg_parent(op, 2))


def _index_layer(func_name: str, args: tuple[Any, ...], slot: int) -> SimpleNamespace:
    """Return a minimal op record with the index parent at ``slot``."""

    return SimpleNamespace(
        func_name=func_name,
        saved_args=args,
        saved_kwargs={},
        parent_arg_positions={"args": {slot: "index"}, "kwargs": {}},
    )


def test_new_index_rows_are_exact_at_the_domain_boundary() -> None:
    """Each newly listed op is exempt at a one-value domain and strict at two."""

    cases = (
        ("index_add", lambda n: (torch.zeros(n, 2), 0, torch.tensor([0]), torch.ones(1, 2)), 2),
        ("index_copy_", lambda n: (torch.zeros(n, 2), 0, torch.tensor([0]), torch.ones(1, 2)), 2),
        ("index_fill", lambda n: (torch.zeros(n, 2), 0, torch.tensor([0]), 1.0), 2),
        ("take_along_dim", lambda n: (torch.zeros(2, n), torch.tensor([[0], [0]]), 1), 1),
        ("nll_loss", lambda n: (torch.zeros(3, n), torch.tensor([0, 0, 0])), 1),
    )
    for func_name, build, slot in cases:
        one = _index_layer(func_name, build(1), slot)
        two = _index_layer(func_name, build(2), slot)
        assert _check_index_domain_degenerate(None, one, ["index"]) is True, func_name
        assert _check_index_domain_degenerate(None, two, ["index"]) is False, func_name
        assert _check_index_domain_degenerate(None, one, ["other"]) is False, func_name


def test_negative_index_fill_entries_count_as_in_domain() -> None:
    """index_fill accepts ``-1`` for the last row, so an all-negative index stays strict."""

    layer = _index_layer("index_fill", (torch.zeros(3, 2), 0, torch.tensor([-1, -3]), 1.0), 2)
    assert _check_index_domain_degenerate(None, layer, ["index"]) is False
    ignored = _index_layer("nll_loss", (torch.zeros(3, 4), torch.tensor([-100, -100, -100])), 1)
    assert _check_index_domain_degenerate(None, ignored, ["index"]) is True


# ---------------------------------------------------------------------------
# masked_scatter mask over an empty source
# ---------------------------------------------------------------------------


class _ScatterIntoMask(nn.Module):
    """Scatter values into masked positions (megablocks expert-routing shape)."""

    def __init__(self, empty_source: bool) -> None:
        super().__init__()
        self.empty_source = empty_source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """An expert with no tokens has an all-False mask and an empty source."""

        if self.empty_source:
            mask, source = x < -100.0, x.new_zeros(0)
        else:
            mask, source = x > 0, x.flatten() * 3.0
        return x.masked_scatter(mask, source) * 2.0


def test_masked_scatter_mask_over_an_empty_source_is_exempt() -> None:
    """Only the all-False mask is valid over an empty source, so the mask edge is exempt."""

    trace, _ = _capture(_ScatterIntoMask(empty_source=True), _NODES)
    op = _op(trace, "masked_scatter")
    assert _perturb(trace, op, _arg_parent(op, 1)).decision == "exempted"


def test_masked_scatter_mask_over_a_live_source_still_fails_when_frozen() -> None:
    """A non-empty source admits other masks, so a dead mask edge still fails."""

    trace, _ = _capture(_ScatterIntoMask(empty_source=False), _NODES)
    op = _op(trace, "masked_scatter")
    mask_label = _arg_parent(op, 1)
    assert check_masked_scatter_empty_source_exempt(trace, op, [mask_label]) is False
    _assert_frozen_live_edge_fails(trace, op, mask_label)


# ---------------------------------------------------------------------------
# ctc_loss target lengths of a batch of one
# ---------------------------------------------------------------------------


class _CtcHead(nn.Module):
    """A CTC head over concatenated targets with lengths derived in the forward."""

    def __init__(self, lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 5)
        self.lengths = lengths
        self.register_buffer("targets", torch.tensor([1, 2, 3, 4, 1, 2])[: sum(lengths)])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Return the summed CTC loss; ``target_lengths`` is a recorded parent."""

        log_probs = self.proj(feats).log_softmax(-1).transpose(0, 1)
        target_lengths = (feats[:, 0, 0] * 0).long() + torch.tensor(self.lengths)
        input_lengths = [feats.shape[1]] * feats.shape[0]
        return F.ctc_loss(log_probs, self.targets, input_lengths, target_lengths, reduction="sum")


def test_ctc_target_length_of_a_batch_of_one_is_exempt() -> None:
    """With one sample the target length must equal the target count, so it is exempt."""

    trace, _ = _capture(_CtcHead((3,)), torch.randn(1, 6, 4))
    op = _op(trace, "ctc_loss")
    assert _perturb(trace, op, _arg_parent(op, 3)).decision == "exempted"


def test_ctc_target_lengths_of_a_batch_of_two_still_fail_when_frozen() -> None:
    """Two samples admit a sum-preserving alternate, so the row refuses them."""

    trace, _ = _capture(_CtcHead((3, 2)), torch.randn(2, 6, 4))
    op = _op(trace, "ctc_loss")
    label = _arg_parent(op, 3)
    assert check_ctc_single_target_length_exempt(trace, op, [label]) is False
    _assert_frozen_live_edge_fails(trace, op, label)
    equal, _ = _capture(_CtcHead((3, 3)), torch.randn(2, 6, 4))
    equal_op = _op(equal, "ctc_loss")
    equal_label = _arg_parent(equal_op, 3)
    assert check_ctc_single_target_length_exempt(equal, equal_op, [equal_label]) is False


def test_ctc_padded_targets_of_a_batch_of_one_are_not_exempt() -> None:
    """Padded 2-D targets allow any length up to the pad width, so they stay strict."""

    layer = SimpleNamespace(
        func_name="ctc_loss",
        saved_args=(torch.zeros(6, 1, 5), torch.tensor([[1, 2, 3]]), [6], torch.tensor([3])),
        saved_kwargs={},
        parent_arg_positions={"args": {3: "lengths"}, "kwargs": {}},
    )
    assert check_ctc_single_target_length_exempt(None, layer, ["lengths"]) is False


# ---------------------------------------------------------------------------
# conv_transpose output_padding with unit stride and dilation
# ---------------------------------------------------------------------------


def _conv_transpose_layer(stride: Any, dilation: Any, output_padding: Any, slot: Any) -> Any:
    """Return a minimal ``conv_transpose1d`` op record with a parent at ``slot``."""

    args = (torch.zeros(1, 2, 5), torch.zeros(2, 2, 3), None, stride, 0, output_padding, 1)
    return SimpleNamespace(
        func_name="conv_transpose1d",
        saved_args=(*args, dilation),
        saved_kwargs={},
        parent_arg_positions={"args": {slot: "padding_parent"}, "kwargs": {}},
    )


def test_conv_transpose_output_padding_under_unit_stride_is_exempt() -> None:
    """Stride and dilation 1 leave 0 as the only valid output_padding."""

    for padding, slot in ((torch.tensor(0), 5), ((torch.tensor(0),), (5, 0)), (0, 5)):
        layer = _conv_transpose_layer(1, (1,), padding, slot)
        assert check_conv_transpose_unit_output_padding_exempt(None, layer, ["padding_parent"])


def test_conv_transpose_output_padding_under_stride_two_is_not_exempt() -> None:
    """A stride or dilation above 1 admits a second padding; other slots stay strict."""

    refused = (
        _conv_transpose_layer(2, 1, torch.tensor(1), 5),
        _conv_transpose_layer((1,), (2,), torch.tensor(0), 5),
        _conv_transpose_layer(1, 1, torch.tensor(0), 0),
        _conv_transpose_layer(1, 1, 1.0, 5),
    )
    for layer in refused:
        assert not check_conv_transpose_unit_output_padding_exempt(None, layer, ["padding_parent"])


# ---------------------------------------------------------------------------
# argsort over a single element
# ---------------------------------------------------------------------------


class _RankFeatures(nn.Module):
    """Rank features along the last dim and feed the ranks onward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ranks added to the input so the argsort output is used."""

        return torch.argsort(x * 2.0, dim=-1).float() + x


def test_argsort_over_one_element_is_exempt() -> None:
    """Sorting one element always returns index 0, so the input edge is exempt."""

    trace, _ = _capture(_RankFeatures(), _NODES[:, :1])
    op = _op(trace, "argsort")
    assert _perturb(trace, op, _arg_parent(op, 0)).decision == "exempted"


def test_argsort_over_two_elements_still_fails_when_frozen() -> None:
    """Two elements can be reordered, so a dead argsort input edge still fails."""

    trace, _ = _capture(_RankFeatures(), _NODES)
    op = _op(trace, "argsort")
    label = _arg_parent(op, 0)
    assert check_argsort_singleton_dim_exempt(trace, op, [label]) is False
    _assert_frozen_live_edge_fails(trace, op, label)


# ---------------------------------------------------------------------------
# setitem value under a bool mask that selects nothing
# ---------------------------------------------------------------------------


class _MaskedWrite(nn.Module):
    """Write a derived row into masked rows of a copy of the input."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rows with a first feature above ``threshold`` take the written value."""

        out = x.clone()
        out[x[:, 0] > self.threshold] = x[0] * 2.0
        return out * 1.5


def test_setitem_value_under_an_all_false_mask_is_exempt() -> None:
    """An all-False mask writes nothing, so the value edge is proved irrelevant."""

    trace, _ = _capture(_MaskedWrite(threshold=100.0), _NODES)
    op = _op(trace, "__setitem__")
    result = _perturb(trace, op, _arg_parent(op, 2))
    assert result.decision == "exempted", (result.decision, result.reason)
    assert result.reason == "setitem_empty_mask_value"


def test_setitem_value_under_a_live_mask_still_fails_when_frozen() -> None:
    """A mask with True entries writes the value, so a dead value edge still fails."""

    trace, _ = _capture(_MaskedWrite(threshold=0.0), _NODES)
    op = _op(trace, "__setitem__")
    label = _arg_parent(op, 2)
    args = tuple(op.call.saved_args)
    assert setitem_empty_mask_value_decision(op, {2}, args) is None
    _assert_frozen_live_edge_fails(trace, op, label)


def test_setitem_empty_mask_proof_refuses_other_slots_and_forged_outputs() -> None:
    """Only the value slot qualifies, and the saved output must equal the destination."""

    destination = torch.zeros(3, 2)
    mask = torch.zeros(3, dtype=torch.bool)
    layer = SimpleNamespace(out=destination.clone())
    args = (destination, mask, torch.ones(2))
    assert setitem_empty_mask_value_decision(layer, {2}, args) is not None
    assert setitem_empty_mask_value_decision(layer, {0}, args) is None
    assert setitem_empty_mask_value_decision(layer, {1}, args) is None
    assert setitem_empty_mask_value_decision(layer, {2}, (destination, ~mask, args[2])) is None
    forged = SimpleNamespace(out=destination + 1.0)
    assert setitem_empty_mask_value_decision(forged, {2}, args) is None
