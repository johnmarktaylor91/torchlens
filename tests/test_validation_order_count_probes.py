"""Non-uniform retry probes for integer and bool parents.

Every uniform retry step preserves what an order-reading or counting child
reads: adding one to every token id keeps the ``argmax`` position, and
``logical_not`` keeps the true count of a balanced mask. A real edge then
passed only when the random draw happened to move the child, and an unlucky
draw false-failed it. The ``lift_min_entry`` rung now lifts the smallest
integer entry above the largest and flips one bool entry, and the unique
family gets a ``merge_adjacent_entry`` rung: the counts of distinct entries
(``unique_consecutive([0, 1, 2])`` counts ``[1, 1, 1]``) survive every shift
and the lift, and move only when two entries merge. Children that read only a
parent's shape (``_shape_as_tensor``, the ``embedding_bag`` outputs that count
entries per bag) get a ``trim_last_entry`` rung: a parent one entry shorter
keeps the call valid and moves the output. A child whose replay ignores its
parent still fails: no probe can move a frozen output.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from _validation_capture import _capture

from torchlens.validation import validate_forward_pass
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
    _execute_func_with_restored_state,
    _perturbation_retry_strategies,
    _prepare_input_args_for_validating_layer,
)

_UNIFORM_STEPS = ("step_up", "step_down", "unit_step_up", "unit_step_down")


class _EndTokenPosition(nn.Module):
    """``ids.argmax(-1)``: the end-of-sequence id is the largest token id."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int32).argmax(dim=-1)


class _MaskCount(nn.Module):
    """``(x > 0).sum()`` over a balanced two-entry mask."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).sum()


class _RunCounts(nn.Module):
    """``unique_consecutive(ids, return_counts=True)`` over distinct sorted ids."""

    def forward(self, x: torch.Tensor):
        return torch.unique_consecutive(x.long(), return_counts=True)


class _ClassCounts(nn.Module):
    """``unique(ids, return_inverse=True, return_counts=True)`` over distinct ids."""

    def forward(self, x: torch.Tensor):
        return torch.unique(x.long(), return_inverse=True, return_counts=True)


class _ShapeOfIds(nn.Module):
    """``torch._shape_as_tensor(ids)``, as ``torch.onnx.operators.shape_as_tensor`` calls it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch._shape_as_tensor(x.long())


class _Bags(nn.Module):
    """``EmbeddingBag(mode=mode)`` over two bags of token ids."""

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        torch.manual_seed(0)
        self.bag = nn.EmbeddingBag(10, 3, mode=mode)
        self.register_buffer("offsets", torch.tensor([0, 2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bag(x.long(), self.offsets)


def _bag_ids() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 4.0, 5.0, 7.0])


def _token_ids() -> torch.Tensor:
    return torch.tensor([[49406.0, 74.0, 117.0, 33.0, 49407.0]])


def _child_and_parent(trace, func_name: str, output_index: int | None = None):
    child = [
        op
        for op in trace._layer_list
        if op.func_name == func_name
        and (output_index is None or op.multi_output_index == output_index)
    ][0]
    return child, child.parents[0]


def _changed_by(trace, child, parent_label: str, strategy: str) -> bool:
    args, _ = _prepare_input_args_for_validating_layer(
        trace, child, [parent_label], perturb_strategy=strategy
    )
    assert args is not None
    out = _execute_func_with_restored_state(child, args, [parent_label], child.label, False)
    # A replay that raises (an index stepped out of range) is no evidence either way.
    return out is not None and not torch.equal(out, child.out)


def test_uniform_steps_keep_an_integer_argmax_and_the_lift_moves_it() -> None:
    """The lift is the only retry that moves an argmax over integer ids."""

    trace, _ = _capture(_EndTokenPosition(), _token_ids())
    argmax_op, parent_label = _child_and_parent(trace, "argmax")
    assert "lift_min_entry" in _perturbation_retry_strategies(argmax_op, [parent_label])
    for strategy in _UNIFORM_STEPS:
        assert not _changed_by(trace, argmax_op, parent_label, strategy), strategy
    assert _changed_by(trace, argmax_op, parent_label, "lift_min_entry")


def test_logical_not_keeps_a_balanced_count_and_the_flip_moves_it() -> None:
    """The single-entry flip is the only bool retry that changes a true count."""

    trace, _ = _capture(_MaskCount(), torch.tensor([-1.0, 2.0]))
    sum_op, parent_label = _child_and_parent(trace, "sum")
    for strategy in _UNIFORM_STEPS:
        assert not _changed_by(trace, sum_op, parent_label, strategy), strategy
    assert _changed_by(trace, sum_op, parent_label, "lift_min_entry")


def test_only_the_merge_moves_unique_counts_of_distinct_entries() -> None:
    """Counts of distinct entries survive every shift and the lift; a merge moves them."""

    cases = (
        (_RunCounts(), torch.tensor([0.0, 1.0, 2.0]), "unique_consecutive", 2),
        (_ClassCounts(), torch.tensor([3.0, 1.0, 2.0]), "_unique2", 2),
    )
    for model, x, func_name, counts_index in cases:
        trace, _ = _capture(model, x)
        counts_op, parent_label = _child_and_parent(trace, func_name, counts_index)
        assert counts_op.out.tolist() == [1, 1, 1]
        strategies = _perturbation_retry_strategies(counts_op, [parent_label])
        assert strategies.index("merge_adjacent_entry") > strategies.index("lift_min_entry")
        for strategy in (*_UNIFORM_STEPS, "lift_min_entry"):
            assert not _changed_by(trace, counts_op, parent_label, strategy), strategy
        assert _changed_by(trace, counts_op, parent_label, "merge_adjacent_entry")


def test_merge_rung_is_only_for_the_unique_family() -> None:
    """Other children keep the plain value-step rungs."""

    trace, _ = _capture(_EndTokenPosition(), _token_ids())
    argmax_op, parent_label = _child_and_parent(trace, "argmax")
    assert "merge_adjacent_entry" not in _perturbation_retry_strategies(argmax_op, [parent_label])


def test_merge_adjacent_entry_values() -> None:
    """The merge copies the first differing entry's predecessor; a constant splits."""

    from torchlens.validation.core import _directional_step_perturb

    merged = _directional_step_perturb(torch.tensor([[4, 4], [7, 9]]), "merge_adjacent_entry")
    assert merged.tolist() == [[4, 4], [4, 9]]
    constant = torch.tensor([5, 5, 5])
    assert _directional_step_perturb(constant, "merge_adjacent_entry").tolist() == [5, 5, 6]
    assert _directional_step_perturb(
        torch.tensor([True, False]), "merge_adjacent_entry"
    ).tolist() == [True, True]
    single = torch.tensor([2.0])
    assert torch.equal(
        _directional_step_perturb(single, "merge_adjacent_entry"),
        _directional_step_perturb(single, "step_up"),
    )


def test_only_the_trim_moves_a_shape_only_output() -> None:
    """A shape-only output survives every value rung; the one-entry trim moves it."""

    trace, _ = _capture(_ShapeOfIds(), _token_ids())
    shape_op, parent_label = _child_and_parent(trace, "_shape_as_tensor")
    cases = [(trace, shape_op, parent_label)]
    bag_trace, _ = _capture(_Bags(), _bag_ids())
    for output_index in (1, 2):
        aux_op = [
            op
            for op in bag_trace._layer_list
            if op.func_name == "embedding_bag" and op.multi_output_index == output_index
        ][0]
        indices_label = aux_op.parent_arg_positions["args"][1]
        cases.append((bag_trace, aux_op, indices_label))
    for case_trace, child, label in cases:
        strategies = _perturbation_retry_strategies(child, [label])
        assert "trim_last_entry" in strategies
        for strategy in (*_UNIFORM_STEPS, "lift_min_entry"):
            assert not _changed_by(case_trace, child, label, strategy), strategy
        assert _changed_by(case_trace, child, label, "trim_last_entry")


def test_sum_mode_placeholder_outputs_still_fail() -> None:
    """Sum-mode ``embedding_bag`` leaves bag sizes as zeros the indices never reach.

    Neither the index values nor their number reach that placeholder, so no
    probe can move it and the edge still fails ``perturbation_insensitive``:
    the shape probe does not bless it.
    """

    trace, _ = _capture(_Bags("sum"), _bag_ids())
    bag_size = [
        op
        for op in trace._layer_list
        if op.func_name == "embedding_bag" and op.multi_output_index == 2
    ][0]
    assert bag_size.out.tolist() == [0, 0]
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace,
        bag_size.label,
        perturb=True,
        layers_to_perturb=[bag_size.parent_arg_positions["args"][1]],
    )
    assert result.decision == "failed"
    assert result.reason == "perturbation_insensitive"


def test_trim_rung_is_only_for_shape_only_outputs() -> None:
    """The pooled embeddings and ordinary children never get the shape probe."""

    bag_trace, _ = _capture(_Bags(), _bag_ids())
    pooled = [
        op
        for op in bag_trace._layer_list
        if op.func_name == "embedding_bag" and op.multi_output_index in (0, None)
    ][0]
    label = pooled.parent_arg_positions["args"][1]
    assert "trim_last_entry" not in _perturbation_retry_strategies(pooled, [label])
    trace, _ = _capture(_EndTokenPosition(), _token_ids())
    argmax_op, parent_label = _child_and_parent(trace, "argmax")
    assert "trim_last_entry" not in _perturbation_retry_strategies(argmax_op, [parent_label])


def test_integer_argmax_bool_count_and_unique_counts_validate() -> None:
    """End to end, the real edges validate whatever the default draw does."""

    cases = (
        (_EndTokenPosition(), _token_ids()),
        (_MaskCount(), torch.tensor([-1.0, 2.0])),
        (_RunCounts(), torch.tensor([0.0, 1.0, 2.0])),
        (_ClassCounts(), torch.tensor([3.0, 1.0, 2.0])),
        (_ShapeOfIds(), _token_ids()),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(4):
            for model, x in cases:
                assert validate_forward_pass(model, x, seed=seed) is True, type(model).__name__


def test_frozen_probed_children_still_fail() -> None:
    """Armed proof: a child whose replay ignores its parent fails under every probe."""

    cases = (
        (_EndTokenPosition(), _token_ids(), "argmax", None),
        (_MaskCount(), torch.tensor([-1.0, 2.0]), "sum", None),
        (_RunCounts(), torch.tensor([0.0, 1.0, 2.0]), "unique_consecutive", 2),
        (_ShapeOfIds(), _token_ids(), "_shape_as_tensor", None),
        (_Bags(), _bag_ids(), "embedding_bag", 1),
    )
    for model, x, func_name, output_index in cases:
        trace, _ = _capture(model, x)
        child, parent_label = _child_and_parent(trace, func_name, output_index)
        saved = child.out.detach().clone()
        object.__setattr__(child, "func", lambda *args, saved=saved, **kwargs: saved.clone())
        result = _check_whether_func_on_saved_parents_yields_saved_tensor(
            trace, child.label, perturb=True, layers_to_perturb=[parent_label]
        )
        assert result.decision == "failed", (func_name, result.reason)
        assert result.reason == "perturbation_insensitive", func_name
