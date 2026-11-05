"""Declared unread arguments of known transformers mask builders (ruling d-aad71aaad456).

Transformers 4.57 builds its eager and sdpa masks as nested ``torch.vmap`` over
``and_masks(causal_mask_function, packed_sequence_mask_function(...))`` called with
``(batch_arange, head_arange, cache_position, kv_arange)``. No builder reads ``head_idx``, so
before the declared table the head edge failed ``perturbation_insensitive`` and GPT-2 eager
validation returned False. These tests build the same chain from the installed
``transformers.masking_utils`` (the builders are unchanged in 4.57 and 5.x) and pin both
sides: the declared position of a recognized builder is excused, and everything else stays
strict (a user mask function composed in, a table function edited to read ``head_idx``, a
planted capture bug, a wrong tensor at a non-excused position).
"""

from __future__ import annotations

import types
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
from _validation_capture import _capture, _quiet_validate
from transformers import masking_utils

_BHQKV_DIMS = (
    (None, None, None, 0),
    (None, None, 0, None),
    (None, 0, None, None),
    (0, None, None, None),
)


def _vmap_bhqkv(mask_function: Callable[..., Any]) -> Callable[..., Any]:
    """Nest ``torch.vmap`` over (kv, q, head, batch) as ``masking_utils._vmap_for_bhqkv`` does."""

    for dims in _BHQKV_DIMS:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def _user_reads_head(batch_idx: Any, head_idx: Any, q_idx: Any, kv_idx: Any) -> Any:
    """A user mask function that reads ``head_idx`` (for its dtype and device) and allows all."""

    return head_idx.new_ones((), dtype=torch.bool) & (kv_idx >= 0)


def _user_ignores_head(batch_idx: Any, head_idx: Any, q_idx: Any, kv_idx: Any) -> Any:
    """A user mask function outside the table that never reads ``head_idx``."""

    return kv_idx >= 0


class _VmapMaskScores(nn.Module):
    """Self-attention scores masked by a transformers mask built with nested ``torch.vmap``."""

    def __init__(self, *extra: Callable[..., Any]) -> None:
        """Compose ``extra`` user mask functions after causal and packed-sequence."""

        super().__init__()
        self.extra = extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask ``x @ x^T`` with a (1, 1, seq, seq) mask from a batch-1, one-head vmap call."""

        seq = x.shape[1]
        cache_position = torch.arange(seq)
        kv_arange = torch.arange(seq)
        kv_arange += 0
        packed = torch.zeros(1, seq, dtype=torch.long)
        mask_function = masking_utils.and_masks(
            masking_utils.causal_mask_function,
            masking_utils.packed_sequence_mask_function(packed),
            *self.extra,
        )
        batch_arange = torch.arange(1)
        head_arange = torch.arange(1)
        with masking_utils.TransformGetItemToIndex():
            mask = _vmap_bhqkv(mask_function)(batch_arange, head_arange, cache_position, kv_arange)
        scores = x @ x.transpose(-1, -2)
        return scores.masked_fill(~mask[:, 0], -1.0e4)


def _input() -> torch.Tensor:
    """Return a (1, 5, 4) input."""

    return torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(0))


def _vmap_op(trace: Any) -> Any:
    """Return the trace's single ``vmap`` boundary op."""

    ops = [op for op in trace._layer_list if op.func_name == "vmap"]
    assert len(ops) == 1, [op.label for op in ops]
    return ops[0]


def _parent_at(op: Any, position: int) -> str:
    """Return the parent label at positional argument ``position`` of ``op``."""

    return op.parent_arg_positions["args"][position]


def _edge(trace: Any, op: Any, parent: str) -> tuple[str, str]:
    """Return (decision, reason) of the perturbation check on one parent edge."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _check_whether_func_on_saved_parents_yields_saved_tensor(
            trace, op.label, perturb=True, layers_to_perturb=[parent]
        )
    return result.decision, result.reason


def _validate(trace: Any, ground_truth: torch.Tensor) -> bool:
    """Validate a captured trace against ground truth with warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(trace.validate_forward_pass(ground_truth))


# ---------------------------------------------------------------------------
# The excuse: the GPT-2 4.57 mask chain validates
# ---------------------------------------------------------------------------


def test_vmapped_transformers_mask_chain_validates() -> None:
    """Regression: the 4.57 eager mask chain (batch 1, one head) validates True.

    Red before the table: the head edge failed ``perturbation_insensitive``.
    """

    assert _quiet_validate(_VmapMaskScores(), _input())


def test_only_the_head_edge_is_excused() -> None:
    """The head edge is a pre-perturbation exemption; the other three keep their verdicts."""

    trace, _ = _capture(_VmapMaskScores(), _input())
    op = _vmap_op(trace)
    try:
        assert _edge(trace, op, _parent_at(op, 1)) == ("exempted", "pre_perturbation_exemption")
        # batch (position 0): the substitute index is out of range for the packed mask, so the
        # bool-predicate probe cannot run (the hf4-validate-index-perturb fix), as before.
        assert _edge(trace, op, _parent_at(op, 0)) == ("exempted", "discrete_bool_output")
        for position in (2, 3):
            assert _edge(trace, op, _parent_at(op, position))[0] == "validated", position
    finally:
        trace.cleanup()


def test_check_excuses_position_one_of_the_recorded_chain_only() -> None:
    """The custom check is True for the position-1 parent and False for every other one."""

    from torchlens.validation._mask_builder_dependence import check_mask_builder_unread_exempt

    trace, _ = _capture(_VmapMaskScores(), _input())
    op = _vmap_op(trace)
    try:
        assert check_mask_builder_unread_exempt(trace, op, [_parent_at(op, 1)])
        for position in (0, 2, 3):
            assert not check_mask_builder_unread_exempt(trace, op, [_parent_at(op, position)])
        assert not check_mask_builder_unread_exempt(
            trace, op, [_parent_at(op, 1), _parent_at(op, 2)]
        )
        assert not check_mask_builder_unread_exempt(trace, op, [])
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Negatives: everything outside the declared carve-out stays strict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_fn", [_user_reads_head, _user_ignores_head], ids=["reads", "ignores"]
)
def test_and_mask_with_a_user_function_keeps_the_head_edge_strict(user_fn: Any) -> None:
    """An ``and_mask`` composing a function outside the table gets no excuse: False."""

    from torchlens.validation._mask_builder_dependence import check_mask_builder_unread_exempt

    model = _VmapMaskScores(user_fn)
    assert not _quiet_validate(model, _input())
    trace, _ = _capture(model, _input())
    op = _vmap_op(trace)
    try:
        assert not check_mask_builder_unread_exempt(trace, op, [_parent_at(op, 1)])
        assert _edge(trace, op, _parent_at(op, 1)) == ("failed", "perturbation_insensitive")
    finally:
        trace.cleanup()


def test_planted_capture_bug_on_the_mask_model_still_fails() -> None:
    """A corrupted saved activation of the same model fails validation."""

    trace, ground_truth = _capture(_VmapMaskScores(), _input())
    try:
        scores = next(op for op in trace._layer_list if op.func_name in ("matmul", "__matmul__"))
        scores.out.add_(1.0)
        assert not _validate(trace, ground_truth)
    finally:
        trace.cleanup()


def test_corrupted_saved_mask_still_fails() -> None:
    """A flipped saved mask (the excused op's own output) fails validation."""

    trace, ground_truth = _capture(_VmapMaskScores(), _input())
    try:
        op = _vmap_op(trace)
        op.out.logical_not_()
        assert not _validate(trace, ground_truth)
    finally:
        trace.cleanup()


def test_wrong_tensor_wired_into_a_non_excused_position_fails() -> None:
    """Position 2 recorded as the head arange (a mis-wired edge) is not excused: False."""

    from torchlens.validation._mask_builder_dependence import check_mask_builder_unread_exempt

    trace, ground_truth = _capture(_VmapMaskScores(), _input())
    try:
        op = _vmap_op(trace)
        head, query = _parent_at(op, 1), _parent_at(op, 2)
        op.parent_arg_positions["args"][2] = head
        assert not check_mask_builder_unread_exempt(trace, op, [head])
        assert not check_mask_builder_unread_exempt(trace, op, [query])
        assert not _validate(trace, ground_truth)
    finally:
        trace.cleanup()


def test_frozen_callable_claiming_the_builder_still_fails_on_query() -> None:
    """A recorded callable that maps the builder but ignores q fails the q edge (dead edge)."""

    from torchlens.validation._mask_builder_dependence import vmapped_callable

    trace, _ = _capture(_VmapMaskScores(), _input())
    try:
        op = _vmap_op(trace)
        saved = op.out.detach().clone()

        def frozen(*args: Any, **kwargs: Any) -> torch.Tensor:
            """Return the saved mask whatever the arguments (a planted dead edge)."""

            return saved.clone()

        frozen.__tl_transform_kind__ = "vmap"  # type: ignore[attr-defined]
        frozen.__tl_transform_inner_fn__ = vmapped_callable(op.func)  # type: ignore[attr-defined]
        object.__setattr__(op, "func", frozen)
        assert _edge(trace, op, _parent_at(op, 1)) == ("exempted", "pre_perturbation_exemption")
        assert _edge(trace, op, _parent_at(op, 2)) == ("failed", "perturbation_insensitive")
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# The table: identity, composition and the leaf guard
# ---------------------------------------------------------------------------

_FAKE_MASKING_UTILS = """
def and_masks(*mask_functions):
    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result
    return and_mask

def causal_mask_function(batch_idx, head_idx, q_idx, kv_idx):
    return kv_idx <= q_idx + HEAD_READ
"""


def _fake_masking_utils(head_read: str) -> types.ModuleType:
    """Return a module named ``transformers.masking_utils`` (not registered) from fake source."""

    module = types.ModuleType("transformers.masking_utils")
    module.__dict__["torch"] = torch
    exec(_FAKE_MASKING_UTILS.replace("HEAD_READ", head_read), module.__dict__)
    return module


def test_table_builders_of_the_installed_transformers_declare_head_unread() -> None:
    """Each table builder of the installed transformers declares position 1 unread."""

    from torchlens.validation._mask_builder_dependence import builder_unread_positions

    packed = torch.zeros(1, 3, dtype=torch.long)
    leaves = (
        masking_utils.causal_mask_function,
        masking_utils.packed_sequence_mask_function(packed),
        masking_utils.padding_mask_function(packed.bool()),
        masking_utils.sliding_window_overlay(2),
    )
    for leaf in leaves:
        assert builder_unread_positions(leaf) == frozenset({1}), leaf
    assert builder_unread_positions(masking_utils.and_masks(*leaves)) == frozenset({1})
    nested = masking_utils.and_masks(masking_utils.and_masks(*leaves), leaves[0])
    assert builder_unread_positions(nested) == frozenset({1})


def test_builders_outside_the_table_declare_nothing() -> None:
    """or_masks, the chunked overlay, offsets and user functions have no row (fail closed)."""

    from torchlens.validation._mask_builder_dependence import builder_unread_positions

    causal = masking_utils.causal_mask_function
    outside: list[Any] = [
        masking_utils.or_masks(causal),
        masking_utils.and_masks(causal, _user_ignores_head),
        masking_utils.and_masks(causal, masking_utils.and_masks(_user_reads_head)),
        _user_ignores_head,
        types.FunctionType(causal.__code__, {"__name__": "user_module"}),
        None,
    ]
    optional: dict[str, Callable[[Any], Any]] = {
        "chunked_overlay": lambda factory: factory(2, torch.zeros(1, dtype=torch.long)),
        "_legacy_chunked_overlay": lambda factory: factory(2),
        "add_offsets_to_mask_function": lambda factory: factory(causal, 0, 0),
    }
    for name, build in optional.items():
        if hasattr(masking_utils, name):
            outside.append(build(getattr(masking_utils, name)))
    for fn in outside:
        assert builder_unread_positions(fn) == frozenset(), fn


def test_a_table_function_edited_to_read_head_loses_the_excuse() -> None:
    """Identity matches, but a leaf whose code reads ``head_idx`` declares nothing."""

    from torchlens.validation._mask_builder_dependence import builder_unread_positions

    faithful = _fake_masking_utils("0")
    edited = _fake_masking_utils("head_idx")
    assert builder_unread_positions(faithful.causal_mask_function) == frozenset({1})
    assert builder_unread_positions(faithful.and_masks(faithful.causal_mask_function)) == {1}
    assert builder_unread_positions(edited.causal_mask_function) == frozenset()
    assert builder_unread_positions(edited.and_masks(edited.causal_mask_function)) == frozenset()


def test_vmapped_callable_walks_only_instrumented_vmap_layers() -> None:
    """Anything but a chain of TorchLens vmap layers ending at a plain callable gives None."""

    from torchlens.validation._mask_builder_dependence import vmapped_callable

    def inner(*args: Any) -> None:
        """A plain mapped callable."""

    def layer(kind: str, wrapped: Any) -> Any:
        """Return a stub transform layer of ``kind`` wrapping ``wrapped``."""

        def stub(*args: Any) -> None:
            """A stub boundary callable."""

        stub.__tl_is_transform_boundary__ = True  # type: ignore[attr-defined]
        stub.__tl_transform_kind__ = kind  # type: ignore[attr-defined]
        stub.__tl_transform_inner_fn__ = wrapped  # type: ignore[attr-defined]
        return stub

    assert vmapped_callable(layer("vmap", layer("vmap", inner))) is inner
    assert vmapped_callable(layer("vmap", layer("grad", inner))) is None
    assert vmapped_callable(layer("grad", inner)) is None
    assert vmapped_callable(layer("vmap", None)) is None
    assert vmapped_callable(inner) is None
    assert vmapped_callable(None) is None
    deep: Any = inner
    for _ in range(9):
        deep = layer("vmap", deep)
    assert vmapped_callable(deep) is None


def test_check_refuses_keywords_arity_and_missing_callables() -> None:
    """A keyword call, another arity or a callable-less op (a loaded bundle) stays strict."""

    from torchlens.validation._mask_builder_dependence import check_mask_builder_unread_exempt

    trace, _ = _capture(_VmapMaskScores(), _input())
    op = _vmap_op(trace)
    try:
        head = _parent_at(op, 1)
        base = {
            "func": op.func,
            "num_pos_args": 4,
            "num_kwargs": 0,
            "parent_arg_positions": {"args": {1: head}, "kwargs": {}},
        }
        assert check_mask_builder_unread_exempt(trace, types.SimpleNamespace(**base), [head])
        for change in (
            {"num_kwargs": 1},
            {"num_kwargs": None},
            {"num_pos_args": 3},
            {"func": None},
            {"parent_arg_positions": {"args": {1: head}, "kwargs": {"h": head}}},
            {"parent_arg_positions": {"args": {(1, 0): head}, "kwargs": {}}},
        ):
            stub = types.SimpleNamespace(**{**base, **change})
            assert not check_mask_builder_unread_exempt(trace, stub, [head]), change
    finally:
        trace.cleanup()
