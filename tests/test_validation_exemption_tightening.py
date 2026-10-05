"""F2 tripwire tightening: legacy index/mask/``*_like`` exemption blankets.

The pre-sprint blankets (``STRUCTURAL_ARG_POSITIONS`` entries for
``embedding``/``gather``/``index_select``/``scatter*``/``cross_entropy``
index-class args, the ``masked_fill`` mask entries, and the posthoc
``*_like`` structural-template excuse) skipped perturbation-sensitivity
checks entirely for those parents, so a genuinely missed or broken
dependency on them could never fail validation. This module locks the
tightened standard:

- Index-class parents are perturbed IN-DOMAIN (``(v + 1) % n`` rotation of
  valid indices), so their recorded edges are now sensitivity-verified.
- A frozen/dead edge on any of these parents now FAILS validation with
  ``perturbation_insensitive`` (armed-proof tests, following the
  ``test_validation_hardening`` freeze-the-func pattern).
- The legitimately-exempt cases stay exempt through NARROW proofs only:
  degenerate index domains, provable value-irrelevance (uniform embedding
  rows, sources constant along the indexed dim, constant full-coverage
  scatter), a ``masked_fill`` input that equals the fill value everywhere,
  and the ``*_like`` TEMPLATE parent (never a runtime fill-value parent).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation._index_domain import index_domain_rotation_values
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
)
from torchlens.validation.exemptions import (
    CUSTOM_EXEMPTION_CHECKS,
    STRUCTURAL_ARG_KWARG_ALIASES,
    STRUCTURAL_ARG_POSITIONS,
    _check_index_domain_degenerate,
    _check_masked_fill_exempt,
    _posthoc_structural_output_decision,
    posthoc_perturb_check,
)


def _capture(model: nn.Module, x: torch.Tensor, seed: int = 0):
    """Capture a full-save trace the way ``validate_forward_pass`` does.

    Parameters
    ----------
    model:
        Module to trace.
    x:
        Input tensor.
    seed:
        RNG seed for the capture.

    Returns
    -------
    Trace
        Captured trace with saved argument values.
    """

    torch.manual_seed(seed)
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=seed),
    )


def _quiet_validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Public-path validation with provenance warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, x))


def _op_with_func_name(trace: Any, func_name: str) -> Any:
    """Return the first op in ``trace`` whose ``func_name`` matches."""

    return next(op for op in trace.layer_list if op.func_name == func_name)


def _freeze_op_replay(op: Any) -> None:
    """Freeze ``op``'s replay callable to return its saved out unconditionally.

    This simulates a genuinely dead recorded edge -- the parent provably does
    not influence the output -- exactly as in the established armed-proof
    tests in ``test_validation_hardening``.
    """

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


def _assert_edge_now_fails(trace: Any, op: Any, parent_label: str) -> None:
    """Assert a frozen edge fails ``perturbation_insensitive`` for this parent."""

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed", (result.decision, result.reason)
    assert result.reason == "perturbation_insensitive"


def _assert_edge_sensitivity_verified(trace: Any, op: Any, parent_label: str) -> None:
    """Assert a healthy edge is now perturbation-VERIFIED, not blanket-exempted."""

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "validated", (result.decision, result.reason)
    assert result.reason == "perturbation_changed"


# ---------------------------------------------------------------------------
# Registry narrowing is locked
# ---------------------------------------------------------------------------


def test_index_and_mask_blankets_removed_from_structural_registry() -> None:
    """The OOB-justified index/target/mask blankets are gone from the registry."""

    for func_name in (
        "cross_entropy",
        "embedding",
        "gather",
        "index_select",
        "scatter_",
        "scatter_add_",
        "scatter_add",
        "scatteradd",
        "maskedfill",
        "masked_fill",
        "masked_fill_",
    ):
        assert func_name not in STRUCTURAL_ARG_POSITIONS
        assert func_name not in STRUCTURAL_ARG_KWARG_ALIASES
    # The narrow custom checks that replaced the index blankets are registered.
    for func_name in (
        "embedding",
        "gather",
        "index_select",
        "cross_entropy",
        "scatter",
        "scatter_",
        "scatter_add",
        "scatter_add_",
        "scatteradd",
    ):
        assert func_name in CUSTOM_EXEMPTION_CHECKS


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _EmbeddingModel(nn.Module):
    """Look up embeddings for the integer input indices."""

    def __init__(self, num_embeddings: int = 7) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, 4)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return summed embeddings so the index edge reaches the output."""

        return self.emb(idx).sum(dim=-1)


class _GatherModel(nn.Module):
    """Gather from the input with a derived index tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather one column per row selected by ``argmax``."""

        index = torch.argmax(x, dim=1, keepdim=True).expand(-1, 2)
        return torch.gather(x, 1, index) + x[:, :2]


class _CrossEntropyModel(nn.Module):
    """Compute a cross-entropy loss against derived integer targets."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the loss so the target edge reaches the output."""

        logits = self.fc(x)
        targets = torch.arange(logits.size(0)) % logits.size(1)
        return torch.nn.functional.cross_entropy(logits, targets)


class _ScatterIndexModel(nn.Module):
    """Scatter non-constant sources through a derived index tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scatter ``x``-derived values so the index edge is value-carrying."""

        index = torch.argsort(x, dim=-1)
        dest = torch.zeros_like(x)
        return dest.scatter(dim=-1, index=index, src=x * 2.0) + x


class _MaskedFillMaskModel(nn.Module):
    """Apply ``masked_fill`` with a mask derived from the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill above-threshold positions so the mask edge matters."""

        mask = x > 0.5
        return x.masked_fill(mask, -2.0)


class _FullLikeTensorFillModel(nn.Module):
    """``full_like`` with a RUNTIME tensor fill value (the F2 example)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill a template with a value computed from the input."""

        fill_value = x.sum() * 0.5
        return torch.full_like(x, fill_value) + x


# ---------------------------------------------------------------------------
# Newly-caught cases: a dead edge on a previously-blanketed parent now FAILS
# ---------------------------------------------------------------------------


def test_dead_embedding_index_edge_now_fails() -> None:
    """Armed-proof: a frozen embedding index edge fails instead of being excused.

    Before F2 tightening, ``STRUCTURAL_ARG_POSITIONS["embedding"] == {1}``
    skipped the check pre-execution, so this dead edge validated True.
    """

    trace = _capture(_EmbeddingModel(), torch.randint(0, 7, (3, 5)))
    op = _op_with_func_name(trace, "embedding")
    index_parent = op.parent_arg_positions["args"][1]
    _assert_edge_sensitivity_verified(trace, op, index_parent)
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, index_parent)


def test_dead_gather_index_edge_now_fails() -> None:
    """Armed-proof: a frozen gather index edge fails instead of being excused."""

    trace = _capture(_GatherModel(), torch.randn(4, 5))
    op = _op_with_func_name(trace, "gather")
    index_parent = op.parent_arg_positions["args"][2]
    _assert_edge_sensitivity_verified(trace, op, index_parent)
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, index_parent)


def test_dead_cross_entropy_target_edge_now_fails() -> None:
    """Armed-proof: a frozen cross-entropy target edge fails instead of passing."""

    trace = _capture(_CrossEntropyModel(), torch.randn(3, 4))
    op = _op_with_func_name(trace, "cross_entropy")
    target_parent = op.parent_arg_positions["args"][1]
    _assert_edge_sensitivity_verified(trace, op, target_parent)
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, target_parent)


def test_dead_scatter_index_edge_now_fails() -> None:
    """Armed-proof: a frozen scatter index edge fails instead of being excused."""

    trace = _capture(_ScatterIndexModel(), torch.randn(4, 6))
    op = _op_with_func_name(trace, "scatter")
    index_parent = op.parent_arg_positions["kwargs"]["index"]
    _assert_edge_sensitivity_verified(trace, op, index_parent)
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, index_parent)


def test_dead_masked_fill_mask_edge_now_fails() -> None:
    """Armed-proof: a frozen masked_fill mask edge fails instead of being excused."""

    trace = _capture(_MaskedFillMaskModel(), torch.randn(4, 4))
    op = _op_with_func_name(trace, "masked_fill")
    mask_parent = op.parent_arg_positions["args"][1]
    _assert_edge_sensitivity_verified(trace, op, mask_parent)
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, mask_parent)


def test_dead_full_like_fill_value_edge_now_fails() -> None:
    """Armed-proof: a frozen ``full_like`` runtime fill-value edge now fails.

    This is the exact F2 example: the posthoc ``structural_output_template``
    blanket used to excuse ANY perturbed ``full_like`` parent, including a
    runtime fill_value dependency. The blanket is now template-slot-only.
    """

    trace = _capture(_FullLikeTensorFillModel(), torch.randn(3, 4))
    op = _op_with_func_name(trace, "full_like")
    fill_parent = op.parent_arg_positions["args"][1]
    _assert_edge_sensitivity_verified(trace, op, fill_parent)
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, fill_parent)


# ---------------------------------------------------------------------------
# Legitimately-exempt cases still pass (no false-fails)
# ---------------------------------------------------------------------------


def test_healthy_index_and_mask_models_still_validate_true() -> None:
    """End-to-end: the healthy index/mask/full_like models all validate True."""

    torch.manual_seed(0)
    assert _quiet_validate(_GatherModel(), torch.randn(4, 5))
    assert _quiet_validate(_CrossEntropyModel(), torch.randn(3, 4))
    assert _quiet_validate(_ScatterIndexModel(), torch.randn(4, 6))
    assert _quiet_validate(_MaskedFillMaskModel(), torch.randn(4, 4))
    assert _quiet_validate(_FullLikeTensorFillModel(), torch.randn(3, 4))


def test_embedding_index_validates_true_end_to_end() -> None:
    """End-to-end embedding validation passes through the public path."""

    model = _EmbeddingModel()
    x = torch.randint(0, 7, (3, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert bool(validate_forward_pass(model, x, random_seed=1))


def test_degenerate_embedding_domain_is_exempt_not_failed() -> None:
    """A single-row embedding admits no in-domain index perturbation.

    ``n <= 1`` means value-irrelevance is FORCED by the domain constraint;
    the narrow degenerate exemption keeps this from false-failing, and it
    stays exempt even for a frozen replay (the domain proof is what excuses
    it, so freezing changes nothing).
    """

    model = _EmbeddingModel(num_embeddings=1)
    x = torch.zeros(3, dtype=torch.long)
    assert _quiet_validate(model, x)

    trace = _capture(model, x)
    op = _op_with_func_name(trace, "embedding")
    index_parent = op.parent_arg_positions["args"][1]
    assert _check_index_domain_degenerate(trace, op, [index_parent]) is True
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[index_parent]
    )
    assert result.decision == "exempted"
    assert result.reason == "pre_perturbation_exemption"


def test_uniform_embedding_rows_are_proved_irrelevant_not_failed() -> None:
    """All-identical embedding rows exempt the index edge via a value proof."""

    model = _EmbeddingModel(num_embeddings=5)
    with torch.no_grad():
        model.emb.weight.fill_(0.25)
    x = torch.randint(0, 5, (3, 4))
    assert _quiet_validate(model, x)

    trace = _capture(model, x)
    op = _op_with_func_name(trace, "embedding")
    index_parent = op.parent_arg_positions["args"][1]
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[index_parent]
    )
    assert result.decision == "exempted"
    assert result.reason == "index_domain_value_irrelevant"


def test_gather_from_constant_source_is_proved_irrelevant() -> None:
    """A source constant along the gather dim exempts the index edge by proof."""

    class _ConstantGather(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            source = x[:, :1].expand(-1, x.shape[1])
            index = torch.argmax(x, dim=1, keepdim=True)
            return torch.gather(source, 1, index) + x[:, :1]

    torch.manual_seed(0)
    assert _quiet_validate(_ConstantGather(), torch.randn(4, 5))


def test_masked_fill_input_equal_to_value_is_proved_irrelevant() -> None:
    """A mask over an input that equals the fill value everywhere is exempt."""

    class _NoOpMaskedFill(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            zeros = x * 0.0
            mask = x > 0.5
            return zeros.masked_fill(mask, 0.0) + x

    torch.manual_seed(0)
    assert _quiet_validate(_NoOpMaskedFill(), torch.randn(4, 4))

    trace = _capture(_NoOpMaskedFill(), torch.randn(4, 4))
    op = _op_with_func_name(trace, "masked_fill")
    mask_parent = op.parent_arg_positions["args"][1]
    assert _check_masked_fill_exempt(trace, op, [mask_parent]) is True


def test_full_like_template_parent_stays_structurally_exempt() -> None:
    """The ``*_like`` TEMPLATE parent keeps its structural excuse; others do not."""

    layer = SimpleNamespace(
        func_name="full_like",
        saved_args=(torch.zeros(2, 2), torch.tensor(3.0)),
        saved_kwargs={},
        parent_arg_positions={
            "args": {0: "template_parent", 1: "fill_parent"},
            "kwargs": {},
        },
        out=torch.full((2, 2), 3.0),
        dtype=torch.float32,
    )
    template_decision = _posthoc_structural_output_decision(
        layer, layer.saved_args, ["template_parent"]
    )
    assert template_decision.exempt
    assert template_decision.reason == "structural_output_template"
    fill_decision = _posthoc_structural_output_decision(layer, layer.saved_args, ["fill_parent"])
    assert not fill_decision.exempt


def test_posthoc_check_no_longer_excuses_index_parents_blanketly() -> None:
    """Posthoc exemption for a healthy-domain index parent requires a proof."""

    weight = torch.randn(6, 3)
    layer = SimpleNamespace(
        func_name="embedding",
        saved_args=(weight, torch.tensor([0, 2, 4]), -1, False, False),
        saved_kwargs={},
        parent_arg_positions={"args": {1: "index_parent"}, "kwargs": {}},
        out=weight[torch.tensor([0, 2, 4])],
        dtype=torch.float32,
    )
    decision = posthoc_perturb_check(None, layer, ["index_parent"])
    assert not decision.exempt


def test_index_domain_rotation_is_valid_and_distinct() -> None:
    """Rotation stays in-domain, differs everywhere in-domain, keeps sentinels."""

    weight = torch.randn(5, 3)
    layer = SimpleNamespace(
        func_name="cross_entropy",
        saved_args=(torch.randn(4, 5), torch.tensor([0, 4, -100, 2])),
        saved_kwargs={},
        parent_arg_positions={"args": {1: "target_parent"}, "kwargs": {}},
    )
    rotated = index_domain_rotation_values(layer, "target_parent", torch.tensor([0, 4, -100, 2]))
    assert rotated is not None
    assert rotated.tolist() == [1, 0, -100, 3]

    emb_layer = SimpleNamespace(
        func_name="embedding",
        saved_args=(weight, torch.tensor([[0, 1], [4, 3]]), -1, False, False),
        saved_kwargs={},
        parent_arg_positions={"args": {1: "index_parent"}, "kwargs": {}},
    )
    emb_rotated = index_domain_rotation_values(
        emb_layer, "index_parent", torch.tensor([[0, 1], [4, 3]])
    )
    assert emb_rotated is not None
    assert emb_rotated.tolist() == [[1, 2], [0, 4]]
    # A NON-index parent (the weight) gets no rotation -- it stays on the
    # strict generic perturbation path.
    assert index_domain_rotation_values(emb_layer, "weight_parent", weight) is None


# ---------------------------------------------------------------------------
# R08-1 (b1-sol round-2): Tensor.new is OVERLOADED. Only the argless /
# integer-sizes / torch.Size forms return uninitialized memory; new(tensor)
# and new(data) are initialized value-bearing calls whose replay must run.
# The blanket registry membership blessed a wrong new(tensor) replay
# 'exempted' without execution.
# ---------------------------------------------------------------------------


class _NewFromTensorModel(nn.Module):
    """Model exercising the value-bearing ``Tensor.new(tensor)`` overload."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``x.new(y)`` plus a value anchor on both inputs.

        Parameters
        ----------
        x:
            Prototype tensor supplying dtype/device.
        y:
            VALUE-BEARING source tensor copied into the result.

        Returns
        -------
        torch.Tensor
            The initialized copy of ``y``.
        """

        return x.new(y)


class _NewSizesModel(nn.Module):
    """Model exercising the genuinely uninitialized size-only overload."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Allocate with ``x.new(2, 3)`` and erase the garbage values.

        Parameters
        ----------
        x:
            Prototype tensor.

        Returns
        -------
        torch.Tensor
            Deterministic output built on the uninitialized allocation.
        """

        return x.new(2, 3).zero_() + x.sum()


def test_value_bearing_tensor_new_corruption_fails_not_exempted() -> None:
    """A corrupted ``new(tensor)`` replay must FAIL, never bless 'exempted'.

    Red-capable: pre-narrowing, the registry-1 early exit returned
    ``exempted('uninitialized_by_design')`` before reading the saved output,
    so this planted corruption validated clean (the b1-sol reproduction).
    """

    trace = _capture(_NewFromTensorModel(), [torch.tensor([9.0]), torch.tensor([1.0, 2.0])])
    op = _op_with_func_name(trace, "new")
    # Corrupt the op's own retained output payload: replay recomputes
    # new(y) honestly from the saved parents and must now disagree.
    # Pre-narrowing the registry early-exit blessed this exempted without
    # ever executing the comparison.
    payload = op._slot("out")
    assert payload is not None
    payload.add_(999.0)
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(trace, op.label)
    assert result.decision == "failed", (result.decision, result.reason)
    assert result.reason == "replay_mismatch"


def test_value_bearing_tensor_new_replays_honestly() -> None:
    """An honest ``new(tensor)`` capture must replay-VALIDATE, not exempt."""

    trace = _capture(_NewFromTensorModel(), [torch.tensor([9.0]), torch.tensor([1.0, 2.0])])
    op = _op_with_func_name(trace, "new")
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(trace, op.label)
    assert result.decision == "validated", (result.decision, result.reason)


def test_size_only_tensor_new_stays_exempt() -> None:
    """The genuinely uninitialized size-only overload keeps its exemption."""

    trace = _capture(_NewSizesModel(), torch.tensor([4.0]))
    op = _op_with_func_name(trace, "new")
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(trace, op.label)
    assert result.decision == "exempted", (result.decision, result.reason)
    assert result.reason == "uninitialized_by_design"


def test_uninitialized_by_design_proof_is_fail_closed() -> None:
    """The per-call proof rejects every non-size-only shape."""

    from torchlens.validation.exemptions import uninitialized_by_design_applies

    healthy = SimpleNamespace(
        func_name="new", parents=("input_1",), non_tensor_kwargs={}, non_tensor_pos_args=[2, 3]
    )
    assert uninitialized_by_design_applies(healthy)
    assert uninitialized_by_design_applies(
        SimpleNamespace(
            func_name="new",
            parents=("input_1",),
            non_tensor_kwargs={},
            non_tensor_pos_args=[torch.Size([2, 3])],
        )
    )
    # Second tensor parent = value-bearing new(tensor).
    assert not uninitialized_by_design_applies(
        SimpleNamespace(
            func_name="new",
            parents=("input_1", "input_2"),
            non_tensor_kwargs={},
            non_tensor_pos_args=[],
        )
    )
    # Sequence positional arg = legacy DATA constructor.
    assert not uninitialized_by_design_applies(
        SimpleNamespace(
            func_name="new",
            parents=("input_1",),
            non_tensor_kwargs={},
            non_tensor_pos_args=[[1.0, 2.0]],
        )
    )
    # Unknown kwargs and bools are not provably sizes.
    assert not uninitialized_by_design_applies(
        SimpleNamespace(
            func_name="new",
            parents=("input_1",),
            non_tensor_kwargs={"weird": 1},
            non_tensor_pos_args=[],
        )
    )
    assert not uninitialized_by_design_applies(
        SimpleNamespace(
            func_name="new", parents=("input_1",), non_tensor_kwargs={}, non_tensor_pos_args=[True]
        )
    )
    # Non-overloaded registry members stay membership-exempt.
    assert uninitialized_by_design_applies(
        SimpleNamespace(
            func_name="empty_like",
            parents=("input_1",),
            non_tensor_kwargs={},
            non_tensor_pos_args=[],
        )
    )


def test_new_from_tensor_full_validation_still_passes() -> None:
    """Public-path regression: an honest new(tensor) model validates green."""

    assert _quiet_validate(_NewFromTensorModel(), [torch.tensor([9.0]), torch.tensor([1.0, 2.0])])


# ---------------------------------------------------------------------------
# R08-2 (b1-sol round-2, KNOWN b1:D2 re-filed): meshgrid/broadcast_tensors
# outputs DO carry input values, but the whole-op perturbation skip (and the
# posthoc structural_output_template blanket) exempted EVERY parent -- so a
# dropped or misattributed value edge on these zipped multi-output ops was
# never perturbation-proved. Per-output parent projection: output j must be
# sensitive to input j; only cross-member siblings are structural.
# ---------------------------------------------------------------------------


class _MeshgridModel(nn.Module):
    """Two-input meshgrid whose outputs feed a value-mixing sum."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the summed meshgrid expansion of both inputs.

        Parameters
        ----------
        a:
            First 1-D input.
        b:
            Second 1-D input.

        Returns
        -------
        torch.Tensor
            Sum of both expanded grids.
        """

        grid_a, grid_b = torch.meshgrid(a, b, indexing="ij")
        return grid_a.sum() + grid_b.sum()


class _BroadcastModel(nn.Module):
    """Two-input broadcast_tensors whose outputs feed a value-mixing sum."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the summed broadcast expansion of both inputs.

        Parameters
        ----------
        a:
            First input.
        b:
            Second input (unsqueezed to force a real broadcast).

        Returns
        -------
        torch.Tensor
            Sum of both broadcast tensors.
        """

        expanded_a, expanded_b = torch.broadcast_tensors(a, b.unsqueeze(1))
        return expanded_a.sum() + expanded_b.sum()


def _zipped_op_for_index(trace: Any, func_names: tuple[str, ...], index: int) -> Any:
    """Return the zipped multi-output op with ``multi_output_index == index``."""

    return next(
        op
        for op in trace.layer_list
        if op.func_name in func_names and op.multi_output_index == index
    )


def test_meshgrid_own_value_edge_is_perturbation_tested() -> None:
    """Output j must be perturbation-SENSITIVE to its own input j.

    Red-capable: pre-narrowing the whole-op skip / posthoc blanket recorded
    this edge 'exempted' without ever proving the value dependency.
    """

    trace = _capture(_MeshgridModel(), [torch.randn(3), torch.randn(4)])
    for index in (0, 1):
        op = _zipped_op_for_index(trace, ("meshgrid",), index)
        own_parent = op.parent_arg_positions["args"][(0, index)]
        result = _check_whether_func_on_saved_parents_yields_saved_tensor(
            trace, op.label, perturb=True, layers_to_perturb=[own_parent]
        )
        assert result.decision == "validated", (index, result.decision, result.reason)


def test_meshgrid_dead_own_edge_now_fails_perturbation() -> None:
    """A provably dead own-value edge must FAIL, never launder structural.

    Red-capable: freezing the op's replay (the dropped-dependency simulation
    used across this file) was blessed 'structural_output_template' by the
    posthoc blanket pre-narrowing.
    """

    trace = _capture(_MeshgridModel(), [torch.randn(3), torch.randn(4)])
    op = _zipped_op_for_index(trace, ("meshgrid",), 0)
    own_parent = op.parent_arg_positions["args"][(0, 0)]
    _freeze_op_replay(op)
    _assert_edge_now_fails(trace, op, own_parent)


def test_meshgrid_cross_member_perturbation_stays_exempt() -> None:
    """Cross-member zipped siblings remain provably structural (no false-fail)."""

    trace = _capture(_MeshgridModel(), [torch.randn(3), torch.randn(4)])
    op = _zipped_op_for_index(trace, ("meshgrid",), 0)
    sibling_parent = op.parent_arg_positions["args"][(0, 1)]
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[sibling_parent]
    )
    assert result.decision == "exempted", (result.decision, result.reason)


def test_broadcast_tensors_own_value_edge_is_perturbation_tested() -> None:
    """The canonical 'broadcasttensors' spelling gets the same projection.

    Doubly red-capable: besides the whole-op-skip class, the old registry
    row was keyed 'broadcast_tensors' while capture canonicalizes to
    'broadcasttensors' -- a silently dead row.
    """

    trace = _capture(_BroadcastModel(), [torch.randn(3), torch.randn(4)])
    for index in (0, 1):
        op = _zipped_op_for_index(trace, ("broadcasttensors", "broadcast_tensors"), index)
        own_parent = op.parent_arg_positions["args"][(0, index)]
        result = _check_whether_func_on_saved_parents_yields_saved_tensor(
            trace, op.label, perturb=True, layers_to_perturb=[own_parent]
        )
        assert result.decision == "validated", (index, result.decision, result.reason)
        sibling_parent = op.parent_arg_positions["args"][(0, 1 - index)]
        sibling_result = _check_whether_func_on_saved_parents_yields_saved_tensor(
            trace, op.label, perturb=True, layers_to_perturb=[sibling_parent]
        )
        assert sibling_result.decision == "exempted", (
            index,
            sibling_result.decision,
            sibling_result.reason,
        )


def test_zipped_models_full_validation_still_passes() -> None:
    """Public-path regression: honest zipped-op models validate green."""

    assert _quiet_validate(_MeshgridModel(), [torch.randn(3), torch.randn(4)])
    assert _quiet_validate(_BroadcastModel(), [torch.randn(3), torch.randn(4)])


def test_meshgrid_perturbation_decisions_are_real_not_whole_op_skipped() -> None:
    """The BFS records REAL perturbation decisions for zipped ops.

    Red-capable: pre-narrowing the whole-op registry skip recorded every
    meshgrid parent edge as 'skip_perturbation_entirely:meshgrid' without
    running the check.
    """

    from torchlens.validation import core as validation_core

    model = _MeshgridModel()
    inputs = [torch.randn(3), torch.randn(4)]
    with torch.no_grad():
        ground_truth = model(*inputs)
    trace = _capture(model, inputs)
    status = validation_core.validate_saved_outs(trace, [ground_truth])
    assert bool(status)
    meshgrid_perturbations = [
        decision
        for decision in status.decisions
        if decision["func_name"] == "meshgrid" and decision["phase"] == "perturbation"
    ]
    assert meshgrid_perturbations, "no perturbation decisions recorded for meshgrid"
    assert not any(
        str(decision["reason"]).startswith("skip_perturbation_entirely")
        for decision in meshgrid_perturbations
    )
    # At least one own-value edge is genuinely perturbation-VALIDATED.
    assert any(decision["decision"] == "validated" for decision in meshgrid_perturbations)
