"""Immunizer liveness: prove each exemption narrowing is LOAD-BEARING.

Companion to ``tests/test_validation_exemption_ledger.py`` (grind r2 row 19 /
matrix R08). The ledger proves every exemption is registered with a contract
citation. This module proves the narrowings actually hold the tripwire up.

The pattern, per belt:

1. Build a model whose recorded edge on a previously-blanketed parent is REAL.
2. Freeze the op's replay callable so the edge is provably DEAD -- the
   established armed-proof pattern from
   ``tests/test_validation_exemption_tightening.py``.
3. Assert validation FAILS ``perturbation_insensitive`` with the belt intact.
4. NEUTRALIZE the belt -- restore the exact pre-tightening blanket, or widen the
   proof predicate to always-exempt -- and assert the SAME dead edge is now
   silently EXCUSED.

Step 4 is the point. An immunizer that would pass with its belt removed is
proving nothing; these tests demonstrate that each narrowing is what stands
between a dead recorded edge and a green validation run. Nothing here weakens a
check outside its own ``monkeypatch`` scope.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import exemptions as ex
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
)

# ---------------------------------------------------------------------------
# Helpers (kept local rather than imported from a sibling test module).
# ---------------------------------------------------------------------------


def _capture(model: nn.Module, x: torch.Tensor, seed: int = 0) -> Any:
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
    Any
        Captured trace with saved argument values.
    """

    torch.manual_seed(seed)
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=seed),
    )


def _op_with_func_name(trace: Any, func_name: str) -> Any:
    """Return the first op in ``trace`` whose ``func_name`` matches.

    Parameters
    ----------
    trace:
        Captured trace.
    func_name:
        Recorded function name to find.

    Returns
    -------
    Any
        The matching op.
    """

    return next(op for op in trace.layer_list if op.func_name == func_name)


def _freeze_op_replay(op: Any) -> None:
    """Freeze ``op``'s replay callable so its recorded edges are provably dead.

    Parameters
    ----------
    op:
        Op whose replay callable is replaced by a constant.
    """

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


def _edge_decision(trace: Any, op: Any, parent_label: str) -> tuple[str, str]:
    """Return the ``(decision, reason)`` validation settles for one edge.

    Parameters
    ----------
    trace:
        Captured trace.
    op:
        Child op being replayed.
    parent_label:
        Parent label to perturb.

    Returns
    -------
    tuple[str, str]
        Validation decision and its reason code.
    """

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    return result.decision, result.reason


def _assert_dead_edge_is_caught(trace: Any, op: Any, parent_label: str) -> None:
    """Assert a frozen edge fails perturbation sensitivity.

    Parameters
    ----------
    trace:
        Captured trace.
    op:
        Op with a frozen replay callable.
    parent_label:
        Parent label to perturb.
    """

    decision, reason = _edge_decision(trace, op, parent_label)
    assert decision == "failed", (decision, reason)
    assert reason == "perturbation_insensitive", reason


def assert_decision_is_excused(decision: str, reason: str) -> None:
    """Assert a settled decision is an exemption.

    Parameters
    ----------
    decision:
        Validation decision kind.
    reason:
        Reason code the decision carried.
    """

    assert decision == "exempted", (
        "belt neutralized but the dead edge was NOT excused -- this immunizer's "
        f"belt may not be the load-bearing one: {(decision, reason)}"
    )


def _assert_dead_edge_is_excused(trace: Any, op: Any, parent_label: str) -> None:
    """Assert the same frozen edge is silently exempted (belt neutralized).

    Parameters
    ----------
    trace:
        Captured trace.
    op:
        Op with a frozen replay callable.
    parent_label:
        Parent label to perturb.
    """

    assert_decision_is_excused(*_edge_decision(trace, op, parent_label))


# ---------------------------------------------------------------------------
# Models with a genuinely value-carrying edge on a historically blanketed parent.
# ---------------------------------------------------------------------------


class _EmbeddingModel(nn.Module):
    """Look up embeddings for the integer input indices."""

    def __init__(self, num_embeddings: int = 7) -> None:
        """Initialize the embedding table.

        Parameters
        ----------
        num_embeddings:
            Row count of the embedding table (the index domain).
        """

        super().__init__()
        self.emb = nn.Embedding(num_embeddings, 4)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return summed embeddings so the index edge reaches the output.

        Parameters
        ----------
        idx:
            Integer index tensor.

        Returns
        -------
        torch.Tensor
            Summed embedding rows.
        """

        return self.emb(idx).sum(dim=-1)


class _GatherModel(nn.Module):
    """Gather from the input with a derived index tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather one column per row selected by ``argmax``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Gathered values plus a live slice of the input.
        """

        index = torch.argmax(x, dim=1, keepdim=True).expand(-1, 2)
        return torch.gather(x, 1, index) + x[:, :2]


class _MaskedFillMaskModel(nn.Module):
    """Apply ``masked_fill`` with a mask derived from the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill above-threshold positions so the mask edge matters.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Masked-fill output.
        """

        mask = x > 0.5
        return x.masked_fill(mask, -2.0)


class _FullLikeTensorFillModel(nn.Module):
    """``full_like`` with a RUNTIME tensor fill value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill a template with a value computed from the input.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Filled template plus the live input.
        """

        fill_value = x.sum() * 0.5
        return torch.full_like(x, fill_value) + x


class _ScatterIndexModel(nn.Module):
    """Scatter non-constant sources through a derived index tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scatter ``x``-derived values so the index edge is value-carrying.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scatter output plus the live input.
        """

        index = torch.argsort(x, dim=-1)
        dest = torch.zeros_like(x)
        return dest.scatter(dim=-1, index=index, src=x * 2.0) + x


# ---------------------------------------------------------------------------
# Belt 1: the embedding index structural blanket.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_embedding_index_immunizer_needs_the_removed_structural_blanket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring ``STRUCTURAL_ARG_POSITIONS['embedding'] = {1}`` excuses a dead edge."""

    trace = _capture(_EmbeddingModel(), torch.randint(0, 7, (3, 5)))
    try:
        op = _op_with_func_name(trace, "embedding")
        index_parent = op.parent_arg_positions["args"][1]
        _freeze_op_replay(op)
        _assert_dead_edge_is_caught(trace, op, index_parent)

        monkeypatch.setitem(ex.STRUCTURAL_ARG_POSITIONS, "embedding", {1})
        _assert_dead_edge_is_excused(trace, op, index_parent)
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Belt 2: the gather index structural blanket.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_gather_index_immunizer_needs_the_removed_structural_blanket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring ``STRUCTURAL_ARG_POSITIONS['gather'] = {2}`` excuses a dead edge."""

    trace = _capture(_GatherModel(), torch.randn(4, 5))
    try:
        op = _op_with_func_name(trace, "gather")
        index_parent = op.parent_arg_positions["args"][2]
        _freeze_op_replay(op)
        _assert_dead_edge_is_caught(trace, op, index_parent)

        monkeypatch.setitem(ex.STRUCTURAL_ARG_POSITIONS, "gather", {2})
        _assert_dead_edge_is_excused(trace, op, index_parent)
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Belt 3: the masked_fill mask structural blanket.
# ---------------------------------------------------------------------------


def test_masked_fill_mask_immunizer_needs_the_removed_structural_blanket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring the ``masked_fill`` mask blanket excuses a dead mask edge."""

    trace = _capture(_MaskedFillMaskModel(), torch.randn(3, 4))
    try:
        op = _op_with_func_name(trace, "masked_fill")
        mask_parent = op.parent_arg_positions["args"][1]
        _freeze_op_replay(op)
        _assert_dead_edge_is_caught(trace, op, mask_parent)

        monkeypatch.setitem(ex.STRUCTURAL_ARG_POSITIONS, "masked_fill", {1})
        _assert_dead_edge_is_excused(trace, op, mask_parent)
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Belt 4: the degenerate-index-domain proof (must not become a blanket).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_index_domain_proof_must_stay_a_proof_not_a_blanket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening ``_check_index_domain_degenerate`` to always-true excuses a dead edge.

    The tightened predicate exempts an index parent ONLY when no in-domain
    perturbation exists at all. This proves that narrowness is load-bearing: as a
    blanket it swallows the same dead edge the strict form catches.
    """

    trace = _capture(_ScatterIndexModel(), torch.randn(4, 6))
    try:
        op = _op_with_func_name(trace, "scatter")
        index_parent = op.parent_arg_positions["kwargs"]["index"]
        _freeze_op_replay(op)
        _assert_dead_edge_is_caught(trace, op, index_parent)

        monkeypatch.setitem(
            ex.CUSTOM_EXEMPTION_CHECKS,
            "scatter",
            lambda *_args, **_kwargs: True,
        )
        _assert_dead_edge_is_excused(trace, op, index_parent)
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Belt 5: the posthoc structural-template decision.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_structural_template_decision_must_exclude_runtime_fill_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An always-exempt posthoc structural decision excuses a dead fill-value edge.

    ``full_like``'s TEMPLATE parent is legitimately structural, but its runtime
    fill VALUE is not. The posthoc decision distinguishes them; as a blanket over
    every ``*_like`` op it excuses the dead value edge instead.
    """

    trace = _capture(_FullLikeTensorFillModel(), torch.randn(3, 4))
    try:
        op = _op_with_func_name(trace, "full_like")
        positions = op.parent_arg_positions["args"]
        fill_parent = positions[1]
        _freeze_op_replay(op)
        _assert_dead_edge_is_caught(trace, op, fill_parent)

        monkeypatch.setattr(
            ex,
            "_posthoc_structural_output_decision",
            lambda *_args, **_kwargs: ex.PosthocPerturbDecision(True, "structural_output_template"),
        )
        _assert_dead_edge_is_excused(trace, op, fill_parent)
    finally:
        trace.cleanup()


def test_neutralization_assertion_cannot_pass_vacuously() -> None:
    """The excused-assertion helper rejects a still-firing tripwire.

    Without this, a belt-neutralization test could pass while validation was
    still failing the edge, which would prove the opposite of what it claims.
    """

    assert_decision_is_excused("exempted", "pre_perturbation_exemption")
    with pytest.raises(AssertionError, match="NOT excused"):
        assert_decision_is_excused("failed", "perturbation_insensitive")
    with pytest.raises(AssertionError, match="NOT excused"):
        assert_decision_is_excused("validated", "perturbation_changed")
