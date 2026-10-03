"""W1_F6 regression tests: facet scatter wrapper survival and spec graph-compat honesty.

review-1 (CRITICAL): a sticky facet-slice hook (``tl.head(0, "q")`` + ``tl.zero_ablate()``)
must fire through the slice-scatter wrapper on rerun; losing the wrapper silently
applied the helper to the WHOLE home tensor (zeroing every head).

review-2 (HIGH, REWORKED): ``check_spec_compat`` returns ``COMPATIBLE_WITH_CONFIRMATION``
for an EXECUTABLE spec whose saved ``graph_shape_hash`` mismatches but whose targets
still resolve -- the honest coarse-preview verdict, since a hash mismatch cannot be told
apart from cross-version hash drift on the SAME graph (the shipped v2.16 backcompat
fixtures). The narrow version-stable refusal (unresolvable targets on a mismatched
graph) stays. Promoting the confirmation verdict to a hard refusal is an owner decision
(see W1_F6_REPORT.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import (
    GraphShapeMismatchError,
    OpaqueCallableInExecutableSaveError,
    ReplayPreconditionError,
)
from torchlens.intervention.hooks import normalize_hooks_from_spec
from torchlens.intervention.save import check_spec_compat, load_intervention_spec
from torchlens.intervention.types import InterventionSpec, TargetSpec


class MultiHeadSelfAttention(nn.Module):
    """Tiny attention block matching the DistilBERT facet recipe class name."""

    def __init__(self) -> None:
        """Initialize projection children used by the built-in recipe."""

        super().__init__()
        self.n_heads = 2
        self.dim = 8
        self.q_lin = nn.Linear(8, 8)
        self.k_lin = nn.Linear(8, 8)
        self.v_lin = nn.Linear(8, 8)
        self.out_lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run projection children so q/k/v facets are op-anchored."""

        return self.out_lin(self.q_lin(x) + self.k_lin(x) + self.v_lin(x))


class _AttnWrapper(nn.Module):
    """Wrapper exposing the attention block under a stable module address."""

    def __init__(self) -> None:
        """Initialize the attention child."""

        super().__init__()
        self.attn = MultiHeadSelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention child."""

        return self.attn(x)


class ReluModel(nn.Module):
    """Single-relu model for graph-compat tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relu."""

        return torch.relu(x)


class SigmoidReluModel(nn.Module):
    """Two-op model whose graph shape differs from ``ReluModel``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid(relu(x))."""

        return torch.sigmoid(torch.relu(x))


def _traced_attention() -> tuple[nn.Module, torch.Tensor, Any]:
    """Trace the attention fixture and return (model, input, clean trace)."""

    torch.manual_seed(0)
    model = _AttnWrapper()
    x = torch.randn(2, 3, 8)
    clean = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    return model, x, clean


# ---------------------------------------------------------------------------
# review-1: facet scatter wrapper must survive sticky storage and rerun.
# ---------------------------------------------------------------------------


def test_facet_head_ablate_zeroes_only_selected_head() -> None:
    """``tl.head(0, "q")`` + zero_ablate zeroes head 0's q and ONLY head 0's q."""

    model, x, clean = _traced_attention()
    clean_h1 = clean.modules["attn"].facets.head(1).q.clone()
    assert int(torch.count_nonzero(clean_h1)) > 0

    zeroed = clean.fork("zero_q_head")
    zeroed.attach_hooks(tl.head(0, "q"), tl.zero_ablate())
    zeroed.run(model, x)

    h0 = zeroed.modules["attn"].facets.head(0).q
    h1 = zeroed.modules["attn"].facets.head(1).q
    assert int(torch.count_nonzero(h0)) == 0
    # The unselected head must be byte-identical to the clean capture: the wrapper
    # was previously dropped and the raw helper zeroed the whole home tensor.
    assert torch.equal(h1, clean_h1)


def test_facet_helper_hook_stores_scatter_wrapper_with_helper_provenance() -> None:
    """Sticky facet entries store the scatter wrapper as the fire-time hook."""

    _, _, clean = _traced_attention()
    edited = clean.fork("wrapper_storage")
    edited.attach_hooks(tl.head(0, "q"), tl.zero_ablate())

    facet_specs = [
        hook_spec
        for hook_spec in edited._intervention_spec.hook_specs
        if hook_spec.metadata.get("facet_write")
    ]
    assert facet_specs, "facet attachment must store facet_write hook specs"
    for hook_spec in facet_specs:
        assert getattr(hook_spec.hook, "_tl_facet_scatter", False), (
            "stored fire-time hook must be the facet scatter wrapper"
        )
        assert hook_spec.helper is not None, "raw helper must be kept as provenance"


def test_facet_write_spec_without_wrapper_refuses_typed() -> None:
    """A facet_write hook spec whose wrapper was lost refuses to normalize.

    This is the class tripwire: ANY path that stores a facet_write entry without
    the live scatter wrapper (e.g. a spec reconstructed from disk) must fail
    closed instead of silently firing the helper against the whole home tensor.
    """

    helper = tl.zero_ablate()
    spec = InterventionSpec()
    spec.add_hook(
        TargetSpec("label", "linear_1_1"),
        helper,
        helper=helper,
        metadata={
            "facet_write": True,
            "facet_name": "q",
            "facet_home_label": "linear_1_1",
            "direction": "forward",
        },
    )
    with pytest.raises(ReplayPreconditionError, match="facet-slice hook lost its scatter wrapper"):
        normalize_hooks_from_spec(spec)


@pytest.mark.parametrize("level", ["executable_with_callables", "portable"])
def test_save_facet_hook_refuses_at_executable_levels(tmp_path: Any, level: str) -> None:
    """Executable-level saves of facet-slice hooks refuse instead of persisting
    a spec whose replay would write the whole home tensor."""

    _, _, clean = _traced_attention()
    edited = clean.fork("facet_save_refusal")
    edited.attach_hooks(tl.head(0, "q"), tl.zero_ablate())

    with pytest.raises(OpaqueCallableInExecutableSaveError, match="facet-slice hook"):
        edited.save_intervention(tmp_path / f"facet_{level}.tlspec", level=level)


# ---------------------------------------------------------------------------
# review-2: executable spec compat verdict on graph_shape_hash mismatch.
#
# REWORK NOTE: the original W1_F6 change hard-raised GraphShapeMismatchError at
# compat-preview time for ANY executable spec whose saved graph_shape_hash did not
# match the target log. That was over-broad: a hash mismatch cannot distinguish a
# genuinely different target graph from cross-version hash drift on the SAME graph
# (identical resolved labels; only the version-unstable hash differs -- exactly what
# the shipped v2.16 backcompat fixtures encode). check_spec_compat is a coarse PREVIEW
# and COMPATIBLE_WITH_CONFIRMATION is its honest "graph shape differs, confirm before
# applying" verdict. These tests lock that contract; the fixture regression below
# proves cross-version reuse still loads-and-matches. Whether that confirmation verdict
# should be promoted to a hard refusal for executable specs (a shipped-contract change)
# is escalated to the owner in W1_F6_REPORT.md.
# ---------------------------------------------------------------------------

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tlspec_v2_16"


def _saved_relu_scale_spec(tmp_path: Any, *, level: str = "executable_with_callables") -> Any:
    """Save and reload a scale-relu intervention spec captured on ``ReluModel``."""

    torch.manual_seed(0)
    x = torch.randn(2, 3)
    log_a = tl.trace(
        ReluModel(),
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    log_a.attach_hooks(tl.func("relu"), tl.scale(2.0), confirm_mutation=True)
    path = tmp_path / f"scale_relu_{level}.tlspec"
    log_a.save_intervention(path, level=level)
    return load_intervention_spec(path)


def test_check_spec_compat_executable_hash_mismatch_returns_confirmation(tmp_path: Any) -> None:
    """Contract: an executable spec whose targets resolve identically on a
    hash-mismatched graph returns COMPATIBLE_WITH_CONFIRMATION (NOT a raise).

    A hash mismatch alone is indistinguishable from cross-version drift; the
    confirmation verdict flags the shape difference without breaking legitimate
    cross-version executable-spec reuse.
    """

    spec = _saved_relu_scale_spec(tmp_path)
    assert spec.metadata.get("executable") is True

    torch.manual_seed(0)
    log_b = tl.trace(
        SigmoidReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    compat = check_spec_compat(spec, log_b)
    assert compat.outcome == "COMPATIBLE_WITH_CONFIRMATION"
    assert compat.targets_resolve_identically is True


def test_check_spec_compat_executable_unresolvable_mismatch_still_raises(tmp_path: Any) -> None:
    """The narrow, version-stable refusal stays: an executable spec whose saved
    target CANNOT resolve on a mismatched graph raises GraphShapeMismatchError."""

    spec = _saved_relu_scale_spec(tmp_path)
    torch.manual_seed(0)
    # A model with no relu op: the saved 'relu' target cannot resolve -> FAIL + mismatch.
    other = tl.trace(
        nn.Sigmoid(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    with pytest.raises(GraphShapeMismatchError):
        check_spec_compat(spec, other)


def test_check_spec_compat_same_graph_stays_exact(tmp_path: Any) -> None:
    """No false refusal: the untampered spec against the same graph is EXACT."""

    spec = _saved_relu_scale_spec(tmp_path)
    torch.manual_seed(0)
    same_log = tl.trace(
        ReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )

    compat = check_spec_compat(spec, same_log)
    assert compat.outcome == "EXACT"
    assert compat.targets_resolve_identically is True


def test_check_spec_compat_nonexecutable_mismatch_still_confirmation(tmp_path: Any) -> None:
    """Audit-level (non-executable) specs keep the confirmation verdict on a
    mismatched graph so inspection-level reuse stays possible."""

    spec = _saved_relu_scale_spec(tmp_path, level="audit")
    assert bool(spec.metadata.get("executable", False)) is False

    torch.manual_seed(0)
    log_b = tl.trace(
        SigmoidReluModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    compat = check_spec_compat(spec, log_b)
    assert compat.outcome == "COMPATIBLE_WITH_CONFIRMATION"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "F1_intervention_default.tlspec",
        "F5_intervention_executable_with_callables.tlspec",
        "F6_intervention_portable.tlspec",
    ],
)
def test_v2_16_executable_fixtures_compat_is_confirmation_not_refusal(fixture_name: str) -> None:
    """Regression: the shipped v2.16 executable fixtures load and their loaded
    spec compat-checks against a fresh same-recipe counterpart WITHOUT raising.

    These fixtures carry a v2.16-era graph_shape_hash that differs from the
    current-algorithm hash for the identical CNN graph; the over-broad W1_F6
    compat raise made all three FAIL at the gate. This pins that cross-version
    executable-spec reuse stays load-and-matchable.
    """

    fixture_path = FIXTURE_ROOT / fixture_name
    spec = tl.load(fixture_path, trust_custom_callables=True)
    assert spec.metadata.get("executable") is True
    # Cross-version drift: saved hash differs from the current-algo hash for the
    # same graph, yet targets resolve identically.
    saved_hashes = {e.get("graph_shape_hash") for e in spec.metadata["target_manifest"]}

    log = _capture_cnn_relu_ablation()
    assert log.graph_shape_hash not in saved_hashes  # confirm the mismatch is real
    compat = check_spec_compat(spec, log)
    assert compat.outcome in {"EXACT", "COMPATIBLE_WITH_CONFIRMATION"}
    assert compat.targets_resolve_identically is True


def _capture_cnn_relu_ablation() -> Any:
    """Fresh in-memory counterpart matching the v2.16 intervention fixtures."""

    class _CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.conv(x))

    torch.manual_seed(1101)
    log = tl.trace(
        _CNN(),
        torch.randn(1, 3, 8, 8),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    return log
