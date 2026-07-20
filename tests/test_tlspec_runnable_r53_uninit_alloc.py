"""r53 hon_2: uninitialized-memory (``torch.empty`` family) value-source honesty.

The ``empty`` factory family (and a GROWING ``resize_``) produces bytes that are
not a function of the recorded computation, yet carried no nondeterministic-source
recognition (r52 finding): an escape was a silent nondeterministic VERIFIED with
zero report signal, an archived family slot RAISED ``numeric_attestation_failed``
(inconsistent with the declared-RNG ``not_applicable`` posture), and an
empty-driven branch had no ceiling.

Whole-class immunizers pinned here:
- ONE load-side classifier (``_nondeterministic_value_sources``) feeds the branch
  ceiling, the attestation gate, and ``RunReport.nondeterministic_sources`` -- a
  source-scan meta-test forbids re-deriving family nondeterminism elsewhere.
- ONE closed family table in ``utils/rng.py`` defended by an aten-namespace drift
  meta-test (no torch Tag exists for uninit allocation).
- Total-write sanitization (``out=``/``copy_``/``zero_``/``fill_``/RNG fills):
  empty-then-fully-written flows stay clean end to end.
- The byte-exact tripwire keeps its teeth: an UNTAINTED archived mismatch still
  raises.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.runnable import NumericAttestationError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


@pytest.fixture(autouse=True)
def _realistic_nondeterministic_fill():
    """Capture under the DEFAULT user context (deterministic algorithms OFF).

    ``tests/conftest.py`` enables ``torch.use_deterministic_algorithms(True)``
    globally, under which the ``empty`` family is deterministically NaN-filled
    (the hon_2 refinement, pinned by
    ``test_deterministic_fill_capture_is_clean_and_attestable``) -- so the
    r52 nondeterminism class only exists in the ordinary eager default this
    fixture restores for the capture window.
    """

    was = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(was, warn_only=warn_only)


def _run_report(model: nn.Module, x: torch.Tensor, tmp_path: Path, **save_kwargs: Any):
    trace = tl.trace(model, x.clone(), capture=_CAPTURE)
    path = tmp_path / f"{type(model).__name__}.tlspec"
    trace.save(path, level="runnable", include_weights=True, **save_kwargs)
    return tl.load(path).run(inputs=x.clone()).report


class EmptyEscapeModel(nn.Module):
    """r52 consequence A: uninitialized bytes escape to the model output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.empty_like(x)


class EmptyCopyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        e.copy_(x)
        return e * 2


class EmptyZeroModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        e.zero_()
        return x + e


class EmptyFillModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        e.fill_(1.0)
        return x + e


class EmptyOutModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        torch.add(x, 1.0, out=e)
        return e * 2


class EmptyRngFillModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        e.uniform_()
        return x + e


class EmptyBranchInDagModel(nn.Module):
    """The empty product stays in the DAG (``empty_like``) and steers a branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        if float(e.sum()) > 1e30:
            return x * 2.0
        return x + 1.0


class EmptyBranchPrunedModel(nn.Module):
    """The empty chain is input-disconnected: orphan-pruned from the DAG."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty(2, 4)
        if float(e.sum()) > 1e30:
            return x * 2.0
        return x + 1.0


class EmptyFillBranchPrunedModel(nn.Module):
    """Pruned empty -> TOTAL WRITE -> branch: the sanitized walk must not flag it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty(3)
        e.fill_(1.0)
        if float(e.sum()) > 0:
            return x * 2.0
        return x + 1.0


class ZeroNumelEmptyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.empty(0).sum()


class EmptyScratchLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.empty_like(x)
        e.copy_(x)
        return self.lin(e)


class ZerosScratchLinear(nn.Module):
    """Deterministic control: byte-identical structure, no family op."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.zeros_like(x)
        e.copy_(x)
        return self.lin(e)


class ResizeGrowBranchModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.reshape(-1).clone()
        t.resize_(64)
        if float(t.sum()) > 1e30:
            return x * 2.0
        return x + 1.0


class ResizeShrinkModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.reshape(-1).clone()
        t.resize_(4)
        return x + t.mean()


@pytest.mark.smoke
def test_empty_escape_is_declared_verified_path_only(tmp_path: Path) -> None:
    """Consequence A closed: the output escape stays path-only VERIFIED but is
    now DECLARED -- distinguishable from a deterministic run by the report."""

    x = torch.randn(2, 4)
    report = _run_report(EmptyEscapeModel(), x, tmp_path)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert report.nondeterministic_sources == ("uninitialized_alloc",)


@pytest.mark.smoke
def test_archived_empty_activation_is_not_applicable_never_raises(tmp_path: Path) -> None:
    """The r52 inconsistency: an archived family slot must report
    ``not_applicable`` upfront (like a declared RNG source), never raise."""

    x = torch.randn(2, 4)
    report = _run_report(EmptyEscapeModel(), x, tmp_path, include_activations=True)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert "uninitialized_alloc" in report.nondeterministic_sources


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls", (EmptyCopyModel, EmptyZeroModel, EmptyFillModel, EmptyOutModel)
)
def test_empty_then_total_writer_declares_nothing(model_cls: type, tmp_path: Path) -> None:
    """Sanitizer pin (one test per total-writer entry): empty-then-fully-written
    is the idiomatic scratch init -- VERIFIED, no ceiling, no declared source."""

    x = torch.randn(2, 4)
    report = _run_report(model_cls(), x, tmp_path)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.nondeterministic_sources == ()


def test_empty_then_rng_fill_reclassifies_as_seeded(tmp_path: Path) -> None:
    """An RNG fill removes the uninit taint and hands the product to the seeded
    nets: declared ``seeded_rng``, never ``uninitialized_alloc``."""

    x = torch.randn(2, 4)
    report = _run_report(EmptyRngFillModel(), x, tmp_path)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.nondeterministic_sources == ("seeded_rng",)


@pytest.mark.smoke
def test_empty_driven_branch_in_dag_is_unverifiable(tmp_path: Path) -> None:
    """Consequence B closed (in-DAG layer): a control fact fed by surviving
    uninit taint ceilings structurally -- parity with the RNG-driven branch --
    for EVERY run, not only when the fresh garbage happens to differ."""

    x = torch.randn(2, 4)
    report = _run_report(EmptyBranchInDagModel(), x, tmp_path)
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert "uninitialized_alloc" in report.nondeterministic_sources


@pytest.mark.smoke
def test_pruned_empty_driven_branch_is_unverifiable(tmp_path: Path) -> None:
    """Consequence B closed (pruned layer): the orphan-pruned empty-driven
    predicate chain is recorded by the shared pruned-nondeterministic-control
    recorder, so the run ceilings even though the DAG never sees the chain."""

    from torchlens.backends.torch.completeness_witness import (
        pruned_rng_control_source_labels,
    )

    x = torch.randn(2, 4)
    trace = tl.trace(EmptyBranchPrunedModel(), x.clone(), capture=_CAPTURE)
    assert pruned_rng_control_source_labels(trace)  # the recorder fired
    path = tmp_path / "pruned-branch.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_pruned_empty_then_fill_walk_is_sanitized(tmp_path: Path) -> None:
    """The sanitized child walk stops at a total-writer edge whose destination
    is the walked tensor: a pruned ``empty -> fill_(1) -> branch`` chain is a
    DETERMINISTIC predicate and must NOT be recorded as a pruned
    nondeterministic control source (the over-trigger an unsanitized
    generalization would have added). Pre-existing conservative nets may still
    govern the overall verdict; this pins MY layer's no-fire."""

    from torchlens.backends.torch.completeness_witness import (
        pruned_rng_control_source_labels,
    )

    x = torch.randn(2, 4)
    trace = tl.trace(EmptyFillBranchPrunedModel(), x.clone(), capture=_CAPTURE)
    assert pruned_rng_control_source_labels(trace) == frozenset()


@pytest.mark.smoke
def test_zero_numel_empty_stays_clean_and_attested(tmp_path: Path) -> None:
    """A zero-element family product exposes no bytes: fully clean."""

    x = torch.randn(2, 4)
    report = _run_report(ZeroNumelEmptyModel(), x, tmp_path, include_activations=True)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert report.nondeterministic_sources == ()


def test_deterministic_fill_capture_is_clean_and_attestable(tmp_path: Path) -> None:
    """Under ``use_deterministic_algorithms(True)`` with the fill knob on, the
    family's bytes ARE a function of the recorded computation (NaN fill):
    no taint, byte-exact attestation, replay restores the recorded fill mode."""

    x = torch.randn(2, 4)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill_before = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        trace = tl.trace(EmptyEscapeModel(), x.clone(), capture=_CAPTURE)
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill_before
    path = tmp_path / "det-fill.tlspec"
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert report.nondeterministic_sources == ()


def test_archive_pair_honest_asymmetry_and_no_raise(tmp_path: Path) -> None:
    """Default save archives the PRE-write garbage slot too: the empty variant
    honestly reports ``not_applicable`` (garbage bytes can never byte-attest;
    the r52 posture here was a RAISE) while the byte-identical deterministic
    control still ATTESTS -- the eligibility gate is flow-scoped, not a blanket
    family downgrade."""

    torch.manual_seed(7)
    x = torch.randn(2, 4)
    empty_report = _run_report(EmptyScratchLinear().eval(), x, tmp_path, include_activations=True)
    assert empty_report.path_faithfulness is PathFaithfulness.VERIFIED
    assert empty_report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE

    torch.manual_seed(7)
    control_report = _run_report(ZerosScratchLinear().eval(), x, tmp_path, include_activations=True)
    assert control_report.path_faithfulness is PathFaithfulness.VERIFIED
    assert control_report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert control_report.nondeterministic_sources == ()


@pytest.mark.smoke
def test_untainted_archived_mismatch_still_raises(tmp_path: Path) -> None:
    """LOCKED tripwire: a byte mismatch on an UNTAINTED slot still raises
    ``numeric_attestation_failed`` -- the uninit gate is never a tolerance."""

    torch.manual_seed(7)
    model = ZerosScratchLinear().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x.clone(), capture=_CAPTURE)
    path = tmp_path / "tripwire.tlspec"
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    loaded = tl.load(path)
    archive = loaded.__dict__["_runnable_archived_activations"]
    victim = next(key for key in archive if "linear" in key)
    record = archive[victim]
    archive[victim] = dataclasses.replace(record, value=record.value + 0.5)
    with pytest.raises(NumericAttestationError):
        loaded.run(inputs=x.clone(), on_divergence="return_diverged")


def _resize_disposition(model: nn.Module, x: torch.Tensor, tmp_path: Path):
    """Save-or-run a resize model, returning ``("refused", None)`` or ``("ran", report)``.

    Today an unmodeled ``resize_`` (storage reallocation) is refused by the
    pre-existing producer preflight BEFORE any runnable artifact exists -- an
    even stronger fail-closed posture than an unverifiable run. The r53 grow
    gate (classifier + pruned walk, keyed on the shared family table) stays as
    defense-in-depth should resize ever become a modeled call, so these tests
    accept either disposition but NEVER a false VERIFIED.
    """

    from torchlens.errors.runnable import RunnablePreflightError

    trace = tl.trace(model, x.clone(), capture=_CAPTURE)
    path = tmp_path / f"{type(model).__name__}.tlspec"
    try:
        trace.save(path, level="runnable", include_weights=True)
    except RunnablePreflightError:
        return "refused", None
    return "ran", tl.load(path).run(inputs=x.clone()).report


def test_resize_grow_branch_never_false_verified(tmp_path: Path) -> None:
    """A GROWING resize exposing stale bytes into a branch must never bless
    VERIFIED: refused at producer preflight today, ceilinged by the grow-gated
    classifier if ever modeled."""

    x = torch.randn(2, 4)
    disposition, report = _resize_disposition(ResizeGrowBranchModel(), x, tmp_path)
    if disposition == "ran":
        assert report is not None
        assert report.path_faithfulness is not PathFaithfulness.VERIFIED
        assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_resize_shrink_declares_no_uninit_source(tmp_path: Path) -> None:
    """Shrink/same-size preserves the element prefix (probed byte-clean): the
    grow gate must never taint it -- no declared uninitialized source on any
    runnable disposition."""

    x = torch.randn(2, 4)
    disposition, report = _resize_disposition(ResizeShrinkModel(), x, tmp_path)
    if disposition == "ran":
        assert report is not None
        assert "uninitialized_alloc" not in report.nondeterministic_sources


def test_uninit_family_table_matches_live_aten_registry() -> None:
    """Family-drift immunizer: torch gives uninit allocation NO Tag, so the
    closed name table is defended by enumeration -- an ``empty*``/``resize*``
    aten name that is neither in the family table nor in the justified
    non-family allowlist below FAILS this test on any torch upgrade."""

    from torchlens.utils.rng import (
        _UNINIT_ALLOC_FACTORY_TAILS,
        _UNINIT_ALLOC_RESIZE_TAILS,
    )

    # Justified NON-family names matching the patterns:
    # - ``_resize_output``/``_resize_output_``: internal ``out=`` plumbing whose
    #   destination is always fully overwritten by the kernel (out= sanitizer).
    # - sparse resizes: sparse layout metadata growth materializes implicit
    #   zeros (``and_clear_`` zeroes), never dense stale bytes.
    allowlist = {
        "_resize_output",
        "_resize_output_",
        "sparse_resize_",
        "sparse_resize_and_clear_",
        "resize_as_sparse_",
    }
    aten_names = set(dir(torch.ops.aten))
    empty_pattern = {
        name
        for name in aten_names
        if name.lstrip("_") == "empty"
        or name.lstrip("_").startswith("empty_")
        or name.endswith("_empty")
        or "_empty_" in name
    }
    resize_pattern = {name for name in aten_names if "resize" in name}
    family = _UNINIT_ALLOC_FACTORY_TAILS | _UNINIT_ALLOC_RESIZE_TAILS
    unaccounted = (empty_pattern | resize_pattern) - family - allowlist
    assert not unaccounted, (
        "New uninitialized-allocation-pattern aten ops are neither in the "
        f"closed family table nor in the justified allowlist: {sorted(unaccounted)}"
    )


def test_single_classifier_owns_qualname_derivation() -> None:
    """Source-scan immunizer: inside the execution module, the family
    predicates are consulted ONLY by ``_nondeterministic_value_sources`` -- no
    second call site can re-derive uninit nondeterminism and drift from the
    single classifier (the r52 raise-vs-not_applicable inconsistency class)."""

    source = (
        Path(__file__).resolve().parents[1] / "torchlens" / "_runnable_execution.py"
    ).read_text()
    functions = re.split(r"(?m)^def ", source)
    users = [
        chunk.split("(", 1)[0]
        for chunk in functions
        if "qualname_is_uninitialized_alloc(" in chunk
        or "qualname_is_uninit_growth_resize(" in chunk
        or "qualname_is_uninit_total_writer(" in chunk
    ]
    assert users == ["_nondeterministic_value_sources"], users


class SeededRandModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand(x.shape[0], x.shape[1])


def test_nondeterministic_sources_field_vocabulary(tmp_path: Path) -> None:
    """F4: the report field is closed-vocabulary, sorted, finalizer-derived --
    a seeded model declares ``seeded_rng``; the live provider declares
    nothing (it gathers no descriptor evidence)."""

    from torchlens.runnable import NONDETERMINISTIC_SOURCE_VOCABULARY

    x = torch.randn(2, 4)
    report = _run_report(SeededRandModel(), x, tmp_path)
    assert report.nondeterministic_sources == ("seeded_rng",)
    assert set(report.nondeterministic_sources) <= NONDETERMINISTIC_SOURCE_VOCABULARY

    live_trace = tl.trace(nn.Linear(4, 3).eval(), x.clone(), capture=_CAPTURE)
    live_report = live_trace.run(inputs=x.clone()).report
    assert live_report.nondeterministic_sources == ()
