"""r53 hon_2: uninitialized-memory (``torch.empty`` family) value-source honesty.

The ``empty`` factory family (and a GROWING ``resize_``) produces bytes that are
not a function of the recorded computation, yet carried no nondeterministic-source
recognition (r52 finding): an escape was a silent nondeterministic VERIFIED with
zero report signal, an archived family slot RAISED ``replay_value_mismatch``
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

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)


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


# Kept: sole branch coverage of _runnable/verification.py.
def test_empty_then_rng_fill_reclassifies_as_seeded(tmp_path: Path) -> None:
    """An RNG fill removes the uninit taint and hands the product to the seeded
    nets: declared ``seeded_rng``, never ``uninitialized_alloc``."""

    x = torch.randn(2, 4)
    report = _run_report(EmptyRngFillModel(), x, tmp_path)
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.nondeterministic_sources == ("seeded_rng",)


# Kept: sole branch coverage of _runnable/path_faithfulness.py; a sole mutation kill.
def test_empty_driven_branch_in_dag_is_unverifiable(tmp_path: Path) -> None:
    """Consequence B closed (in-DAG layer): a control fact fed by surviving
    uninit taint ceilings structurally -- parity with the RNG-driven branch --
    for EVERY run, not only when the fresh garbage happens to differ."""

    x = torch.randn(2, 4)
    report = _run_report(EmptyBranchInDagModel(), x, tmp_path)
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert "uninitialized_alloc" in report.nondeterministic_sources


# Kept: a sole mutation kill.
def test_pruned_empty_driven_branch_is_unverifiable(tmp_path: Path) -> None:
    """Consequence B closed (pruned layer): the orphan-pruned empty-driven
    predicate chain is recorded by the shared pruned-nondeterministic-control
    recorder, so the run ceilings even though the DAG never sees the chain."""

    from torchlens.backends.torch._completeness_escape_state import pruned_rng_control_source_labels

    x = torch.randn(2, 4)
    trace = tl.trace(EmptyBranchPrunedModel(), x.clone(), capture=_CAPTURE)
    assert pruned_rng_control_source_labels(trace)  # the recorder fired
    path = tmp_path / "pruned-branch.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# Kept: sole branch coverage of _runnable/attestation.py; a sole mutation kill.
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


def _materialized_aten_op_names() -> set[str]:
    """Return the complete live ATen packet namespace after eager materialization.

    Returns
    -------
    set[str]
        Names exposed by ``torch.ops.aten`` after every dispatcher-registered
        ATen base name has been resolved once.
    """

    dispatch_names = getattr(torch._C, "_dispatch_get_all_op_names")()
    aten_base_names = {
        name.removeprefix("aten::").split(".", maxsplit=1)[0]
        for name in dispatch_names
        if name.startswith("aten::")
    }
    for name in aten_base_names:
        getattr(torch.ops.aten, name)
    return set(dir(torch.ops.aten))


# Kept: sole branch coverage of _utils/_uninit_alloc.py.
def test_size_gated_new_recognition_and_arg_form() -> None:
    """r55 hon_1: ``Tensor.new`` recognition is size-gated -- the size form is
    family, the data form is NOT (over-ceiling guard), and an undecidable form
    fails closed to tainted (the grow-gate posture)."""

    import numpy as np

    from torchlens._utils._uninit_alloc import (
        qualname_is_uninit_size_gated_alloc,
        qualname_is_uninitialized_alloc,
        uninit_new_call_is_size_form,
    )

    assert qualname_is_uninit_size_gated_alloc("torch.Tensor", "Tensor.new")
    assert qualname_is_uninit_size_gated_alloc("torch.Tensor", "new")
    # NOT in the unconditional table: qualname-only consumers (which cannot see
    # argument forms) must never taint the deterministic data spelling.
    assert not qualname_is_uninitialized_alloc("torch.Tensor", "new")
    assert not qualname_is_uninit_size_gated_alloc("torch.Tensor", "new_empty")
    assert not qualname_is_uninit_size_gated_alloc("my.custom.ns", "new")
    assert not qualname_is_uninit_size_gated_alloc("torch.Tensor", None)

    # Probed live semantics: SIZE forms (uninitialized allocation).
    assert uninit_new_call_is_size_form(()) is True
    assert uninit_new_call_is_size_form((3, 4)) is True
    assert uninit_new_call_is_size_form((np.int64(3),)) is True
    assert uninit_new_call_is_size_form((3, np.int64(4))) is True
    assert uninit_new_call_is_size_form((torch.Size([2, 2]),)) is True
    # Probed live semantics: DATA forms (deterministic copy constructors).
    assert uninit_new_call_is_size_form(([2, 2],)) is False
    assert uninit_new_call_is_size_form((torch.ones(2),)) is False
    assert uninit_new_call_is_size_form((np.array([1.0, 2.0]),)) is False
    assert uninit_new_call_is_size_form((range(3),)) is False
    # UNDECIDABLE -> ``None`` (caller fails closed to tainted): a live plain
    # int-tuple is the data form, but the portable literal grammar erases
    # ``torch.Size`` to a plain tuple, so a decoded int-tuple is ambiguous.
    assert uninit_new_call_is_size_form(((2, 2),)) is None
    assert uninit_new_call_is_size_form((True,)) is None  # bool is not a size (raises live)
    assert uninit_new_call_is_size_form(("x",)) is None


# Kept: sole branch coverage of backends/torch/_completeness_origins.py; a sole mutation kill.
def test_python_method_surface_witness_recognition() -> None:
    """r55 hon_1: the capture-side recognition of the Python-``torch.Tensor``
    method surface consults the single ``utils/rng.py`` table block and applies
    the size-vs-data gate; undecidable forms fail closed to recognized."""

    from torchlens.backends.torch._completeness_origins import (
        _python_tensor_method_uninit_family_tail as tail_of,
    )

    assert tail_of("torch.Tensor", "Tensor.new", (64,)) == "new"
    assert tail_of("torch.Tensor", "Tensor.new", ()) == "new"
    assert tail_of("torch.Tensor", "Tensor.new", ([1.0, 2.0],)) is None  # data form
    assert tail_of("torch.Tensor", "Tensor.new", ((2, 2),)) == "new"  # undecidable: fail closed
    assert tail_of("torch.Tensor", "Tensor.empty_like", ()) == "empty_like"
    assert tail_of("torch.Tensor", "Tensor.resize_", ()) == "resize_"
    assert tail_of("my.custom.ns", "Tensor.new", (64,)) is None  # namespace-gated
    assert tail_of("torch.Tensor", "Tensor.add", ()) is None
    assert tail_of("torch.Tensor", None, (64,)) is None


# Kept: a sole mutation kill.
def test_new_size_form_redispatches_uninit_family_aten() -> None:
    """r55 hon_1 live-surface pin: ``Tensor.new(sizes)`` redispatches an aten
    uninit-family op (``aten.empty.memory_format``, probed), which is exactly
    how the dispatch-level origin ledger covers it transitively; the data form
    redispatches a NON-family op. A torch upgrade that changes the
    decomposition away from the family turns this RED instead of silently
    breaking the transitive-coverage claim."""

    from torch.utils._python_dispatch import TorchDispatchMode

    from torchlens.backends.torch._completeness_origins import _operator_uninit_family_tail

    calls: list[Any] = []

    class _Spy(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
        ) -> Any:
            calls.append(func)
            return func(*args, **(kwargs or {}))

    t = torch.randn(2)
    with _Spy():
        t.new(4)
    assert any(_operator_uninit_family_tail(func) is not None for func in calls)
    calls.clear()
    with _Spy():
        t.new([1.0, 2.0])
    assert not any(_operator_uninit_family_tail(func) is not None for func in calls)


class SeededRandModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand(x.shape[0], x.shape[1])
