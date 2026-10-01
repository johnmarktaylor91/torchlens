"""Pinned tests for the mechanical belt (protocol-invisible coverage).

The belt membership is DERIVED per build (never hand-listed): every wrapped
entry outside torch's override registries whose probe call fires zero
``TorchFunctionMode`` callbacks and touches tensors. On this build that is
exactly ``{torch.from_numpy, torch.from_dlpack, torch.frombuffer,
torch.Tensor.as_subclass, torch.Tensor._make_subclass}`` (``from_dlpack``
joined the wrap inventory with the 9bea6649 inventory-gap closure;
``_make_subclass`` got its probe recipe with the round-3 b6-fable carried
fix); ``torch.from_file`` measures VISIBLE here and must stay excluded (the
build-dependent case the mechanical derivation exists to settle).

NOTE: raw originals are held in function locals throughout — module-level or
``__main__``-level raw references get rewritten by the (pre-deletion)
crawler, which would silently vacuate these tests.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import belt

_EXPECTED_MEMBERS = {
    ("torch", "from_numpy"),
    ("torch", "from_dlpack"),
    ("torch", "frombuffer"),
    ("torch.Tensor", "as_subclass"),
    ("torch.Tensor", "_make_subclass"),
}


@pytest.fixture(autouse=True)
def _wrapped_torch() -> Iterator[None]:
    """Belt derivation needs installed wrappers."""
    tl.trace(torch.nn.Linear(2, 2), torch.randn(1, 2))
    yield


@pytest.mark.smoke
def test_belt_membership_is_derived_and_pinned() -> None:
    """The measured protocol-invisible set on this build, exactly."""

    report = belt.belt_report()
    assert report is not None
    assert set(report.members) == _EXPECTED_MEMBERS
    assert not report.probe_failures
    if hasattr(torch, "from_file"):
        # Build-dependent visibility: measured VISIBLE on this build, so the
        # net covers it and the belt must NOT claim it.
        assert ("torch", "from_file") in report.probed_visible
        assert ("torch", "from_file") not in report.members


@pytest.mark.smoke
def test_belt_members_fire_zero_mode_callbacks() -> None:
    """Independent re-measurement: no mode can see a belt member's call."""

    report = belt.belt_report()
    assert report is not None
    for entry in report.members:
        recipe = belt.PROBE_RECIPES[entry]
        namespace = belt._resolve_namespace(entry[0])
        current = getattr(namespace, entry[1])
        original = _state._decorated_to_orig.get(id(current), current)
        args, kwargs = recipe()
        mode = belt._CountingMode()
        with _state.pause_logging(), mode:
            original(*args, **kwargs)
        assert mode.calls == 0, f"{entry} fired the mode; it is not protocol-invisible"


def _original_and_wrapper(name: str) -> tuple[Any, Any]:
    wrapper = getattr(torch, name)
    original = _state._decorated_to_orig[id(wrapper)]
    return original, wrapper


@pytest.mark.smoke
def test_belt_sweep_patches_stale_module_ref_and_restores() -> None:
    """Module-attr stale refs to belt members round-trip through the ledger."""

    original, wrapper = _original_and_wrapper("from_numpy")
    mod = types.ModuleType("_tl_belt_sweep_check")
    mod.op = original
    sys.modules[mod.__name__] = mod
    try:
        patched = belt.sweep_stale_belt_references()
        assert patched >= 1
        assert mod.op is wrapper
        belt.restore_belt_references()
        assert mod.op is original
    finally:
        sys.modules.pop(mod.__name__, None)
        belt.restore_belt_references()


@pytest.mark.smoke
def test_belt_restore_preserves_user_reassignment() -> None:
    """Reversal is conditional: a slot the user rewrote is left alone."""

    original, wrapper = _original_and_wrapper("frombuffer")
    mod = types.ModuleType("_tl_belt_reassign_check")
    mod.op = original
    sys.modules[mod.__name__] = mod
    try:
        belt.sweep_stale_belt_references()
        assert mod.op is wrapper
        sentinel = object()
        mod.op = sentinel
        belt.restore_belt_references()
        assert mod.op is sentinel
    finally:
        sys.modules.pop(mod.__name__, None)
        belt.restore_belt_references()


@pytest.mark.smoke
def test_belt_sweep_is_epoch_incremental() -> None:
    """A module identity is scanned once; new imports are picked up later."""

    original, wrapper = _original_and_wrapper("from_numpy")
    belt.sweep_stale_belt_references()  # drain: everything live is now swept
    late = types.ModuleType("_tl_belt_late_import")
    late.op = original
    sys.modules[late.__name__] = late
    try:
        assert belt.sweep_stale_belt_references() >= 1
        assert late.op is wrapper
    finally:
        sys.modules.pop(late.__name__, None)
        belt.restore_belt_references()


@pytest.mark.smoke
def test_belt_sweep_prefilter_evicts_dead_module_ids() -> None:
    """The O(new) pre-filter never turns a reused id into a silent skip.

    A drained sweep returns 0 through the id-set fast path; a swept module's
    death evicts its id from the live set (weakref callback), so an unrelated
    later allocation reusing that id reads as NEW and gets scanned.
    """

    original, wrapper = _original_and_wrapper("from_numpy")
    belt.sweep_stale_belt_references()  # drain: everything live is now swept
    assert belt.sweep_stale_belt_references() == 0  # fast path: nothing new
    doomed = types.ModuleType("_tl_belt_doomed")
    sys.modules[doomed.__name__] = doomed
    try:
        belt.sweep_stale_belt_references()
        doomed_id = id(doomed)
        assert doomed_id in belt._swept_ids_live
        sys.modules.pop(doomed.__name__)
        del doomed
        assert doomed_id not in belt._swept_ids_live
        # A genuinely new module is still found after the eviction churn.
        late = types.ModuleType("_tl_belt_post_eviction")
        late.op = original
        sys.modules[late.__name__] = late
        try:
            assert belt.sweep_stale_belt_references() >= 1
            assert late.op is wrapper
        finally:
            sys.modules.pop(late.__name__, None)
    finally:
        sys.modules.pop("_tl_belt_doomed", None)
        belt.restore_belt_references()


@pytest.mark.smoke
def test_probe_rng_bracket_restores_global_seed() -> None:
    """The probe framework is RNG-neutral by construction (b8-fable R56).

    The candidate inventory is build-derived, so a state-mutating factory
    row entering it (``manual_seed`` already has a recipe that would call
    ``manual_seed(7)``) would silently clobber the user's global torch seed
    at first wrap inside the user's first capture. The bracket must restore
    the exact pre-probe state even when the probed call reseeds and draws.
    """

    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    with belt._probe_rng_bracket():
        torch.manual_seed(7)
        torch.rand(4)
    assert torch.equal(torch.random.get_rng_state(), before), (
        "probe bracket leaked RNG state: a state-mutating probe recipe would "
        "clobber the user's global seed"
    )


@pytest.mark.smoke
def test_belt_derivation_is_rng_neutral() -> None:
    """End-to-end: a full ``_derive()`` pass leaves the global RNG untouched."""

    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    belt._derive()
    assert torch.equal(torch.random.get_rng_state(), before)


def _forged_failure_report() -> belt.BeltReport:
    """A belt report carrying one probe failure, for disclosure-row tests."""

    return belt.BeltReport(
        members=(("torch", "from_numpy"),),
        probed_visible=(),
        probe_failures=(("torch", "frombuffer"),),
        unprobed_candidate_count=2,
        unprobed_candidates=(("torch", "abs_"), ("torch", "acos_")),
        probe_failure_details=(("torch", "frombuffer", "RuntimeError('probe exploded')"),),
    )


@pytest.mark.heavy  # doctor-row probe crossed the 7s smoke budget at T56 (8.0s cpu)
def test_probe_failure_reaches_doctor_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """grind-r6 b3 R02 (sol MED): doctor() must consume belt probe failures.

    A failed probe means the candidate is neither belt-patched nor proven
    mode-visible, so a stale pre-wrap reference to it drops ops with zero
    signal while ``capture_verified`` stays True. Before the fix NO doctor
    row consumed the belt report at all.
    """

    import torchlens.utils as tl_utils

    monkeypatch.setattr(belt, "_report", _forged_failure_report())
    monkeypatch.setattr(belt, "_member_map", {})
    rows = {check.name: check for check in tl_utils.doctor().checks}
    assert "mechanical belt" in rows, "no doctor row consumes the belt report"
    row = rows["mechanical belt"]
    assert row.status == "WARN"
    assert "torch.frombuffer" in row.detail
    assert "probe exploded" in row.detail
    assert "unprobed_candidates=2" in row.detail


@pytest.mark.smoke
def test_probe_failure_reaches_compat_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """grind-r6 b3 R02 (sol MED): compat.report() must carry the belt row."""

    from torchlens.compat import report as compat_report

    monkeypatch.setattr(belt, "_report", _forged_failure_report())
    monkeypatch.setattr(belt, "_member_map", {})
    row = compat_report(torch.nn.Linear(2, 2), torch.randn(1, 2)).row("mechanical_belt")
    assert row.status == "scope"
    assert row.severity == "warning"
    assert row.detected is True
    assert "torch.frombuffer" in row.details


@pytest.mark.smoke
def test_clean_belt_reports_pass_not_false_alarm() -> None:
    """Healthy build: PASS rows that still DISCLOSE the unprobed count.

    Hundreds of in-place variants legitimately have no probe recipe on every
    healthy build; that standing limitation is disclosed as a count but must
    never flip the status (a permanent false alarm trains users to ignore
    the row -- r-b4 R26-4).
    """

    import torchlens.utils as tl_utils
    from torchlens.compat import report as compat_report

    real = belt.belt_report()
    assert real is not None
    assert not real.probe_failures

    # Direct probe call: the full doctor() sweep (graphviz subprocess, extras
    # imports) is exercised by the failure test and is too slow to repeat here.
    doctor_row = tl_utils._probe_mechanical_belt()
    assert doctor_row.status == "PASS"
    assert "unprobed_candidates=" in doctor_row.detail

    compat_row = compat_report(torch.nn.Linear(2, 2), torch.randn(1, 2)).row("mechanical_belt")
    assert compat_row.status == "pass"
    assert compat_row.severity == "ok"
    assert "unprobed_candidates=" in compat_row.details
