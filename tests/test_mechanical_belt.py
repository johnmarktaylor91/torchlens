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


def _original_and_wrapper(name: str) -> tuple[Any, Any]:
    wrapper = getattr(torch, name)
    original = _state._decorated_to_orig[id(wrapper)]
    return original, wrapper


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/belt.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/belt.py, campaign mutant kills.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of compat/_report.py, report/_environment.py.
def test_probe_failure_reaches_compat_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """grind-r6 b3 R02 (sol MED): report.doctor() must carry the belt row."""

    from torchlens.report import doctor as compat_report

    monkeypatch.setattr(belt, "_report", _forged_failure_report())
    monkeypatch.setattr(belt, "_member_map", {})
    row = compat_report(torch.nn.Linear(2, 2), torch.randn(1, 2)).check("mechanical_belt")
    assert row.status == "scope"
    assert row.severity == "warning"
    assert row.is_detected is True
    assert "torch.frombuffer" in row.details


def test_clean_belt_reports_pass_not_false_alarm() -> None:
    """Healthy build: PASS rows that still DISCLOSE the unprobed count.

    Hundreds of in-place variants legitimately have no probe recipe on every
    healthy build; that standing limitation is disclosed as a count but must
    never flip the status (a permanent false alarm trains users to ignore
    the row -- r-b4 R26-4).
    """

    from torchlens.report import _environment, doctor as compat_report

    real = belt.belt_report()
    assert real is not None
    assert not real.probe_failures

    # Direct probe call: the full doctor() sweep (graphviz subprocess, extras
    # imports) is exercised by the failure test and is too slow to repeat here.
    doctor_row = _environment._probe_mechanical_belt()
    assert doctor_row.status == "PASS"
    assert "unprobed_candidates=" in doctor_row.detail

    compat_row = compat_report(torch.nn.Linear(2, 2), torch.randn(1, 2)).check("mechanical_belt")
    assert compat_row.status == "pass"
    assert compat_row.severity == "ok"
    assert "unprobed_candidates=" in compat_row.details


def _capture_from_file_probe_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the temp path every ``from_file`` probe recipe creates."""

    real_recipe = belt.PROBE_RECIPES[("torch", "from_file")]
    paths: list[str] = []

    def recording_recipe() -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = real_recipe()
        paths.append(str(args[0]))
        return args, kwargs

    monkeypatch.setitem(belt.PROBE_RECIPES, ("torch", "from_file"), recording_recipe)
    return paths


def _from_file_classification(report: belt.BeltReport) -> tuple[bool, bool, bool]:
    key = ("torch", "from_file")
    return (key in report.members, key in report.probed_visible, key in report.probe_failures)


def test_from_file_probe_releases_mapping_before_unlink(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe tensor is gone before its temp file is removed (Windows).

    Windows refuses to delete a file a live mapping still holds; the
    ``from_file`` result maps the probe file, so unlinking while it is alive
    raised ``PermissionError`` out of ``_derive`` and failed every
    ``tl.trace`` on Windows. The fake ``unlink`` reproduces that refusal
    whenever a probe result tensor is still alive.
    """

    import os
    import weakref

    expected = _from_file_classification(belt._derive()[0])
    paths = _capture_from_file_probe_paths(monkeypatch)
    live_results: list[weakref.ref[torch.Tensor]] = []
    real_touches = belt._touches_tensor

    def track(result: Any) -> None:
        if isinstance(result, torch.Tensor):
            live_results.append(weakref.ref(result))

    # The result is seen by the counting mode where the call dispatches
    # (mode-visible builds) and by the touches check where it does not.
    class RecordingMode(belt._CountingMode):
        def __torch_function__(
            self,
            func: Any,
            types_: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            result = super().__torch_function__(func, types_, args, kwargs)
            track(result)
            return result

    def spying_touches(result: Any, func: Any) -> bool:
        track(result)
        return real_touches(result, func)

    real_unlink = os.unlink

    def windows_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        if str(path) in paths and any(ref() is not None for ref in live_results):
            raise PermissionError(13, "simulated WinError 5: file is mapped", str(path))
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(belt, "_CountingMode", RecordingMode)
    monkeypatch.setattr(belt, "_touches_tensor", spying_touches)
    monkeypatch.setattr(belt, "_deferred_probe_paths", [])
    monkeypatch.setattr(os, "unlink", windows_unlink)

    report, _ = belt._derive()

    assert paths, "the from_file probe never ran"
    assert _from_file_classification(report) == expected
    assert belt._deferred_probe_paths == []
    assert not any(os.path.exists(path) for path in paths)


def test_trace_survives_refused_probe_file_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused removal defers cleanup; ``tl.trace`` and the report are unchanged.

    Even with the mapping released, Windows (or a scanner holding a handle)
    can refuse the first removal. The capture must still succeed, the probe's
    classification must not change, and the file must be removed on retry.
    """

    import os

    expected = _from_file_classification(belt._derive()[0])
    paths = _capture_from_file_probe_paths(monkeypatch)
    refused: list[str] = []
    real_unlink = os.unlink

    def refuse_first_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        if str(path) in paths and not refused:
            refused.append(str(path))
            raise PermissionError(13, "simulated WinError 32: file in use", str(path))
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(belt, "_deferred_probe_paths", [])
    monkeypatch.setattr(os, "unlink", refuse_first_unlink)
    # Force a fresh derivation inside the next capture, as on a first wrap.
    monkeypatch.setattr(belt, "_report", None)
    monkeypatch.setattr(belt, "_member_map", None)

    trace = tl.trace(torch.nn.Linear(2, 2), torch.randn(1, 2))

    assert trace is not None
    assert refused == paths[:1]
    report = belt._report
    assert report is not None
    assert _from_file_classification(report) == expected
    assert belt._deferred_probe_paths == refused
    assert os.path.exists(refused[0])
    belt._retry_deferred_probe_files()
    assert belt._deferred_probe_paths == []
    assert not os.path.exists(refused[0])
