"""Tests for the TorchLens doctor utility."""

from __future__ import annotations

import importlib.metadata

import pytest
from packaging.requirements import Requirement

import torchlens as tl
from torchlens import _state
from torchlens._utils import _DOCTOR_EXCLUDED_EXTRAS
from torchlens._utils._torch_compat import get_torch_capability_snapshot
from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
from torchlens.backends.torch.wrappers import wrap_torch


def test_doctor_returns_sane_report() -> None:
    """Doctor returns structured checks and a printable report."""

    report = tl.report.doctor()
    assert report.checks
    names = {check.name for check in report.checks}
    assert {
        "pytorch",
        "runtime capabilities",
        "torch wrapper bindings",
        "cuda",
        "graphviz",
        "safetensors",
        "extras",
        "model fingerprint",
    } <= names
    assert all(check.status in {"PASS", "FAIL", "SKIP", "WARN"} for check in report.checks)
    text = report.show()
    assert "TorchLens doctor report" in text
    assert "pytorch" in text


def test_doctor_surfaces_every_runtime_capability() -> None:
    """Doctor capability surface stays in lockstep with defined capability flags.

    The row cell carries the grouped absences-first summary (bounded-reprs
    doctrine, sumfam wave-0 item 2); the full ``name=value`` dump is served by
    the detail accessor the row points at. Lockstep means: the accessor covers
    every defined flag, and the row itself names every absent flag plus the
    accessor spelling, so no capability can go silently invisible.
    """

    report = tl.report.doctor()
    row = next(check for check in report.checks if check.name == "runtime capabilities")
    expected = set(get_torch_capability_snapshot()) | set(get_tf_capability_snapshot())

    snapshot = tl._utils.capability_snapshot()
    assert set(snapshot) == expected
    assert report.capability_snapshot() == snapshot

    absent = sorted(name for name, available in snapshot.items() if not available)
    present = len(snapshot) - len(absent)
    assert row.detail.split(";")[0] == f"{present}/{len(snapshot)} capabilities present"
    for name in absent:
        assert name in row.detail
    assert "tl._utils.capability_snapshot()" in row.detail


def test_doctor_warns_on_stale_torch_wrapper_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor reports torch namespace attrs that point at original callables."""

    wrap_torch()
    original_relu = _state._decorated_to_orig[id(__import__("torch").relu)]
    monkeypatch.setattr("torch.relu", original_relu)

    report = tl.report.doctor()
    row = next(check for check in report.checks if check.name == "torch wrapper bindings")

    assert row.status == "WARN"
    assert "torch.relu" in row.detail


def test_declared_extra_probes_track_packaging_metadata() -> None:
    """Doctor extra probes must stay in lockstep with declared package extras."""

    distribution = importlib.metadata.distribution("torchlens")
    expected = {
        extra
        for extra in (distribution.metadata.get_all("Provides-Extra") or [])
        if extra not in _DOCTOR_EXCLUDED_EXTRAS
    }
    requirement_extras = set()
    for requirement_line in distribution.requires or ():
        requirement = Requirement(requirement_line)
        if requirement.marker is None:
            continue
        requirement_extras.update(tl._utils._extras_from_requirement_marker(requirement))

    probes = tl._utils._declared_extra_probes()

    assert set(probes) == expected
    assert requirement_extras <= set(probes)
    # the no-op [profiler] extra is gone (A12: the empty-extra class is closed), so
    # it is no longer a probe; the test_packaging_docs_a12 lockstep pins its absence.
    assert {"jax", "mlx", "paddle", "sae", "tensorflow", "tf", "tinygrad"} <= set(probes)


def test_capability_row_optional_absences_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """r-b4 R26-4: absent OPTIONAL features keep a healthy install at PASS.

    Interpreter-version surfaces (PEP 657), upstream-removed APIs, and
    not-installed optional backends are reported with their true value under
    ``optional_absent=`` but never drive WARN -- a permanent false alarm
    trains users to ignore the row.
    """

    from torchlens._utils import _probe_torch_capabilities, _torch_compat as tc

    for flag in ("HAS_CODE_POSITIONS", "HAS_CODE_QUALNAME", "HAS_NAMED_TENSOR_API"):
        monkeypatch.setattr(tc, flag, False)
    row = _probe_torch_capabilities()
    assert row.status == "PASS"
    assert "optional_absent=" in row.detail
    assert "missing=" not in row.detail


def test_capability_snapshot_force_probes_meta_item_and_mha_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic snapshot resolves the two cheap lazy probes it used to skip.

    ``get_meta_item_guard_support`` and ``get_mha_fastpath_switch_support``
    document that diagnostic surfaces force them; the snapshot did not, so the
    doctor read both as ``unprobed`` (and never as a real absence on torch
    2.1-2.2) on any install that had not yet consumed them.
    """

    from torchlens._utils import _torch_compat as tc

    for latch, flag in (
        ("_META_ITEM_GUARD_PROBED", "HAS_META_ITEM_GUARD"),
        ("_MHA_FASTPATH_SWITCH_PROBED", "HAS_MHA_FASTPATH_SWITCH"),
    ):
        monkeypatch.setattr(tc, latch, False)
        monkeypatch.setattr(tc, flag, False)
    snapshot = get_torch_capability_snapshot()
    assert tc._META_ITEM_GUARD_PROBED is True
    assert tc._MHA_FASTPATH_SWITCH_PROBED is True
    assert snapshot["HAS_META_ITEM_GUARD"] is tc._probe_meta_item_guard()
    assert snapshot["HAS_MHA_FASTPATH_SWITCH"] is tc._probe_mha_fastpath_switch()


def _row_segment(text: str, key: str) -> str:
    """Return one ``key=a,b`` group of a capability row (empty when absent)."""

    for part in text.split("; "):
        if part.startswith(key + "="):
            return part
    return ""


def test_compat_capability_row_reads_unprobed_lazy_flag_as_unprobed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tl.compat.report()`` classifies absences exactly like the doctor row.

    A lazily probed flag nothing has consumed yet (here the memory profile,
    which needs a live profiler session) is ``unprobed``, not ``missing``, so
    it must not turn a healthy install's row into a warning. The same flag
    after its probe ran and found nothing stays ``missing`` and still warns.
    """

    from torchlens._utils import _probe_torch_capabilities, _torch_compat as tc
    from torchlens.compat._report import _torch_capabilities_row

    monkeypatch.setattr(tc, "HAS_MEMORY_PROFILE", False)
    monkeypatch.setattr(tc, "_MEMORY_PROFILE_PROBED", False)
    compat_row = _torch_capabilities_row()
    doctor_row = _probe_torch_capabilities()
    assert "HAS_MEMORY_PROFILE" in _row_segment(compat_row.details, "unprobed")
    assert "HAS_MEMORY_PROFILE" not in _row_segment(compat_row.details, "missing")
    assert "HAS_MEMORY_PROFILE" in _row_segment(doctor_row.detail, "unprobed")

    monkeypatch.setattr(tc, "_MEMORY_PROFILE_PROBED", True)
    compat_row = _torch_capabilities_row()
    assert "HAS_MEMORY_PROFILE" in _row_segment(compat_row.details, "missing")
    assert compat_row.severity == "warning"
    assert _probe_torch_capabilities().status == "WARN"


def test_capability_row_pre_graphsafe_generator_flags_stay_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r7 R26 (opus b4 LM): a pre-graph-safe torch install is healthy, not degraded.

    The three ``torch.Generator`` method probes gate rows of the RNG
    ``GENERATOR_METHOD_TABLE``: a ``False`` DROPS the row, so there is no
    method left to monitor and nothing degrades. On the older half of the
    declared ``torch>=2.1`` matrix these flags are legitimately absent; they
    must report under ``optional_absent=`` and keep the doctor row at PASS
    instead of a permanent unactionable WARN.
    """

    from torchlens._utils import _probe_torch_capabilities, _torch_compat as tc

    for flag in (
        "HAS_GENERATOR_CLONE_STATE",
        "HAS_GENERATOR_GRAPHSAFE_GET_STATE",
        "HAS_GENERATOR_GRAPHSAFE_SET_STATE",
    ):
        monkeypatch.setattr(tc, flag, False)
    row = _probe_torch_capabilities()
    assert row.status == "PASS"
    assert "optional_absent=" in row.detail
    assert "missing=" not in row.detail


def test_capability_row_genuine_degradation_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine degradation (non-optional flag False) still drives WARN."""

    from torchlens._utils import _probe_torch_capabilities, _torch_compat as tc

    monkeypatch.setattr(tc, "HAS_VARIABLE_FUNCTIONS", False)
    row = _probe_torch_capabilities()
    assert row.status == "WARN"
    assert "missing=HAS_VARIABLE_FUNCTIONS" in row.detail


def test_tf_snapshot_empty_without_tensorflow() -> None:
    """r-b4 R26-4: a torch-only install merges NO TF flags into the snapshot."""

    import importlib.util

    from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot

    snapshot = get_tf_capability_snapshot()
    if importlib.util.find_spec("tensorflow") is None:
        assert snapshot == {}
    else:
        assert "HAS_TF_OP_CALLBACKS" in snapshot


def test_probe_graphviz_routes_through_the_bounded_group_killing_runner(monkeypatch):
    """The doctor dot probe uses the ONE spawn seam, not bare subprocess.run (R40).

    subprocess.run's timeout kills only the direct child, so a wedged ``dot``
    wrapper's grandchild survived the "bounded" probe for the life of the box
    (probe-proven, b6 sol HIGH). The shared runner tears down the whole
    process group.
    """

    import subprocess

    from torchlens._utils import _probe_graphviz, _subprocess as sp

    calls = {}

    def _fake_runner(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, "dot - graphviz version 9.0", "")

    monkeypatch.setattr(sp, "run_bounded_subprocess", _fake_runner)
    check = _probe_graphviz()
    assert calls["cmd"] == ["dot", "-V"]
    assert calls["kwargs"]["timeout"] == 5
    assert check.status == "PASS"


def test_bounded_subprocess_children_are_armed_to_die_with_the_parent():
    """Children carry PR_SET_PDEATHSIG=SIGKILL on glibc hosts (R40).

    Group teardown runs in the PARENT, so a hard parent SIGKILL left the
    session-leading renderer running with nothing to reap it (probe-proven,
    b6 sol HIGH). The kernel-side parent-death signal closes that hole for
    the direct child.
    """

    import signal
    import sys

    from torchlens._utils import _subprocess as sp

    if sp._PRCTL is None:
        pytest.skip("no glibc prctl on this host")

    probe = (
        "import ctypes; v = ctypes.c_int();"
        "ctypes.CDLL('libc.so.6').prctl(2, ctypes.byref(v), 0, 0, 0);"
        "print(v.value)"
    )
    completed = sp.run_bounded_subprocess([sys.executable, "-c", probe], timeout=30, text=True)
    assert completed.stdout.strip() == str(int(signal.SIGKILL))
