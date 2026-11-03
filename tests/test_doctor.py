"""Tests for the TorchLens doctor utility."""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens import _state
from torchlens._utils._torch_compat import get_torch_capability_snapshot
from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.report import _environment


# Kept by shrink-r4 (per-id subset rule): sole coverage of _doctor_types.py.
def test_doctor_surfaces_every_runtime_capability() -> None:
    """Doctor capability surface stays in lockstep with defined capability flags.

    The row cell carries the grouped absences-first summary (bounded-reprs
    doctrine, sumfam wave-0 item 2); the full ``name=value`` dump is served by
    the detail accessor the row points at. Lockstep means: the accessor covers
    every defined flag, and the row itself names every absent flag plus the
    accessor spelling, so no capability can go silently invisible.
    """

    report = tl.report.doctor()
    row = next(check for check in report.checks if check.key == "torch_capabilities")
    expected = set(get_torch_capability_snapshot()) | set(get_tf_capability_snapshot())

    snapshot = _environment.capability_snapshot()
    assert set(snapshot) == expected
    assert report.capability_snapshot() == snapshot

    absent = sorted(name for name, available in snapshot.items() if not available)
    present = len(snapshot) - len(absent)
    assert row.details.split(";")[0] == f"{present}/{len(snapshot)} capabilities present"
    for name in absent:
        assert name in row.details
    assert "report.capability_snapshot()" in row.details


# Kept by shrink-r4 (per-id subset rule): sole coverage of report/_environment.py.
def test_doctor_warns_on_stale_torch_wrapper_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor reports torch namespace attrs that point at original callables."""

    wrap_torch()
    original_relu = _state._decorated_to_orig[id(__import__("torch").relu)]
    monkeypatch.setattr("torch.relu", original_relu)

    report = tl.report.doctor()
    row = next(check for check in report.checks if check.key == "torch wrapper bindings")

    assert row.status == "scope"
    assert "torch.relu" in row.details


def test_probe_graphviz_routes_through_the_bounded_group_killing_runner(monkeypatch):
    """The doctor dot probe uses the ONE spawn seam, not bare subprocess.run (R40).

    subprocess.run's timeout kills only the direct child, so a wedged ``dot``
    wrapper's grandchild survived the "bounded" probe for the life of the box
    (probe-proven, b6 sol HIGH). The shared runner tears down the whole
    process group.
    """

    import subprocess

    from torchlens._utils import _subprocess as sp
    from torchlens.report._environment import _probe_graphviz

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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _doctor_types.py.
def test_doctor_inspects_model_without_executing_it(capsys: pytest.CaptureFixture[str]) -> None:
    """One door combines environment/model rows without running forward."""

    import torch
    from torch import nn

    class NeverExecute(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("doctor must never execute the supplied model")

    report = tl.report.doctor(model=NeverExecute(), inputs=torch.ones(1))
    assert report.model_type.endswith("NeverExecute")
    assert report.check("pytorch").status == "pass"
    assert report.check("torch_capabilities") in report.checks
    assert report.check("quantized_tensor").status == "pass"
    assert len({row.key for row in report.checks}) == len(report.checks)
    capsys.readouterr()
    assert str(report) == report.show()
    assert capsys.readouterr().out == str(report) + "\n"
    assert "| Row | Status |" in report.to_markdown()
    with pytest.raises(KeyError, match="absent-key"):
        report.check("absent-key")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _utils/facade.py, compat/__init__.py, report/__init__.py.
def test_doctor_has_one_public_home() -> None:
    """Removed compat and utility spellings have no compatibility aliases."""

    from torchlens import _utils
    from torchlens.report import DoctorCheck, DoctorReport, doctor

    assert doctor is tl.report.doctor
    assert DoctorCheck is tl.report.DoctorCheck
    assert DoctorReport is tl.report.DoctorReport
    for name in ("report", "CompatReport", "CompatRow"):
        assert not hasattr(tl.compat, name)
        assert name not in tl.compat.__all__
    for name in ("doctor", "DoctorReport", "DoctorCheck"):
        assert not hasattr(_utils, name)
        assert name not in _utils.__all__
        assert not hasattr(getattr(tl, "utils", None), name)
