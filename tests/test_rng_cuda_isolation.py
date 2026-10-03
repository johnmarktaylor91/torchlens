"""W11-F1 -- a CPU capture must not touch a visible-but-unusable CUDA device.

``log_current_rng_states`` used to snapshot CUDA RNG state whenever
``torch.cuda.is_available()`` was True, for every logged operation, regardless of
where the model and inputs actually lived.  ``torch.cuda.get_rng_state_all()``
reads one generator per visible device and torch documents it as eagerly
initializing CUDA, so a pure-CPU capture on a host whose CUDA stack is visible
but unusable (stale driver, mismatched build, one bad device in a multi-GPU box)
aborted with the driver error instead of completing on the CPU.  That is the bug
that forced ``CUDA_VISIBLE_DEVICES=""`` on every command in this repo.

The fix (``torchlens/utils/rng.py::_snapshot_cuda_rng_states``) skips the CUDA
snapshot entirely until the process has actually initialized CUDA, and degrades
to a warning if the read fails anyway.  Real CUDA captures are unaffected: a
capture that touches CUDA has initialized it by definition.

Subprocesses are the only faithful test: ``_is_cuda_available()`` caches its
probe per process, the "CUDA RNG unusable" latch is process-global, and the
simulation replaces ``torch.cuda`` entry points before ``import torchlens``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from torchlens.utils import rng as tl_rng, tensor_utils

# Markers are additive: a file-level smoke pytestmark would keep the heavy tests
# in the `-m smoke` tier, so tier marks are applied per test instead.


# ======================================================================================
# Subprocess simulations of a visible-but-unusable CUDA device
# ======================================================================================

_CPU_CAPTURE_WITH_UNUSABLE_VISIBLE_DEVICE = textwrap.dedent(
    """
    import sys
    import torch

    # Simulate a host that reports two visible CUDA devices whose runtime cannot
    # actually be initialized -- exactly the "visible but unusable" shape.
    touched = []

    def _make_exploder(name):
        # A DISTINCT function object per patched name: torch._dynamo's trace-rule
        # map asserts on one function object bound to several torch entry points.
        def _explode(*args, **kwargs):
            touched.append(name)
            raise RuntimeError("simulated incompatible visible CUDA driver")

        return _explode

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    for _name in ("_lazy_init", "get_rng_state_all", "get_rng_state", "set_rng_state_all"):
        setattr(torch.cuda, _name, _make_exploder(_name))
    # NOTE: is_initialized() is left at its real value (False) -- nothing in this
    # process has touched CUDA. torch's own empty_cache() self-guards on it, so the
    # other opportunistic CUDA callsites stay no-ops without any stubbing.

    import torch.nn as nn
    import torchlens as tl

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            # Dropout consumes RNG, so per-op RNG snapshots are genuinely taken.
            return self.drop(torch.relu(self.fc(x)))

    log = tl.trace(M().train(), torch.randn(2, 4))
    assert len(list(log)) > 0, "no ops captured"
    assert not touched, f"CPU capture reached into CUDA: {touched}"
    print("OK", len(list(log)))
    """
)


_CPU_CAPTURE_WITH_FAILING_INITIALIZED_CUDA = textwrap.dedent(
    """
    import sys
    import torch

    # Harsher shape: CUDA reports itself initialized, but reading generator state
    # raises. The capture must still finish, with a warning, not abort.
    reads = []

    def _explode_read():
        reads.append("get_rng_state_all")
        raise RuntimeError("simulated incompatible visible CUDA driver")

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    torch.cuda.is_initialized = lambda: True
    torch.cuda.get_rng_state_all = _explode_read
    # Stubbed only because is_initialized() is now a lie: torch's real
    # empty_cache() would dispatch into the (genuinely broken) allocator. The
    # RNG snapshot path is what is under test here.
    torch.cuda.empty_cache = lambda: None

    import torch.nn as nn
    import torchlens as tl

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            return self.drop(torch.relu(self.fc(x)))

    log = tl.trace(M().train(), torch.randn(2, 4))
    assert len(list(log)) > 0, "no ops captured"
    # Latched within the capture: the failing read is attempted only at the
    # capture-level snapshot seams (the rescue snapshot, then the capture
    # snapshot after set_random_seed re-arms the per-capture retry), never
    # once per logged operation. Dropout consumes RNG per op, so an unlatched
    # path would read once per op here.
    assert 1 <= len(reads) <= 2, f"CUDA RNG read not latched off: {reads}"
    assert len(list(log)) > len(reads), f"per-op CUDA RNG reads leaked: {reads}"
    print("OK", len(list(log)))
    """
)


def _run_child(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter and return the completed process.

    Parameters
    ----------
    script:
        Python source to execute.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured result, decoded as text.
    """

    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.heavy
def test_cpu_capture_ignores_unusable_visible_cuda_device() -> None:
    """A CPU capture completes without initializing a visible, unusable device."""

    completed = _run_child(_CPU_CAPTURE_WITH_UNUSABLE_VISIBLE_DEVICE)

    assert completed.returncode == 0, (
        "CPU capture aborted on a visible-but-unusable CUDA device:\n"
        f"STDOUT:{completed.stdout}\nSTDERR:{completed.stderr}"
    )
    assert "simulated incompatible visible CUDA driver" not in completed.stderr
    assert "OK" in completed.stdout


@pytest.mark.heavy
def test_cpu_capture_tolerates_failing_cuda_rng_read() -> None:
    """A failing CUDA generator read degrades to a warning, not a capture abort."""

    completed = _run_child(_CPU_CAPTURE_WITH_FAILING_INITIALIZED_CUDA)

    assert completed.returncode == 0, (
        "CPU capture aborted when the CUDA RNG read failed:\n"
        f"STDOUT:{completed.stdout}\nSTDERR:{completed.stderr}"
    )
    assert "Could not read CUDA RNG state" in completed.stderr
    assert "OK" in completed.stdout


# ======================================================================================
# In-process pins for the two helpers
# ======================================================================================


def test_snapshot_skips_cuda_until_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """An available-but-never-initialized CUDA runtime is never read."""

    def _forbidden() -> list[torch.Tensor]:
        """Fail if the snapshot reaches the eagerly-initializing torch API."""

        raise AssertionError("get_rng_state_all() must not run before CUDA init")

    monkeypatch.setattr(tl_rng, "_cuda_rng_unusable", False)
    monkeypatch.setattr(tl_rng, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", _forbidden)

    assert tl_rng._snapshot_cuda_rng_states() == []
    states = tl_rng.log_current_rng_states(torch_only=True)
    assert "torch_cuda_all" not in states
    assert "torch_cuda" not in states
    # A snapshot without CUDA keys must still round-trip through the restore path.
    tl_rng.set_rng_from_saved_states(states)


def test_initialized_cuda_snapshot_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With CUDA live, the snapshot dict is identical to the pre-fix contract.

    This is the pin for the real-CUDA path: same call
    (``torch.cuda.get_rng_state_all()``), same per-device list, same
    ``torch_cuda_all`` + ``torch_cuda`` keys. It cannot be run against real
    devices on this host, so the device layer is a fiction and the assertion is
    on the shape TorchLens builds from it.
    """

    fake_states = [torch.tensor([7], dtype=torch.uint8), torch.tensor([9], dtype=torch.uint8)]

    monkeypatch.setattr(tl_rng, "_cuda_rng_unusable", False)
    monkeypatch.setattr(tl_rng, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake_states)

    states = tl_rng.log_current_rng_states(torch_only=True)

    assert set(states) == {"torch", "torch_cuda_all", "torch_cuda"}
    assert states["torch_cuda_all"] is fake_states
    assert states["torch_cuda"] is fake_states[0]


@pytest.mark.smoke
def test_failing_cuda_rng_read_warns_latches_and_skips_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising CUDA generator read warns once and disables both directions."""

    calls: list[str] = []

    def _explode() -> list[torch.Tensor]:
        """Simulate a broken CUDA stack failing a host-side generator read."""

        calls.append("read")
        raise RuntimeError("simulated incompatible visible CUDA driver")

    def _forbidden_restore(states: object) -> None:
        """Fail if a latched-off CUDA RNG is written back during restore."""

        raise AssertionError("set_rng_state_all() must not run after a latched failure")

    monkeypatch.setattr(tl_rng, "_cuda_rng_unusable", False)
    monkeypatch.setattr(tl_rng, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", _explode)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", _forbidden_restore)

    with pytest.warns(UserWarning, match="Could not read CUDA RNG state"):
        assert tl_rng._snapshot_cuda_rng_states() == []

    # Latched: no second attempt, no second warning.
    assert tl_rng._snapshot_cuda_rng_states() == []
    assert calls == ["read"]
    # A snapshot taken before the latch must not be written back to a dead CUDA RNG.
    tl_rng.set_rng_from_saved_states({"torch": torch.random.get_rng_state(), "torch_cuda_all": []})


def test_transient_cuda_rng_failure_rearms_at_next_capture_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TRANSIENT CUDA RNG read failure degrades one capture, not the process.

    The latch used to be process-lifetime (and its inventory row said "must
    never be reset"), so one busy-device/OOM moment silently downgraded the
    replay fidelity of EVERY later capture (grind p5, B2P3-16). Every capture
    runs ``set_random_seed`` at entry, which now re-arms the retry; a
    recovered CUDA stack is snapshotted again.
    """

    attempts: list[str] = []
    healthy = [torch.tensor([7], dtype=torch.uint8)]

    def _transiently_broken() -> list[torch.Tensor]:
        """Fail the first read (transient), succeed afterwards."""

        attempts.append("read")
        if len(attempts) == 1:
            raise RuntimeError("simulated transient CUDA failure (busy device)")
        return healthy

    monkeypatch.setattr(tl_rng, "_cuda_rng_unusable", False)
    monkeypatch.setattr(tl_rng, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", _transiently_broken)

    with pytest.warns(UserWarning, match="Could not read CUDA RNG state"):
        assert tl_rng._snapshot_cuda_rng_states() == []
    # Within the same capture the latch holds: per-op cost stays bounded.
    assert tl_rng._snapshot_cuda_rng_states() == []
    assert attempts == ["read"]

    # The next capture entry (every capture seeds at entry) re-arms the retry
    # and the recovered stack is snapshotted again.
    tl_rng.set_random_seed(11)
    assert tl_rng._snapshot_cuda_rng_states() is healthy
    assert attempts == ["read", "read"]


def test_cuda_availability_probe_failure_degrades_to_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising ``torch.cuda.is_available()`` reads as "no CUDA", with a warning."""

    def _explode() -> bool:
        """Simulate a driver probe that raises instead of returning False."""

        raise RuntimeError("simulated incompatible visible CUDA driver")

    monkeypatch.setattr(tensor_utils, "_cuda_available", None)
    monkeypatch.setattr(torch.cuda, "is_available", _explode)

    with pytest.warns(UserWarning, match="torch.cuda.is_available"):
        assert tensor_utils._is_cuda_available() is False
    # Cached: the failing probe is not repeated.
    assert tensor_utils._is_cuda_available() is False


def test_cuda_initialized_probe_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_is_cuda_initialized`` tracks torch's flag instead of caching it."""

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert tensor_utils._is_cuda_initialized() is False
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    assert tensor_utils._is_cuda_initialized() is True


def test_seeding_degrades_when_accelerator_seed_leg_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising accelerator seed leg warns and still seeds the CPU engine.

    ``torch.manual_seed`` seeds every accelerator engine BEFORE the CPU default
    generator, so an abort from a broken CUDA stack (first observed on real
    H200 hardware as an ``IndexError`` from ``torch.cuda.default_generators``)
    used to escape ``set_random_seed`` with the CPU engine unseeded and kill
    the capture. The degrade path must leave the CPU generator in exactly the
    state a healthy ``torch.manual_seed`` would have produced.
    """

    torch.default_generator.manual_seed(1234)
    expected = torch.randn(4)

    def _explode(seed: int) -> None:
        """Simulate the CUDA seed leg raising from inside torch."""

        raise IndexError("tuple index out of range")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", _explode)

    with pytest.warns(UserWarning, match="Could not seed torch accelerator RNG"):
        tl_rng.set_random_seed(1234)
    assert torch.equal(torch.randn(4), expected)
