"""Transactionality of the host-nondeterminism monitor's arm and unwind (B3 L4).

The monitor mutates ~40 process-global attributes and both profile slots before it can
return. Every leak class below was live on main:

* a ``BaseException`` mid-arm (a Ctrl-C in the O(model-size) generator sweep) escaped the
  ``except Exception`` guard, so ``__exit__`` never ran and the patches plus the
  ``sys.setprofile`` hook stayed installed FOREVER -- and the next window snapshotted the
  leaked wrapper as its own "original", stacking the leak monotonically;
* one raising install surface aborted every LATER surface, silently costing the whole
  profile belt;
* re-arming the same instance recorded its own patches as the "prior" state;
* a second ``__exit__`` (an ``ExitStack`` double-close) clobbered a later window's hook;
* a non-LIFO overlap destroyed the process profile hook and stranded the true originals;
* threads started in-window kept the hook (and the model graph) as their profile function
  forever, misclassifying the NEXT capture's draws into this dead window's result.

These are witness-integrity defects, not cosmetics: a dead monitor's marks land in a
discarded result, so a real host draw reads as no consumption.
"""

import os
import random
import sys
import threading
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.utils import rng as rng_module
from torchlens.utils._torch_compat import autocast_is_enabled
from torchlens.utils.rng import AutocastRestore, host_nondeterminism_monitor

# Surfaces the monitor patches that are NOT also decorated by ``wrap_torch`` (the torch
# RNG capture wrappers legitimately outlive a capture until ``unwrap_torch()``).
_WATCHED_SURFACES: tuple[tuple[object, str], ...] = (
    (time, "time"),
    (time, "monotonic"),
    (os, "urandom"),
    (np.random, "default_rng"),
    (random.Random, "random"),
)


def _surface_snapshot() -> dict[tuple[str, str], object]:
    """Capture the identity of every watched process-global surface."""

    return {
        (getattr(holder, "__name__", str(holder)), name): getattr(holder, name)
        for holder, name in _WATCHED_SURFACES
    }


def _surface_leaks(before: dict[tuple[str, str], object]) -> list[tuple[str, str]]:
    """Return the watched surfaces whose identity did not survive the window."""

    return [key for key, original in before.items() if _surface_snapshot()[key] is not original]


@pytest.fixture(autouse=True)
def _profile_slots_clean():
    """Fail loudly on any test that leaves a profile hook or patch stack behind."""

    yield
    sys.setprofile(None)
    threading.setprofile(None)
    rng_module._PATCH_STACKS.clear()


def _tiny_model() -> nn.Module:
    """Return a two-op model that is cheap to trace repeatedly."""

    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def test_base_exception_during_arm_leaks_nothing(monkeypatch):
    """A Ctrl-C in the generator sweep must unwind every patch and both hooks."""

    before = _surface_snapshot()

    def _interrupt(self):
        raise KeyboardInterrupt("injected mid-arm")

    monkeypatch.setattr(host_nondeterminism_monitor, "_sweep_model_generators", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        tl.trace(
            _tiny_model(),
            torch.randn(1, 4),
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )

    assert _surface_leaks(before) == []
    assert sys.getprofile() is None
    assert threading.getprofile() is None
    assert rng_module._ACTIVE_MONITOR is None


def test_base_exception_after_profile_install_leaks_nothing(monkeypatch):
    """Interrupting AFTER ``sys.setprofile`` must still hand the slot back."""

    before = _surface_snapshot()
    original_install = host_nondeterminism_monitor._install_profile_hooks

    def _install_then_interrupt(self):
        original_install(self)
        raise KeyboardInterrupt("injected post-profile-install")

    monkeypatch.setattr(
        host_nondeterminism_monitor, "_install_profile_hooks", _install_then_interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        tl.trace(
            _tiny_model(),
            torch.randn(1, 4),
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )

    assert _surface_leaks(before) == []
    assert sys.getprofile() is None


def test_one_failing_install_surface_does_not_disarm_later_surfaces(monkeypatch):
    """A raising model sweep must not cost the profile belt installed after it."""

    def _raise(self):
        raise RuntimeError("hostile model attribute")

    monkeypatch.setattr(host_nondeterminism_monitor, "_sweep_model_generators", _raise)
    monitor = host_nondeterminism_monitor(_tiny_model())
    with monitor as result:
        assert sys.getprofile() is monitor._sys_hook
    assert result.uncertain
    assert "monitor_install_failed:generator_belt" in result.uncertain_detail
    assert sys.getprofile() is None


def test_rearming_the_same_instance_is_refused():
    """A second ``__enter__`` must install nothing and degrade completeness."""

    monitor = host_nondeterminism_monitor(None)
    monitor.__enter__()
    try:
        first_hook = monitor._sys_hook
        monitor.__enter__()
        assert monitor._sys_hook is first_hook
        assert "monitor_reentered" in monitor.result.uncertain_detail
    finally:
        monitor.__exit__(None, None, None)


def test_second_exit_does_not_clobber_a_later_hook():
    """Teardown is idempotent: a double close must leave the live hook alone."""

    monitor = host_nondeterminism_monitor(None)
    with monitor:
        pass

    def _sentinel(frame, event, arg):
        return None

    sys.setprofile(_sentinel)
    monitor.__exit__(None, None, None)
    assert sys.getprofile() is _sentinel
    sys.setprofile(None)


@pytest.mark.smoke
def test_non_lifo_overlap_preserves_both_the_hook_and_the_originals():
    """Out-of-order unwind must not destroy the live hook or strand a wrapper."""

    before = _surface_snapshot()
    outer = host_nondeterminism_monitor(None)
    inner = host_nondeterminism_monitor(None)
    outer.__enter__()
    inner.__enter__()
    # Non-LIFO: the OUTER window exits first, while the inner one is still live.
    outer.__exit__(None, None, None)
    assert sys.getprofile() is inner._sys_hook
    inner.__exit__(None, None, None)

    # A torn-down hook handed back by a successor self-uninstalls on its first event.
    for _ in range(3):
        (lambda: None)()
    assert sys.getprofile() is None
    assert _surface_leaks(before) == []
    assert "monitor_overlap" in inner.result.uncertain_detail


def test_patch_stacks_do_not_grow_across_captures():
    """The splice registry must be empty again after every clean capture."""

    model = _tiny_model()
    for _ in range(3):
        tl.trace(
            model, torch.randn(1, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
        )
    assert rng_module._PATCH_STACKS == {}


def test_in_window_thread_hook_self_uninstalls_after_teardown():
    """A worker started in-window must not keep the dead window's hook forever."""

    observed: dict[str, object] = {}
    started = threading.Event()
    release = threading.Event()

    def _worker():
        observed["hook_in_window"] = sys.getprofile()
        started.set()
        release.wait(timeout=10)
        # First Python calls after teardown: the hook must retire itself.
        for _ in range(3):
            (lambda: None)()
        observed["hook_after"] = sys.getprofile()

    monitor = host_nondeterminism_monitor(None)
    with monitor:
        worker = threading.Thread(target=_worker)
        worker.start()
        assert started.wait(timeout=10)
    release.set()
    worker.join(timeout=10)

    # ``threading.setprofile`` seeds NEW threads, so the worker carries the threading
    # copy of the hook -- the one ``__exit__`` provably cannot reach.
    assert observed["hook_in_window"] is monitor._threading_hook
    assert observed["hook_after"] is None


def test_patch_restore_survives_a_raising_tamper_probe():
    """A holder whose attribute read raises must still get its original back."""

    class _HostileHolder:
        """Holder whose reads raise once the monitor probes it at teardown."""

        def __init__(self) -> None:
            self.value = "original"
            self.hostile = False

        def __getattribute__(self, name):
            if name == "value" and object.__getattribute__(self, "hostile"):
                raise RuntimeError("hostile read")
            return object.__getattribute__(self, name)

    holder = _HostileHolder()
    monitor = host_nondeterminism_monitor(None)
    monitor._patch_attr(holder, "value", "wrapper")
    holder.hostile = True
    monitor._teardown()
    holder.hostile = False
    assert holder.value == "original"


def test_suppress_self_marks_is_thread_local():
    """A probe on one thread must never suppress another thread's host marks."""

    monitor = host_nondeterminism_monitor(None)
    inside = threading.Event()
    release = threading.Event()
    seen: dict[str, int] = {}

    def _other_thread():
        seen["depth"] = monitor._suppress_self_marks
        monitor._mark("clock")
        release.set()

    with monitor._monitor_internal_probe():
        assert monitor._suppress_self_marks == 1
        worker = threading.Thread(target=_other_thread)
        worker.start()
        assert release.wait(timeout=10)
        worker.join(timeout=10)
        inside.set()
    assert inside.is_set()
    assert seen["depth"] == 0
    assert "clock" in monitor.result.channels
    assert monitor._suppress_self_marks == 0


def test_suppress_depth_never_rests_negative():
    """A lost update must not leave a truthy resting depth that drops every mark."""

    monitor = host_nondeterminism_monitor(None)
    monitor._suppress_state.depth = 0
    with monitor._monitor_internal_probe():
        # Simulate the interleaved decrement that used to store -1.
        monitor._suppress_state.depth = 0
    assert monitor._suppress_self_marks == 0
    monitor._mark("clock")
    assert "clock" in monitor.result.channels


def test_autocast_restore_unwinds_a_partial_enter(monkeypatch):
    """A failing second device entry must not leave the first autocast entered.

    The second entry is failed by monkeypatching the constructor rather than by naming an
    absent device: a real CUDA-less ``torch.amp.autocast("cuda", ...)`` warns before it
    raises, and this suite runs with warnings promoted to errors, which would test the
    warning rather than the unwind.
    """

    real_autocast = torch.amp.autocast
    calls = {"n": 0}

    def _second_entry_fails(*args, **kwargs):
        """Build a real cpu autocast first, then fail."""

        calls["n"] += 1
        if calls["n"] == 1:
            return real_autocast("cpu", enabled=True, dtype=torch.bfloat16)
        raise TypeError("injected second-device autocast failure")

    monkeypatch.setattr(torch.amp, "autocast", _second_entry_fails)
    assert not autocast_is_enabled("cpu")
    with pytest.raises(TypeError, match="injected second-device autocast failure"):
        with AutocastRestore(
            {
                "cpu": {"enabled": True, "dtype": torch.bfloat16},
                "second": {"enabled": True, "dtype": torch.bfloat16},
            }
        ):
            pass
    monkeypatch.undo()
    assert not autocast_is_enabled("cpu")
    assert calls["n"] == 2


def test_autocast_restore_exits_outer_contexts_when_inner_raises():
    """A raising innermost exit must not strand the outer contexts."""

    exited: list[str] = []

    class _Ctx:
        """Minimal autocast stand-in recording its own exit."""

        def __init__(self, name: str, raises: bool) -> None:
            self.name = name
            self.raises = raises

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            exited.append(self.name)
            if self.raises:
                raise RuntimeError(f"{self.name} exit failed")

    restore = AutocastRestore({})
    restore._contexts = [_Ctx("outer", False), _Ctx("inner", True)]
    with pytest.raises(RuntimeError):
        restore.__exit__(None, None, None)
    assert exited == ["inner", "outer"]
    assert restore._contexts == []


def test_rng_monitor_verdict_is_prestamped_fail_closed(monkeypatch):
    """An unreached verdict stamp must read as UNCERTAIN, never as a clean window."""

    import torchlens.capture.trace as capture_trace

    def _raise(*args, **kwargs):
        raise RuntimeError("injected after the monitor window")

    monkeypatch.setattr(capture_trace, "host_rng_advanced", _raise, raising=False)
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            _tiny_model(),
            torch.randn(1, 4),
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )
    partial = getattr(excinfo.value, "partial_log", None)
    assert partial is not None, "the failed capture must still expose its partial trace"
    runnable = partial.trace._runnable
    assert runnable.rng_monitor_uncertain is True
    assert runnable.rng_monitor_uncertain_detail == ("monitor_verdict_not_stamped",)
