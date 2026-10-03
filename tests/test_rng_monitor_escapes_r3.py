"""Round-3 RNG monitor closures: uuid1's C funnel, raw ``_thread`` spawns,
and bounded ``uncertain_detail`` growth.

Three hunt-b8 findings on ``utils/rng.py``:

* ``uuid.uuid1()`` escaped the seal completely on Linux -- the libuuid
  ``uuid._generate_time_safe`` C path (wall clock + clock-seq entropy + node)
  touches no monitored Python surface, so an in-window ``uuid1()`` reported
  ``channels=[]`` / ``uncertain=False``: a clean false-VERIFIED. The Python
  fallback path was already caught.
* Threads started IN-WINDOW via raw ``_thread.start_new_thread`` were never
  profile-hooked (``threading.setprofile`` rides only ``threading.Thread``'s
  bootstrap), so a raw thread drawing an externally-held numpy generator was
  unwitnessed -- outside the documented residual, which covers PRE-EXISTING
  threads only.
* ``_flag_uncertain`` grew ``uncertain_detail`` by a full tuple copy per call
  with no dedupe or cap -- measured O(N^2); a persistently-raising profiled
  object turned a capture into an effective hang.
"""

from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest
import torch
from torch import nn

from torchlens.utils.rng import (
    _UNCERTAIN_DETAIL_CAP,
    host_nondeterminism_monitor,
)


def test_uuid1_c_path_is_marked_and_restored() -> None:
    """In-window ``uuid.uuid1()`` marks its channel through the C funnel too."""

    if getattr(uuid, "_generate_time_safe", None) is None and (
        getattr(uuid, "_UuidCreate", None) is None
    ):
        pytest.skip("no platform C uuid1 funnel to exercise; fallback path is covered")
    pre_window_funnel = getattr(uuid, "_generate_time_safe", None)

    with host_nondeterminism_monitor(nn.Identity()) as result:
        uuid.uuid1()
    assert "uuid.uuid1" in result.channels, (
        "uuid.uuid1() drew wall-clock/entropy through the libuuid C path without "
        f"marking any channel: {sorted(result.channels)!r}"
    )
    # Exact restoration: the module attr holds the pre-window funnel again.
    assert getattr(uuid, "_generate_time_safe", None) is pre_window_funnel


@pytest.mark.smoke
def test_uuid4_still_marks_through_os_urandom() -> None:
    """Sibling pin: uuid4's os.urandom feed stays witnessed."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        uuid.uuid4()
    assert "os.urandom" in result.channels


def test_raw_thread_spawned_in_window_is_profile_hooked() -> None:
    """A raw ``_thread.start_new_thread`` thread's host draws are witnessed."""

    import _thread

    external_generator = np.random.default_rng(7)  # seeded OUTSIDE the window
    pre_window_spawn = _thread.start_new_thread
    done = threading.Event()

    def draw_from_raw_thread() -> None:
        """Draw from an externally-held generator, then signal completion."""

        try:
            external_generator.random()
        finally:
            done.set()

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _thread.start_new_thread(draw_from_raw_thread, ())
        assert done.wait(timeout=10.0), "raw thread never ran"
    assert result.channels, (
        "an in-window raw _thread.start_new_thread thread drew from an "
        "externally-held numpy generator with no channel marked: the spawn "
        "was never profile-hooked"
    )
    assert _thread.start_new_thread is pre_window_spawn, (
        "the _thread.start_new_thread patch did not restore"
    )


@pytest.mark.smoke
def test_threading_thread_in_window_draw_still_witnessed() -> None:
    """Sibling pin: the threading.Thread in-window positive control holds."""

    external_generator = np.random.default_rng(11)

    def draw() -> None:
        """Draw through a Python frame so the frame-digest belt can witness."""

        external_generator.random()

    with host_nondeterminism_monitor(nn.Identity()) as result:
        worker = threading.Thread(target=draw)
        worker.start()
        worker.join(timeout=10.0)
    assert result.channels


@pytest.mark.smoke
def test_flag_uncertain_detail_is_deduped_and_capped() -> None:
    """Per-event uncertainty reasons cannot grow the detail quadratically."""

    monitor = host_nondeterminism_monitor(nn.Identity())

    # Repeated identical reason: one retained entry, O(1) per repeat.
    for _ in range(10_000):
        monitor._flag_uncertain("profile_rng_state_read_failed:BrokenThing")
    assert monitor.result.uncertain is True
    assert monitor.result.uncertain_detail == ("profile_rng_state_read_failed:BrokenThing",)

    # Distinct reasons past the cap: retained set bounded, overflow disclosed.
    for index in range(10_000):
        monitor._flag_uncertain(f"reason:{index}")
    detail = monitor.result.uncertain_detail
    assert len(detail) <= _UNCERTAIN_DETAIL_CAP + 1
    assert detail[-1] == "uncertain_detail_capped"
    assert monitor.result.uncertain is True


def test_rng_state_digest_is_printoptions_independent() -> None:
    """The verdict-steering RandomState digest never rides np.set_printoptions.

    ``repr(RandomState.get_state())`` obeys the user-global ``threshold``
    (commonly small in notebooks), truncating the 624-word MT19937 key. The
    digest must be bytes-exact: identical under any display options, and
    distinct for two states differing only INSIDE the truncated region.
    """

    digest = host_nondeterminism_monitor._digest_rng_instance
    state = np.random.RandomState(3)
    saved_printoptions = np.get_printoptions()
    try:
        baseline = digest(state)
        np.set_printoptions(threshold=5)
        assert digest(state) == baseline

        # Two states differing only mid-key (the region repr truncates away).
        keys, pos = state.get_state()[1], state.get_state()[2]
        twin = np.random.RandomState(3)
        twin_key = keys.copy()
        twin_key[300] ^= 1
        twin.set_state(("MT19937", twin_key, pos, 0, 0.0))
        assert digest(twin) != digest(state)
    finally:
        np.set_printoptions(**saved_printoptions)


def test_flag_uncertain_hot_loop_is_fast() -> None:
    """1e5 repeated flags finish in well under a second (was O(N^2) copies)."""

    import time

    monitor = host_nondeterminism_monitor(nn.Identity())
    start = time.perf_counter()
    for _ in range(100_000):
        monitor._flag_uncertain("profile_classifier_error:X")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"_flag_uncertain hot loop took {elapsed:.2f}s for 1e5 calls"
    assert torch is not None


def test_random_subclass_draw_override_fail_closes_to_uncertain() -> None:
    """A user ``random.Random`` subclass overriding a draw method cannot read clean.

    The override escapes every witness at once: the class patches sit on the
    library base (shadowed), the ``c_call`` classifier never fires for a
    pure-Python method, and the inherited C-state digest does not advance when
    the override draws from its own attributes. Probe-proven FALSE-CLEAN
    (grind p5 §3.9): channels=() / uncertain=False on a genuinely
    nondeterministic-under-replay model. Possession now downgrades
    completeness with a detail naming the type and method.
    """

    import random

    import torchlens as tl
    from torchlens.options import CaptureOptions

    class _CounterRand(random.Random):
        """Draws from its own attribute; base MT19937 state never advances."""

        def __init__(self) -> None:
            super().__init__(0)
            self.counter = 0.0

        def random(self) -> float:
            self.counter += 0.125
            return self.counter % 1.0

    class _Model(nn.Module):
        """Scales its output by the unwitnessable subclass draw."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.rng = _CounterRand()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) * self.rng.random()

    capture = CaptureOptions(
        intervention_ready=True,
        capture_container_structure=True,
        cache=False,
        random_seed=7,
    )
    trace = tl.trace(_Model(), torch.randn(2, 4), capture=capture)
    assert trace._runnable.rng_monitor_uncertain is True
    assert any(
        detail.startswith("rng_subclass_override_unwitnessable:")
        and detail.endswith("._CounterRand.random")
        for detail in trace._runnable.rng_monitor_uncertain_detail
    ), trace._runnable.rng_monitor_uncertain_detail


def test_library_rng_subclasses_do_not_over_trigger() -> None:
    """Held library engines stay clean: the override check trusts library definers.

    ``SystemRandom`` (its draws ARE witnessed by the class patches) and a
    numpy ``default_rng`` Generator over a ``PCG64`` bit generator (its C
    state IS digested) must not ceiling a deterministic capture -- the
    no-over-trigger gate for the subclass-override fail-close.
    """

    import random

    import torchlens as tl
    from torchlens.options import CaptureOptions

    class _Plain(nn.Module):
        """Holds undrawn library engines only."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.base_rng = random.Random(3)
            self.sys_rng = random.SystemRandom()
            self.np_gen = np.random.default_rng(5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    capture = CaptureOptions(
        intervention_ready=True,
        capture_container_structure=True,
        cache=False,
        random_seed=7,
    )
    trace = tl.trace(_Plain(), torch.randn(2, 4), capture=capture)
    assert trace._runnable.rng_monitor_uncertain is False
    assert trace._runnable.rng_monitor_uncertain_detail == ()


def test_balanced_in_window_setprofile_swap_flags_uncertain() -> None:
    """grind-r4 b8 R57: a balanced in-window profile swap must not stay silent.

    User code that saves the monitor's hook, installs its own profile
    function, draws entropy, and restores the hook BEFORE window exit left no
    teardown evidence -- the slot held our hook at exit -- so the draws in
    the blind sub-window escaped uncertain=False (false-VERIFIED). The slot
    write itself is now the witness.
    """

    import sys

    with host_nondeterminism_monitor(nn.Identity()) as result:
        saved = sys.getprofile()
        sys.setprofile(None)  # the blind sub-window opens
        sys.setprofile(saved)  # balanced: our hook is back before exit
    assert result.uncertain is True, (
        "a balanced in-window sys.setprofile swap opened an unwitnessed "
        "sub-window with no uncertainty flag"
    )
    assert any(
        detail.startswith("profile_slot_swapped_in_window:sys.setprofile")
        for detail in result.uncertain_detail
    ), sorted(result.uncertain_detail)
    # Exact restoration: the module attr holds the real builtin again.
    assert sys.setprofile is not saved
    assert "torchlens" not in getattr(sys.setprofile, "__module__", "")


def test_balanced_threading_setprofile_swap_flags_uncertain() -> None:
    """Sibling slot: threading.setprofile blinds threads started after it."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        saved = threading._profile_hook if hasattr(threading, "_profile_hook") else None
        threading.setprofile(None)
        threading.setprofile(saved)
    assert result.uncertain is True
    assert any(
        detail.startswith("profile_slot_swapped_in_window:threading.setprofile")
        for detail in result.uncertain_detail
    ), sorted(result.uncertain_detail)


def test_swap_free_window_stays_certain() -> None:
    """Control: the monitor's own installs/restores never trip the detector."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        pass
    assert not any(
        detail.startswith("profile_slot_swapped_in_window") for detail in result.uncertain_detail
    ), sorted(result.uncertain_detail)


def test_raw_thread_hook_install_does_not_trip_swap_detector() -> None:
    """The in-window raw-thread hook install is monitor-internal: no flag."""

    import _thread

    done = threading.Event()

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _thread.start_new_thread(done.set, ())
        assert done.wait(timeout=10.0), "raw thread never ran"
    assert not any(
        detail.startswith("profile_slot_swapped_in_window") for detail in result.uncertain_detail
    ), sorted(result.uncertain_detail)


def test_in_window_thread_start_does_not_trip_swap_detector() -> None:
    """Thread._bootstrap_inner re-installs the window's own threading hook via
    sys.setprofile on every in-window thread start: machinery, never a swap."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join(timeout=10.0)
    assert not any(
        detail.startswith("profile_slot_swapped_in_window") for detail in result.uncertain_detail
    ), sorted(result.uncertain_detail)


def test_generator_spawn_is_witnessed_as_consumption() -> None:
    """``Generator.spawn()`` + child draw must not settle a clean window (R57).

    ``spawn()`` advances ONLY ``seed_seq.n_children_spawned`` -- hidden
    verdict-steering state that keys every future child's stream --
    while ``bit_generator.state`` stays untouched, so the state-only digest
    read ``channels=[] / uncertain=False`` and identical re-runs produced
    different outputs: a probe-proven false VERIFIED (r5 b8-fable HIGH).
    """

    class _SpawningModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gen = np.random.default_rng(1234)

    model = _SpawningModel()
    with host_nondeterminism_monitor(model) as result:
        child = model.gen.spawn(1)[0]
        child.standard_normal()
    assert "model_attribute_generator" in result.channels, (
        "Generator.spawn() escaped the seal: "
        f"channels={sorted(result.channels)!r} uncertain={result.uncertain}"
    )


@pytest.mark.smoke
def test_bitgenerator_and_seedsequence_spawn_are_witnessed() -> None:
    """The same-family escapes: ``BitGenerator.spawn`` and bare ``SeedSequence`` (R57)."""

    class _BitGenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bg = np.random.PCG64(7)

    model = _BitGenModel()
    with host_nondeterminism_monitor(model) as result:
        model.bg.spawn(1)
    assert "model_attribute_generator" in result.channels

    class _SeedSeqModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ss = np.random.SeedSequence(42)

    seed_model = _SeedSeqModel()
    with host_nondeterminism_monitor(seed_model) as seed_result:
        seed_model.ss.spawn(1)
    assert "model_attribute_generator" in seed_result.channels

    # Control: an un-drawn, un-spawned holder still settles clean.
    clean_model = _BitGenModel()
    with host_nondeterminism_monitor(clean_model) as clean_result:
        pass
    assert not clean_result.channels and not clean_result.uncertain


def test_held_implicit_now_converter_with_explicit_none_marks() -> None:
    """A held ``localtime(None)`` reads the clock NOW and must mark (R57).

    The held-ref decision decoded the positional argcount only, which is
    value-blind: ``argcount 1 > time_arg_index 0`` classified the call as a
    pure transform while the explicit ``None`` argument means "read the
    current clock" -- channels=[] / uncertain=False, a false VERIFIED
    (r5 b8-fable MED). Only a literal non-None time proves a transform.
    """

    from time import ctime, localtime

    held_localtime = localtime
    held_ctime = ctime

    with host_nondeterminism_monitor(nn.Identity()) as result:
        held_localtime(None)
    assert "time.localtime" in result.channels

    # The common wrapper idiom: a defaulted variable that IS None at runtime.
    def fmt(ts=None):
        return held_ctime(ts)

    with host_nondeterminism_monitor(nn.Identity()) as idiom_result:
        fmt()
    assert "time.ctime" in idiom_result.channels

    # A literal explicit time stays a pure transform (no false ceiling).
    with host_nondeterminism_monitor(nn.Identity()) as literal_result:
        held_localtime(1234567890)
    assert "time.localtime" not in literal_result.channels

    # The bare implicit-now spelling still marks.
    with host_nondeterminism_monitor(nn.Identity()) as bare_result:
        held_localtime()
    assert "time.localtime" in bare_result.channels
