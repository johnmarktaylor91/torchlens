"""Round-5 RNG seal closures (grind-r5 b8 R57).

* ``Generator.spawn()`` advanced no digested state (it mutates only
  ``seed_seq._n_children_spawned``), so a spawn+child-draw on a digest-rooted
  model generator escaped with ``channels=[] / uncertain=False`` -- a clean
  false-VERIFIED. Spawn state is now folded into every Generator/BitGenerator
  digest, and bare ``SeedSequence`` holders are digested too.
* A held pre-window alias of an implicit-now converter called with an
  explicit literal ``None`` (``localtime(None)``) read the clock but escaped
  the value-blind positional-count check unmarked.
* ``_skip_retired_hooks`` followed the restored SLOT's predecessor attr even
  when the dead link was the owner's OTHER hook, restoring the wrong chain.
"""

from __future__ import annotations

import pickle
import random
import time

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.utils.rng import _skip_retired_hooks, host_nondeterminism_monitor

_HELD_LOCALTIME = time.localtime  # pre-window held alias (module import time)


class _SpawningModel(nn.Module):
    """Model holding a digest-rooted generator that spawns mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = np.random.default_rng(1234)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_generator_spawn_changes_state_digest() -> None:
    """The digest must witness spawn(): it is verdict-steering hidden state."""

    generator = np.random.default_rng(5)
    before = host_nondeterminism_monitor._digest_rng_instance(generator)
    generator.spawn(1)
    after = host_nondeterminism_monitor._digest_rng_instance(generator)
    assert before != after, "Generator.spawn() left the state digest unchanged"


def test_bit_generator_spawn_changes_state_digest() -> None:
    """Same seal for a bare model-held BitGenerator."""

    bit_generator = np.random.PCG64(7)
    before = host_nondeterminism_monitor._digest_rng_instance(bit_generator)
    bit_generator.spawn(1)
    after = host_nondeterminism_monitor._digest_rng_instance(bit_generator)
    assert before != after, "BitGenerator.spawn() left the state digest unchanged"


def test_seed_sequence_holder_is_digestable_and_spawn_witnessed() -> None:
    """A bare model-held SeedSequence is a spawnable entropy root."""

    seed_seq = np.random.SeedSequence(42)
    before = host_nondeterminism_monitor._digest_rng_instance(seed_seq)
    seed_seq.spawn(1)
    after = host_nondeterminism_monitor._digest_rng_instance(seed_seq)
    assert before != after, "SeedSequence.spawn() left the state digest unchanged"


def test_model_held_generator_spawn_draw_is_witnessed_in_window() -> None:
    """The r5 probe scenario: spawn a child from a model-held generator and
    draw from it inside the window -- the window must NOT settle clean."""

    model = _SpawningModel()
    with host_nondeterminism_monitor(model) as result:
        child = model.gen.spawn(1)[0]
        child.standard_normal()
    assert result.channels or result.uncertain, (
        "spawn+child-draw on a digest-rooted model generator settled "
        "channels=[] / uncertain=False (false-VERIFIED escape)"
    )


def test_held_alias_localtime_none_is_marked() -> None:
    """``held_localtime(None)`` reads the clock exactly like ``held_localtime()``."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(None)
    assert any("localtime" in channel for channel in result.channels), (
        f"held localtime(None) escaped unmarked: {sorted(result.channels)!r}"
    )


def test_held_alias_localtime_literal_timestamp_stays_a_transform() -> None:
    """A provably non-None literal keeps the pure-transform classification."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(1700000000.0)
    assert not any("localtime" in channel for channel in result.channels)
    assert not result.uncertain


def test_held_alias_localtime_unresolvable_argument_flags_uncertainty() -> None:
    """An UNRESOLVABLE argument is runtime-dependent: neither a clock-draw
    claim nor a clean pass is provable, so the window flags uncertainty.

    A simple bound name resolves from the frame at ``c_call`` time and stays
    a pure transform (the fixwave-5 value-resolving decode; see
    test_held_alias_localtime_bound_variable_stays_a_transform), so the
    uncertainty lane is exercised by an attribute argument the bytecode
    walk-back cannot resolve.
    """

    class _Holder:
        ts = 1700000000.0

    holder = _Holder()
    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(holder.ts)
    assert result.uncertain, (
        "an unresolvable explicit-time argument settled certain; the value "
        "could have been None at runtime"
    )


def test_held_alias_localtime_bound_variable_stays_a_transform() -> None:
    """A bound non-None local resolves at ``c_call`` time: pure transform.

    Nothing can rebind a simple name between its argument load and the call
    in the same thread, so the resolved value IS the value the converter
    received -- no over-ceiling and no uncertainty (the runnable-contract
    no-over-trigger pin exercises the same decode end-to-end).
    """

    timestamp = float(len("x")) * 1700000000.0
    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(timestamp)
    assert not any("localtime" in channel for channel in result.channels)
    assert not result.uncertain


@pytest.mark.smoke
def test_call_site_time_arg_proof_skips_interposed_call_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytecode walk-back must not mistake a call-bookkeeping opcode for
    the last argument instruction (grind-pyver R8).

    Python 3.11 alone splits a call into ``PRECALL`` (dispatch) followed by
    ``CALL`` (the invocation), with ``PRECALL`` sitting directly before
    ``CALL`` in the instruction stream -- after the real last argument
    instruction. A fixed-width slice ending at the ``CALL`` position then
    swallows ``PRECALL`` as if it were that argument, starving the decode of
    the actual ``LOAD_CONST(None)`` and misclassifying every held-alias call
    as monitor uncertainty ("unknown") instead of "now_read" -- the exact
    3.11-only failure this test pins directly against FABRICATED
    instructions, so it holds on every interpreter regardless of which
    Python actually ships ``PRECALL`` today or reintroduces a similar
    bookkeeping opcode tomorrow.
    """

    import torchlens.utils.rng as rng_module

    class _FakeInstruction:
        def __init__(self, offset: int, opname: str, argval: object = None) -> None:
            self.offset = offset
            self.opname = opname
            self.argval = argval

    class _FakeFrame:
        f_lasti = 100  # the CALL instruction's offset, below
        f_code = None  # unused by the stubbed get_instructions below
        f_locals: dict = {}
        f_globals: dict = {}

    # Mirrors the real 3.11 shape for ``held_localtime(None)``: a LOAD_GLOBAL
    # for the callable, a LOAD_CONST(None) argument, a PRECALL bookkeeping
    # opcode, then the CALL at f_lasti.
    fake_instructions = [
        _FakeInstruction(0, "LOAD_GLOBAL", "held_localtime"),
        _FakeInstruction(50, "LOAD_CONST", None),
        _FakeInstruction(75, "PRECALL"),
        _FakeInstruction(100, "CALL"),
    ]

    monkeypatch.setattr(
        rng_module._dis_module, "get_instructions", lambda code: iter(fake_instructions)
    )
    proof = rng_module._call_site_time_arg_proof(_FakeFrame(), argcount=1, time_arg_index=0)

    assert proof == "now_read", (
        f"PRECALL-interposed call site misclassified as {proof!r}; the walk-back "
        "must skip call-bookkeeping opcodes to find the real last argument"
    )


@pytest.mark.smoke
def test_skip_retired_hooks_follows_the_dead_links_own_chain() -> None:
    """A dead THREADING hook found while restoring the sys slot must resolve
    through the dead owner's threading predecessor, not the sys one."""

    class _DeadOwner:
        pass

    def dead_threading_hook(frame, event, arg):
        return None

    def sentinel_threading_predecessor(frame, event, arg):
        return None

    def sentinel_sys_predecessor(frame, event, arg):
        return None

    owner = _DeadOwner()
    owner._hooks_retired = True
    owner._threading_hook = dead_threading_hook
    owner._sys_hook = None
    owner._previous_threading_profile = sentinel_threading_predecessor
    owner._previous_sys_profile = sentinel_sys_predecessor
    dead_threading_hook._tl_owner = owner

    resolved = _skip_retired_hooks(dead_threading_hook, "_previous_sys_profile")
    assert resolved is sentinel_threading_predecessor, (
        "restoring the sys slot through a dead threading hook resolved the "
        "dead owner's SYS predecessor instead of its threading chain"
    )


class _TinyNet(nn.Module):
    """Minimal deterministic model for the restore-half assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _RaisesInForward(nn.Module):
    """Model whose forward raises after consuming RNG-free work."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected forward failure for the RNG restore path")


def _global_rng_fingerprint() -> tuple[bytes, bytes, bytes]:
    """Byte-exact snapshot of the three global engines' states."""

    return (
        pickle.dumps(random.getstate()),
        pickle.dumps(np.random.get_state()),
        bytes(torch.random.get_rng_state().tolist()),
    )


def test_capture_restores_user_global_rng_streams() -> None:
    """``tl.trace`` must not leave the process reseeded (r7 restore half).

    Capture seeding reseeds ``random``/``numpy``/``torch`` at entry; the
    refresh and fast-run siblings snapshot and restore around their reseeds,
    but the primary capture left the user's global streams permanently on the
    capture's stream -- every post-capture ``randn``/``randint`` in user code
    silently changed meaning. With an explicit seed the capture must restore
    all three engines byte-exactly.
    """

    model = _TinyNet()  # parameter init draws torch RNG; construct first
    random.seed(20260816)
    np.random.seed(4711)
    torch.manual_seed(99)
    before = _global_rng_fingerprint()
    tl.trace(model, torch.ones(2, 3), capture=tl.options.CaptureOptions(random_seed=1234))
    assert _global_rng_fingerprint() == before, (
        "capture left the user's global RNG engines reseeded"
    )


def test_failed_capture_restores_user_global_rng_streams() -> None:
    """A capture failing mid-forward restores the streams on the unwind."""

    model = _RaisesInForward()
    random.seed(313)
    np.random.seed(626)
    torch.manual_seed(939)
    before = _global_rng_fingerprint()
    with pytest.raises(RuntimeError, match="injected forward failure"):
        tl.trace(model, torch.ones(2, 3), capture=tl.options.CaptureOptions(random_seed=77))
    assert _global_rng_fingerprint() == before, (
        "failed capture leaked the seeded RNG engines to the user"
    )


def test_auto_seed_freshness_survives_the_restore() -> None:
    """Auto-seeded captures still draw FRESH seeds after the restore.

    The seed pick (``random.randint``) deliberately stays OUTSIDE the restore
    bracket: restoring the pick too would make every ``random_seed=None``
    capture reuse the identical seed, silently correlating dropout patterns
    across runs.
    """

    random.seed(555)
    first = tl.trace(_TinyNet(), torch.ones(2, 3))
    second = tl.trace(_TinyNet(), torch.ones(2, 3))
    assert first.random_seed != second.random_seed, (
        "restore bracket swallowed the auto-seed draw; captures now reuse one seed"
    )


def test_torch_generator_draw_changes_state_digest() -> None:
    """A model-held ``torch.Generator`` is digestable like the numpy analog.

    r7 b8-sol: the digest raised ``_NotADigestableRng`` for torch.Generator,
    so a model-held instance drawn on a pre-existing (non-hooked) thread
    advanced state with NO witness while ``np.random.default_rng`` analogs
    were digest-caught -- and the residual enumeration claimed the residual
    was "only an EXTERNALLY-HELD generator".
    """

    generator = torch.Generator()
    generator.manual_seed(7)
    before = host_nondeterminism_monitor._digest_rng_instance(generator)
    torch.randn(4, generator=generator)
    after = host_nondeterminism_monitor._digest_rng_instance(generator)
    assert before != after, "torch.Generator draw left the state digest unchanged"


@pytest.mark.smoke
def test_model_held_torch_generator_pre_existing_thread_draw_is_witnessed() -> None:
    """The exact sol scenario: pre-existing thread draws from a model-held
    ``torch.Generator`` mid-window -- the window must NOT settle clean."""

    import threading

    class _TorchGenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gen = torch.Generator()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = _TorchGenModel()
    start = threading.Event()
    done = threading.Event()

    def worker() -> None:
        start.wait(10.0)
        torch.randn(4, generator=model.gen)
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()  # pre-existing (never-hooked) thread
    try:
        with host_nondeterminism_monitor(model) as result:
            start.set()
            assert done.wait(10.0)
    finally:
        thread.join(10.0)
    assert result.channels or result.uncertain, (
        "model-held torch.Generator drawn on a pre-existing thread settled "
        "channels=[] / uncertain=False (false-VERIFIED escape)"
    )


def test_seeded_global_torch_draw_stays_clean() -> None:
    """The replayable global torch engine stays identity-exempt (no
    over-trigger): a seeded ``torch.randn`` model draw must not ceiling."""

    torch.manual_seed(3)
    with host_nondeterminism_monitor(nn.Identity()) as result:
        torch.randn(4)
    assert not any("torch" in channel.lower() for channel in result.channels), (
        f"seeded global torch draw over-triggered: {sorted(result.channels)!r}"
    )


def test_capture_is_rng_neutral_to_the_host_process() -> None:
    """grind-r6 b8 R57 (opus MED, measured): capture restores all engines.

    Capture entry seeds python random, NumPy, and torch (for a reproducible
    forward) and never restored them, so ONE instrumented forward inside a
    seeded evaluation loop silently diverged every subsequent host draw --
    dropout masks, augmentation, shuffling -- from the uninstrumented run.
    The runnable-transaction and fast-run paths already bracket the same
    seeding; the primary capture entry now does too. The reseed POLICY is
    untouched (queued RNG-reseed fork); only the leak is closed. The
    auto-seed pick itself draws from a PRIVATE entropy stream (never the
    user's global engines), so neutrality holds even with
    ``random_seed=None`` while consecutive captures still get fresh seeds.
    """

    import random as _random

    import numpy as _np

    import torchlens as tl

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(1, 4)

    torch.manual_seed(0)
    _np.random.seed(0)
    _random.seed(0)
    control = (torch.rand(1).item(), _np.random.rand(), _random.random())

    torch.manual_seed(0)
    _np.random.seed(0)
    _random.seed(0)
    tl.trace(model, x)
    after = (torch.rand(1).item(), _np.random.rand(), _random.random())

    assert after == control, (
        f"capture perturbed the host RNG engines: control={control} after={after}"
    )
