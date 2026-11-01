"""Exact state digests of witnessed RNG instances (NumPy generators, bit generators, ``random``)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]


from .inventory_types import (
    _NotADigestableRng,
)


def _exact_state_repr(state: Any) -> str:
    """Render an RNG state tree EXACTLY, independent of display options.

    ``repr`` of an ndarray obeys the user-global ``np.set_printoptions``
    ``threshold`` (commonly small in notebooks), truncating the 624-word
    MT19937 key to head/tail -- hanging a verdict-steering digest off a
    DISPLAY knob. Arrays render as ``(dtype, shape, tobytes)`` and
    containers recurse, so the digest is bytes-exact and
    printoptions-independent.
    """

    if isinstance(state, np.ndarray):
        return f"ndarray({state.dtype!s},{state.shape!r},{state.tobytes()!r})"
    if isinstance(state, dict):
        rendered = ",".join(f"{key!r}:{_exact_state_repr(value)}" for key, value in state.items())
        return "{" + rendered + "}"
    if isinstance(state, (tuple, list)):
        rendered = ",".join(_exact_state_repr(item) for item in state)
        return f"{type(state).__name__}({rendered})"
    return repr(state)


def _seed_seq_spawn_fragment(bit_generator: Any) -> str:
    """Digest the spawn-relevant SeedSequence state behind one BitGenerator.

    ``Generator.spawn()`` / ``BitGenerator.spawn()`` advance NO sampled
    state: they mutate ``seed_seq._n_children_spawned``, which is
    verdict-steering hidden state (a fresh oracle-1 run spawns a
    differently-keyed child). Folding the spawn counter plus the seeding
    identity into the digest makes an in-window spawn on a digest-rooted
    engine witnessable (grind-r5 b8 R57 HIGH). Absent/opaque seed
    sequences digest to the empty fragment (nothing spawnable to hide).
    """

    seed_seq = getattr(bit_generator, "seed_seq", None)
    if seed_seq is None:
        seed_seq = getattr(bit_generator, "_seed_seq", None)
    if seed_seq is None:
        return ""
    return _seed_seq_state_fragment(seed_seq)


def _seed_seq_state_fragment(seed_seq: Any) -> str:
    """Digest one SeedSequence's seeding identity and spawn counter."""

    return _exact_state_repr(
        (
            "seed_seq",
            getattr(seed_seq, "entropy", None),
            getattr(seed_seq, "spawn_key", None),
            getattr(seed_seq, "pool_size", None),
            getattr(seed_seq, "n_children_spawned", None),
        )
    )


def _digest_rng_instance(holder: Any) -> str:
    """Return a comparable state digest for one RNG holder.

    Covers numpy ``Generator``/``RandomState``/bare ``BitGenerator``,
    bare ``SeedSequence`` holders, ``torch.Generator`` (state bytes plus
    device identity), and ``random.Random``. Generator and
    BitGenerator digests fold in the underlying SeedSequence spawn state
    so ``spawn()`` -- which advances no sampled state -- is witnessed. A
    stateless ``Random`` subclass whose ``getstate()``
    raises ``NotImplementedError`` (``SystemRandom``) is classified
    monitored-not-digestible rather than an inventory error: possessing an
    undrawn stateless engine is not nondeterminism.
    """

    exact = _exact_state_repr
    spawn_fragment = _seed_seq_spawn_fragment
    if isinstance(holder, np.random.Generator):
        bit_generator = holder.bit_generator
        return exact(bit_generator.state) + spawn_fragment(bit_generator)
    if isinstance(holder, np.random.RandomState):
        return exact(holder.get_state())
    # r41 (review): a BARE model-held BitGenerator (``self.bg = PCG64(...)`` drawn
    # through a wrapping Generator) advances its own ``state``; digest it directly
    # so the registry's BitGenerator claim is digest-true.
    if isinstance(holder, np.random.BitGenerator):
        return exact(holder.state) + spawn_fragment(holder)
    # grind-r5 b8 R57: a model-held bare ``SeedSequence`` is a spawnable
    # entropy root; ``seed_seq.spawn()`` mid-window is the same hidden
    # verdict-steering mutation as ``Generator.spawn()``.
    if isinstance(holder, np.random.SeedSequence):
        return _seed_seq_state_fragment(holder)
    if isinstance(holder, torch.Generator):
        # r7 b8-sol R57: a model-held ``torch.Generator`` drawn on a
        # pre-existing (non-hooked) thread advanced state with NO witness
        # while the numpy analog was digest-caught, so the residual
        # enumeration's "only an EXTERNALLY-HELD generator" claim was
        # false. Digest the exact state bytes plus device identity so the
        # before/after sweeps witness any draw thread-independently. A
        # state-read failure propagates to the fail-closed inventory
        # error path, downgrading completeness rather than reading clean.
        state_bytes = holder.get_state().cpu().numpy().tobytes()
        return exact(("torch.Generator", str(holder.device), state_bytes))
    if isinstance(holder, random.Random):
        try:
            state = holder.getstate()
        except NotImplementedError:
            # r55 corr_1: ``random.SystemRandom`` (and any stateless ``Random``
            # subclass following its documented protocol) INTENTIONALLY has no
            # digestible state -- ``getstate()`` raises ``NotImplementedError``
            # by design. Possession of an UNDRAWN stateless engine is not
            # nondeterminism: classify monitored-not-digestible (the sweep then
            # walks it structurally like any holder) instead of letting the
            # generic inventory error path over-trigger
            # ``inventory_state_read_failed`` on a deterministic model. Actual
            # draws stay witnessed by the class-method patches on
            # ``random.SystemRandom.{random,getrandbits,randbytes}``. Any OTHER
            # exception from ``getstate()`` (a genuinely broken state read)
            # still propagates to the fail-closed inventory error path.
            raise _NotADigestableRng from None
        return _exact_state_repr(state)
    raise _NotADigestableRng
