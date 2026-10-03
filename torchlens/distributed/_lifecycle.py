"""Arming, group-lifecycle wraps, restricted seeding, and seq counters.

``torchlens.distributed.arm()`` is the explicit process-start opt-in for
distributed capture. Arming:

1. derives and verifies the five-namespace collective recognizer (fail-closed
   arm-time census, :mod:`._recognizer`);
2. installs the group-lifecycle wraps IMMEDIATELY (no first-capture laziness):
   ``init_process_group`` / ``new_group`` / ``split_group`` assign lifetime
   ordinals at creation; ``destroy_process_group`` (and ``_abort_process_group``
   where present) are wrapped observationally;
3. stamps the install-epoch record -- ``armed_before_any_group`` when no
   process group has ever been observed in this process, else ``seeded``.

Group identity is ``group_uid = (membership_digest, lifetime_ordinal)``
(exactly two fields). Ordinals are TorchLens-owned, monotone, ever-created:
unobserved DESTRUCTION is harmless (a dead ordinal is simply retired), and
every rule here exists to make unobserved CREATION either impossible or
fail-closed. Registry seeding of groups that predate arming is RESTRICTED to
the provably unambiguous case: only ordinal 0, only when exactly one
same-membership group is alive and no churn of that membership was ever
observed; anything else refuses typed (``ambiguous_group_lifetime``).

SPMD programs that first-capture symmetrically may be armed lazily by capture
entry (:func:`maybe_auto_arm`); ``arm()`` before any group creation is the
REQUIRED spelling for MPMD programs and the only way a rank can be a COMPLETE
WITNESS in the merge-time pre-join lineage audit.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors._base import CompatibilityError
from ._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    InstallEpoch,
    membership_digest_for_ranks,
)
from ._recognizer import (
    CollectiveRecognizer,
    UncapturedCollectiveOpError,
    derive_collective_recognizer,
)

__all__ = [
    "AMBIGUOUS_GROUP_LIFETIME",
    "AmbiguousGroupLifetimeError",
    "ArmingRecord",
    "GroupIdentity",
    "arm",
    "armed_state",
    "auto_arm_degradation",
    "disarm",
    "is_armed",
    "maybe_auto_arm",
]

AMBIGUOUS_GROUP_LIFETIME = "ambiguous_group_lifetime"
"""Finding kind for refused local seeding of a group with ambiguous lifetime."""


class AmbiguousGroupLifetimeError(CompatibilityError, RuntimeError):
    """Raised when a pre-arming group's lifetime ordinal cannot be proven.

    Structured context on ``fields``: ``kind`` (always
    ``"ambiguous_group_lifetime"``), ``membership_digest``, and ``reason``.
    """


@dataclass(frozen=True)
class GroupIdentity:
    """Resolved lifetime identity of one process group on this rank."""

    membership_digest: str
    lifetime_ordinal: int
    ordinal_source: str
    global_ranks: tuple[int, ...]
    backend: str | None

    @property
    def group_uid(self) -> tuple[str, int]:
        """The exactly-two-field correlation group identity."""

        return (self.membership_digest, self.lifetime_ordinal)


@dataclass(frozen=True)
class ArmingRecord:
    """What ``arm()`` established, stamped on every boundary record."""

    install_epoch: InstallEpoch
    recognizer_snapshot: str
    source: str


@dataclass
class _ArmedState:
    """Process-global armed distributed-capture state."""

    arming: ArmingRecord
    recognizer: CollectiveRecognizer
    ledger: GroupLifecycleLedger = field(default_factory=GroupLifecycleLedger)
    identities: dict[int, GroupIdentity] = field(default_factory=dict)
    seq_counters: dict[tuple[str, int, str], int] = field(default_factory=dict)
    originals: dict[tuple[Any, str], Any] = field(default_factory=dict)
    broken: bool = False
    """Poison flag: a failed arm whose restore ALSO failed published this
    state only so ``disarm()`` can retry restoration. Some collective sites
    are wrapped and others pristine, so presenting it as armed would let a
    capture proceed while unwrapped collectives are silently omitted --
    exactly the fail-open arming exists to prevent. Every arming entry point
    refuses typed while this is set."""
    ledger_gaps: list[str] = field(default_factory=list)
    """Unenumerable lifecycle observations (round-3 R18 fail-closed). The
    ledger holds only PROVEN events, so a wrapped creation/destroy whose
    membership cannot be read is recorded here instead of being silently
    dropped. While non-empty, no new group identity is minted and restricted
    seeding refuses: the dropped event may have been a generation of ANY
    membership, so every later ordinal derivation is unprovable."""


_LOCK = threading.Lock()
_STATE: _ArmedState | None = None
_AUTO_ARM_DEGRADATION: str | None = None


def auto_arm_degradation() -> str | None:
    """Why the LAST lazy-arming attempt degraded to unarmed capture, if it did.

    ``None`` when the last :func:`maybe_auto_arm` call armed, found
    distributed provably not in play, or was never called. A non-``None``
    reason means the process may be issuing collectives that captures are NOT
    recording -- the durable, in-band answer to "why are my collective
    boundary nodes missing?" that a stderr warning alone cannot give.
    """

    return _AUTO_ARM_DEGRADATION


def is_armed() -> bool:
    """Whether the distributed opt-in is currently armed in this process."""

    return _STATE is not None


def armed_state() -> _ArmedState | None:
    """Return the live armed state, or ``None`` when unarmed."""

    return _STATE


def capture_armed_state() -> _ArmedState | None:
    """Return the armed state when it should observe a capture starting now.

    Arming is process-lifetime by design and survives ``destroy_process_group``:
    the group-lifecycle ledger must outlive every group so a re-initialized
    group gets a fresh lifetime ordinal, and a disarm-and-rearm after teardown
    would read the emptied group registry as "no history" and claim the
    ``armed_before_any_group`` epoch. But with no process group initialized no
    collective can run, so the per-capture machinery (plane-P dispatch
    observation, the funcol completion session) has nothing to observe; it
    stays dormant and the capture takes the unarmed zero-interference path. An
    armed process that tore its group down (or armed before creating one) must
    capture exactly like an unarmed one. A probe failure reads as initialized
    (fail closed to observing).

    Returns
    -------
    _ArmedState | None
        The live armed state when armed AND a process group is initialized,
        else ``None``.
    """

    state = _STATE
    if state is None:
        return None
    try:
        if not torch.distributed.is_initialized():
            return None
    except (AttributeError, RuntimeError):
        # A build without a usable c10d registry cannot prove "no group":
        # stay armed (fail closed).
        return state
    return state


def _refuse_if_broken(state: _ArmedState) -> None:
    """Refuse typed on a poisoned half-armed state (deep-hunt F9).

    A failed arm whose restore also failed leaves SOME collective sites
    wrapped and others pristine; proceeding "armed" would silently omit the
    unwrapped collectives from an armed capture. ``disarm()`` retries the
    restoration and clears the state.
    """

    if state.broken:
        raise CompatibilityError(
            "torchlens.distributed arming previously failed and its rollback "
            "could not restore every wrapped collective site; this process is "
            "half-armed and captures would silently omit unwrapped "
            "collectives. Call torchlens.distributed.disarm() to retry the "
            "restoration, then arm() again.",
            kind="distributed_arming_broken",
        )


def _dist() -> Any:
    """Return ``torch.distributed``, refusing typed when unavailable."""

    if not torch.distributed.is_available():
        raise CompatibilityError(
            "torchlens.distributed.arm() requires a torch build with "
            "distributed support (torch.distributed.is_available() is False).",
            kind="distributed_unavailable",
        )
    return torch.distributed


def _any_group_history() -> bool:
    """Whether any process group has been observed alive in this process.

    FAIL-CLOSED (round-3 R18): when this returns ``False``, ``arm()`` stamps
    ``armed_before_any_group`` -- the complete-witness epoch whose lineage
    vector the merge-time audit lets override other ranks' evidence. A probe
    failure or an unreadable private registry therefore reads as "history
    assumed" (``True``, demoting the claim to ``seeded``); only a successfully
    read, empty registry proves the negative.
    """

    dist = torch.distributed
    try:
        if not dist.is_available():
            return False
        if dist.is_initialized():
            return True
    except Exception:
        return True
    world = getattr(getattr(dist, "distributed_c10d", None), "_world", None)
    pg_map = getattr(world, "pg_map", None)
    if pg_map is None:
        # Private-registry drift: absence of the map is not proof of absence
        # of groups.
        return True
    return bool(pg_map)


def _group_global_ranks(group: Any) -> tuple[int, ...]:
    """Return a process group's member ranks in global numbering."""

    dist = torch.distributed
    return tuple(int(rank) for rank in dist.get_process_group_ranks(group))


def _group_backend_name(group: Any) -> str | None:
    """Best-effort backend name for a group; diagnostics only."""

    try:
        return str(torch.distributed.get_backend(group))
    except Exception:
        return None


def _group_display_name(group: Any) -> tuple[str | None, str | None]:
    """Return (group_name, name_scheme) diagnostics; never identity."""

    try:
        name = getattr(group, "group_name", None)
        if name is not None:
            return str(name), "pg.group_name"
    except Exception:
        pass
    return None, None


def _note_ledger_gap(state: _ArmedState, reason: str) -> None:
    """Record an unenumerable lifecycle observation and disclose it loudly.

    The ledger holds only PROVEN events, so a wrapped observation whose
    membership cannot be read cannot be appended -- but silently dropping it
    read as "no churn" downstream and let restricted seeding bless
    generation-0 claims this rank cannot prove (round-3 R18 fail-open). The
    gap instead poisons every later lifetime decision: no new identity is
    minted, restricted seeding refuses typed, and the warning names the
    consequence.
    """

    state.ledger_gaps.append(reason)
    warnings.warn(
        "torchlens.distributed observed a process-group lifecycle event whose "
        f"membership could not be enumerated: {reason}. Group-lifetime "
        "bookkeeping on this rank is unprovable from this point: groups "
        "created from now on refuse capture correlation typed "
        "(ambiguous_group_lifetime) and restricted seeding is disabled. "
        "Recover by re-arming in a fresh process whose groups all enumerate.",
        stacklevel=4,
    )


def _record_created_group(state: _ArmedState, group: Any) -> None:
    """Assign a wrapped-creation ordinal to a newly created group."""

    if group is None or not _is_member_group(group):
        return
    if state.ledger_gaps:
        # A prior enumeration gap means the dropped event may have been a
        # generation of ANY membership, so this creation's ordinal derivation
        # is unprovable. Mint nothing and append nothing: the group stays
        # unidentified and resolve_group_identity() refuses it typed instead
        # of fabricating lineage evidence on a possibly-stale count.
        return
    try:
        global_ranks = _group_global_ranks(group)
    except Exception as exc:
        _note_ledger_gap(
            state,
            f"wrapped creation not enumerable ({type(exc).__name__}: {exc})",
        )
        return
    digest = membership_digest_for_ranks(global_ranks)
    ordinal = state.ledger.next_ordinal(digest)
    name, scheme = _group_display_name(group)
    state.ledger.append(
        GroupLifecycleEvent(
            event_index=state.ledger.next_event_index(),
            kind="create",
            membership_digest=digest,
            ordinal=ordinal,
            ordinal_source="wrapped",
            install_epoch=state.arming.install_epoch,
            local_creation_index=state.ledger.creation_count(),
            group_name=name,
            name_scheme=scheme,
        )
    )
    state.identities[id(group)] = GroupIdentity(
        membership_digest=digest,
        lifetime_ordinal=ordinal,
        ordinal_source="wrapped",
        global_ranks=global_ranks,
        backend=_group_backend_name(group),
    )


def _is_member_group(group: Any) -> bool:
    """Whether ``group`` is a real ProcessGroup this rank belongs to."""

    dist = torch.distributed
    non_member = getattr(getattr(dist, "GroupMember", None), "NON_GROUP_MEMBER", object())
    return group is not None and group is not non_member


def _record_destroyed_group(state: _ArmedState, group: Any) -> None:
    """Observationally mark a group (or, for None, all groups) destroyed."""

    dist = torch.distributed
    targets: list[GroupIdentity] = []
    if group is None:
        # destroy_process_group(None) tears down the world and every subgroup.
        targets = list(state.identities.values())
        state.identities.clear()
    else:
        identity = state.identities.pop(id(group), None)
        if identity is None and _is_member_group(group):
            if state.ledger_gaps:
                # Same poisoning rule as creation: with a gap open, minting a
                # churn ordinal from the ledger's generation count could
                # collide with the dropped generation. Seeding is already
                # disabled by the gap, so the churn-guard purpose is served.
                return
            # A destroy of a group we never identified: try to at least record
            # the membership churn so restricted seeding refuses it later.
            try:
                ranks = _group_global_ranks(group)
            except Exception as exc:
                _note_ledger_gap(
                    state,
                    f"wrapped destroy not enumerable ({type(exc).__name__}: {exc})",
                )
                return
            identity = GroupIdentity(
                membership_digest=membership_digest_for_ranks(ranks),
                lifetime_ordinal=state.ledger.next_ordinal(membership_digest_for_ranks(ranks)),
                ordinal_source="wrapped",
                global_ranks=ranks,
                backend=None,
            )
        if identity is not None:
            targets = [identity]
    for identity in targets:
        state.ledger.append(
            GroupLifecycleEvent(
                event_index=state.ledger.next_event_index(),
                kind="destroy",
                membership_digest=identity.membership_digest,
                ordinal=identity.lifetime_ordinal,
                ordinal_source=identity.ordinal_source,  # type: ignore[arg-type]
                install_epoch=state.arming.install_epoch,
            )
        )
        # Bounded growth: a destroyed uid can never tick again (a recreated
        # group gets a NEW lifetime_ordinal, so its counters start fresh at 0
        # under a different key). Dropping the dead keys keeps seq_counters
        # sized to the LIVE group population in create/destroy-loop programs
        # instead of growing without bound; issued correlation keys are
        # unaffected (they were stamped at issue time).
        dead_uid = (identity.membership_digest, identity.lifetime_ordinal)
        for key in [k for k in state.seq_counters if (k[0], k[1]) == dead_uid]:
            del state.seq_counters[key]
    if group is None:
        return
    _ = dist  # narrow: dist retained for parity with the None branch above


_CREATE_WRAP_TARGETS = ("init_process_group", "new_group", "split_group")
_DESTROY_WRAP_TARGETS = ("destroy_process_group", "_abort_process_group")


def restore_wrapped_attr(module: Any, name: str, original: Any) -> None:
    """Restore one wrapped attribute WITHOUT clobbering a foreign patch.

    Every distributed teardown used to ``setattr`` the pristine original
    blindly: a third-party library (or user code) that patched the same
    attribute AFTER TorchLens wrapped it had its patch silently destroyed at
    disarm/rollback time. When the live attribute is not the TorchLens wrap
    recorded for ``original`` (marker + exact ``__wrapped__`` identity), the
    foreign patch is left in place with a warning -- the TorchLens shim
    underneath is an inert passthrough once the armed state is gone, so
    correctness never depended on peeling it.

    Raises whatever ``setattr`` raises when the restore itself fails.
    """

    current = getattr(module, name, None)
    if current is original:
        # Already pristine: an arm that failed between recording the original
        # and installing the wrap has nothing to peel.
        return
    ours = getattr(current, "__tl_distributed_wrap__", False) and (
        getattr(current, "__wrapped__", None) is original
    )
    if not ours:
        warnings.warn(
            f"torchlens.distributed left {getattr(module, '__name__', module)}."
            f"{name} untouched at teardown: the attribute was re-patched by a "
            "third party after TorchLens wrapped it, and restoring the "
            "pristine function would destroy that patch. The TorchLens shim "
            "beneath it is an inert passthrough while disarmed.",
            stacklevel=3,
        )
        return
    setattr(module, name, original)


def _patch_modules() -> list[Any]:
    """Modules whose lifecycle-function attributes are patched at arm time."""

    dist = torch.distributed
    modules = [dist]
    c10d = getattr(dist, "distributed_c10d", None)
    if c10d is not None:
        modules.append(c10d)
    device_mesh_mod = getattr(dist, "device_mesh", None)
    if device_mesh_mod is not None:
        modules.append(device_mesh_mod)
    return modules


def _install_lifecycle_wraps(state: _ArmedState) -> None:
    """Install creation/destroy wraps on every module holding a reference."""

    def make_create_wrap(original: Any, returns_group: bool) -> Any:
        """Build the group-creation wrap for one c10d entry point.

        ``returns_group`` distinguishes ``new_group``-style factories, which return
        the group, from ``init_process_group``, which returns ``None`` and leaves
        the world group as the default.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Create the group, then record its identity in the live armed state."""

            result = original(*args, **kwargs)
            group = result if returns_group else None
            if not returns_group:
                # init_process_group returns None; the world group is the
                # default group after it completes.
                try:
                    group = torch.distributed.group.WORLD
                except Exception:
                    group = None
            with _LOCK:
                if _STATE is state and id(group) not in state.identities:
                    _record_created_group(state, group)
            return result

        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        wrapped.__name__ = getattr(original, "__name__", "wrapped")
        wrapped.__tl_distributed_wrap__ = True  # type: ignore[attr-defined]
        return wrapped

    def make_destroy_wrap(original: Any) -> Any:
        """Build the group-destruction wrap for one c10d entry point."""

        def wrapped(group: Any = None, *args: Any, **kwargs: Any) -> Any:
            """Delegate destruction, then record it only after backend success."""

            result = original(group, *args, **kwargs)
            with _LOCK:
                if _STATE is state:
                    resolved = group
                    if resolved is not None and not _is_member_group(resolved):
                        resolved = None
                    _record_destroyed_group(state, resolved)
            return result

        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        wrapped.__name__ = getattr(original, "__name__", "wrapped")
        wrapped.__tl_distributed_wrap__ = True  # type: ignore[attr-defined]
        return wrapped

    for module in _patch_modules():
        for name in _CREATE_WRAP_TARGETS:
            original = getattr(module, name, None)
            if original is None or (module, name) in state.originals:
                continue
            state.originals[(module, name)] = original
            returns_group = name != "init_process_group"
            setattr(module, name, make_create_wrap(original, returns_group))
        for name in _DESTROY_WRAP_TARGETS:
            original = getattr(module, name, None)
            if original is None or (module, name) in state.originals:
                continue
            state.originals[(module, name)] = original
            setattr(module, name, make_destroy_wrap(original))


def arm() -> ArmingRecord:
    """Arm distributed capture for this process.

    Returns
    -------
    ArmingRecord
        The install-epoch record and recognizer snapshot established. Calling
        ``arm()`` again is idempotent and returns the original record.

    Raises
    ------
    UncapturedCollectiveOpError
        When the arm-time recognizer derivation refuses this torch runtime.
    CompatibilityError
        When torch has no distributed support.
    """

    return _arm(source="explicit")


def _arm(source: str) -> ArmingRecord:
    """Arm collective capture once per process and return the install-epoch record.

    Shared by the public :func:`arm` and by lazy arming at capture entry;
    ``source`` records which of the two installed. Idempotent: an already-armed
    process returns its original ``ArmingRecord`` untouched.
    """

    global _STATE
    with _LOCK:
        if _STATE is not None:
            _refuse_if_broken(_STATE)
            return _STATE.arming
        _dist()
        recognizer = derive_collective_recognizer()
        epoch: InstallEpoch = "seeded" if _any_group_history() else "armed_before_any_group"
        arming = ArmingRecord(
            install_epoch=epoch,
            recognizer_snapshot=recognizer.snapshot_name,
            source=source,
        )
        state = _ArmedState(arming=arming, recognizer=recognizer)
        try:
            _install_lifecycle_wraps(state)
            from ..backends.torch.collectives import install_collective_wraps
            from ..backends.torch.funcol import install_funcol_wraps

            install_collective_wraps(state.originals)
            install_funcol_wraps(state.originals)
        except BaseException as arm_error:
            # Arming installs TWO independent wrap families before publishing ``_STATE``.
            # A failure (or a Ctrl-C) between them left unguarded wraps installed with
            # ``_STATE is None``, so nothing owned them: a later ``disarm()`` had no
            # record to restore from, and a re-arm wrapped the STALE WRAPS -- each failed
            # arm adding another passthrough layer that could never be peeled back to the
            # pristine functions. Restore whatever was recorded, then re-raise.
            restore_error: Exception | None = None
            for (module, name), original in list(state.originals.items()):
                try:
                    restore_wrapped_attr(module, name, original)
                except Exception as exc:
                    if restore_error is None:
                        restore_error = exc
                    continue
                state.originals.pop((module, name), None)
            if restore_error is not None:
                # Published ONLY so disarm() can retry restoration; poisoned
                # so no arming entry point presents the half-wrapped process
                # as armed (deep-hunt F9).
                state.broken = True
                _STATE = state
                raise restore_error from arm_error
            raise
        _STATE = state
        return arming


def maybe_auto_arm() -> ArmingRecord | None:
    """Lazily arm at capture entry when torch.distributed is initialized.

    Returns
    -------
    ArmingRecord | None
        The arming record when armed (newly or already), ``None`` when
        distributed is not in play or lazy arming was refused.

    Notes
    -----
    Explicit ``arm()`` raises on recognizer refusal; the lazy path degrades to
    unarmed capture instead, because refusing every capture in a process that
    merely initialized a process group would break previously-working dense
    captures that issue no collectives at all. Unarmed capture records no
    collective boundaries -- the pre-tier-(b) status quo. Each degraded
    capture entry warns (a once-per-process latch left every capture after
    the first with ZERO disclosure; Python's default warning filter still
    dedupes repeats for interactive users) and records its reason durably in
    :func:`auto_arm_degradation`.
    """

    global _AUTO_ARM_DEGRADATION
    # Already-armed fast path under the lock: the unlocked triple read of
    # _STATE raced disarm() (None between the check and the attribute read).
    with _LOCK:
        state = _STATE
        if state is not None:
            _refuse_if_broken(state)
            _AUTO_ARM_DEGRADATION = None
            return state.arming
    try:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            _AUTO_ARM_DEGRADATION = None
            return None
    except Exception as error:
        # Fail-open sibling of the arm-time history probe: a probe failure
        # must not SILENTLY skip arming in a process that may be issuing
        # collectives. Degrading to unarmed capture is the documented lazy
        # path, but it is disclosed on EVERY degraded capture entry, not
        # latched away after the first.
        _AUTO_ARM_DEGRADATION = (
            f"probe_failed: {type(error).__name__}: {error}; lazy arming was "
            "skipped and collective boundary nodes are NOT recorded"
        )
        warnings.warn(
            "torchlens could not probe torch.distributed state at capture "
            f"entry ({type(error).__name__}: {error}); lazy arming was "
            "skipped and collective boundary nodes will NOT be recorded.",
            stacklevel=3,
        )
        return None
    try:
        record = _arm(source="auto")
        _AUTO_ARM_DEGRADATION = None
        return record
    except UncapturedCollectiveOpError as error:
        _AUTO_ARM_DEGRADATION = (
            "uncaptured_collective_op: the arm-time recognizer refused this "
            f"torch runtime; collective boundary nodes are NOT recorded. {error}"
        )
        warnings.warn(
            "torchlens could not arm distributed collective capture on this "
            "torch runtime (uncaptured_collective_op); collective boundary "
            f"nodes will NOT be recorded. {error}",
            stacklevel=3,
        )
        return None


def disarm() -> None:
    """Remove the lifecycle wraps and drop all armed state.

    Primarily for tests; ordinary programs stay armed for process lifetime.
    """

    global _STATE, _AUTO_ARM_DEGRADATION
    with _LOCK:
        state = _STATE
        if state is None:
            _AUTO_ARM_DEGRADATION = None
            return
        first_failure: Exception | None = None
        for (module, name), original in list(state.originals.items()):
            try:
                restore_wrapped_attr(module, name, original)
            except Exception as exc:
                if first_failure is None:
                    first_failure = exc
                continue
            state.originals.pop((module, name), None)
        if first_failure is not None:
            raise first_failure
        _STATE = None
        _AUTO_ARM_DEGRADATION = None


def resolve_group_identity(group: Any) -> GroupIdentity:
    """Resolve a process group's lifetime identity, seeding if provably safe.

    Parameters
    ----------
    group:
        A live ProcessGroup (or ``None`` for the default/world group).

    Returns
    -------
    GroupIdentity
        The two-field ``group_uid`` identity plus diagnostics.

    Raises
    ------
    AmbiguousGroupLifetimeError
        When the group predates arming and restricted seeding cannot prove
        its lifetime ordinal (two or more same-membership groups alive, or
        observed churn of that membership).
    RuntimeError
        When called while unarmed.
    """

    dist = torch.distributed
    if group is None:
        group = dist.group.WORLD
    # Snapshot _STATE only INSIDE the lock: disarm() retires the armed state
    # under this same lock, so a pre-lock snapshot could seed identities and
    # ledger events into a RETIRED state after the public API already
    # reported the process unarmed.
    with _LOCK:
        state = _STATE
        if state is None:
            raise RuntimeError(
                "resolve_group_identity() requires torchlens.distributed to be armed"
            )
        identity = state.identities.get(id(group))
        if identity is not None:
            return identity
        return _seed_group_locked(state, group)


def _alive_same_membership_count(digest: str) -> int | None:
    """Count matching live groups, or return ``None`` when any membership is unreadable."""

    dist = torch.distributed
    world = getattr(getattr(dist, "distributed_c10d", None), "_world", None)
    pg_map = getattr(world, "pg_map", None)
    if pg_map is None:
        # Unreadable private registry (API drift) is not a provable zero:
        # the caller refuses instead of seeding on missing evidence.
        return None
    if not pg_map:
        return 0
    count = 0
    for candidate in list(pg_map):
        try:
            ranks = _group_global_ranks(candidate)
        except Exception:
            return None
        if membership_digest_for_ranks(ranks) == digest:
            count += 1
    return count


def _seed_group_locked(state: _ArmedState, group: Any) -> GroupIdentity:
    """Restricted registry seeding: only ordinal 0, only provably unambiguous."""

    try:
        global_ranks = _group_global_ranks(group)
    except Exception as exc:
        # A group whose own membership cannot be read has no provable
        # identity at all; refuse typed instead of leaking the raw torch
        # error mid-capture.
        raise AmbiguousGroupLifetimeError(
            "torchlens cannot read this process group's membership "
            f"(get_process_group_ranks failed: {type(exc).__name__}: {exc}), "
            "so its lifetime identity is unprovable.",
            kind=AMBIGUOUS_GROUP_LIFETIME,
            membership_digest=None,
            reason="membership_unenumerable",
        ) from exc
    digest = membership_digest_for_ranks(global_ranks)

    def refuse(reason: str) -> AmbiguousGroupLifetimeError:
        """Build the typed refusal for a pre-arming group whose lifetime is unprovable."""

        return AmbiguousGroupLifetimeError(
            "torchlens cannot assign a provable lifetime ordinal to a process "
            f"group created before arming: {reason}. Call "
            "torchlens.distributed.arm() at process start, before any process "
            "group is created.",
            kind=AMBIGUOUS_GROUP_LIFETIME,
            membership_digest=digest,
            reason=reason,
        )

    if state.ledger_gaps:
        raise refuse(
            "a group lifecycle event on this rank could not be enumerated "
            f"({state.ledger_gaps[0]}; {len(state.ledger_gaps)} gap(s) total), "
            "so no membership's generation-0 claim is provable"
        )
    churn = [event for event in state.ledger.events if event.membership_digest == digest]
    if churn:
        raise refuse(
            "lifecycle churn of this membership was already observed "
            f"({len(churn)} ledger event(s)), so an unwrapped group of the same "
            "membership cannot be generation 0"
        )
    alive = _alive_same_membership_count(digest)
    if alive is None:
        raise refuse(
            "the live process-group registry (or one of its entries' "
            "memberships) cannot be read, so the same-membership alive count "
            "is incomplete"
        )
    if alive > 1:
        raise refuse(
            f"{alive} live groups share this membership; the seed cannot prove "
            "which generation this group is"
        )
    name, scheme = _group_display_name(group)
    state.ledger.append(
        GroupLifecycleEvent(
            event_index=state.ledger.next_event_index(),
            kind="seed",
            membership_digest=digest,
            ordinal=0,
            ordinal_source="seeded",
            install_epoch=state.arming.install_epoch,
            local_creation_index=state.ledger.creation_count(),
            group_name=name,
            name_scheme=scheme,
        )
    )
    identity = GroupIdentity(
        membership_digest=digest,
        lifetime_ordinal=0,
        ordinal_source="seeded",
        global_ranks=global_ranks,
        backend=_group_backend_name(group),
    )
    state.identities[id(group)] = identity
    return identity


def next_seq(identity: GroupIdentity, channel: str) -> int:
    """Tick and return the issue-time seq counter for ``(group_uid, channel)``.

    Seq counters key on the FULL two-field ``group_uid``: a recreated
    communicator is a new uid and its counters start fresh at 0.
    """

    key = (identity.membership_digest, identity.lifetime_ordinal, channel)
    # Snapshot _STATE only INSIDE the lock (same atomicity contract as
    # resolve_group_identity): a pre-lock snapshot raced disarm() and ticked
    # the seq counters of a RETIRED state while reporting success.
    with _LOCK:
        state = _STATE
        if state is None:
            raise RuntimeError("next_seq() requires torchlens.distributed to be armed")
        value = state.seq_counters.get(key, 0)
        state.seq_counters[key] = value + 1
        return value
