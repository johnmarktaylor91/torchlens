"""Functional-collective (funcol) boundary capture + plane-W completion authority.

Merge-ranks rung C2, recording slice (design-merge-ranks-c v5, 5.2 planes S/W;
L8 census plan sec 1.1/2). The pinned wave-0 census finding this module closes:
``torch.distributed._functional_collectives`` calls were INVISIBLE to the
shipped C1 python-wrap capture -- a captured forward issuing
``funcol.all_reduce`` completed with no boundary record and only a
provenance-warning escape. Two cooperating layers close the gap:

* **Plane S (semantic issue records).** The public funcol entry points are
  wrapped at arm time exactly like the c10d python API
  (:mod:`.collectives`): each captured call becomes a first-class boundary op
  node emitted through the ordinary producer, ticks the issue-time seq counter
  on the same ``(group_uid, "coll")`` channel space (v5 1.2: implicit/funcol
  collectives are not nested inside a public c10d call and tick normally), and
  journals a portable payload into ``annotations["distributed"]["boundaries"]``
  beside the c10d entries. The payload schema is
  ``functional_collective_boundary_v0`` -- a DOCUMENTED-UNSTABLE sibling of the
  frozen ``collective_boundary_v1``, deliberately distinct so no consumer keyed
  on the frozen v1 vocabulary ever reads v0 fields, and the frozen C1 merge
  contract is untouched (the merge evidence extractor refuses v0-bearing cores
  typed; see ``merged/_evidence.py``). The funcol event mapping applies
  (v5 1.4b): issue IS launch, completion is the OBSERVED wait.

* **Plane W (mode-independent completion authority).** funcol returns
  :class:`~torch.distributed._functional_collectives.AsyncCollectiveTensor`
  (ACT), whose wait fires below every python mode (probe P2: a
  ``TorchDispatchMode`` provably cannot see an ACT-triggered ``wait_tensor``).
  A CAPTURE-SCOPED dispatcher interposition -- ``torch.library.Library
  ("_c10d_functional", "IMPL")`` at the CPU key, redispatching below itself
  (probe P3a) -- observes every ``wait_tensor`` regardless of mode-stack state
  and binds it to the pending boundary by destination identity. Honesty never
  depends on the interposition succeeding: an unobserved completion keeps
  ``completion_binding="unobserved"`` plus the typed disclosures; observation
  is a quality upgrade, not a soundness precondition (v5 1.4c).

ACT discipline (v5 1.4c, binding): an unwaited ACT is NEVER materialized by
this module. Metadata reads go through the plain inner ``.elem`` tensor; the
boundary node's logged copy is taken from the inner tensor pre-completion and
carries the ``read_of_inflight_destination`` disclosure; digest witnesses of
destinations are taken only at an OBSERVED completion, never forced.

Disclosed wave-1 residuals (each keeps the census red rather than lying):
direct ``torch.ops._c10d_functional.*`` invocations bypassing the python API
(DTensor internals included) get plane-P visibility but no boundary node; the
``legacy_*`` / ``*_inplace`` / ``batch_p2p_ops_inplace`` funcol surfaces are
unwrapped; pre-arm ``from ... import all_reduce`` references escape the wrap
(same class as the c10d wraps); the wait interposition registers the CPU
dispatch key only (CUDA rows are NOT-RUN on this box and stay honestly
``unobserved`` elsewhere); an ACT returned as a MODEL OUTPUT is materialized by
ordinary output extraction after the completion-authority window closes.
"""

from __future__ import annotations

import copy
import inspect
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

from ...errors._base import CompatibilityError
from .collectives import (
    COLLECTIVE_BOUNDARY_FASTLOG_UNSUPPORTED,
    _BoundaryScope,
    _c10d_group_seq,
    _digest_tensor,
    _emit_boundary_op,
    _inside_boundary,
    _journal_boundary,
    _role_entry,
)

__all__ = [
    "FUNCOL_BOUNDARY_SCHEMA",
    "FUNCOL_SITES",
    "FuncolSite",
    "distributed_recording_session",
    "install_funcol_wraps",
    "remove_funcol_wraps",
]

FUNCOL_BOUNDARY_SCHEMA = "functional_collective_boundary_v0"
"""Schema tag on funcol boundary payloads (DOCUMENTED-UNSTABLE, pre-S2)."""

_FUNCOL_MODULE = "torch.distributed._functional_collectives"

#: Completion-binding values inside the v0 payload (DOCUMENTED-UNSTABLE; the
#: frozen v1 vocabulary in ``merged/_evidence.py`` is deliberately NOT edited
#: -- new closed vocabulary rides an S2 amendment, never a build lane).
_FUNCOL_COMPLETION_OBSERVED = "observed_wait"
_FUNCOL_COMPLETION_UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class FuncolSite:
    """One wrapped public funcol entry point.

    Parameters
    ----------
    attr:
        Function name on ``torch.distributed._functional_collectives``.
    kind:
        Canonical collective kind recorded on the boundary.
    func_name:
        Sanitized TorchLens label type (drives ``op.type`` and labels).
    tensor_arg:
        Bound-argument name carrying this rank's contribution tensor(s).
    detail_args:
        Bound-argument names copied (JSON-safe) into the payload's ``detail``.
    """

    attr: str
    kind: str
    func_name: str
    tensor_arg: str
    detail_args: tuple[str, ...] = ()
    #: Journal-only flag for parity with c10d sites; funcol sites always
    #: return destination tensors, so every site emits an op node.
    tensorless: bool = False


FUNCOL_SITES: tuple[FuncolSite, ...] = (
    FuncolSite("all_reduce", "all_reduce", "funcolallreduce", "self", ("reduceOp",)),
    FuncolSite("all_gather_tensor", "all_gather", "funcolallgather", "self", ("gather_dim",)),
    FuncolSite("all_gather_single", "all_gather", "funcolallgathersingle", "self", ("gather_dim",)),
    FuncolSite(
        "reduce_scatter_tensor",
        "reduce_scatter",
        "funcolreducescatter",
        "self",
        ("reduceOp", "scatter_dim"),
    ),
    FuncolSite(
        "reduce_scatter_single",
        "reduce_scatter",
        "funcolreducescattersingle",
        "self",
        ("reduceOp", "scatter_dim"),
    ),
    FuncolSite("broadcast", "broadcast", "funcolbroadcast", "self", ("src",)),
    FuncolSite(
        "all_to_all_single",
        "all_to_all",
        "funcolalltoall",
        "self",
        ("output_split_sizes", "input_split_sizes"),
    ),
    FuncolSite("permute_tensor", "permute_tensor", "funcolpermute", "self", ("src_dst",)),
    FuncolSite(
        "all_reduce_coalesced",
        "all_reduce_coalesced",
        "funcolallreducecoalesced",
        "self",
        ("reduceOp",),
    ),
    FuncolSite(
        "all_gather_into_tensor_coalesced",
        "all_gather_coalesced",
        "funcolallgathercoalesced",
        "self",
    ),
    FuncolSite(
        "reduce_scatter_tensor_coalesced",
        "reduce_scatter_coalesced",
        "funcolreducescattercoalesced",
        "inputs",
        ("reduceOp", "scatter_dim"),
    ),
    FuncolSite(
        "all_gather_tensor_autograd",
        "all_gather",
        "funcolallgatherautograd",
        "self",
        ("gather_dim",),
    ),
    FuncolSite(
        "all_gather_single_autograd",
        "all_gather",
        "funcolallgathersingleautograd",
        "self",
        ("gather_dim",),
    ),
    FuncolSite(
        "reduce_scatter_tensor_autograd",
        "reduce_scatter",
        "funcolreducescatterautograd",
        "self",
        ("reduceOp", "scatter_dim"),
    ),
    FuncolSite(
        "reduce_scatter_single_autograd",
        "reduce_scatter",
        "funcolreducescattersingleautograd",
        "self",
        ("reduceOp", "scatter_dim"),
    ),
    FuncolSite(
        "all_to_all_single_autograd",
        "all_to_all",
        "funcolalltoallautograd",
        "self",
        ("output_split_sizes", "input_split_sizes"),
    ),
)


def _act_type() -> type[Any] | None:
    """Return the AsyncCollectiveTensor class, or ``None`` off-build.

    Routed through the compat chokepoint (lazy, ``sys.modules``-deferred);
    absence flips ``HAS_ASYNC_COLLECTIVE_TENSOR`` there.
    """

    from ...utils._torch_compat import get_async_collective_tensor_type

    return get_async_collective_tensor_type()


def _inner_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return the plain inner tensor of an ACT (metadata-safe), else the value.

    ``.elem`` is a wrapper attribute read; it never triggers the ACT wait.
    """

    act_cls = _act_type()
    if act_cls is not None and isinstance(value, act_cls):
        return value.elem
    return value


def _is_unwaited_act(value: torch.Tensor) -> bool:
    """Whether ``value`` is an ACT whose completion has not yet been triggered."""

    act_cls = _act_type()
    return act_cls is not None and isinstance(value, act_cls) and not bool(value.completed)


def _contribution_tensors(value: Any) -> list[torch.Tensor]:
    """Flatten a funcol contribution argument (tensor or tensor list)."""

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def _result_tensors(result: Any) -> list[torch.Tensor]:
    """Flatten a funcol return value (tensor or tensor list) preserving order."""

    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (list, tuple)):
        return [item for item in result if isinstance(item, torch.Tensor)]
    return []


def _jsonable_detail(value: Any) -> Any:
    """Coerce a funcol non-tensor argument into a JSON-safe payload value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_detail(item) for item in value]
    return str(value)


class _PendingFuncolCompletion:
    """One issued funcol boundary awaiting plane-W completion observation.

    Parameters
    ----------
    payload:
        The journaled boundary payload; its ``events`` / ``witness`` /
        ``disclosures`` sub-objects are shared with the trace journal entry,
        so in-place settlement propagates there by construction.
    destinations:
        Strong references to the INNER destination tensors, index-aligned with
        the payload's destination bookkeeping. Strong refs pin object identity
        (and storage) for the capture window so completion matching can never
        alias a recycled ``id()``.
    witness_policy:
        The capture's resolved ``distributed_witness`` level.
    """

    __slots__ = ("payload", "destinations", "observed", "witness_policy")

    def __init__(
        self,
        payload: dict[str, Any],
        destinations: list[torch.Tensor],
        witness_policy: str,
    ) -> None:
        self.payload = payload
        self.destinations = destinations
        self.observed = [False] * len(destinations)
        self.witness_policy = witness_policy

    def mark_observed(self, index: int) -> None:
        """Settle one destination slot as completion-observed (post-redispatch).

        Parameters
        ----------
        index:
            Destination slot index within this boundary.
        """

        if self.observed[index]:
            return
        self.observed[index] = True
        events = self.payload["events"]
        events["destination_completions"][index] = True
        if all(self.observed):
            events["completion_binding"] = _FUNCOL_COMPLETION_OBSERVED
        if self.witness_policy == "digest":
            digests = self.payload["witness"].setdefault(
                "destination_digests", [None] * len(self.destinations)
            )
            # Post-completion read: the redispatched wait returned, so the
            # destination bytes are final and a digest is legal (v5 payload
            # policy: copies only after observed completion).
            digests[index] = _digest_tensor(self.destinations[index])
            if all(digest is not None for digest in digests):
                self.payload["witness"]["not_present_reason"] = None


def _make_observed_wait_tensor(
    session_ref: weakref.ref[_FuncolCaptureSession],
    wait_op: Any,
    exclude_cpu_guard: Callable[[], Any],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the dispatcher-level ``wait_tensor`` kernel for one session.

    The kernel holds its session through ``weakref`` ONLY. On current torch
    builds the C++ dispatcher retains the python kernel past
    ``torch.library.Library._destroy()`` with zero gc-visible referrers, so a
    strong reference would pin the session -- and through ``session.trace``
    the whole latest armed capture's Trace, retained activations included --
    for the life of the process. During the capture window the session is
    strongly held elsewhere (``_ACTIVE_FUNCOL_SESSION`` plus the
    ``distributed_recording_session`` frame), so the ref can never go dead
    while its own capture is live.

    Dead-ref behavior (the leaked-kernel path): the REAL wait is still
    executed via redispatch -- a leaked bookkeeping hook must never swallow
    or refuse the user's collective completion -- and only the dead session's
    completion bookkeeping is skipped. That is honest by construction: the
    owning capture already settled fail-closed
    (``completion_binding="unobserved"`` + typed disclosures), and a later
    armed capture's own install replaces this kernel, so no live capture's
    observation quality ever depends on a dead kernel.

    Parameters
    ----------
    session_ref:
        Weak reference to the owning :class:`_FuncolCaptureSession`.
    wait_op:
        The resolved ``wait_tensor`` op handle to redispatch through.
    exclude_cpu_guard:
        Zero-arg factory for the below-CPU-key redispatch guard.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The kernel to register at the CPU dispatch key.
    """

    def observed_wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Redispatch ``wait_tensor`` below this key, then bind completion."""

        with exclude_cpu_guard():
            result = wait_op(tensor)
        session = session_ref()
        if session is not None:
            session.record_completion(tensor)
        return result

    return observed_wait_tensor


class _FuncolCaptureSession:
    """Capture-scoped plane-W state: pending completions + wait interposition.

    One session exists per armed capture (installed by
    :func:`distributed_recording_session`); the dispatcher interposition it
    registers is process-global, so the session owns its lifetime exactly like
    the census harness's counter fixture -- registered on entry, destroyed on
    exit, with completion recording gated on the session being live.
    """

    def __init__(self, trace: Any) -> None:
        """Initialize the session for one capture.

        Parameters
        ----------
        trace:
            The live capture's Trace.
        """

        self.trace = trace
        self.owner_thread_id = threading.get_ident()
        self.entries: list[_PendingFuncolCompletion] = []
        self.annotation_copies: list[dict[str, Any]] = []
        self._pending_by_id: dict[int, tuple[_PendingFuncolCompletion, int]] = {}
        self._lock = threading.Lock()
        self._library: Any | None = None
        self.interposition_status = "uninstalled"

    # -- plane-W install / teardown ----------------------------------------

    def install_interposition(self) -> None:
        """Register the scoped ``wait_tensor`` dispatcher wrapper (CPU key).

        Degrades typed instead of raising: on any failure the session runs
        with ``interposition_status`` naming the cause, and every funcol
        boundary of this capture keeps its fail-closed ``unobserved``
        completion disclosure.
        """

        from ...utils._torch_compat import get_funcol_wait_redispatch

        redispatch = get_funcol_wait_redispatch()
        if redispatch is None:
            self.interposition_status = "unavailable: HAS_FUNCOL_WAIT_INTERPOSITION is False"
            return
        wait_op, exclude_cpu_guard = redispatch
        try:
            library = torch.library.Library("_c10d_functional", "IMPL")  # noqa: TOR901
            # weakref, never self: the dispatcher can retain this kernel past
            # _destroy(), and a strong session hold would pin session.trace
            # (the latest Trace + activations) for the life of the process.
            kernel = _make_observed_wait_tensor(weakref.ref(self), wait_op, exclude_cpu_guard)
            library.impl("wait_tensor", kernel, "CPU")
        except Exception as exc:
            self.interposition_status = f"unavailable: {type(exc).__name__}: {exc}"
            return
        self._library = library
        self.interposition_status = "installed"

    def teardown_interposition(self) -> None:
        """Destroy the scoped dispatcher registration (idempotent)."""

        library = self._library
        self._library = None
        if library is not None:
            library._destroy()

    # -- issue / completion bookkeeping -------------------------------------

    def register_boundary(
        self,
        payload: dict[str, Any],
        inner_destinations: list[torch.Tensor],
        witness_policy: str,
    ) -> None:
        """Track one issued funcol boundary for completion observation.

        Parameters
        ----------
        payload:
            The journaled boundary payload (shared sub-objects).
        inner_destinations:
            The INNER (plain) destination tensors, in role order.
        witness_policy:
            The capture's resolved ``distributed_witness`` level.
        """

        entry = _PendingFuncolCompletion(payload, inner_destinations, witness_policy)
        with self._lock:
            self.entries.append(entry)
            for index, tensor in enumerate(inner_destinations):
                self._pending_by_id[id(tensor)] = (entry, index)

    def record_completion(self, tensor: torch.Tensor) -> None:
        """Bind an observed ``wait_tensor`` to its pending boundary, if any.

        Parameters
        ----------
        tensor:
            The tensor the dispatcher-level wait ran on (already complete).
        """

        with self._lock:
            match = self._pending_by_id.get(id(tensor))
        if match is None:
            return
        entry, index = match
        entry.mark_observed(index)

    def register_annotation_copy(self, annotation: dict[str, Any]) -> None:
        """Track one op-record deep copy of a funcol payload for settlement.

        The op producer deep-copies boundary payloads per output record
        (sibling isolation), so wait-time settlement of the master payload
        never reaches those copies; they are synced once at session close.

        Parameters
        ----------
        annotation:
            The per-record ``annotations["collective"]`` deep copy.
        """

        with self._lock:
            self.annotation_copies.append(annotation)

    # -- settlement ----------------------------------------------------------

    def settle(self) -> None:
        """Settle every pending boundary and sync the op-record copies.

        Runs once at capture close, after the forward completed and the
        interposition tore down: never-waited destinations gain the typed
        ``async_unwaited_output_unwitnessed`` disclosure (v5 1.4b), and each
        op-record annotation copy is refreshed from its settled master so the
        persisted per-op disclosure and the trace journal cannot diverge.
        """

        by_key: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
        for entry in self.entries:
            payload = entry.payload
            if not all(entry.observed):
                disclosures = payload["disclosures"]
                if "async_unwaited_output_unwitnessed" not in disclosures:
                    disclosures.append("async_unwaited_output_unwitnessed")
            correlation = payload["correlation"]
            by_key[
                (
                    correlation["membership_digest"],
                    correlation["lifetime_ordinal"],
                    correlation["channel"],
                    correlation["seq"],
                )
            ] = payload
        for annotation in self.annotation_copies:
            correlation = annotation.get("correlation")
            if not isinstance(correlation, dict):
                continue
            master = by_key.get(
                (
                    correlation.get("membership_digest"),
                    correlation.get("lifetime_ordinal"),
                    correlation.get("channel"),
                    correlation.get("seq"),
                )
            )
            if master is None:
                continue
            annotation["events"] = copy.deepcopy(master["events"])
            annotation["witness"] = copy.deepcopy(master["witness"])
            annotation["disclosures"] = list(master["disclosures"])


#: The live capture-scoped session. Lifecycle: installed/cleared exclusively by
#: :func:`distributed_recording_session` (entered around active logging in the
#: torch backend, armed captures only); never survives a capture.
_ACTIVE_FUNCOL_SESSION: _FuncolCaptureSession | None = None


def active_funcol_session() -> _FuncolCaptureSession | None:
    """Return the live capture-scoped funcol session, or ``None``."""

    return _ACTIVE_FUNCOL_SESSION


@contextmanager
def distributed_recording_session(trace: Any) -> Iterator[None]:
    """Install the plane-W completion authority around one armed capture.

    A no-op context for unarmed captures and for armed captures with no
    initialized process group (zero-interference: the plain dense path is
    untouched). For armed captures it registers the capture-scoped
    ``wait_tensor`` interposition, publishes the session for the funcol wraps,
    and settles every pending boundary on exit -- teardown and settlement run
    on the failure path too, so an aborted forward never leaks a dispatcher
    registration.

    Parameters
    ----------
    trace:
        The live capture's Trace.

    Yields
    ------
    None
        The capture runs with the completion authority installed.
    """

    from torchlens.distributed._lifecycle import capture_armed_state

    global _ACTIVE_FUNCOL_SESSION
    if capture_armed_state() is None:
        yield
        return
    session = _FuncolCaptureSession(trace)
    session.install_interposition()
    previous = _ACTIVE_FUNCOL_SESSION
    _ACTIVE_FUNCOL_SESSION = session
    try:
        yield
    finally:
        _ACTIVE_FUNCOL_SESSION = previous
        session.teardown_interposition()
        session.settle()


def _resolve_funcol_process_group(group: Any, tag: str) -> Any | None:
    """Resolve a funcol group spelling to a live ProcessGroup, or ``None``.

    Parameters
    ----------
    group:
        Any public funcol group spelling (ProcessGroup, group-name string,
        DeviceMesh, ``(mesh, dim)``, rank list).
    tag:
        The funcol legacy tag argument.

    Returns
    -------
    Any | None
        The live ProcessGroup, or ``None`` when resolution is impossible
        (missing capability, unknown name, malformed spelling).
    """

    from ...utils._torch_compat import get_funcol_group_resolvers

    resolvers = get_funcol_group_resolvers()
    if resolvers is None:
        return None
    resolve_group, resolve_name = resolvers
    try:
        resolved = resolve_group(group, tag)
    except Exception:
        return None
    if isinstance(resolved, str):
        try:
            return resolve_name(resolved)
        except Exception:
            return None
    return resolved


def _build_funcol_payload(
    site: FuncolSite,
    bound: dict[str, Any],
    identity: Any,
    seq: int,
    arming: Any,
    contributions: list[torch.Tensor],
    inner_destinations: list[torch.Tensor],
    contribution_digests: list[str] | None,
    contribution_witness_blocked: bool,
    witness_policy: str,
    interposition_status: str,
    group: Any,
) -> dict[str, Any]:
    """Assemble the portable ``functional_collective_boundary_v0`` payload.

    Parameters
    ----------
    site:
        The wrapped funcol site.
    bound:
        Bound call arguments.
    identity:
        Resolved :class:`~torchlens.distributed._lifecycle.GroupIdentity`.
    seq:
        Issue-time seq on the ``(group_uid, "coll")`` channel.
    arming:
        The armed state's :class:`~torchlens.distributed._lifecycle.ArmingRecord`.
    contributions:
        This rank's contribution tensors (possibly ACTs from chained funcol).
    inner_destinations:
        INNER destination tensors, role order.
    contribution_digests:
        Digest witnesses of the contributions, when taken.
    contribution_witness_blocked:
        Whether a contribution digest was withheld because taking it would
        have forced an unwaited ACT's completion (v5 1.4c).
    witness_policy:
        Resolved ``distributed_witness`` level.
    interposition_status:
        The session's plane-W install status for this capture.
    group:
        The resolved ProcessGroup.

    Returns
    -------
    dict[str, Any]
        The journal-ready payload. ``events`` / ``witness`` / ``disclosures``
        are the settlement-shared sub-objects.
    """

    dist = torch.distributed
    my_global_rank = int(dist.get_rank())
    try:
        my_group_rank: int | None = int(dist.get_group_rank(group, my_global_rank))
    except Exception:
        my_group_rank = None

    roles: list[dict[str, Any]] = []
    for index, tensor in enumerate(contributions):
        entry = _role_entry("contribution", index, _inner_tensor(tensor))
        entry["async_collective_tensor"] = _is_unwaited_act(tensor)
        roles.append(entry)
    for index, tensor in enumerate(inner_destinations):
        entry = _role_entry("destination", index, tensor)
        entry["async_collective_tensor"] = True
        roles.append(entry)

    disclosures: list[str] = ["read_of_inflight_destination"]
    if contribution_witness_blocked:
        disclosures.append("witness_would_force_completion")
    if interposition_status != "installed":
        disclosures.append("completion_interposition_unavailable")

    witness: dict[str, Any] = {
        "policy_resolved": witness_policy,
        "contribution_digests": None,
        "destination_digests": None,
        "not_present_reason": None,
    }
    if witness_policy == "digest":
        witness["contribution_digests"] = contribution_digests
        witness["destination_digests"] = [None] * len(inner_destinations)
        witness["not_present_reason"] = "async_completion_unobserved"

    reduce_op = bound.get("reduceOp")
    detail = {name: _jsonable_detail(bound.get(name)) for name in site.detail_args}

    group_seq, group_seq_disclosure = _c10d_group_seq(group)
    if group_seq_disclosure is not None:
        disclosures.append(group_seq_disclosure)

    return {
        "schema": FUNCOL_BOUNDARY_SCHEMA,
        "kind": site.kind,
        "func": f"{_FUNCOL_MODULE}.{site.attr}",
        "correlation": {
            "membership_digest": identity.membership_digest,
            "lifetime_ordinal": identity.lifetime_ordinal,
            "channel": "coll",
            "seq": seq,
        },
        "group": {
            "global_ranks": list(identity.global_ranks),
            "size": len(identity.global_ranks),
            "backend": identity.backend,
            "my_global_rank": my_global_rank,
            "my_group_rank": my_group_rank,
            "coord_provenance": "torch.distributed.get_rank/get_group_rank",
        },
        "reduce_op": None if reduce_op is None else str(reduce_op),
        "detail": detail,
        "events": {
            "async_op": True,
            "event_model": "funcol_issue_is_launch",
            "completion_binding": _FUNCOL_COMPLETION_UNOBSERVED,
            "completion_interposition": interposition_status,
            "destination_completions": [False] * len(inner_destinations),
        },
        "roles": roles,
        "witness": witness,
        "lifetime_evidence": {
            "ordinal_source": identity.ordinal_source,
            "install_epoch": arming.install_epoch,
            "arming_source": arming.source,
        },
        "c10d_group_seq": group_seq,
        "disclosures": disclosures,
        "op_node": True,
    }


def _make_funcol_wrap(site: FuncolSite, original: Callable[..., Any]) -> Callable[..., Any]:
    """Build the armed wrapper for one public funcol entry point.

    Parameters
    ----------
    site:
        The site definition.
    original:
        The pristine funcol function.

    Returns
    -------
    Callable[..., Any]
        The wrapper, marked with the standard distributed-wrap dunders.
    """

    signature = inspect.signature(original)

    @wraps(original)
    def wrapped_funcol(*args: Any, **kwargs: Any) -> Any:
        """Record the boundary op around one funcol call, then delegate.

        Falls through untouched -- under the reentrancy scope, so no seq tick
        and no record -- when arming is inactive, when this is a nested inner
        collective call, or when argument binding fails (the call is about to
        raise its own ``TypeError`` anyway).
        """

        from torchlens.distributed._lifecycle import armed_state, next_seq, resolve_group_identity

        state = armed_state()
        if state is None or _inside_boundary():
            with _BoundaryScope():
                return original(*args, **kwargs)

        from ... import _state

        try:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            bound = dict(bound_args.arguments)
        except TypeError:
            with _BoundaryScope():
                return original(*args, **kwargs)

        active_trace, logging_enabled = _state.active_capture()
        capturing = bool(
            logging_enabled
            and active_trace is not None
            and _state._active_owner_thread_id == threading.get_ident()
        )
        tag = str(bound.get("tag") or "")
        group = _resolve_funcol_process_group(bound.get("group"), tag)
        if group is None:
            if capturing:
                # Fail closed: letting an uncorrelatable collective execute
                # inside a capture would silently omit it from the trace --
                # exactly the invisibility class this module exists to close.
                from ..._distributed import DistributedCaptureUnsupportedError, DistributedFinding

                raise DistributedCaptureUnsupportedError(
                    "TorchLens cannot resolve this functional collective's "
                    f"process group ({site.attr!r} group spelling "
                    f"{type(bound.get('group')).__name__}), so the boundary "
                    "cannot be correlated. Pass a live ProcessGroup, group "
                    "name, or DeviceMesh -- or capture without the "
                    "distributed opt-in armed.",
                    findings=[
                        DistributedFinding(
                            kind="uncaptured_collective_op",
                            detail=(
                                f"funcol {site.attr} group spelling unresolvable at capture time"
                            ),
                            suggestion=(
                                "pass a live ProcessGroup, group name, or "
                                "DeviceMesh to the funcol call"
                            ),
                        )
                    ],
                )
            with _BoundaryScope():
                return original(*args, **kwargs)

        try:
            identity = resolve_group_identity(group)
        except Exception:
            if capturing:
                raise
            with _BoundaryScope():
                return original(*args, **kwargs)

        # Issue-time tick, always while armed: the counter is the correlation
        # authority and must not depend on which forwards were captured.
        seq = next_seq(identity, "coll")

        if not capturing:
            with _BoundaryScope():
                return original(*args, **kwargs)

        trace = active_trace
        if getattr(trace, "capture_mode", None) == "predicate":
            raise CompatibilityError(
                "Fastlog cannot represent collective boundary journals; refusing before "
                "the collective executes instead of returning a Recording that omits it.",
                kind=COLLECTIVE_BOUNDARY_FASTLOG_UNSUPPORTED,
                func=f"{_FUNCOL_MODULE}.{site.attr}",
            )

        contributions = _contribution_tensors(bound.get(site.tensor_arg))
        witness_policy = str(getattr(trace, "distributed_witness", "none") or "none")

        from ...utils.rng import log_current_autocast_state, log_current_rng_states

        save_rng = getattr(trace, "save_rng_states", False)
        rng_states = log_current_rng_states(torch_only=True) if save_rng else {}
        autocast_state = log_current_autocast_state()
        session = active_funcol_session()
        interposition_status = (
            session.interposition_status if session is not None else "uninstalled"
        )

        contribution_digests: list[str] | None = None
        contribution_witness_blocked = False
        with _BoundaryScope(), _state.pause_logging():
            if witness_policy == "digest":
                if any(_is_unwaited_act(tensor) for tensor in contributions):
                    # v5 1.4c: digesting an unwaited ACT would force its wait;
                    # the witness records not_present instead of perturbing.
                    contribution_witness_blocked = True
                else:
                    contribution_digests = [
                        _digest_tensor(_inner_tensor(tensor)) for tensor in contributions
                    ]
            start = time.time()
            result = original(*args, **kwargs)
            elapsed = time.time() - start
            destinations = _result_tensors(result)
            inner_destinations = [_inner_tensor(tensor) for tensor in destinations]
            payload = _build_funcol_payload(
                site,
                bound,
                identity,
                seq,
                state.arming,
                contributions,
                inner_destinations,
                contribution_digests,
                contribution_witness_blocked,
                witness_policy,
                interposition_status,
                group,
            )

        op_labels = _emit_boundary_op(
            trace,
            site,
            original,
            args,
            kwargs,
            contributions,
            inner_destinations,
            payload,
            elapsed,
            rng_states,
            autocast_state,
        )
        # Downstream user ops hold the ACT wrapper, not the labeled inner
        # tensor; provenance still resolves because the ONE label chokepoint
        # (``_tl.get_tensor_label``) delegates ACT reads to ``.elem``, whose
        # label the ordinary relabel dance inside _emit_boundary_op just set.
        if session is not None:
            session.register_boundary(payload, inner_destinations, witness_policy)
        _journal_boundary(trace, payload, op_labels)
        return result

    wrapped_funcol.__wrapped__ = original
    wrapped_funcol.__tl_distributed_wrap__ = True  # type: ignore[attr-defined]
    return wrapped_funcol


def install_funcol_wraps(originals: dict[tuple[Any, str], Any]) -> None:
    """Install the funcol boundary wrappers (arm-time, beside the c10d wraps).

    Parameters
    ----------
    originals:
        The armed state's original-function registry; pristine references are
        recorded here so disarm can restore them through the standard
        ``restore_wrapped_attr`` path.
    """

    if not torch.distributed.is_available():
        return
    from ...utils._torch_compat import get_funcol_module

    funcol_module = get_funcol_module()
    if funcol_module is None:
        return
    for site in FUNCOL_SITES:
        current = getattr(funcol_module, site.attr, None)
        if current is None or (funcol_module, site.attr) in originals:
            continue
        originals[(funcol_module, site.attr)] = current
        setattr(funcol_module, site.attr, _make_funcol_wrap(site, current))


def remove_funcol_wraps(originals: dict[tuple[Any, str], Any]) -> None:
    """Restore pristine funcol functions recorded at install time.

    Parameters
    ----------
    originals:
        The armed state's original-function registry.
    """

    from torchlens.distributed._lifecycle import restore_wrapped_attr

    from ...utils._torch_compat import get_funcol_module

    funcol_module = get_funcol_module()
    if funcol_module is None:
        return
    first_failure: Exception | None = None
    for (module, attr), original in list(originals.items()):
        if module is not funcol_module:
            continue
        try:
            restore_wrapped_attr(module, attr, original)
        except Exception as exc:
            if first_failure is None:
                first_failure = exc
            continue
        originals.pop((module, attr), None)
    if first_failure is not None:
        raise first_failure
