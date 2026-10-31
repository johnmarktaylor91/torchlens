"""Collective boundary capture for explicit ``torch.distributed`` python calls.

Merge-ranks tier (b): a hand-rolled tensor-parallel or data-parallel forward
that issues explicit c10d python collectives (``dist.all_reduce`` and friends)
used to capture with the collectives silently ABSENT -- the rank-local graph
was incomplete with no disclosure. This module wraps the public c10d python
API under the distributed opt-in (:func:`torchlens.distributed.arm`, or lazy
arming at capture entry for already-initialized SPMD processes) so every
explicit collective becomes a first-class BOUNDARY NODE in the trace, shaped
exactly like the shipped ``torch.func`` transform boundaries: the real call
runs inside ``pause_logging()`` (interior collapse), the node is emitted by
the ordinary producer, and downstream ops parent on it through the standard
label machinery.

Every boundary carries the C0 ``collective_boundary_v1`` payload in its
portable ``annotations["collective"]`` namespace: the two-field
``group_uid`` correlation key with an issue-ticked per-``(uid, channel)``
seq, role-indexed tensor entries with dual-geometry slots, the three-phase
event disclosure (sync all-in-one; an async or batched issue whose completion
TorchLens does not observe is marked ``completion_binding="unobserved"`` --
honesty never depends on interposition succeeding), the witness digests the
session-time ``distributed_witness`` knob asked for, and the per-record
``lifetime_evidence`` projection of the group-lifecycle ledger.

Boundaries with no tensor dataflow on this rank (``barrier``, the object
collectives, send-side p2p) are journaled on the trace's distributed record
without an op node; nested collectives issued inside another public c10d call
(object collectives delegating to ``all_gather``) neither tick nor record --
the OUTER user-level call is the correlating boundary (design v5, 1.2).
"""

from __future__ import annotations

import hashlib
import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Protocol

import torch

from ..._transport import to_cpu_contiguous
from ...errors._base import CompatibilityError

__all__ = [
    "COLLECTIVE_SITES",
    "CollectiveSite",
    "WILDCARD_RECV_UNSUPPORTED",
    "WildcardRecvUnsupportedError",
    "install_collective_wraps",
    "remove_collective_wraps",
]

WILDCARD_RECV_UNSUPPORTED = "wildcard_recv_unsupported"
"""Finding kind for refused wildcard (any-source) recv during capture."""

COLLECTIVE_BOUNDARY_FASTLOG_UNSUPPORTED = "collective_boundary_fastlog_unsupported"
"""Finding kind for fastlog's refusal of an executed collective boundary."""

BOUNDARY_SCHEMA = "collective_boundary_v1"

# Backends whose runtime honors p2p tags; on these the tag is part of the
# channel. On tag-ignoring backends (NCCL) the tag demotes to a check field.
_TAG_HONORING_BACKENDS = frozenset({"gloo", "mpi"})


class BoundarySiteLike(Protocol):
    """Structural site contract shared by the c10d and funcol boundary tables.

    ``func_name`` drives the emitted node's TorchLens label type;
    ``tensorless`` marks journal-only boundaries that emit no op node.
    """

    @property
    def func_name(self) -> str:
        """Sanitized TorchLens label type for the emitted boundary node."""
        ...

    @property
    def tensorless(self) -> bool:
        """Whether the boundary journals without emitting an op node."""
        ...


class WildcardRecvUnsupportedError(CompatibilityError, RuntimeError):
    """Raised when a captured forward issues an any-source recv.

    A wildcard recv has no determinate peer, so no correlation channel exists
    for it; capture refuses rather than recording an uncorrelatable boundary.
    ``fields["kind"]`` is ``"wildcard_recv_unsupported"``.
    """


@dataclass(frozen=True)
class CollectiveSite:
    """One wrapped public c10d entry point.

    Parameters
    ----------
    attr:
        Function name on ``torch.distributed``.
    kind:
        Canonical collective kind recorded on the boundary.
    func_name:
        Sanitized TorchLens label type (drives ``op.type`` and labels).
    inputs:
        Extractor returning this rank's CONTRIBUTION tensors from bound args.
    outputs:
        Extractor returning this rank's DESTINATION tensors (mutated at
        completion) from bound args. Receives the rank-role context so
        root-only aggregates (gather/reduce/scatter) stay per-rank honest.
    p2p:
        ``None`` for symmetric-issue collectives; ``"send"`` / ``"recv"``
        for point-to-point.
    has_reduce_op:
        Whether the site carries a ``ReduceOp`` argument.
    tensorless:
        Journal-only boundary (object collectives, barrier): no op node.
    """

    attr: str
    kind: str
    func_name: str
    inputs: Callable[[dict[str, Any], _RankRole], list[torch.Tensor]]
    outputs: Callable[[dict[str, Any], _RankRole], list[torch.Tensor]]
    p2p: str | None = None
    has_reduce_op: bool = False
    tensorless: bool = False


@dataclass(frozen=True)
class _RankRole:
    """Per-call rank-role context for root-aware role extraction."""

    my_global_rank: int
    root_global_rank: int | None

    @property
    def is_root(self) -> bool:
        """Whether this rank is the collective's root (``False`` for rootless calls)."""

        return self.root_global_rank is not None and (self.my_global_rank == self.root_global_rank)


def _tensors(value: Any) -> list[torch.Tensor]:
    """Flatten a tensor / list of tensors argument into a list."""

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def _arg(name: str) -> Callable[[dict[str, Any], _RankRole], list[torch.Tensor]]:
    """Build a role extractor reading one bound argument on every rank."""

    return lambda bound, role: _tensors(bound.get(name))


def _arg_if_root(name: str) -> Callable[[dict[str, Any], _RankRole], list[torch.Tensor]]:
    """Build a role extractor reading one bound argument on the root rank only."""

    return lambda bound, role: _tensors(bound.get(name)) if role.is_root else []


def _arg_if_not_root(name: str) -> Callable[[dict[str, Any], _RankRole], list[torch.Tensor]]:
    """Build a role extractor reading one bound argument on non-root ranks only."""

    return lambda bound, role: [] if role.is_root else _tensors(bound.get(name))


def _nothing(bound: dict[str, Any], role: _RankRole) -> list[torch.Tensor]:
    """Role extractor for a slot that never carries tensors on any rank."""

    return []


COLLECTIVE_SITES: tuple[CollectiveSite, ...] = (
    CollectiveSite(
        "all_reduce", "all_reduce", "allreduce", _arg("tensor"), _arg("tensor"), has_reduce_op=True
    ),
    CollectiveSite("all_gather", "all_gather", "allgather", _arg("tensor"), _arg("tensor_list")),
    CollectiveSite(
        "all_gather_into_tensor",
        "all_gather_into_tensor",
        "allgatherintotensor",
        _arg("input_tensor"),
        _arg("output_tensor"),
    ),
    CollectiveSite(
        "reduce_scatter",
        "reduce_scatter",
        "reducescatter",
        _arg("input_list"),
        _arg("output"),
        has_reduce_op=True,
    ),
    CollectiveSite(
        "reduce_scatter_tensor",
        "reduce_scatter_tensor",
        "reducescattertensor",
        _arg("input"),
        _arg("output"),
        has_reduce_op=True,
    ),
    CollectiveSite(
        "broadcast", "broadcast", "broadcast", _arg("tensor"), _arg_if_not_root("tensor")
    ),
    CollectiveSite(
        "reduce", "reduce", "reduce", _arg("tensor"), _arg_if_root("tensor"), has_reduce_op=True
    ),
    CollectiveSite(
        "all_to_all",
        "all_to_all",
        "alltoall",
        _arg("input_tensor_list"),
        _arg("output_tensor_list"),
    ),
    CollectiveSite(
        "all_to_all_single", "all_to_all_single", "alltoallsingle", _arg("input"), _arg("output")
    ),
    CollectiveSite("gather", "gather", "gather", _arg("tensor"), _arg_if_root("gather_list")),
    CollectiveSite("scatter", "scatter", "scatter", _arg_if_root("scatter_list"), _arg("tensor")),
    CollectiveSite("send", "send", "send", _arg("tensor"), _nothing, p2p="send"),
    CollectiveSite("isend", "send", "isend", _arg("tensor"), _nothing, p2p="send"),
    CollectiveSite("recv", "recv", "recv", _nothing, _arg("tensor"), p2p="recv"),
    CollectiveSite("irecv", "recv", "irecv", _nothing, _arg("tensor"), p2p="recv"),
    CollectiveSite("barrier", "barrier", "barrier", _nothing, _nothing, tensorless=True),
    CollectiveSite(
        "all_gather_object",
        "all_gather_object",
        "allgatherobject",
        _nothing,
        _nothing,
        tensorless=True,
    ),
    CollectiveSite(
        "broadcast_object_list",
        "broadcast_object_list",
        "broadcastobjectlist",
        _nothing,
        _nothing,
        tensorless=True,
    ),
    CollectiveSite(
        "gather_object", "gather_object", "gatherobject", _nothing, _nothing, tensorless=True
    ),
    CollectiveSite(
        "scatter_object_list",
        "scatter_object_list",
        "scatterobjectlist",
        _nothing,
        _nothing,
        tensorless=True,
    ),
)


_BOUNDARY_DEPTH = threading.local()


def _inside_boundary() -> bool:
    """Whether the calling thread is already inside a recorded collective boundary."""

    return getattr(_BOUNDARY_DEPTH, "depth", 0) > 0


class _BoundaryScope:
    """Reentrancy guard: nested public c10d calls neither tick nor record."""

    __slots__ = ()

    def __enter__(self) -> None:
        _BOUNDARY_DEPTH.depth = getattr(_BOUNDARY_DEPTH, "depth", 0) + 1

    def __exit__(self, *_: Any) -> None:
        _BOUNDARY_DEPTH.depth = getattr(_BOUNDARY_DEPTH, "depth", 1) - 1


def _digest_tensor(tensor: torch.Tensor) -> str:
    """Byte-exact SHA-256 of a tensor's values (dtype-agnostic)."""

    # detach-ok: read-only digest snapshot; no retained gradient payload crosses this boundary.
    flat = to_cpu_contiguous(tensor).reshape(-1)
    if flat.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    # Buffer-protocol digest (r7 R35-3): identical bytes to the old
    # ``tobytes`` spelling (this digest PERSISTS in collective annotations,
    # so byte identity is load-bearing) without the whole-payload copy.
    return hashlib.sha256(flat.view(torch.uint8).numpy().data).hexdigest()


def _role_entry(role: str, index: int, tensor: torch.Tensor) -> dict[str, Any]:
    """One role-indexed entry; dual-geometry slots stay None for plain tensors.

    The physical ``shape`` is ALWAYS the local bytes actually held; for a
    DTensor value (reachable once C2 relaxes the dtensor refusal) the logical
    side is declared explicitly alongside it.
    """

    entry = {
        "role": role,
        "index": index,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "logical_shape": None,
        "placements": None,
        "mesh_coords": None,
        "shard_offset": None,
    }
    from torchlens.distributed._dtensor import dtensor_dual_geometry

    geometry = dtensor_dual_geometry(tensor)
    if geometry is not None:
        entry["shape"] = geometry.get("local_shape") or entry["shape"]
        entry["logical_shape"] = geometry.get("logical_shape")
        entry["placements"] = geometry.get("placements")
        entry["mesh_coords"] = geometry.get("mesh_coords")
        entry["shard_offset"] = geometry.get("shard_offset")
    return entry


def _resolve_root(bound: dict[str, Any], group: Any, src_or_dst: str) -> int | None:
    """Canonicalize a root/peer spelling to a GLOBAL rank, or None."""

    dist = torch.distributed
    global_rank = bound.get(src_or_dst)
    group_relative = bound.get(f"group_{src_or_dst}")
    if global_rank is not None:
        return int(global_rank)
    if group_relative is not None:
        resolved_group: Any = group if group is not None else dist.group.WORLD
        return int(dist.get_global_rank(resolved_group, int(group_relative)))
    return None


def _c10d_group_seq(group: Any) -> tuple[int | None, str | None]:
    """Read c10d's private per-group sequence number, or disclose why not.

    Routed through ``_torch_compat`` (``HAS_C10D_GROUP_SEQ``, r-b4 R26-2
    shape): a private-API rename used to silently and permanently disable
    this redundant correlation cross-check -- the only in-band detector in
    the base-misalignment neighborhood -- with zero visibility in
    ``tl.report.doctor()``.

    Returns
    -------
    tuple[int | None, str | None]
        ``(seq, disclosure)``. A getter that RAISES while the probed flag
        claims the capability present used to swallow to a bare ``None`` --
        indistinguishable downstream from honest capability absence, so the
        only in-band base-misalignment cross-check vanished with zero
        visibility. It now demotes the capability through the standard
        degradation channel (visible in ``tl.report.doctor()``)
        and returns the ``"c10d_group_seq_read_failed"`` disclosure token for
        the boundary record where the witness vanished.
    """

    from torchlens._utils._torch_compat import (
        mark_torch_capability_missing,
        probe_c10d_capabilities,
    )

    if not probe_c10d_capabilities()["HAS_C10D_GROUP_SEQ"]:
        return None, None
    dist = torch.distributed
    try:
        target: Any = group if group is not None else dist.group.WORLD
        return int(target._get_sequence_number_for_group()), None
    except Exception as exc:
        mark_torch_capability_missing(
            "HAS_C10D_GROUP_SEQ",
            "the probed _get_sequence_number_for_group getter raised at read "
            f"time ({type(exc).__name__}: {exc}); collective boundary records "
            "omit the redundant c10d_group_seq cross-check field from here on",
        )
        return None, "c10d_group_seq_read_failed"


def _build_payload(
    site: CollectiveSite,
    bound: dict[str, Any],
    identity: Any,
    channel: str,
    seq: int,
    arming: Any,
    peer_info: dict[str, Any] | None,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    contribution_digests: list[str] | None,
    async_op: bool,
    witness_policy: str,
    group: Any,
) -> dict[str, Any]:
    """Assemble the portable ``collective_boundary_v1`` payload."""

    dist = torch.distributed
    my_global_rank = int(dist.get_rank())
    try:
        payload_group: Any = group if group is not None else dist.group.WORLD
        my_group_rank: int | None = int(dist.get_group_rank(payload_group, my_global_rank))
    except Exception:
        my_group_rank = None
    roles: list[dict[str, Any]] = []
    input_ids = {id(t) for t in inputs}
    for index, tensor in enumerate(inputs):
        role = (
            "contribution_destination"
            if any(id(o) == id(tensor) for o in outputs)
            else "contribution"
        )
        roles.append(_role_entry(role, index, tensor))
    for index, tensor in enumerate(outputs):
        if id(tensor) in input_ids:
            continue
        roles.append(_role_entry("destination", index, tensor))

    disclosures: list[str] = []
    completion_binding = "issue_sync"
    if async_op:
        # The recorded wait is not (yet) interposed for the c10d python API;
        # fail closed per 1.4c: completion unobserved, destination witnesses
        # not_present, and every pre-wait read is a read of an in-flight
        # destination.
        completion_binding = "unobserved"
        disclosures.append("read_of_inflight_destination")

    witness: dict[str, Any] = {
        "policy_resolved": witness_policy,
        "contribution_digests": None,
        "destination_digests": None,
        "not_present_reason": None,
    }
    if witness_policy == "digest":
        witness["contribution_digests"] = contribution_digests
        if async_op:
            witness["not_present_reason"] = "async_completion_unobserved"
        else:
            witness["destination_digests"] = [_digest_tensor(t) for t in outputs]

    reduce_op = bound.get("op") if site.has_reduce_op else None

    group_seq, group_seq_disclosure = _c10d_group_seq(group)
    if group_seq_disclosure is not None:
        disclosures.append(group_seq_disclosure)

    return {
        "schema": BOUNDARY_SCHEMA,
        "kind": site.kind,
        "func": f"torch.distributed.{site.attr}",
        "correlation": {
            "membership_digest": identity.membership_digest,
            "lifetime_ordinal": identity.lifetime_ordinal,
            "channel": channel,
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
        "peer": peer_info,
        "events": {
            "async_op": bool(async_op),
            "completion_binding": completion_binding,
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
        "op_node": not site.tensorless,
    }


def _journal_boundary(trace: Any, payload: dict[str, Any], op_labels: list[str]) -> None:
    """Append a boundary to the trace-level distributed journal."""

    from torchlens.distributed._lifecycle import armed_state

    entry = dict(payload)
    entry["op_labels_raw"] = list(op_labels)
    record = trace.annotations.setdefault("distributed", {})
    record.setdefault("boundaries", []).append(entry)
    state = armed_state()
    if state is not None:
        record["group_lifecycle_ledger"] = state.ledger.to_payload()
        record["install_epoch"] = state.arming.install_epoch
        record["arming_source"] = state.arming.source
        record["recognizer_snapshot"] = state.arming.recognizer_snapshot


def _emit_boundary_op(
    trace: Any,
    site: BoundarySiteLike,
    original: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    payload: dict[str, Any],
    elapsed: float,
    rng_states: dict[str, Any],
    autocast_state: dict[str, Any],
) -> list[str]:
    """Emit the boundary op node through the ordinary producer.

    Returns the raw labels of the emitted node's output tensors (one per
    destination; empty when no node was emitted).
    """

    from ... import _state
    from ..._data_classes.internal_types import FuncExecutionContext
    from ..._utils.tensor_utils import safe_copy
    from ._tl import get_tensor_label, set_tensor_label
    from .ops import (
        _record_label_version_snapshot,
        apply_live_hooks_to_outputs,
        log_function_output_tensors,
    )
    from .wrappers import (
        _propagate_mutation_label_to_storage_aliases,
        _register_inplace_live_grad_hook,
    )

    raw_replay = partial(original, *args, **kwargs)
    raw_replay.__tl_collective_info__ = payload  # type: ignore[attr-defined]

    # The node's logged output: safe copies of the mutated destinations (or,
    # for send-side p2p, of the contribution -- a leaf node that shows the
    # boundary and its provenance without claiming to produce data).
    mutated = list(outputs)
    logged_sources = mutated if mutated else list(inputs)
    if not logged_sources:
        return []
    with _state.pause_logging():
        logged_copies = [safe_copy(t) for t in logged_sources]
    out_for_log: Any = logged_copies[0] if len(logged_copies) == 1 else list(logged_copies)

    exec_ctx = FuncExecutionContext(
        time_elapsed=elapsed,
        rng_states=rng_states,
        autocast_state=autocast_state,
    )
    call_args = tuple(inputs)
    func_call_id = _state.next_func_call_id()
    out_for_log = apply_live_hooks_to_outputs(
        trace,
        raw_replay,
        site.func_name,
        call_args,
        {},
        out_for_log,
        exec_ctx,
        True,
        func_call_id,
    )
    if _state._completeness_witness_mode == "shadow":
        with _state.pause_logging():
            log_function_output_tensors(
                trace,
                raw_replay,
                site.func_name,
                call_args,
                {},
                call_args,
                {},
                out_for_log,
                exec_ctx,
                True,
                func_call_id,
            )
    else:
        log_function_output_tensors(
            trace,
            raw_replay,
            site.func_name,
            call_args,
            {},
            call_args,
            {},
            out_for_log,
            exec_ctx,
            True,
            func_call_id,
        )

    final_list = out_for_log if isinstance(out_for_log, list) else [out_for_log]
    labels: list[str] = []
    for index, logged in enumerate(final_list):
        if not isinstance(logged, torch.Tensor):
            continue
        label = get_tensor_label(logged)
        if label is None:
            continue
        labels.append(label)
        if index < len(mutated):
            # Advance the LIVE mutated destination to the boundary's label so
            # downstream consumers parent on the collective, not on the
            # pre-collective producer (the standard in-place relabel dance).
            live = mutated[index]
            set_tensor_label(live, label)
            _register_inplace_live_grad_hook(trace, live, label)
            _record_label_version_snapshot(live)
            _propagate_mutation_label_to_storage_aliases(trace, live, label)
    return labels


def _make_collective_wrap(site: CollectiveSite, original: Callable[..., Any]) -> Callable[..., Any]:
    """Build the armed wrapper for one public c10d entry point."""

    signature = inspect.signature(original)

    @wraps(original)
    def wrapped_collective(*args: Any, **kwargs: Any) -> Any:
        """Record the boundary op around one public c10d call, then delegate.

        Falls through to the original untouched -- under a reentrancy scope, so no
        seq tick and no record -- when arming is inactive, when this is a nested
        inner c10d call, or when argument binding fails (the call is about to raise
        its own ``TypeError`` anyway).
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
            # A binding failure means the call is about to fail anyway; let
            # the original raise its own error.
            with _BoundaryScope():
                return original(*args, **kwargs)

        capturing = bool(
            _state._logging_enabled
            and _state._active_trace is not None
            and _state._active_owner_thread_id == threading.get_ident()
        )
        group = bound.get("group")

        # Wildcard recv has no determinate peer: refuse during capture.
        peer_info: dict[str, Any] | None = None
        root_global: int | None = None
        dist = torch.distributed
        if site.p2p is not None:
            peer_field = "dst" if site.p2p == "send" else "src"
            peer_global = _resolve_root(bound, group, peer_field)
            if peer_global is None:
                if site.p2p == "recv" and capturing:
                    raise WildcardRecvUnsupportedError(
                        "torchlens cannot correlate an any-source recv: it has no "
                        "determinate peer, so no correlation channel exists. Give "
                        "the recv an explicit src (or group_src).",
                        kind=WILDCARD_RECV_UNSUPPORTED,
                        func=f"torch.distributed.{site.attr}",
                    )
                with _BoundaryScope():
                    return original(*args, **kwargs)
        elif site.attr in ("broadcast", "scatter", "broadcast_object_list", "scatter_object_list"):
            root_global = _resolve_root(bound, group, "src")
        elif site.attr in ("reduce", "gather", "gather_object"):
            root_global = _resolve_root(bound, group, "dst")

        # Group identity resolution (may seed; may refuse typed). Outside a
        # capture a refusal degrades to a skipped tick: the group is
        # permanently uncorrelatable on this rank and any captured boundary
        # on it will raise the same typed error then.
        try:
            identity = resolve_group_identity(group)
        except Exception:
            if capturing:
                raise
            with _BoundaryScope():
                return original(*args, **kwargs)

        # Channel derivation (1.4).
        tag = bound.get("tag", 0) or 0
        if site.p2p is not None:
            my_rank = int(dist.get_rank())
            src = my_rank if site.p2p == "send" else int(peer_global)  # type: ignore[arg-type]
            dst = int(peer_global) if site.p2p == "send" else my_rank  # type: ignore[arg-type]
            backend = (identity.backend or "").lower()
            tag_in_channel = any(name in backend for name in _TAG_HONORING_BACKENDS)
            channel = f"p2p/{src}->{dst}"
            if tag_in_channel:
                channel = f"{channel}/{tag}"
            peer_info = {
                "raw": {
                    "dst" if site.p2p == "send" else "src": bound.get(
                        "dst" if site.p2p == "send" else "src"
                    ),
                    "group_peer": bound.get(f"group_{'dst' if site.p2p == 'send' else 'src'}"),
                    "tag": tag,
                },
                "canonical": {"src": src, "dst": dst},
                "provenance": "explicit_global"
                if bound.get("dst" if site.p2p == "send" else "src") is not None
                else "group_relative_resolved",
                "tag_in_channel": tag_in_channel,
            }
        else:
            channel = "coll"

        # Issue-time tick, always while armed: the counter is the correlation
        # authority and must not depend on which forwards were captured.
        seq = next_seq(identity, channel)

        if not capturing:
            with _BoundaryScope():
                return original(*args, **kwargs)

        trace = _state._active_trace
        if getattr(trace, "capture_mode", None) == "predicate":
            raise CompatibilityError(
                "Fastlog cannot represent collective boundary journals; refusing before "
                "the collective executes instead of returning a Recording that omits it.",
                kind=COLLECTIVE_BOUNDARY_FASTLOG_UNSUPPORTED,
                func=f"torch.distributed.{site.attr}",
            )
        role = _RankRole(my_global_rank=int(dist.get_rank()), root_global_rank=root_global)
        inputs = site.inputs(bound, role)
        async_op = bool(bound.get("async_op", False)) or site.attr in ("isend", "irecv")
        witness_policy = str(getattr(trace, "distributed_witness", "none") or "none")

        from ..._utils.rng import log_current_autocast_state, log_current_rng_states

        save_rng = getattr(trace, "save_rng_states", False)
        rng_states = log_current_rng_states(torch_only=True) if save_rng else {}
        autocast_state = log_current_autocast_state()
        contribution_digests: list[str] | None = None
        with _BoundaryScope(), _state.pause_logging():
            if witness_policy == "digest":
                contribution_digests = [_digest_tensor(tensor) for tensor in inputs]
            start = time.time()
            result = original(*args, **kwargs)
            elapsed = time.time() - start
            outputs = site.outputs(bound, role)
            payload = _build_payload(
                site,
                bound,
                identity,
                channel,
                seq,
                state.arming,
                peer_info,
                inputs,
                outputs,
                contribution_digests,
                async_op,
                witness_policy,
                group,
            )
        op_labels: list[str] = []
        if not site.tensorless:
            op_labels = _emit_boundary_op(
                trace,
                site,
                original,
                args,
                kwargs,
                inputs,
                outputs,
                payload,
                elapsed,
                rng_states,
                autocast_state,
            )
        _journal_boundary(trace, payload, op_labels)
        return result

    wrapped_collective.__wrapped__ = original
    wrapped_collective.__tl_distributed_wrap__ = True  # type: ignore[attr-defined]
    return wrapped_collective


def install_collective_wraps(originals: dict[tuple[Any, str], Any]) -> None:
    """Install the boundary wrappers on every module holding a reference.

    Parameters
    ----------
    originals:
        The armed state's original-function registry; pristine references are
        recorded here so :func:`remove_collective_wraps` can restore them.
    """

    if not torch.distributed.is_available():
        return
    from torchlens.distributed._lifecycle import _patch_modules

    for module in _patch_modules():
        for site in COLLECTIVE_SITES:
            current = getattr(module, site.attr, None)
            if current is None or (module, site.attr) in originals:
                continue
            originals[(module, site.attr)] = current
            setattr(module, site.attr, _make_collective_wrap(site, current))


def remove_collective_wraps(originals: dict[tuple[Any, str], Any]) -> None:
    """Restore pristine collective functions recorded at install time.

    A site re-patched by a third party after installation is left in place
    with a warning instead of being clobbered back to the pristine function
    (see :func:`torchlens.distributed._lifecycle.restore_wrapped_attr`).
    """

    from torchlens.distributed._lifecycle import restore_wrapped_attr

    first_failure: Exception | None = None
    for (module, attr), original in list(originals.items()):
        if any(attr == site.attr for site in COLLECTIVE_SITES):
            try:
                restore_wrapped_attr(module, attr, original)
            except Exception as exc:
                if first_failure is None:
                    first_failure = exc
                continue
            originals.pop((module, attr), None)
    if first_failure is not None:
        raise first_failure
