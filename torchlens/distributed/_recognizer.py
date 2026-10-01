"""Five-namespace collective recognizer with arm-time mechanical derivation.

The enumerated collective namespaces are FIVE (design-merge-ranks-c v5, 5.2):
``c10d`` (the ProcessGroup dispatcher surface), ``_c10d_functional``,
``_c10d_functional_autograd``, legacy ``c10d_functional``, and ``_dtensor``.
The list is not curated prose; it is enforced mechanically at arm time in two
layers:

* **Layer 1 (authority):** per-namespace op-set equality against the VETTED
  snapshot table below. Any mismatch -- an op added, removed, or a namespace
  missing -- refuses arming under the ``uncaptured_collective_op`` class.
* **Layer 2 (tripwire for the unknown):** a whole-dispatcher schema scan. Any
  op OUTSIDE the five namespaces whose schema carries the c10d class types
  ``ProcessGroup``, ``Work``, or ``ReduceOp`` refuses arming. This is what
  catches a NEW collective namespace on a future torch.

Disclosed residual, stated honestly: a functional-STYLE namespace with
pure-tensor schemas (like ``_c10d_functional``'s ``group_name: str``
signatures) -- or one-sided ops typed only on other c10d classes such as
``SymmetricMemory`` (``symm_mem::``) -- evades layer 2; the capture-fidelity
census is the version-sensitivity tripwire of last resort for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors._base import CompatibilityError

__all__ = [
    "COLLECTIVE_NAMESPACES",
    "CollectiveRecognizer",
    "UNCAPTURED_COLLECTIVE_OP",
    "UncapturedCollectiveOpError",
    "derive_collective_recognizer",
    "has_vetted_snapshot",
]

UNCAPTURED_COLLECTIVE_OP = "uncaptured_collective_op"
"""Finding kind for arm-time recognizer refusals and runtime allowlist misses."""

COLLECTIVE_NAMESPACES: tuple[str, ...] = (
    "c10d",
    "_c10d_functional",
    "_c10d_functional_autograd",
    "c10d_functional",
    "_dtensor",
)

# Schema type qualifiers whose presence OUTSIDE the five namespaces marks an
# op as an uncaptured collective surface (layer 2). Exactly the three c10d
# class types named by the design; other c10d classes (SymmetricMemory) are
# the disclosed census-caught residual.
_LAYER2_TYPE_MARKERS: tuple[str, ...] = (
    ".c10d.ProcessGroup",
    ".c10d.Work",
    ".c10d.ReduceOp",
)

# Vetted per-namespace dispatcher op sets. One entry per vetted torch surface;
# arming requires the runtime to match one snapshot JOINTLY across all five
# namespaces. A torch upgrade that adds, removes, or renames a collective op
# goes red here, never silent -- extending this table is a reviewed change that
# must ride a capture-fidelity re-census.
VETTED_NAMESPACE_SNAPSHOTS: tuple[tuple[str, dict[str, frozenset[str]]], ...] = (
    (
        "torch-2.13",
        {
            "c10d": frozenset(
                {
                    "_allgather_base_",
                    "_reduce_scatter_base_",
                    "allgather_",
                    "allgather_coalesced_",
                    "allgather_into_tensor_coalesced_",
                    "allreduce_",
                    "allreduce_coalesced_",
                    "alltoall_",
                    "alltoall_base_",
                    "barrier",
                    "check_for_nan",
                    "broadcast_",
                    "gather_",
                    "monitored_barrier_",
                    "recv_",
                    "recv_any_source_",
                    "reduce_",
                    "reduce_scatter_",
                    "reduce_scatter_tensor_coalesced_",
                    "scatter_",
                    "send",
                }
            ),
            "_c10d_functional": frozenset(
                {
                    "_wrap_tensor_autograd",
                    "all_gather_into_tensor",
                    "all_gather_into_tensor_coalesced",
                    "all_gather_into_tensor_out",
                    "all_reduce",
                    "all_reduce_",
                    "all_reduce_coalesced",
                    "all_reduce_coalesced_",
                    "all_to_all_single",
                    "batch_p2p_ops",
                    "broadcast",
                    "broadcast_",
                    "irecv",
                    "isend",
                    "reduce_scatter_tensor",
                    "reduce_scatter_tensor_coalesced",
                    "reduce_scatter_tensor_out",
                    "wait_tensor",
                }
            ),
            "_c10d_functional_autograd": frozenset(
                {
                    "all_gather_into_tensor",
                    "all_to_all_single",
                    "reduce_scatter_tensor",
                }
            ),
            "c10d_functional": frozenset(
                {
                    "all_gather_into_tensor",
                    "all_gather_into_tensor_coalesced",
                    "all_reduce",
                    "all_reduce_coalesced",
                    "all_to_all_single",
                    "batch_p2p_ops",
                    "broadcast",
                    "irecv",
                    "isend",
                    "reduce_scatter_tensor",
                    "reduce_scatter_tensor_coalesced",
                    "wait_tensor",
                }
            ),
            "_dtensor": frozenset({"mesh_get_process_group", "shard_dim_alltoall"}),
        },
    ),
)


class UncapturedCollectiveOpError(CompatibilityError, RuntimeError):
    """Raised when the collective recognizer cannot vouch for this runtime.

    Structured context is retained on ``fields``: ``kind`` is always
    ``"uncaptured_collective_op"``; ``layer`` names which derivation layer
    refused (``1`` for allowlist set-inequality, ``2`` for the dispatcher
    schema scan); ``mismatches`` / ``offending_ops`` carry the evidence.
    Callers branch on these fields, never on message text.
    """


@dataclass(frozen=True)
class CollectiveRecognizer:
    """The armed recognizer: vetted per-namespace collective op sets.

    Parameters
    ----------
    snapshot_name:
        Which vetted snapshot the runtime matched.
    namespace_ops:
        Per-namespace vetted op-name sets, equal to the runtime's dispatcher
        contents at arm time.
    """

    snapshot_name: str
    namespace_ops: dict[str, frozenset[str]]

    def is_collective_namespace(self, namespace: str) -> bool:
        """Whether a dispatcher namespace is one of the enumerated five."""

        return namespace in COLLECTIVE_NAMESPACES

    def classify(self, qualified_op_name: str) -> str | None:
        """Classify a ``namespace::op`` dispatcher name.

        Returns
        -------
        str | None
            ``"collective"`` for a vetted collective op, ``"unknown_collective"``
            for an op inside a collective namespace that the vetted set does
            not know (the caller must raise the typed
            ``uncaptured_collective_op`` refusal and ceiling the rank capture),
            or ``None`` for an op outside the five namespaces.
        """

        namespace, _, op_name = qualified_op_name.partition("::")
        if namespace not in COLLECTIVE_NAMESPACES:
            return None
        if op_name in self.namespace_ops.get(namespace, frozenset()):
            return "collective"
        return "unknown_collective"


# Modules whose import registers collective dispatcher ops lazily. Forced at
# derivation time so the layer-1 comparison sees the FULLY materialized
# surface: without this, the op sets depend on incidental import order (e.g.
# ``_dtensor::mesh_get_process_group`` only registers once
# ``torch.distributed.tensor`` imports) and the equality check would be
# nondeterministic across programs on the SAME torch build.
_REGISTRATION_MODULES: tuple[str, ...] = (
    "torch.distributed.distributed_c10d",
    "torch.distributed._functional_collectives",
    "torch.distributed.tensor",
)


def _force_registration_imports() -> None:
    """Materialize the collective dispatcher surface deterministically."""

    import importlib

    for module_name in _REGISTRATION_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            # A build lacking the module simply presents a smaller surface;
            # layer 1 then refuses against the vetted snapshot, which is the
            # correct fail-closed outcome for an unvetted build.
            continue


def _all_dispatcher_schemas() -> list[Any]:
    """Return every dispatcher schema, or refuse typed when unenumerable.

    r-b4 R26-2: routed through the ``_torch_compat`` accessor
    (``HAS_JIT_SCHEMA_ENUMERATION``). A private-API rename used to escape as a
    raw ``AttributeError`` at ``tl.distributed.arm()``; an unenumerable
    dispatcher now raises the contract's typed ``uncaptured_collective_op``
    refusal -- the census cannot be vetted, so arming must not proceed.
    """

    from ..utils._torch_compat import get_jit_all_schemas

    schemas = get_jit_all_schemas()
    if schemas is None:
        raise UncapturedCollectiveOpError(
            "TorchLens cannot enumerate the torch dispatcher schemas on this build, "
            "so the collective recognizer census cannot be vetted and distributed "
            "arming refuses.",
            kind=UNCAPTURED_COLLECTIVE_OP,
            layer=0,
            reason="jit_schema_enumeration_unavailable",
        )
    return schemas


def _runtime_namespace_sets() -> dict[str, set[str]]:
    """Return the runtime dispatcher's per-namespace op sets for the five."""

    _force_registration_imports()
    sets: dict[str, set[str]] = {namespace: set() for namespace in COLLECTIVE_NAMESPACES}
    for schema in _all_dispatcher_schemas():
        namespace, _, op_name = schema.name.partition("::")
        if namespace in sets:
            sets[namespace].add(op_name)
    return sets


def _layer2_offenders() -> list[str]:
    """Return non-enumerated-namespace ops whose schemas carry c10d classes."""

    offenders: list[str] = []
    for schema in _all_dispatcher_schemas():
        namespace = schema.name.partition("::")[0]
        if namespace in COLLECTIVE_NAMESPACES:
            continue
        rendered = " ".join(str(argument.type) for argument in (*schema.arguments, *schema.returns))
        if any(marker in rendered for marker in _LAYER2_TYPE_MARKERS):
            offenders.append(schema.name)
    return sorted(set(offenders))


def _match_vetted_snapshot(
    runtime_sets: dict[str, set[str]],
) -> tuple[
    tuple[str, dict[str, frozenset[str]]] | None,
    dict[str, dict[str, dict[str, list[str]]]],
]:
    """Layer-1 set-equality match of ``runtime_sets`` against every snapshot.

    Returns ``(matched, per_snapshot_mismatches)``: ``matched`` is the first
    ``(snapshot_name, vetted)`` pair whose five namespace op-sets equal the
    runtime's, or ``None`` when no snapshot matches; ``per_snapshot_mismatches``
    names the added/removed ops per namespace for every snapshot that did not
    match (empty when ``matched`` is not ``None``). Shared by
    :func:`derive_collective_recognizer` (which raises on no match) and
    :func:`has_vetted_snapshot` (which only asks the yes/no question).
    """

    matched: tuple[str, dict[str, frozenset[str]]] | None = None
    per_snapshot_mismatches: dict[str, dict[str, dict[str, list[str]]]] = {}
    for snapshot_name, vetted in VETTED_NAMESPACE_SNAPSHOTS:
        mismatches: dict[str, dict[str, list[str]]] = {}
        for namespace in COLLECTIVE_NAMESPACES:
            runtime_ops = runtime_sets[namespace]
            vetted_ops = vetted[namespace]
            added = sorted(runtime_ops - vetted_ops)
            removed = sorted(vetted_ops - runtime_ops)
            if added or removed:
                mismatches[namespace] = {"added": added, "removed": removed}
        if not mismatches:
            matched = (snapshot_name, vetted)
            break
        per_snapshot_mismatches[snapshot_name] = mismatches
    return matched, per_snapshot_mismatches


def has_vetted_snapshot() -> bool:
    """Capability probe: does this torch build match a censused snapshot?

    Cheap, read-only, and NEVER raises: answers whether the running torch's
    collective dispatcher namespaces (layer 1 of :func:`derive_collective_recognizer`)
    equal one of the rows in :data:`VETTED_NAMESPACE_SNAPSHOTS`. This is a
    TorchLens-owned census fact, not a generic torch capability (it compares
    the runtime against OUR reviewed allowlist, not against a torch API's mere
    presence), so it lives here next to the snapshot table rather than in
    ``torchlens/utils/_torch_compat.py``.

    Use this to gate tests and call sites that require full collective arming
    to succeed: on an unvetted torch, :func:`derive_collective_recognizer` (and
    therefore :func:`torchlens.distributed.arm`) correctly raises
    :class:`UncapturedCollectiveOpError` -- that fail-closed refusal is the
    product's correct behavior, not a bug to route around. A caller that wants
    to assert the refusal itself should call ``arm()``/``derive_collective_recognizer()``
    directly rather than branch on this flag first.
    """

    try:
        runtime_sets = _runtime_namespace_sets()
    except UncapturedCollectiveOpError:
        # Dispatcher schema enumeration itself is unavailable on this build
        # (layer 0): there is no runtime census to match against.
        return False
    matched, _ = _match_vetted_snapshot(runtime_sets)
    return matched is not None


def derive_collective_recognizer() -> CollectiveRecognizer:
    """Derive and verify the collective recognizer against this runtime.

    Returns
    -------
    CollectiveRecognizer
        The armed recognizer, when the runtime's dispatcher matches a vetted
        snapshot (layer 1) and no op outside the five namespaces carries a
        c10d ProcessGroup/Work/ReduceOp-typed schema (layer 2).

    Raises
    ------
    UncapturedCollectiveOpError
        On any layer-1 set inequality or layer-2 scan hit. Arming fails
        closed: an unvetted collective surface must never capture silently.
    """

    runtime_sets = _runtime_namespace_sets()
    matched, per_snapshot_mismatches = _match_vetted_snapshot(runtime_sets)

    if matched is None:
        raise UncapturedCollectiveOpError(
            "torchlens.distributed.arm() refuses: this torch runtime's collective "
            "dispatcher namespaces do not match any vetted snapshot, so TorchLens "
            "cannot promise that every collective op would be captured. "
            f"Per-snapshot set differences: {per_snapshot_mismatches!r}. "
            "Distributed capture on this torch version requires a reviewed "
            "allowlist update plus a capture-fidelity re-census.",
            kind=UNCAPTURED_COLLECTIVE_OP,
            layer=1,
            mismatches=per_snapshot_mismatches,
        )

    offenders = _layer2_offenders()
    if offenders:
        raise UncapturedCollectiveOpError(
            "torchlens.distributed.arm() refuses: dispatcher op(s) outside the "
            "five enumerated collective namespaces carry c10d "
            "ProcessGroup/Work/ReduceOp-typed schemas and would evade collective "
            f"capture: {offenders}. This torch surface requires a reviewed "
            "recognizer update plus a capture-fidelity re-census.",
            kind=UNCAPTURED_COLLECTIVE_OP,
            layer=2,
            offending_ops=offenders,
        )

    snapshot_name, vetted = matched
    return CollectiveRecognizer(snapshot_name=snapshot_name, namespace_ops=dict(vetted))
