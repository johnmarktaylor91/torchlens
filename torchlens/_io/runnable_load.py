"""Safe sparse runnable descriptor loading and torch callable reattachment.

This module performs readiness preflight only. It never imports artifact-selected
modules, binds state, constructs runtime calls, or executes a recorded graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import inspect
import operator
import platform
from typing import Any, cast

import torch

from .. import _state
from ._torch_symbols import torch_attr
from ..constants import get_orig_torch_funcs
from ..intervention.types import FunctionRegistryKey
from ..runnable import (
    LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS,
    RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
    RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
    RUNNABLE_CALL_RECIPE_VERSION,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    AmbientExecutionContext,
    AutocastDeviceContext,
    CallControlObligation,
    CallExecutionContext,
    CallableRegistryEntry,
    ControlDependencyEdge,
    InputAttestationFingerprint,
    InputBoundarySite,
    InputBoundaryTensorSite,
    ControlWitness,
    ControlWitnessKind,
    InputSlotBinding,
    ReplayWitnessStructure,
    WITNESS_GAP_REGISTRY,
    WitnessCoverageGap,
    WitnessGapKind,
    derive_required_witness_members,
    derived_witness_completeness,
    LiteralArgumentRef,
    LiteralAtom,
    LiteralAtomKind,
    LiteralMapping,
    LiteralMappingEntry,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralSlice,
    LiteralTorchSymbol,
    LiteralTupleKey,
    NonTensorLiteral,
    PayloadLayerDescriptor,
    PayloadLayersDescriptor,
    ProducerPreflight,
    ReadinessReport,
    ReadinessStatus,
    RequiredWitnessFamily,
    RequiredWitnessInventory,
    ResolverRecord,
    ResolverStatus,
    RunProvider,
    RunnableCallDescriptor,
    RunnableCompatibility,
    RunnableDiagnostic,
    RunnableErrorCode,
    RunnableRngProfile,
    SparseRunDescriptor,
    SlotByteDigest,
    StateByteDigest,
    StateSlotBinding,
    StateSlotRole,
    StateSource,
    TensorArgumentRef,
    TensorSlotDescriptor,
    TensorSlotRole,
    TensorUseSite,
    WITNESS_FAMILY_REGISTRY,
    WITNESS_FAMILY_REGISTRY_VERSION,
    WitnessCompleteness,
    decode_input_site_position,
)
from ..utils._callable_safety import (
    _PURE_TENSOR_PROPERTY_NAMES,
    is_pure_forward_callable,
    unsafe_callable_reason,
)
from ..utils._torch_compat import resolve_runnable_torch_alias


_ALLOWED_EXACT_ROOTS: Mapping[str, Any] = {
    "torch": torch,
    "torch.Tensor": torch.Tensor,
    "torch.nn.functional": torch.nn.functional,
    "operator": operator,
}
_ENUMERATED_TORCH_NAMESPACES = frozenset(
    {
        "torch.fft",
        "torch.linalg",
        "torch.nested",
        "torch.special",
        "torch.nn.functional",
        "torch.Tensor",
        "torch._C._nn",
        "torch._C._special",
        "torch._C._VariableFunctions",
        "torch._C._VariableFunctionsClass",
        "torch._C._TensorBase",
        "torch._C.TensorBase",
        "torch._VF",
    }
)
_REMOVED_TORCH_CALLABLES = frozenset({"torch.gesv"})
# Canonical copy lives in ``torchlens.utils._callable_safety`` so the capture-side
# keyer, this resolver, and the security gate's recognized-operator predicate can
# never drift apart on the safe pure-read property surface.
_SAFE_TENSOR_PROPERTY_NAMES = _PURE_TENSOR_PROPERTY_NAMES


@dataclass(frozen=True, slots=True)
class _ReverseCandidate:
    """One reverse-index callable candidate and its stable display path."""

    namespace: str
    qualname: str
    func: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _Resolution:
    """Internal result of resolving one callable registry entry."""

    record: ResolverRecord
    func: Callable[..., Any] | None


def parse_sparse_run_descriptor(value: Mapping[str, Any]) -> SparseRunDescriptor:
    """Parse a schema-validated manifest run object into frozen descriptor types.

    Parameters
    ----------
    value:
        Decoded ``manifest.run`` mapping after structural manifest validation.

    Returns
    -------
    SparseRunDescriptor
        Typed sparse descriptor. Version values are retained verbatim so the
        readiness preflight can report unsupported ceilings without executing.
    """

    registry = tuple(
        CallableRegistryEntry(
            registry_id=_string(entry, "registry_id"),
            key=_parse_registry_key(_mapping(entry, "key")),
        )
        for entry in _mapping_sequence(value, "callable_registry")
    )
    calls = tuple(_parse_call(item) for item in _mapping_sequence(value, "calls"))
    slots = tuple(_parse_slot(item) for item in _mapping_sequence(value, "tensor_slots"))
    witnesses = tuple(
        _parse_witness(item) for item in _mapping_sequence(value, "control_witnesses")
    )
    _validate_state_metadata_fact_witnesses(witnesses)
    _validate_input_structure_witnesses(witnesses)
    # r71 A: the REQUIRED witness-free replay-structure records + the explicit gap
    # ledger parse BEFORE the presence ledger; the comprehensive obligation/discharge
    # validation runs on the fully typed descriptor below.
    input_boundary = _parse_input_boundary(value.get("input_boundary"))
    coverage_gaps = _parse_coverage_gaps(value.get("coverage_gaps"))
    # r69 A parse order: slots/witness syntax first (above), then the REQUIRED
    # descriptor-native presence ledger, then independently derived/cross-family
    # sets and exact family/member equality (inside the validator).
    inventory = _parse_required_witness_inventory(value.get("required_witness_inventory"))
    _validate_required_witness_inventory(inventory, witnesses, slots)
    payload = _mapping(value, "payload_layers")
    compatibility = _mapping(value, "compatibility")
    preflight = _mapping(value, "preflight")
    # v2: the capture-scoped ambient execution context is REQUIRED and EXPLICIT.
    # Absence is a parse failure (fail closed), never a defaulted context.
    ambient = _parse_ambient_context(_mapping(value, "ambient_context"))
    descriptor = SparseRunDescriptor(
        capability=cast(Any, _string(value, "capability")),
        backend=_string(value, "backend"),
        call_recipe=cast(Any, _string(value, "call_recipe")),
        callable_ref_schema=cast(Any, _integer(value, "callable_ref_schema")),
        state_binding=cast(Any, _string(value, "state_binding")),
        input_binding=cast(Any, _string(value, "input_binding")),
        control_witness=cast(Any, _string(value, "control_witness")),
        initializer_policy_version=cast(Any, _string(value, "initializer_policy_version")),
        payload_layers=PayloadLayersDescriptor(
            weights=_parse_payload_layer(_mapping(payload, "weights")),
            nonpersistent_buffers=_parse_nonpersistent_buffer_payload_layer(payload),
            activations=_parse_activation_payload_layer(_mapping(payload, "activations")),
        ),
        callable_registry=registry,
        calls=calls,
        tensor_slots=slots,
        input_boundary=input_boundary,
        control_witnesses=witnesses,
        coverage_gaps=coverage_gaps,
        required_witness_inventory=inventory,
        witness_completeness=WitnessCompleteness(_string(value, "witness_completeness")),
        rng_profile=_parse_rng_profile(value.get("rng_profile")),
        ambient_context=ambient,
        compatibility=RunnableCompatibility(
            torchlens_version=_string(compatibility, "torchlens_version"),
            python_version=_string(compatibility, "python_version"),
            backend_version=_string(compatibility, "backend_version"),
            descriptor_version=_string(compatibility, "descriptor_version"),
            call_recipe_version=_string(compatibility, "call_recipe_version"),
            callable_ref_schema_version=_integer(compatibility, "callable_ref_schema_version"),
            initializer_policy_version=_string(compatibility, "initializer_policy_version"),
        ),
        preflight=ProducerPreflight(
            passed=_boolean(preflight, "passed"),
            diagnostics=tuple(
                _parse_diagnostic(item) for item in _mapping_sequence(preflight, "diagnostics")
            ),
        ),
        unsupported_sites=tuple(
            _parse_diagnostic(item) for item in _mapping_sequence(value, "unsupported_sites")
        ),
    )
    # r71 A: the comprehensive obligation/discharge validation on the fully typed
    # descriptor -- independent structural derivation, terminal-slot totality, gap
    # ledger coherence, and the parser-derived completeness floor. The container
    # family's independent anchor is enforced at readiness attach (where the
    # rehydrated container records are in scope).
    validate_witness_obligations(descriptor, container_members=None)
    return descriptor


class ContextFieldInvalidError(ValueError):
    """A persisted execution-context field failed closed-vocabulary validation (INV-4).

    Raised at PARSE time -- before readiness, staging, or any torch setter/callable
    can observe the attacker-controllable bytes -- and surfaced as the frozen
    ``context_field_invalid`` readiness diagnostic.
    """

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"Persisted execution-context field {field!r} is invalid: {detail}")
        self.field = field
        self.detail = detail


_STATE_METADATA_FACT_SITE_PREFIX = "state_metadata:"
"""``site_label`` prefix of a persisted declared state-metadata fact witness (r65 F-1)."""

_STATE_METADATA_FACT_ALLOWED_NAMES = frozenset({"requires_grad", "grad_fn"})
"""CLOSED parse-side vocabulary of declared state-metadata fact names (r65 F-1).

Mirrors ``torchlens._io.runnable._STATE_METADATA_FACT_NAMES``; any other name -- or a
non-bool value, a malformed envelope, a site label disagreeing with the embedded state
name -- refuses the descriptor at parse (``context_field_invalid``), before run
preparation could apply an attacker-chosen bit to staged state.
"""


_INPUT_STRUCTURE_SITE_PREFIX = "input_structure:"
"""``site_label`` prefix of a persisted input-boundary structure fact (r67 C2)."""

_INPUT_STRUCTURE_NODE_KINDS = frozenset(
    {"tensor", "empty", "namedtuple", "dataclass", "mapping", "sequence", "registered", "leaf"}
)
"""Closed node-kind vocabulary accepted from a persisted input-structure fact."""


def _validate_input_structure_witnesses(witnesses: "Sequence[ControlWitness]") -> None:
    """Validate the REQUIRED input-boundary structure facts at PARSE time (r67 C2).

    The complete structure block is required and parse-validated inside existing v2:
    missing, duplicate, malformed, stripped, or internally inconsistent site/node/key
    facts make readiness unavailable through the existing ``context_field_invalid``
    typed path -- absence may never mean the old weaker semantics. Consistency proved
    here: every fact declares the SAME ``site_count``; positions are unique and
    well-formed; the fact count equals the declared count (a stripped per-site fact
    fails set equality); every node record carries a closed-vocabulary kind, an exact
    two-string type ref, and a ``str | int`` component path.
    """

    from .._runnable_execution import _decode_literal  # lazy: layering, not a cycle at import

    field = "control_witnesses.input_structure"
    facts: list[Mapping[str, Any]] = []
    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_INPUT_STRUCTURE_SITE_PREFIX):
            continue
        try:
            decoded = _decode_literal(witness.observed_value)
        except Exception as exc:
            raise ContextFieldInvalidError(field, f"undecodable input-structure fact: {exc}")
        if not isinstance(decoded, Mapping) or decoded.get("input_structure") is not True:
            raise ContextFieldInvalidError(field, "malformed input-structure fact envelope")
        facts.append(decoded)
    if not facts:
        return
    declared_counts = {int(fact.get("site_count", -1)) for fact in facts}
    if len(declared_counts) != 1 or next(iter(declared_counts)) != len(facts):
        raise ContextFieldInvalidError(
            field,
            "input-structure site set is incomplete or inconsistent "
            f"(declared {sorted(declared_counts)}, present {len(facts)})",
        )
    positions: set[tuple[Any, ...]] = set()
    for fact in facts:
        position = fact.get("position")
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
            or position[0] not in {"arg", "kwarg"}
            or not isinstance(position[1], (str, int))
        ):
            raise ContextFieldInvalidError(field, f"malformed site position {position!r}")
        position_key = tuple(position)
        if position_key in positions:
            raise ContextFieldInvalidError(field, f"duplicate site position {position!r}")
        positions.add(position_key)
        nodes = fact.get("nodes")
        if not isinstance(nodes, (list, tuple)) or not nodes:
            raise ContextFieldInvalidError(field, f"site {position!r} has no node records")
        root_seen = False
        for node in nodes:
            if not isinstance(node, Mapping):
                raise ContextFieldInvalidError(field, "malformed input-structure node record")
            node_path = node.get("path")
            if not isinstance(node_path, (list, tuple)) or any(
                not isinstance(component, (str, int)) for component in node_path
            ):
                raise ContextFieldInvalidError(field, f"malformed node path {node_path!r}")
            if len(node_path) == 0:
                root_seen = True
            node_kind = node.get("kind")
            if node_kind not in _INPUT_STRUCTURE_NODE_KINDS:
                raise ContextFieldInvalidError(field, f"unknown node kind {node_kind!r}")
            type_ref = node.get("type")
            if node_kind in {"tensor", "leaf"}:
                # Leaf-value semantics belong to the tensor/literal VALUE contracts; a
                # type fact here would be a forged extra comparison surface.
                if type_ref is not None:
                    raise ContextFieldInvalidError(
                        field, f"{node_kind} node must not carry a type ref"
                    )
            elif (
                not isinstance(type_ref, (list, tuple))
                or len(type_ref) != 2
                or not all(isinstance(part, str) and part for part in type_ref)
            ):
                raise ContextFieldInvalidError(field, f"malformed node type ref {type_ref!r}")
        if not root_seen:
            raise ContextFieldInvalidError(field, f"site {position!r} lacks a root node record")


def _validate_state_metadata_fact_witnesses(witnesses: "Sequence[ControlWitness]") -> None:
    """Validate every declared state-metadata fact witness at PARSE time (r65 F-1).

    Fact names validate against the closed two-name vocabulary and values must be bools;
    run preparation applies ``requires_grad`` facts to staged state, so a malformed fact is
    refused HERE -- ``context_field_invalid``-class, analysis-only load -- never consumed.
    """

    from .._runnable_execution import _decode_literal  # lazy: layering, not a cycle at import

    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_STATE_METADATA_FACT_SITE_PREFIX):
            continue
        field = "control_witnesses.state_metadata"
        try:
            decoded = _decode_literal(witness.observed_value)
        except Exception as exc:
            raise ContextFieldInvalidError(field, f"undecodable state-metadata fact: {exc}")
        if not isinstance(decoded, Mapping) or decoded.get("state_metadata") is not True:
            raise ContextFieldInvalidError(field, "malformed state-metadata fact envelope")
        name = decoded.get("state")
        if not isinstance(name, str) or not name:
            raise ContextFieldInvalidError(field, "state-metadata fact names no state entry")
        if witness.site_label != f"{_STATE_METADATA_FACT_SITE_PREFIX}{name}":
            raise ContextFieldInvalidError(
                field, "state-metadata fact site label disagrees with its state entry"
            )
        facts = decoded.get("facts")
        if not isinstance(facts, Mapping) or not facts:
            raise ContextFieldInvalidError(field, "state-metadata fact carries no facts")
        for fact_name, fact_value in facts.items():
            if str(fact_name) not in _STATE_METADATA_FACT_ALLOWED_NAMES:
                raise ContextFieldInvalidError(
                    field,
                    f"state-metadata fact name {fact_name!r} is outside the closed vocabulary",
                )
            if not isinstance(fact_value, bool):
                raise ContextFieldInvalidError(
                    field, f"state-metadata fact {fact_name!r} carries a non-bool value"
                )


_REQUIRED_WITNESS_INVENTORY_FIELD = "required_witness_inventory"
"""Manifest/diagnostic field name of the r69 A presence ledger."""


def _parse_required_witness_inventory(value: Any) -> RequiredWitnessInventory:
    """Parse the REQUIRED descriptor-native presence ledger (r69 A).

    Missing inventory, unknown registry discriminator, malformed rows, missing/extra/
    duplicate families, non-string or duplicate member IDs, a disposition disagreeing
    with the closed registry, and members on an anchored family all refuse
    ``context_field_invalid`` -- analysis load survives, readiness is unavailable,
    execution cannot attach. Absence NEVER means the old weaker semantics.
    """

    field = _REQUIRED_WITNESS_INVENTORY_FIELD
    if not isinstance(value, Mapping):
        raise ContextFieldInvalidError(field, "required witness inventory is missing or malformed")
    version = value.get("registry_version")
    if version != WITNESS_FAMILY_REGISTRY_VERSION:
        raise ContextFieldInvalidError(
            field, f"unknown witness-family registry discriminator {version!r}"
        )
    raw_families = value.get("families")
    if not isinstance(raw_families, Sequence) or isinstance(raw_families, (str, bytes)):
        raise ContextFieldInvalidError(field, "malformed inventory family rows")
    rows: list[RequiredWitnessFamily] = []
    seen: set[str] = set()
    for raw in raw_families:
        if not isinstance(raw, Mapping):
            raise ContextFieldInvalidError(field, "malformed inventory family row")
        family = raw.get("family")
        disposition = raw.get("disposition")
        members = raw.get("members")
        if not isinstance(family, str) or family not in WITNESS_FAMILY_REGISTRY:
            raise ContextFieldInvalidError(field, f"unknown witness family {family!r}")
        if family in seen:
            raise ContextFieldInvalidError(field, f"duplicate witness family row {family!r}")
        seen.add(family)
        if disposition != WITNESS_FAMILY_REGISTRY[family].disposition:
            raise ContextFieldInvalidError(
                field, f"family {family!r} declares disposition {disposition!r}"
            )
        if (
            not isinstance(members, Sequence)
            or isinstance(members, (str, bytes))
            or not all(isinstance(member, str) for member in members)
        ):
            raise ContextFieldInvalidError(field, f"malformed member set for {family!r}")
        member_tuple = tuple(members)
        if len(set(member_tuple)) != len(member_tuple):
            raise ContextFieldInvalidError(field, f"duplicate member identity in family {family!r}")
        if WITNESS_FAMILY_REGISTRY[family].disposition == "independent_ceiling" and member_tuple:
            raise ContextFieldInvalidError(
                field,
                f"family {family!r} is anchored by an independent structural proof "
                "and carries no inventory members",
            )
        rows.append(
            RequiredWitnessFamily(family=family, disposition=str(disposition), members=member_tuple)
        )
    if seen != set(WITNESS_FAMILY_REGISTRY):
        raise ContextFieldInvalidError(
            field,
            f"inventory family set {sorted(seen)!r} does not equal the closed "
            f"registry {sorted(WITNESS_FAMILY_REGISTRY)!r}",
        )
    return RequiredWitnessInventory(registry_version=str(version), families=tuple(rows))


def _validate_required_witness_inventory(
    inventory: RequiredWitnessInventory,
    witnesses: "Sequence[ControlWitness]",
    slots: "Sequence[TensorSlotDescriptor]",
) -> None:
    """Require EXACT family+member coverage plus the independent cross-checks (r69 A).

    The present member sets are re-derived from the parsed witnesses with the SAME
    single-source function the producer authored the inventory from
    (``torchlens._io.runnable.required_witness_family_members``); exact set equality
    is required for every ``inventory_indexed`` family, so stripping, duplicating,
    or forging any fact refuses ``context_field_invalid``. Independent cross-checks
    anchor the ledger to the rest of the descriptor: every tensor ``MODEL_INPUT``
    binding position, literal-fact position, and metadata-fact position must sit
    inside the inventory site set; positional roots are dense; ``site_count``
    values are redundant consistency only; and literal-leaf paths cross-anchor
    BIDIRECTIONALLY with the structure snapshots' leaf/empty nodes, so per-leaf
    literal stripping cannot hide behind site-level coverage.
    """

    from .._runnable_execution import _decode_literal
    from .runnable import (
        EMPTY_CONTAINER_PATH_MARKER,
        required_witness_family_members,
        witness_family_of,
    )

    field = _REQUIRED_WITNESS_INVENTORY_FIELD
    # Family closure: every SHAPE_STRUCTURE_FACT witness must belong to a registered
    # family -- an unregistered fact family can never ride a v2 descriptor silently.
    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if witness_family_of(witness.site_label) is None:
            raise ContextFieldInvalidError(
                field,
                f"witness site label {witness.site_label!r} belongs to no registered "
                "replay-critical family",
            )
    try:
        present = required_witness_family_members(witnesses)
    except ValueError as exc:
        raise ContextFieldInvalidError(field, f"malformed witness envelope: {exc}")
    rows = {row.family: row for row in inventory.families}
    for family, spec in WITNESS_FAMILY_REGISTRY.items():
        if spec.disposition != "inventory_indexed":
            continue
        present_members = present[family]
        if len(set(present_members)) != len(present_members):
            raise ContextFieldInvalidError(
                field, f"duplicate {family!r} member identity among present facts"
            )
        if sorted(present_members) != list(rows[family].members):
            raise ContextFieldInvalidError(
                field,
                f"{family!r} facts do not equal the required inventory (declared "
                f"{list(rows[family].members)[:8]!r}, present "
                f"{sorted(present_members)[:8]!r})",
            )
    # ---- independent input-site cross-checks -------------------------------------
    try:
        sites = {decode_input_site_position(member) for member in rows["input_structure"].members}
    except ValueError as exc:
        raise ContextFieldInvalidError(field, str(exc))
    arg_indices = sorted(key for kind, key in sites if kind == "arg")
    if arg_indices != list(range(len(arg_indices))):
        raise ContextFieldInvalidError(
            field, f"positional input sites are not dense: {arg_indices[:8]!r}"
        )
    for slot in slots:
        if slot.role is not TensorSlotRole.MODEL_INPUT or slot.input_binding is None:
            continue
        position = slot.input_binding.model_site_position
        if not isinstance(position, tuple) or len(position) != 2 or tuple(position) not in sites:
            raise ContextFieldInvalidError(
                field,
                f"tensor MODEL_INPUT binding position {position!r} is outside the "
                "required input-site inventory",
            )
    # ---- bidirectional literal-leaf <-> structure-leaf cross-anchor --------------
    structure_expected: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    literal_present: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    literal_claims: list[tuple[Any, ...]] = []
    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        fact_family = witness_family_of(witness.site_label)
        if fact_family == "input_structure":
            fact = _decode_literal(witness.observed_value)
            site_position = _fact_path_tuple(
                field, fact.get("position", ()), "structure fact position"
            )
            expected: set[tuple[Any, ...]] = set()
            for node in fact.get("nodes", ()):
                if not isinstance(node, Mapping):
                    continue
                node_path = _fact_path_tuple(field, node.get("path", ()), "structure node path")
                if node.get("kind") == "leaf":
                    expected.add(node_path)
                elif node.get("kind") == "empty":
                    expected.add((*node_path, EMPTY_CONTAINER_PATH_MARKER))
            structure_expected[site_position] = expected
            declared_count = fact.get("site_count")
            if declared_count != len(sites):
                raise ContextFieldInvalidError(
                    field,
                    f"site {site_position!r} declares site_count={declared_count!r}, but the "
                    f"required inventory proves {len(sites)} sites (site_count is a "
                    "redundant consistency value, never authority)",
                )
        elif fact_family == "model_input_literal":
            fact = _decode_literal(witness.observed_value)
            if not isinstance(fact, Mapping):
                raise ContextFieldInvalidError(field, "undecodable model-input literal fact")
            site_position = _fact_position_tuple(
                field, fact.get("position"), "literal fact position"
            )
            if site_position not in sites:
                raise ContextFieldInvalidError(
                    field,
                    f"literal fact position {site_position!r} is outside the required "
                    "input-site inventory",
                )
            path = _fact_path_tuple(field, fact.get("path", ()) or (), "literal fact path")
            claim = (site_position, path)
            if claim in literal_claims:
                raise ContextFieldInvalidError(field, f"duplicate literal fact identity {claim!r}")
            literal_claims.append(claim)
            literal_present.setdefault(site_position, set()).add(path)
        elif fact_family == "model_input_metadata":
            fact = _decode_literal(witness.observed_value)
            if not isinstance(fact, Mapping):
                raise ContextFieldInvalidError(field, "undecodable model-input metadata fact")
            site_position = _fact_position_tuple(
                field, fact.get("position"), "metadata fact position"
            )
            if site_position not in sites:
                raise ContextFieldInvalidError(
                    field,
                    f"metadata fact position {site_position!r} is outside the required "
                    "input-site inventory",
                )
    if set(structure_expected) != sites:
        raise ContextFieldInvalidError(
            field,
            f"structure fact positions {sorted(structure_expected, key=repr)!r} do not "
            f"equal the required inventory sites {sorted(sites, key=repr)!r}",
        )
    for site_position, expected in structure_expected.items():
        actual = literal_present.get(site_position, set())
        if actual != expected:
            missing = sorted(expected - actual, key=repr)[:4]
            extra = sorted(actual - expected, key=repr)[:4]
            raise ContextFieldInvalidError(
                field,
                f"literal facts at site {site_position!r} do not cross-anchor the structure "
                f"leaf nodes (missing {missing!r}, extra {extra!r}); per-leaf literal "
                "stripping cannot hide behind site-level coverage",
            )
    for site_position in literal_present:
        if site_position not in structure_expected:
            raise ContextFieldInvalidError(
                field,
                f"literal facts at site {site_position!r} have no structure fact to anchor",
            )


def _fact_path_tuple(field: str, raw: Any, description: str) -> tuple[Any, ...]:
    """Return a decoded fact's path/position components, refusing non-sequences typed (r75 L1).

    An int-encoded (or otherwise non-sequence) ``path`` in a foreign artifact's
    metadata/literal envelope crashed ``tuple(...)`` with a ``TypeError`` into the
    generic parse catch-all (``run_capability_unavailable``) instead of the closed-
    vocabulary ``context_field_invalid`` analysis-only lane its sibling ``position``
    checks use. A string is refused too: iterating it would silently shred one
    component into characters instead of refusing the malformed encoding.

    r77 L1 extends the same check one nesting level down: the COMPONENTS are decoded
    literals that may be nested list/mapping/slice values -- all unhashable -- which
    crashed the first downstream ``set``/``dict`` consumer with a ``TypeError`` into
    the same untyped catch-all. Every component must be a ``str``/``int`` scalar
    (the r67 structure-node-path precedent); anything else refuses typed here.
    """

    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ContextFieldInvalidError(
            field, f"{description} {raw!r} is not a sequence of path components"
        )
    for component in raw:
        if not isinstance(component, (str, int)):
            raise ContextFieldInvalidError(
                field, f"{description} component {component!r} is not a str/int scalar"
            )
    return tuple(raw)


def _fact_position_tuple(field: str, raw: Any, description: str) -> tuple[Any, ...]:
    """Return a decoded fact's site position as a component-validated tuple (r77 L1).

    The former inline coercions (``tuple(raw) if isinstance(raw, (list, tuple)) else
    (raw,)``) validated only the CONTAINER type; a position carrying a nested
    list/mapping/slice component crashed the inventory-membership ``in set`` check
    with a ``TypeError`` into the untyped parse catch-all. Components must be
    ``str``/``int`` scalars (a well-formed position is ``("arg", i)`` /
    ``("kwarg", name)``); a scalar position is wrapped so the existing inventory
    membership check refuses it typed as before.
    """

    coerced = tuple(raw) if isinstance(raw, (list, tuple)) else (raw,)
    for component in coerced:
        if not isinstance(component, (str, int)):
            raise ContextFieldInvalidError(
                field, f"{description} component {component!r} is not a str/int scalar"
            )
    return coerced


def _parse_input_boundary(value: Any) -> tuple[InputBoundarySite, ...]:
    """Parse the REQUIRED input-boundary record (r71 A2).

    Missing record, malformed sites, duplicate positions, or malformed tensor-site
    entries refuse ``context_field_invalid`` (analysis-only). Absence never means the
    old inventory-authority semantics.
    """

    field = "input_boundary"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContextFieldInvalidError(field, "input-boundary record is missing or malformed")
    sites: list[InputBoundarySite] = []
    seen_positions: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ContextFieldInvalidError(field, "malformed input-boundary site record")
        position = raw.get("position")
        if not isinstance(position, str):
            raise ContextFieldInvalidError(field, f"malformed site position {position!r}")
        try:
            decode_input_site_position(position)
        except ValueError as exc:
            raise ContextFieldInvalidError(field, str(exc))
        if position in seen_positions:
            raise ContextFieldInvalidError(field, f"duplicate boundary site {position!r}")
        seen_positions.add(position)
        raw_tensor_sites = raw.get("tensor_sites")
        if not isinstance(raw_tensor_sites, Sequence) or isinstance(raw_tensor_sites, (str, bytes)):
            raise ContextFieldInvalidError(field, f"malformed tensor sites at {position!r}")
        tensor_sites: list[InputBoundaryTensorSite] = []
        seen_paths: set[tuple[Any, ...]] = set()
        for raw_site in raw_tensor_sites:
            if not isinstance(raw_site, Mapping):
                raise ContextFieldInvalidError(field, "malformed boundary tensor-site record")
            path = _path(raw_site, "container_path")
            if path in seen_paths:
                raise ContextFieldInvalidError(
                    field, f"duplicate tensor container path {path!r} at {position!r}"
                )
            seen_paths.add(path)
            reads = raw_site.get("metadata_reads")
            if (
                not isinstance(reads, Sequence)
                or isinstance(reads, (str, bytes))
                or not all(isinstance(name, str) for name in reads)
            ):
                raise ContextFieldInvalidError(
                    field, f"malformed metadata-read set at {position!r}"
                )
            read_tuple = tuple(reads)
            if list(read_tuple) != sorted(set(read_tuple)):
                raise ContextFieldInvalidError(
                    field, f"metadata-read set at {position!r} is not a sorted unique set"
                )
            tensor_sites.append(
                InputBoundaryTensorSite(
                    container_path=path,
                    slot_id=_string(raw_site, "slot_id"),
                    metadata_reads=read_tuple,
                )
            )
        sites.append(InputBoundarySite(position=position, tensor_sites=tuple(tensor_sites)))
    return tuple(sites)


def _parse_coverage_gaps(value: Any) -> tuple[WitnessCoverageGap, ...]:
    """Parse the REQUIRED explicit gap ledger against the closed gap registry (r71 A3).

    Unknown gap kinds, a source family or resulting completeness disagreeing with the
    registry row, non-dense orders, or malformed members refuse
    ``context_field_invalid`` (analysis-only).
    """

    field = "coverage_gaps"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContextFieldInvalidError(field, "coverage-gap ledger is missing or malformed")
    gaps: list[WitnessCoverageGap] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ContextFieldInvalidError(field, "malformed coverage-gap record")
        kind_text = _string(raw, "gap_kind")
        try:
            kind = WitnessGapKind(kind_text)
        except ValueError:
            raise ContextFieldInvalidError(
                field, f"gap kind {kind_text!r} is outside the closed gap registry"
            )
        spec = WITNESS_GAP_REGISTRY[kind]
        source_family = _string(raw, "source_family")
        if source_family != spec.source_family:
            raise ContextFieldInvalidError(
                field,
                f"gap {kind_text!r} declares source family {source_family!r}; the "
                f"registry requires {spec.source_family!r}",
            )
        resulting = _string(raw, "resulting_completeness")
        if resulting != spec.resulting_completeness.value:
            raise ContextFieldInvalidError(
                field,
                f"gap {kind_text!r} declares resulting completeness {resulting!r}; "
                f"the registry requires {spec.resulting_completeness.value!r}",
            )
        member = _string(raw, "source_member")
        if not member:
            raise ContextFieldInvalidError(field, f"gap {kind_text!r} names no source member")
        gaps.append(
            WitnessCoverageGap(
                gap_kind=kind,
                source_family=source_family,
                source_member=member,
                order=_integer(raw, "order"),
                resulting_completeness=WitnessCompleteness(resulting),
            )
        )
    orders = sorted(gap.order for gap in gaps)
    if orders != list(range(len(gaps))):
        raise ContextFieldInvalidError(field, "coverage-gap orders are not dense from zero")
    return tuple(gaps)


def validate_witness_obligations(
    descriptor: SparseRunDescriptor,
    *,
    container_members: "tuple[str, ...] | None",
) -> None:
    """Enforce the r71 obligation/discharge invariant over one typed descriptor.

    THE shared comprehensive validator: the parser runs it on every load (with the
    container anchor deferred to readiness attach, where the rehydrated container
    records exist) and the producer runs the SAME function as its save-time
    self-check (with the container members from the live trace). It rebuilds the
    witness-free :class:`ReplayWitnessStructure`, re-derives every family's required
    members via ``derive_required_witness_members``, and requires exact discharge --
    an exact witness XOR a typed source-linked gap per the family's registry-v2
    discharge rule -- plus:

    * witness ``order`` density (0..N-1) and ``witness:<order+1>`` id consistency
      (any raw deletion trips; renumbering = coherent reauthoring);
    * strictly increasing unique ``call:<n>`` ids and full referential closure
      (obligation slots, arm-edge labels, boundary slots resolve against the
      descriptor's own calls/slots);
    * terminal-slot totality: every call-produced slot consumed by no call and not
      bound by the output contract is claimed by EXACTLY one of {scalar_bool,
      loop_predicate, tensor_derived_scalar_literal, inert_sink, typed gap};
    * the E2 conditional<->predicate pairing (every conditional id in arm edges owns
      a same-conditional predicate obligation XOR a predicate-class gap);
    * metadata-envelope/owner agreement (the totalized envelope's fact-name set
      equals the boundary record's declared reads) and boundary<->binding equality;
    * state-fact witness/binding agreement + unbound-state disposition totality;
    * the parser-derived completeness FLOOR: the persisted summary must EQUAL
      ``derived_witness_completeness(coverage_gaps)`` -- a stronger claim is an
      internally contradictory descriptor.

    Every violation raises :class:`ContextFieldInvalidError` (analysis-only load /
    typed producer save refusal). The witness stream, the inventory, and the summary
    are never consulted for their own required coverage.
    """

    from .._runnable_execution import _decode_literal

    field = "witness_obligations"
    witnesses = descriptor.control_witnesses
    calls = descriptor.calls
    slots = descriptor.tensor_slots
    slot_by_id = {slot.slot_id: slot for slot in slots}
    call_by_id = {call.call_id: call for call in calls}
    # ---- witness order density + id consistency ----------------------------------
    orders = sorted(witness.order for witness in witnesses)
    if orders != list(range(len(witnesses))):
        raise ContextFieldInvalidError(field, "witness orders are not dense from zero")
    for witness in witnesses:
        if witness.witness_id != f"witness:{witness.order + 1}":
            raise ContextFieldInvalidError(
                field,
                f"witness id {witness.witness_id!r} disagrees with its order {witness.order!r}",
            )
    # ---- call id monotonicity + uniqueness + referential closure ------------------
    numbers: list[int] = []
    for call in calls:
        prefix, separator, suffix = call.call_id.partition(":")
        if prefix != "call" or not separator or not suffix.isdigit():
            raise ContextFieldInvalidError(field, f"malformed call id {call.call_id!r}")
        numbers.append(int(suffix))
    if any(later <= earlier for earlier, later in zip(numbers, numbers[1:])):
        raise ContextFieldInvalidError(field, "call ids are not strictly increasing")
    op_label_set = {label for call in calls for label in call.op_labels}
    slot_label_set = {
        slot.slot_id[len("slot:") :] for slot in slots if slot.slot_id.startswith("slot:")
    }
    # Arm-edge endpoints are pass-free LAYER labels; op/slot labels carry the pass
    # suffix. Referential integrity resolves both spellings, exactly like the
    # producer's attachment and the runtime ``_op_for_label`` lookup.
    label_aliases = set(op_label_set) | set(slot_label_set)
    for label in list(label_aliases):
        if ":" in label:
            label_aliases.add(label.rsplit(":", 1)[0])
    for call in calls:
        call_label_aliases = set(call.op_labels) | {
            label.rsplit(":", 1)[0] for label in call.op_labels if ":" in label
        }
        for parent_id in call.parent_call_ids:
            if parent_id not in call_by_id:
                raise ContextFieldInvalidError(
                    field, f"call {call.call_id!r} names unknown parent {parent_id!r}"
                )
        for obligation in call.control_obligations:
            if obligation.output_slot_id not in slot_by_id:
                raise ContextFieldInvalidError(
                    field,
                    f"control obligation on {call.call_id!r} names unknown slot "
                    f"{obligation.output_slot_id!r}",
                )
            if obligation.output_slot_id not in set(call.output_slot_ids):
                raise ContextFieldInvalidError(
                    field,
                    f"control obligation on {call.call_id!r} names slot "
                    f"{obligation.output_slot_id!r} outside the call's outputs",
                )
        for edge in call.control_dependencies:
            if edge.child_op_label not in call_label_aliases:
                raise ContextFieldInvalidError(
                    field,
                    f"control dependency on {call.call_id!r} names child "
                    f"{edge.child_op_label!r} outside the call's op labels",
                )
            for label in (edge.parent_op_label, edge.child_op_label):
                if label not in label_aliases:
                    raise ContextFieldInvalidError(
                        field,
                        f"control dependency edge names unresolvable op label {label!r}",
                    )
    # ---- structural view + independent required derivation ------------------------
    structure = ReplayWitnessStructure.from_descriptor(
        descriptor, container_members=container_members
    )
    required = derive_required_witness_members(structure)
    try:
        present = required_witness_family_members_shared(witnesses)
    except ValueError as exc:
        raise ContextFieldInvalidError(field, f"malformed witness envelope: {exc}")
    gap_members_by_family: dict[str, set[str]] = {}
    for gap in descriptor.coverage_gaps:
        gap_members_by_family.setdefault(gap.source_family, set()).add(gap.source_member)
    # Exact-witness families: required members == present witnesses. The mode family
    # is a required MINIMUM (honest artifacts declare the mode without mode-sensitive
    # ops); the container family defers to the readiness-attach anchor when the
    # rehydrated records are out of scope.
    for family in ("input_structure", "model_input_metadata", "state_metadata"):
        if sorted(required[family]) != sorted(present[family]):
            raise ContextFieldInvalidError(
                field,
                f"{family!r} witnesses do not equal the structurally derived required "
                f"set (required {sorted(required[family])[:6]!r}, present "
                f"{sorted(present[family])[:6]!r})",
            )
    if not set(required["module_training_mode"]) <= set(present["module_training_mode"]):
        raise ContextFieldInvalidError(
            field,
            "a mode-sensitive call replays without the declared module_training_mode witness",
        )
    if container_members is not None and sorted(required["container"]) != sorted(
        present["container"]
    ):
        raise ContextFieldInvalidError(
            field,
            f"container witnesses do not equal the rehydrated container-record "
            f"snapshots (required {sorted(required['container'])[:6]!r}, present "
            f"{sorted(present['container'])[:6]!r})",
        )
    # Witness-or-gap families: every required member discharged by an exact witness
    # or a typed gap; no orphan witness without a structural obligation.
    for family, gap_kinds in (
        ("scalar_bool", {WitnessGapKind.UNOBSERVED_PREDICATE}),
        ("loop_predicate", {WitnessGapKind.UNOBSERVED_PREDICATE}),
        ("conditional_arm_entry", {WitnessGapKind.UNANCHORABLE_ARM_EDGE}),
        ("tensor_derived_scalar_literal", {WitnessGapKind.UNWITNESSABLE_ESCAPE_SOURCE}),
        ("unbound_state_escape", {WitnessGapKind.UNWITNESSABLE_STATE_ESCAPE}),
    ):
        required_set = set(required[family])
        present_set = set(present[family])
        if not present_set <= required_set:
            raise ContextFieldInvalidError(
                field,
                f"{family!r} witnesses {sorted(present_set - required_set)[:6]!r} have "
                "no owning structural obligation",
            )
        gapped = {
            gap.source_member for gap in descriptor.coverage_gaps if gap.gap_kind in gap_kinds
        }
        undischarged = required_set - present_set - gapped
        if undischarged:
            raise ContextFieldInvalidError(
                field,
                f"{family!r} obligations {sorted(undischarged)[:6]!r} are discharged by "
                "neither an exact witness nor a typed coverage gap",
            )
    # ---- terminal-slot totality ----------------------------------------------------
    produced = {slot_id for call in calls for slot_id in call.output_slot_ids}
    consumed = {argument.slot_id for call in calls for argument in call.tensor_arguments}
    output_bound: set[str] = set()
    for slot in slots:
        if slot.role is TensorSlotRole.OUTPUT:
            output_bound.add(slot.slot_id)
            if slot.producer_slot_id is not None:
                output_bound.add(slot.producer_slot_id)
        elif slot.output_path is not None:
            output_bound.add(slot.slot_id)
    obligation_slots = {
        obligation.output_slot_id for call in calls for obligation in call.control_obligations
    }
    terminal_gap_members = (
        gap_members_by_family.get("scalar_bool", set())
        | gap_members_by_family.get("loop_predicate", set())
        | gap_members_by_family.get("tensor_derived_scalar_literal", set())
    )
    for slot_id in sorted(produced - consumed - output_bound):
        terminal_slot = slot_by_id.get(slot_id)
        if terminal_slot is None:
            raise ContextFieldInvalidError(field, f"call output names unknown slot {slot_id!r}")
        if terminal_slot.version_of is not None:
            continue
        structural_claims = (
            int(slot_id in obligation_slots)
            + int(terminal_slot.host_escape)
            + int(terminal_slot.inert_sink)
        )
        if structural_claims > 1:
            raise ContextFieldInvalidError(
                field, f"terminal slot {slot_id!r} carries conflicting claims"
            )
        if structural_claims == 0 and slot_id not in terminal_gap_members:
            raise ContextFieldInvalidError(
                field,
                f"terminal slot {slot_id!r} is claimed by no control obligation, host "
                "escape, inert_sink claim, or typed coverage gap",
            )
    for slot in slots:
        if slot.host_escape and slot.inert_sink:
            raise ContextFieldInvalidError(
                field, f"slot {slot.slot_id!r} claims both host_escape and inert_sink"
            )
        if slot.inert_sink and (slot.slot_id not in produced or slot.slot_id in consumed):
            raise ContextFieldInvalidError(
                field,
                f"slot {slot.slot_id!r} claims inert_sink but is not a terminal call-produced slot",
            )
    # ---- E2 conditional <-> predicate pairing --------------------------------------
    predicate_conditionals = {
        obligation.conditional_id
        for call in calls
        for obligation in call.control_obligations
        if obligation.conditional_id is not None
    }
    predicate_gap_present = any(
        gap.source_family in {"scalar_bool", "loop_predicate"} for gap in descriptor.coverage_gaps
    )
    for call in calls:
        for edge in call.control_dependencies:
            if edge.conditional_id in predicate_conditionals or predicate_gap_present:
                continue
            raise ContextFieldInvalidError(
                field,
                f"conditional {edge.conditional_id!r} enters an arm with no "
                "same-conditional predicate witness or typed predicate gap",
            )
    # ---- boundary <-> binding equality + metadata envelope/owner agreement ---------
    boundary_tensor_sites: set[tuple[Any, Any, str]] = set()
    for site in descriptor.input_boundary:
        position = decode_input_site_position(site.position)
        for tensor_site in site.tensor_sites:
            if tensor_site.slot_id not in slot_by_id:
                raise ContextFieldInvalidError(
                    field,
                    f"boundary tensor site names unknown slot {tensor_site.slot_id!r}",
                )
            boundary_tensor_sites.add(
                (position, tuple(tensor_site.container_path), tensor_site.slot_id)
            )
    binding_tensor_sites: set[tuple[Any, Any, str]] = set()
    for slot in slots:
        if (
            slot.role is not TensorSlotRole.MODEL_INPUT
            or slot.input_binding is None
            or slot.version_of is not None
        ):
            continue
        binding_position = slot.input_binding.model_site_position
        if not isinstance(binding_position, tuple) or len(binding_position) != 2:
            raise ContextFieldInvalidError(
                field,
                f"tensor MODEL_INPUT binding position {binding_position!r} is outside "
                "the root site grammar",
            )
        binding_tensor_sites.add(
            (tuple(binding_position), tuple(slot.input_binding.container_path), slot.slot_id)
        )
    if boundary_tensor_sites != binding_tensor_sites:
        raise ContextFieldInvalidError(
            field,
            "input-boundary tensor sites do not equal the MODEL_INPUT slot bindings "
            f"(boundary-only {sorted(boundary_tensor_sites - binding_tensor_sites, key=repr)[:4]!r}, "
            f"binding-only {sorted(binding_tensor_sites - boundary_tensor_sites, key=repr)[:4]!r})",
        )
    reads_by_site: dict[tuple[Any, Any], tuple[str, ...]] = {}
    for site in descriptor.input_boundary:
        position = decode_input_site_position(site.position)
        for tensor_site in site.tensor_sites:
            reads_by_site[(position, tuple(tensor_site.container_path))] = (
                tensor_site.metadata_reads
            )
    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith("model_input_metadata:"):
            continue
        fact = _decode_literal(witness.observed_value)
        if not isinstance(fact, Mapping):
            raise ContextFieldInvalidError(field, "undecodable model-input metadata fact")
        fact_position: Any = _fact_position_tuple(
            field, fact.get("position"), "metadata envelope position"
        )
        path = _fact_path_tuple(field, fact.get("path", ()) or (), "metadata envelope path")
        declared_reads = reads_by_site.get((fact_position, path))
        if declared_reads is None:
            raise ContextFieldInvalidError(
                field,
                f"metadata envelope at {fact_position!r}:{list(path)!r} has no owning "
                "input-boundary tensor site",
            )
        facts = fact.get("facts")
        fact_names = (
            tuple(sorted(str(name) for name in facts)) if isinstance(facts, Mapping) else ()
        )
        if fact_names != declared_reads:
            raise ContextFieldInvalidError(
                field,
                f"metadata envelope at {fact_position!r}:{list(path)!r} carries fact "
                f"names {fact_names!r} but the boundary record declares "
                f"{declared_reads!r}",
            )
    # ---- state-fact witness/binding agreement + disposition totality ----------------
    facts_by_name: dict[str, tuple[bool, bool]] = {}
    slots_by_name: dict[str, list[TensorSlotDescriptor]] = {}
    for slot in slots:
        binding = slot.state_binding
        if binding is None:
            continue
        pair = (binding.captured_requires_grad, binding.captured_grad_fn)
        existing = facts_by_name.setdefault(binding.state_dict_name, pair)
        if existing != pair:
            raise ContextFieldInvalidError(
                field,
                f"state name {binding.state_dict_name!r} carries disagreeing declared "
                "metadata facts across its slots",
            )
        slots_by_name.setdefault(binding.state_dict_name, []).append(slot)
    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith("state_metadata:"):
            continue
        fact = _decode_literal(witness.observed_value)
        if not isinstance(fact, Mapping):
            continue  # envelope shape already validated at parse
        name = fact.get("state")
        fact_map = fact.get("facts")
        if not isinstance(name, str) or not isinstance(fact_map, Mapping):
            continue
        declared = facts_by_name.get(name)
        if declared is None:
            raise ContextFieldInvalidError(
                field, f"state-metadata witness names undeclared state {name!r}"
            )
        requires_grad, grad_fn_present = declared
        if (
            bool(fact_map.get("requires_grad")) != requires_grad
            or bool(fact_map.get("grad_fn")) != grad_fn_present
        ):
            raise ContextFieldInvalidError(
                field,
                f"state-metadata witness for {name!r} disagrees with the declared binding facts",
            )
    bound_slot_ids = consumed
    bound_state_names = {
        binding.state_dict_name
        for slot in slots
        if (binding := slot.state_binding) is not None and slot.slot_id in bound_slot_ids
    }
    for name, name_slots in slots_by_name.items():
        for slot in name_slots:
            binding = slot.state_binding
            assert binding is not None
            is_unbound = slot.slot_id not in bound_slot_ids and name not in bound_state_names
            if is_unbound and binding.host_escape_disposition is None:
                raise ContextFieldInvalidError(
                    field,
                    f"unbound state slot {slot.slot_id!r} ({name!r}) carries no "
                    "host-escape disposition claim",
                )
            if binding.host_escape_disposition == "inert" and not is_unbound:
                raise ContextFieldInvalidError(
                    field,
                    f"bound state slot {slot.slot_id!r} ({name!r}) cannot claim "
                    "unbound_state_inert",
                )
    # ---- opaque-leaf gap anchoring ---------------------------------------------------
    opaque_literal_members: set[str] = set()
    for witness in witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith("model_input_literal:"):
            continue
        fact = _decode_literal(witness.observed_value)
        if isinstance(fact, Mapping) and fact.get("encodable") is False:
            raw_position = fact.get("position")
            leaf_position: Any = (
                tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
            )
            opaque_literal_members.add(f"{leaf_position!r}:{list(fact.get('path', ()) or ())!r}")
    opaque_gap_members = {
        gap.source_member
        for gap in descriptor.coverage_gaps
        if gap.gap_kind is WitnessGapKind.OPAQUE_INPUT_LEAF
    }
    unanchored_opaque = opaque_literal_members - opaque_gap_members
    if unanchored_opaque:
        raise ContextFieldInvalidError(
            field,
            f"opaque input leaves {sorted(unanchored_opaque)[:4]!r} carry no "
            "OPAQUE_INPUT_LEAF coverage gap",
        )
    # ---- parser-derived completeness FLOOR -------------------------------------------
    floor = derived_witness_completeness(descriptor.coverage_gaps)
    if descriptor.witness_completeness is not floor:
        raise ContextFieldInvalidError(
            field,
            f"persisted witness_completeness {descriptor.witness_completeness.value!r} "
            f"does not equal the parser-derived floor {floor.value!r} (the summary is "
            "a redundant assertion, never authority)",
        )


def required_witness_family_members_shared(
    witnesses: "Sequence[ControlWitness]",
) -> "dict[str, list[str]]":
    """Resolve the shared witness->present-member derivation (single source)."""

    from .runnable import required_witness_family_members

    return required_witness_family_members(witnesses)


def validate_container_witness_anchor(
    descriptor: SparseRunDescriptor,
    container_members: "tuple[str, ...]",
) -> None:
    """Anchor container witnesses to the rehydrated container records (r71 A).

    Runs at readiness attach, where the loaded trace's container records are in
    scope: the descriptor's ``container:*`` witnesses must equal, bidirectionally,
    the MODEL_INPUT / MODEL_OUTPUT snapshot identities the rehydrated trace itself
    implies (the SAME derivation the producer emits from). A lockstep
    witness+inventory-member strip leaves the surviving container record
    contradicted -- analysis-only, never a silently weaker output-structure check.
    """

    present = [
        witness.site_label
        for witness in descriptor.control_witnesses
        if witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith("container:")
    ]
    if sorted(present) != sorted(container_members):
        raise ContextFieldInvalidError(
            "witness_obligations",
            f"container witnesses do not equal the rehydrated container-record "
            f"snapshots (present {sorted(present)[:6]!r}, required "
            f"{sorted(container_members)[:6]!r})",
        )


def _normalized_callable_name(name: str | None) -> str | None:
    """Return a callable name in the one spelling both records agree on (r83 S3).

    ``sites[].function_path`` (derived from ``layer_list[].func_name``) and
    ``run.callable_registry[].key.qualname`` are persisted independently and
    agree verbatim for every op measured across dunder, in-place, method and
    function dispatch, with two exceptions where the site keeps the operator
    dunder while the registry records the torch method (``__neg__``/``neg``,
    ``__pow__``/``pow``). Stripping the operator dunder underscores collapses
    exactly those, and nothing else: ``__iadd__`` and ``relu_`` keep their
    distinguishing characters, so an in-place op can never normalize onto its
    out-of-place sibling.

    Parameters
    ----------
    name:
        Persisted callable name from either record.

    Returns
    -------
    str | None
        Normalized name, or ``None`` when the record states no opinion.
    """

    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped or stripped == "none":
        return None
    if stripped.startswith("__") and stripped.endswith("__") and len(stripped) > 4:
        return stripped[2:-2]
    return stripped


def validate_callable_registry_anchor(
    descriptor: SparseRunDescriptor,
    func_names_by_op_label: "Mapping[str, str | None]",
) -> None:
    """Reconcile the execution authority against the independently persisted names (r83 S3).

    The callable a loaded artifact EXECUTES comes from
    ``manifest.json -> run.callable_registry[].key``. The SAME call is named
    independently by ``manifest.json -> sites[].function_path`` and by
    ``metadata.pkl -> layer_list[].func_name`` / ``.func_id.qualname``. Nothing
    reconciled them, so editing one JSON field (``Tensor.__add__`` ->
    ``__sub__``) left an internally SELF-CONTRADICTORY artifact that ran happily
    and reported ``VERIFIED`` with different numbers -- three other persisted
    fields still said ``add``.

    This is defence-in-depth against artifact CORRUPTION or a partial rewrite,
    not against an attacker: a competent one would simply capture the wrong
    program honestly, which is out of scope by contract (coherent reauthoring).
    A typed refusal beats a silent ``VERIFIED``. The attestation lane already
    anchors this when ``include_activations=True`` makes it eligible; the gap
    was the DEFAULT artifact, where the run is ``not_applicable``.

    Fails closed only on a definite disagreement: a record that states no name
    (a source op's ``"none"``, a missing label) is no opinion, never a refusal.

    Parameters
    ----------
    descriptor:
        Parsed sparse run descriptor carrying the callable registry and calls.
    func_names_by_op_label:
        Per-op function names implied by the REHYDRATED trace, keyed by op label.

    Raises
    ------
    ContextFieldInvalidError
        When the execution authority and an independent record disagree.
    """

    registry_by_id = {entry.registry_id: entry for entry in descriptor.callable_registry}
    for call in descriptor.calls:
        entry = registry_by_id.get(call.registry_id)
        if entry is None:
            continue
        authority = _normalized_callable_name(entry.key.qualname)
        if authority is None:
            continue
        for op_label in call.op_labels:
            recorded = _normalized_callable_name(func_names_by_op_label.get(op_label.split(":")[0]))
            if recorded is None or recorded == authority:
                continue
            raise ContextFieldInvalidError(
                "callable_registry",
                f"call {call.call_id!r} executes "
                f"{entry.key.namespace}.{entry.key.qualname!r} but the artifact's "
                f"independently recorded name for op {op_label!r} is "
                f"{func_names_by_op_label.get(op_label.split(':')[0])!r}; the "
                f"artifact contradicts itself and is refused rather than run",
            )


class DescriptorStructuralBoundError(ValueError):
    """A persisted runnable-descriptor integer failed structural cross-validation (r53 free_1).

    Raised at PARSE time -- before readiness resolution, signature binding, state
    staging, or any allocation whose size is a function of the persisted integer
    -- so a hostile ``manifest.json`` integer (``num_positional_args=1e10``,
    ``shape=[1e9,1e9]``) can never drive an allocation bomb on the default
    ``tl.load()``/``.run()`` path. Surfaced as an analysis-only readiness
    diagnostic (frozen ``call_arity_mismatch`` / ``state_shape_mismatch`` codes)
    at detection stage ``descriptor_parse``; the load still succeeds for
    analysis and ``.run()`` refuses typed.
    """

    def __init__(self, code: RunnableErrorCode, field: str, detail: str) -> None:
        super().__init__(f"Persisted runnable descriptor field {field!r} is invalid: {detail}")
        self.code = code
        self.field = field
        self.detail = detail


_MAX_RUNNABLE_TENSOR_RANK = 64
"""Structural rank sanity bound: no real tensor produced by torch exceeds it."""

_INT64_MAX = 2**63 - 1
"""Signed 64-bit ceiling: torch sizes/storage byte counts are int64 quantities."""

_MAX_LITERAL_NESTING_DEPTH = 200
"""Structural nesting ceiling for the recursive non-tensor literal grammar (r55 free_2).

Independent belt to the bounded-JSON depth prescan: a hand-edited or
depth-tolerant front end could hand ``_parse_literal`` a nesting bomb that the
stdlib ``json`` boundary did not gate. Over-depth raises ``ValueError``, routed
through the existing descriptor-parse refusal to an ANALYSIS-ONLY load rather than
an uncaught ``RecursionError``. 200 sits far above any real literal-argument
nesting and below the interpreter's default recursion crash depth.
"""


def _validate_call_arity(
    call_id: str,
    num_positional_args: int,
    num_keyword_args: int,
    tensor_arguments: tuple[TensorArgumentRef, ...],
    literal_arguments: tuple[LiteralArgumentRef, ...],
) -> None:
    """Anchor persisted call arity to the DENSE first-level argument leaves (r53 free_1).

    A legitimate producer records ``num_positional_args`` equal to the count of
    distinct first-level positional roots ``("args", i)`` with ``i`` dense in
    ``[0, n)``, and ``num_keyword_args`` equal to the count of distinct keyword
    names -- the run-time tripwire in ``_pre_call_contract_checks`` has always
    required exactly this, so anchoring it at parse cannot over-trigger. The
    density check is the allocation-free pigeonhole (len/min/max over the root
    set); it never materializes ``set(range(n))`` from the untrusted integer.
    """

    def _refuse(detail: str) -> DescriptorStructuralBoundError:
        return DescriptorStructuralBoundError(
            RunnableErrorCode.CALL_ARITY_MISMATCH, f"calls[{call_id}]", detail
        )

    if num_positional_args < 0 or num_keyword_args < 0:
        raise _refuse(
            f"negative arity (num_positional_args={num_positional_args}, "
            f"num_keyword_args={num_keyword_args})"
        )
    positional_roots: set[int] = set()
    keyword_names: set[str] = set()
    paths = tuple(argument.argument_path for argument in tensor_arguments) + tuple(
        argument.argument_path for argument in literal_arguments
    )
    for path in paths:
        if len(path) < 2 or path[0] not in ("args", "kwargs"):
            raise _refuse(f"argument path {path!r} is not rooted at args/kwargs")
        head = path[1]
        if path[0] == "args":
            if not isinstance(head, int) or isinstance(head, bool) or head < 0:
                raise _refuse(f"positional argument root {head!r} is not a nonnegative integer")
            positional_roots.add(head)
        else:
            if not isinstance(head, str):
                raise _refuse(f"keyword argument root {head!r} is not a string")
            keyword_names.add(head)
    n = num_positional_args
    dense = len(positional_roots) == n and (
        n == 0 or (min(positional_roots) == 0 and max(positional_roots) == n - 1)
    )
    if not dense:
        raise _refuse(
            f"num_positional_args={n} does not match the dense first-level positional "
            f"leaf roots (count={len(positional_roots)}, "
            f"span={sorted(positional_roots)[:8]!r}...)"
        )
    if num_keyword_args != len(keyword_names):
        raise _refuse(
            f"num_keyword_args={num_keyword_args} does not match the "
            f"{len(keyword_names)} distinct keyword leaf names"
        )


def _validate_slot_shape(
    slot_id: str,
    shape: tuple[int, ...],
    rank: int,
    dtype_literal: str,
) -> None:
    """Anchor a persisted slot shape to its declared rank and an int64-safe byte product.

    Structural bounds only (r53 free_1): nonnegative dimensions, ``rank ==
    len(shape)``, rank within torch's practical ceiling, and a total byte count
    that fits signed 64-bit (torch storage sizes are int64). Deliberately NO
    absolute byte cap here -- real slots (multi-GiB LLM embeddings) must keep
    loading; run-preparation magnitude gating is the allocation preflight in
    ``_runnable_state.py``, and embedded-payload slots stay value-anchored by
    the strict binder and the safetensors header.
    """

    def _refuse(detail: str) -> DescriptorStructuralBoundError:
        return DescriptorStructuralBoundError(
            RunnableErrorCode.STATE_SHAPE_MISMATCH, f"tensor_slots[{slot_id}]", detail
        )

    if rank < 0:
        raise _refuse(f"negative rank {rank}")
    if rank != len(shape):
        raise _refuse(f"rank={rank} contradicts len(shape)={len(shape)}")
    if rank > _MAX_RUNNABLE_TENSOR_RANK:
        raise _refuse(f"rank={rank} exceeds the structural ceiling {_MAX_RUNNABLE_TENSOR_RANK}")
    numel = 1
    for dim in shape:
        if dim < 0:
            raise _refuse(f"negative dimension {dim} in shape {shape[:8]!r}")
        numel *= dim
        if numel > _INT64_MAX:
            raise _refuse("shape element product exceeds the signed 64-bit ceiling")
    name = dtype_literal.removeprefix("torch.")
    resolved = torch_attr(name)
    itemsize = resolved.itemsize if isinstance(resolved, torch.dtype) else 1
    if numel * itemsize > _INT64_MAX:
        raise _refuse(
            f"shape byte product numel={numel} x itemsize={itemsize} exceeds the "
            "signed 64-bit ceiling"
        )


_MATMUL_PRECISION_VOCABULARY = frozenset({"highest", "high", "medium"})
"""Closed vocabulary for ``float32_matmul_precision`` (torch's exact accepted set)."""

_DEVICE_TYPE_VOCABULARY = frozenset(
    {
        "cpu",
        "cuda",
        "meta",
        "mps",
        "xpu",
        "hpu",
        "mtia",
        "ipu",
        "xla",
        "vulkan",
        "lazy",
        "privateuseone",
    }
)
"""Closed vocabulary of device TYPE literals a persisted context may name."""


def _validated_device_literal(field: str, raw: str) -> str:
    """Validate a persisted device literal against the closed type[:index] grammar."""

    device_type, separator, index = raw.partition(":")
    if device_type not in _DEVICE_TYPE_VOCABULARY:
        raise ContextFieldInvalidError(
            field, f"device type {device_type!r} is outside the closed device vocabulary"
        )
    if separator and not index.isdigit():
        raise ContextFieldInvalidError(
            field, f"device index {index!r} is not a nonnegative integer"
        )
    return raw


def _validated_dtype_literal(field: str, raw: str) -> str:
    """Validate a persisted ``torch.<dtype>`` literal against the live dtype table."""

    name = raw.removeprefix("torch.")
    # r42 secC_1 / r45: the shared ``torch_attr`` helper never fires ``torch.__getattr__`` (no
    # lazy submodule import, no deprecated-attr call). Every real dtype is a ``torch.__dict__``
    # entry.
    resolved = torch_attr(name)
    if not isinstance(resolved, torch.dtype):
        raise ContextFieldInvalidError(field, f"{raw!r} does not name a torch dtype")
    return raw


def _parse_ambient_context(value: Mapping[str, Any]) -> AmbientExecutionContext:
    """Parse the required v2 capture-scoped ambient execution context.

    Every field must be present explicitly; ``None`` means the producing runtime
    did not expose that control. Missing keys raise (fail closed). r37 INV-4:
    every VALUE validates here against a closed vocabulary -- device literals
    against the type[:index] grammar, dtype literals against the live dtype
    table, matmul precision against torch's exact accepted set, Booleans
    strictly -- so no torch setter (and no callable) ever receives an
    unvalidated persisted byte. The ambient-apply guards remain as a second
    belt.
    """

    def _optional_bool_field(name: str) -> bool | None:
        if name not in value:
            # r53 hon_1 posture: an absent ambient field is a typed parse
            # refusal (analysis-only load), never a defaulted control.
            raise ContextFieldInvalidError(
                f"ambient_context.{name}",
                "required ambient field is absent; absent context is never "
                "defaulted (re-capture with this TorchLens version)",
            )
        raw = value[name]
        if raw is None:
            return None
        if not isinstance(raw, bool):
            raise ContextFieldInvalidError(
                f"ambient_context.{name}", f"{raw!r} is not a strict boolean or null"
            )
        return raw

    def _required_bool_field(name: str) -> bool:
        # r53 hon_1: the global autograd/inference mode is REQUIRED and strictly
        # boolean -- there is NO honest default (a defaulted grad mode could
        # bless a different-ambient comparison as verified), and ``null`` is not
        # a legal producer value (every supported torch exposes both queries).
        raw = _optional_bool_field(name)
        if raw is None:
            raise ContextFieldInvalidError(
                f"ambient_context.{name}",
                "null is not a legal value: the global autograd/inference mode "
                "is exposed by every supported torch and is never defaulted",
            )
        return raw

    def _optional_str_field(name: str) -> str | None:
        raw = value[name]
        if raw is None:
            return None
        return _string_item(raw, f"ambient_context.{name}")

    precision = _optional_str_field("float32_matmul_precision")
    if precision is not None and precision not in _MATMUL_PRECISION_VOCABULARY:
        raise ContextFieldInvalidError(
            "ambient_context.float32_matmul_precision",
            f"{precision!r} is outside {sorted(_MATMUL_PRECISION_VOCABULARY)}",
        )
    return AmbientExecutionContext(
        default_dtype=_validated_dtype_literal(
            "ambient_context.default_dtype", _string(value, "default_dtype")
        ),
        default_device=_validated_device_literal(
            "ambient_context.default_device", _string(value, "default_device")
        ),
        float32_matmul_precision=precision,
        deterministic_algorithms=_optional_bool_field("deterministic_algorithms"),
        deterministic_algorithms_warn_only=_optional_bool_field(
            "deterministic_algorithms_warn_only"
        ),
        cuda_matmul_allow_tf32=_optional_bool_field("cuda_matmul_allow_tf32"),
        cudnn_allow_tf32=_optional_bool_field("cudnn_allow_tf32"),
        cudnn_deterministic=_optional_bool_field("cudnn_deterministic"),
        cudnn_benchmark=_optional_bool_field("cudnn_benchmark"),
        cudnn_enabled=_optional_bool_field("cudnn_enabled"),
        flash_sdp_enabled=_optional_bool_field("flash_sdp_enabled"),
        mem_efficient_sdp_enabled=_optional_bool_field("mem_efficient_sdp_enabled"),
        math_sdp_enabled=_optional_bool_field("math_sdp_enabled"),
        grad_enabled=_required_bool_field("grad_enabled"),
        inference_mode=_required_bool_field("inference_mode"),
        fill_uninitialized_memory=_optional_bool_field("fill_uninitialized_memory"),
        attestation_ineligible_context=_boolean(value, "attestation_ineligible_context"),
    )


def _parse_call_execution_context(value: Mapping[str, Any]) -> CallExecutionContext:
    """Parse the required v2 per-call execution context (explicit, never defaulted)."""

    autocast_entries: list[AutocastDeviceContext] = []
    for item in _mapping_sequence(value, "autocast"):
        dtype = item.get("dtype")
        autocast_entries.append(
            AutocastDeviceContext(
                device_type=_validated_device_literal(
                    "execution_context.autocast.device_type", _string(item, "device_type")
                ),
                enabled=_boolean(item, "enabled"),
                dtype=(
                    None
                    if dtype is None
                    else _validated_dtype_literal(
                        "execution_context.autocast.dtype",
                        _string_item(dtype, "autocast.dtype"),
                    )
                ),
            )
        )
    return CallExecutionContext(
        autocast=tuple(autocast_entries),
        grad_enabled=_boolean(value, "grad_enabled"),
        inference_mode=_boolean(value, "inference_mode"),
    )


def _parse_input_fingerprint(value: Mapping[str, Any]) -> InputAttestationFingerprint:
    """Parse one required physical input fingerprint (``selected_activation_v2``)."""

    device_index = value.get("device_index")
    return InputAttestationFingerprint(
        slot_id=_string(value, "slot_id"),
        byte_digest=_string(value, "byte_digest"),
        device_type=_string(value, "device_type"),
        device_index=None if device_index is None else _integer_item(device_index, "device_index"),
        layout=_string(value, "layout"),
        sizes=tuple(_integer_item(item, "sizes") for item in _sequence(value, "sizes")),
        strides=tuple(_integer_item(item, "strides") for item in _sequence(value, "strides")),
        storage_offset=_integer(value, "storage_offset"),
        is_contiguous=_boolean(value, "is_contiguous"),
        is_channels_last=_boolean(value, "is_channels_last"),
        is_channels_last_3d=_boolean(value, "is_channels_last_3d"),
        is_conj=_boolean(value, "is_conj"),
        is_neg=_boolean(value, "is_neg"),
        tensor_class=_string(value, "tensor_class"),
        requires_grad=_boolean(value, "requires_grad"),
        is_inference=_boolean(value, "is_inference"),
        alignment_class=_integer(value, "alignment_class"),
    )


def _parse_rng_profile(value: Any) -> RunnableRngProfile:
    """Parse the optional host-RNG profile, defaulting legacy manifests to deterministic.

    Manifests written before host-RNG honesty tracking omit this object; they are
    treated as ``host_rng_consumed=False`` (deterministic) because their capture
    predates the recorded signal and cannot be recovered.
    """

    if not isinstance(value, Mapping):
        return RunnableRngProfile(host_rng_consumed=False, capture_seed=None)
    consumed = value.get("host_rng_consumed")
    seed = value.get("capture_seed")
    return RunnableRngProfile(
        host_rng_consumed=bool(consumed),
        capture_seed=int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None,
    )


def attach_sparse_run_readiness(
    trace: Any,
    raw_descriptor: Mapping[str, Any] | None,
) -> ReadinessReport:
    """Parse, resolve, and atomically attach runnable callables to a loaded Trace.

    Parameters
    ----------
    trace:
        Analysis Trace already rehydrated by the ordinary portable loader.
    raw_descriptor:
        Decoded manifest run descriptor, or ``None`` for an analysis artifact.

    Returns
    -------
    ReadinessReport
        Complete non-executing readiness report. Resolution failure never
        prevents the already-loaded analysis Trace from being returned.
    """

    if raw_descriptor is None:
        report = _analysis_only_readiness(trace)
        _store_readiness(trace, None, report, None)
        return report
    capability = raw_descriptor.get("capability")
    if isinstance(capability, str) and capability in LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS:
        # Decision G: a legacy v1 descriptor carries no execution-context records
        # (per-call context, ambient backend context, input fingerprints), and
        # "absent" is never interpreted as a default. Legacy artifacts load for
        # analysis with a typed readiness refusal naming the missing context class;
        # re-capture under v2 is the actionable remedy.
        diagnostic = _diagnostic(
            RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
            f"Legacy runnable capability {capability!r} is analysis-only: it "
            "records no per-call execution context, capture-scoped ambient "
            "backend context, or physical input fingerprints, and absent "
            "context is never defaulted. Re-capture and re-save with this "
            f"TorchLens version to produce {RUNNABLE_TLSPEC_SCHEMA_VERSION!r}.",
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="descriptor_legacy_version",
            details=(
                ("capability", capability),
                ("supported", RUNNABLE_TLSPEC_SCHEMA_VERSION),
                ("missing_context_class", "execution_context/ambient_context"),
            ),
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=capability,
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report
    try:
        descriptor = parse_sparse_run_descriptor(raw_descriptor)
    except ContextFieldInvalidError as exc:
        # r37 INV-4: an invalid persisted context VALUE is its own frozen refusal
        # class -- refused at parse, before readiness/staging, so no torch setter
        # or resolved callable ever observes the bytes.
        diagnostic = _diagnostic(
            RunnableErrorCode.CONTEXT_FIELD_INVALID,
            str(exc),
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="context_parse_validation",
            details=(("context_field", exc.field), ("reason", exc.detail)),
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=cast(str | None, raw_descriptor.get("capability")),
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report
    except DescriptorStructuralBoundError as exc:
        # r53 free_1: a structurally impossible persisted integer (arity not
        # matching the dense argument leaves, shape contradicting rank, an
        # int64-overflowing byte product) is refused at PARSE with its frozen
        # code -- before readiness resolution or any allocation the integer
        # could scale. The load stays analysis-only; ``.run()`` refuses typed.
        diagnostic = _diagnostic(
            exc.code,
            str(exc),
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="descriptor_parse",
            details=(("descriptor_field", exc.field), ("reason", exc.detail)),
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=cast(str | None, raw_descriptor.get("capability")),
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report
    except (KeyError, TypeError, ValueError) as exc:
        diagnostic = _diagnostic(
            RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
            f"Sparse runnable descriptor could not be parsed: {exc}",
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="descriptor_parse",
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=cast(str | None, raw_descriptor.get("capability")),
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report

    try:
        # r71 A: the container family's INDEPENDENT anchor -- the descriptor's
        # ``container:*`` witnesses must equal the snapshot identities the
        # REHYDRATED trace's own container records imply (the replay consumes those
        # records for input binding and output reconstruction), so a lockstep
        # witness+member strip fails closed here, analysis-only.
        from .runnable import _container_structure_witnesses

        container_members = tuple(
            witness.site_label for witness in _container_structure_witnesses(trace, start_order=0)
        )
        validate_container_witness_anchor(descriptor, container_members)
        # r83 S3: the execution authority (``run.callable_registry``) reconciled
        # against the name the rehydrated trace records for the same op. A
        # single-field registry edit otherwise leaves a self-contradictory
        # artifact that runs and reports VERIFIED with different numbers.
        validate_callable_registry_anchor(
            descriptor,
            {
                str(getattr(layer, "layer_label", "")): getattr(layer, "func_name", None)
                for layer in getattr(trace, "layer_list", []) or ()
            },
        )
    except ContextFieldInvalidError as exc:
        diagnostic = _diagnostic(
            RunnableErrorCode.CONTEXT_FIELD_INVALID,
            str(exc),
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="container_anchor_validation",
            details=(("context_field", exc.field), ("reason", exc.detail)),
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=cast(str | None, raw_descriptor.get("capability")),
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report

    report, attachments = preflight_sparse_run_descriptor(descriptor)
    _store_readiness(trace, descriptor, report, attachments)
    return report


def _device_capability_diagnostics(
    descriptor: SparseRunDescriptor,
) -> tuple[RunnableDiagnostic, ...]:
    """Capability-check every recorded slot device WITHOUT allocation (r37 R5).

    Validates device class existence and index bounds against this runtime. Only
    accelerator classes with runtime probes are gated; the check must never
    allocate payload memory or initialize a device context beyond the cheap
    availability query.
    """

    diagnostics: list[RunnableDiagnostic] = []
    seen: set[tuple[str, int | None]] = set()
    for slot in descriptor.tensor_slots:
        key = (slot.device_type, slot.device_index)
        if key in seen:
            continue
        seen.add(key)
        unavailable: str | None = None
        if slot.device_type == "cuda":
            if not torch.cuda.is_available():
                unavailable = "CUDA is unavailable on this runtime"
            elif slot.device_index is not None and slot.device_index >= torch.cuda.device_count():
                unavailable = (
                    f"CUDA device index {slot.device_index} exceeds "
                    f"device_count()={torch.cuda.device_count()}"
                )
        elif slot.device_type == "mps":
            mps_module = getattr(torch.backends, "mps", None)
            if mps_module is None or not bool(mps_module.is_available()):
                unavailable = "MPS is unavailable on this runtime"
        if unavailable is not None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                    f"Recorded slot {slot.slot_id!r} requires device "
                    f"{slot.device_type}"
                    + (f":{slot.device_index}" if slot.device_index is not None else "")
                    + f": {unavailable}. The artifact stays loadable for analysis; "
                    "execution requires a runtime exposing the recorded device.",
                    descriptor=descriptor,
                    detection_stage="readiness_device_capability",
                    details=(
                        ("slot_id", slot.slot_id),
                        ("device_type", slot.device_type),
                        ("device_index", repr(slot.device_index)),
                    ),
                )
            )
    return tuple(diagnostics)


def preflight_sparse_run_descriptor(
    descriptor: SparseRunDescriptor,
) -> tuple[ReadinessReport, Mapping[str, Callable[..., Any]] | None]:
    """Resolve a parsed descriptor once and return atomic call attachments.

    Parameters
    ----------
    descriptor:
        Parsed sparse descriptor.

    Returns
    -------
    tuple[ReadinessReport, Mapping[str, Callable[..., Any]] | None]
        Readiness plus a call-id mapping only when every required computational
        group resolved successfully.
    """

    version_diagnostics = _descriptor_version_diagnostics(descriptor)
    device_diagnostics = _device_capability_diagnostics(descriptor)
    if device_diagnostics:
        # r37 R5 readiness gate: recorded slot devices are capability-checked
        # WITHOUT allocating anything -- a CPU-only host loads a CUDA artifact for
        # analysis fine, reports UNAVAILABLE here, and ``.run()`` refuses typed
        # before any callable resolution or payload transfer.
        report = _readiness_report(
            descriptor,
            records=(),
            diagnostics=(*version_diagnostics, *device_diagnostics),
            ready=False,
        )
        return report, None
    if descriptor.backend != "torch":
        diagnostic = _diagnostic(
            RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY,
            f"Sparse runnable replay is unavailable for backend {descriptor.backend!r}.",
            descriptor=descriptor,
            detection_stage="resolver_backend",
        )
        report = _readiness_report(
            descriptor,
            records=(),
            diagnostics=(*version_diagnostics, diagnostic),
            ready=False,
        )
        return report, None

    registry_by_id = {entry.registry_id: entry for entry in descriptor.callable_registry}
    labels_by_registry: dict[str, list[str]] = {}
    structural_diagnostics: list[RunnableDiagnostic] = []
    for call in descriptor.calls:
        labels_by_registry.setdefault(call.registry_id, []).extend(call.op_labels)
        if call.registry_id not in registry_by_id:
            structural_diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CALLABLE_REF,
                    "Computational group references a missing callable registry ID.",
                    descriptor=descriptor,
                    registry_id=call.registry_id,
                    affected_ops=call.op_labels,
                    detection_stage="resolver_registry_validation",
                )
            )

    resolutions: dict[str, _Resolution] = {}
    if not version_diagnostics:
        with _state.pause_logging():
            for entry in descriptor.callable_registry:
                resolutions[entry.registry_id] = _resolve_registry_entry(
                    entry,
                    descriptor=descriptor,
                    affected_ops=tuple(
                        dict.fromkeys(labels_by_registry.get(entry.registry_id, ()))
                    ),
                    calls=tuple(
                        call for call in descriptor.calls if call.registry_id == entry.registry_id
                    ),
                )

    records = tuple(resolution.record for resolution in resolutions.values())
    diagnostics = [*version_diagnostics, *structural_diagnostics]
    diagnostics.extend(diagnostic for record in records for diagnostic in record.diagnostics)
    ready = (
        not version_diagnostics
        and not structural_diagnostics
        and len(resolutions) == len(descriptor.callable_registry)
        and all(item.func is not None for item in resolutions.values())
        and all(call.registry_id in resolutions for call in descriptor.calls)
    )
    report = _readiness_report(
        descriptor,
        records=records,
        diagnostics=tuple(diagnostics),
        ready=ready,
    )
    if not ready:
        return report, None
    attachments = {
        call.call_id: cast(Callable[..., Any], resolutions[call.registry_id].func)
        for call in descriptor.calls
    }
    return report, attachments


def _resolve_registry_entry(
    entry: CallableRegistryEntry,
    *,
    descriptor: SparseRunDescriptor,
    affected_ops: tuple[str, ...],
    calls: tuple[RunnableCallDescriptor, ...],
) -> _Resolution:
    """Resolve one unique registry entry through the locked torch ladder."""

    key = entry.key
    stock_path = _stock_path_from_key(key)
    if key.namespace == "custom" and stock_path is None:
        diagnostic = _diagnostic(
            RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT,
            "Resolver-only readiness refuses custom module imports.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_security",
            provenance="custom_import_default_deny",
            details=(("import_path", str(key.import_path)),),
        )
        return _unavailable_resolution(entry, diagnostic, "custom_import_default_deny")

    exact = _resolve_exact_key(key, stock_path)
    if exact is not None:
        func, resolved_qualname = exact
        return _resolved_callable(
            entry,
            func,
            resolved_qualname=resolved_qualname,
            provenance=f"exact_getattr:{resolved_qualname}",
            status=ResolverStatus.RESOLVED_EXACT,
            descriptor=descriptor,
            affected_ops=affected_ops,
            calls=calls,
            moved=False,
        )

    if _is_captured_internal_torch_builtin_key(key):
        diagnostic = _diagnostic(
            RunnableErrorCode.PRIVATE_API_UNAVAILABLE,
            "Captured internal torch builtin is unavailable; its public wrapper is not a "
            "safe replay substitute for the recorded call recipe.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_exact",
            provenance="internal_builtin_identity_required",
            details=(("recorded_key", _key_display_path(key)),),
        )
        return _unavailable_resolution(entry, diagnostic, "internal_builtin_identity_required")

    alias = resolve_runnable_torch_alias(
        stock_path or _key_display_path(key), descriptor.compatibility.backend_version
    )
    if alias is not None:
        namespace, qualname, provenance = alias
        alias_func = _getattr_allowlisted(namespace, qualname)
        if alias_func is None:
            private = _is_private_path(stock_path or _key_display_path(key))
            code = (
                RunnableErrorCode.PRIVATE_API_UNAVAILABLE
                if private
                else RunnableErrorCode.CALLABLE_REMOVED
            )
            diagnostic = _diagnostic(
                code,
                "Recorded callable has an explicit successor, but that successor is unavailable.",
                descriptor=descriptor,
                registry_id=entry.registry_id,
                affected_ops=affected_ops,
                detection_stage="resolver_alias",
                provenance=provenance,
                details=(("target", f"{namespace}.{qualname}"),),
            )
            return _unavailable_resolution(entry, diagnostic, provenance)
        return _resolved_callable(
            entry,
            alias_func,
            resolved_qualname=f"{namespace}.{qualname}",
            provenance=provenance,
            status=ResolverStatus.RESOLVED_ALIAS,
            descriptor=descriptor,
            affected_ops=affected_ops,
            calls=calls,
            moved=True,
        )

    if (stock_path or _key_display_path(key)) in _REMOVED_TORCH_CALLABLES:
        diagnostic = _diagnostic(
            RunnableErrorCode.CALLABLE_REMOVED,
            "Recorded callable was removed and has no supported sparse-run successor.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_tombstone",
            provenance="explicit_removed_callable_table",
        )
        return _unavailable_resolution(entry, diagnostic, "explicit_removed_callable_table")

    candidates = _reverse_candidates(key.qualname, calls)
    if candidates:
        ranked = _best_ranked_candidates(candidates, key, stock_path)
        if len(ranked) > 1:
            candidate_names = tuple(
                sorted(f"{candidate.namespace}.{candidate.qualname}" for candidate in ranked)
            )
            diagnostic = _diagnostic(
                RunnableErrorCode.AMBIGUOUS_QUALNAME,
                "Multiple reverse-index candidates survived namespace and arity guards.",
                descriptor=descriptor,
                registry_id=entry.registry_id,
                affected_ops=affected_ops,
                detection_stage="resolver_reverse_index",
                provenance="reverse_index_ambiguous",
                details=tuple(("candidate", name) for name in candidate_names),
            )
            return _unavailable_resolution(entry, diagnostic, "reverse_index_ambiguous")
        candidate = ranked[0]
        return _resolved_callable(
            entry,
            candidate.func,
            resolved_qualname=f"{candidate.namespace}.{candidate.qualname}",
            provenance="reverse_index:namespace_rank+name+arity",
            status=ResolverStatus.RESOLVED_ALIAS,
            descriptor=descriptor,
            affected_ops=affected_ops,
            calls=calls,
            moved=True,
        )

    code = (
        RunnableErrorCode.PRIVATE_API_UNAVAILABLE
        if stock_path is not None and _is_private_path(stock_path)
        else RunnableErrorCode.UNRESOLVED_QUALNAME
    )
    diagnostic = _diagnostic(
        code,
        "No exact, explicit alias, or unambiguous reverse-index callable was found.",
        descriptor=descriptor,
        registry_id=entry.registry_id,
        affected_ops=affected_ops,
        detection_stage="resolver_reverse_index",
        provenance="ladder_exhausted_no_guess",
        details=(("recorded_key", _key_display_path(key)),),
    )
    return _unavailable_resolution(entry, diagnostic, "ladder_exhausted_no_guess")


def _resolved_callable(
    entry: CallableRegistryEntry,
    func: Callable[..., Any],
    *,
    resolved_qualname: str,
    provenance: str,
    status: ResolverStatus,
    descriptor: SparseRunDescriptor,
    affected_ops: tuple[str, ...],
    calls: tuple[RunnableCallDescriptor, ...],
    moved: bool,
) -> _Resolution:
    """Unwrap and validate one resolved callable before recording success."""

    original = _unwrap_decorated(func)
    if id(original) in _state._decorated_to_orig:
        diagnostic = _diagnostic(
            RunnableErrorCode.WRAPPER_SHADOWED,
            "Resolved callable remains a TorchLens wrapper after translation.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_wrapper_translation",
            provenance=provenance,
        )
        return _unavailable_resolution(entry, diagnostic, provenance)
    # SECURITY BOUNDARY (tripwire). A bundle is UNTRUSTED input. The torch /
    # torch.Tensor / torch.nn.functional / operator namespaces also expose
    # side-effecting callables -- above all torch.load / torch.save (both in
    # torch.serialization), which unpickle attacker files (RCE) or write to
    # arbitrary paths BEFORE any downstream path/signature check fires. Every
    # success ladder rung (exact, alias, reverse-index) funnels through here, so
    # this gate closes the whole class: only pure, side-effect-free forward/
    # tensor ops may resolve. Note torch.load IS in get_orig_torch_funcs(), so
    # gating on the wrapped-op inventory would NOT suffice.
    if not is_pure_forward_callable(original):
        diagnostic = _diagnostic(
            RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT,
            "Resolved callable is not a pure forward/tensor op and is refused "
            "for security (side-effecting or dangerous namespace).",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_security",
            provenance="nonforward_callable_denied",
            details=(("resolved_callable", unsafe_callable_reason(original)),),
        )
        return _unavailable_resolution(entry, diagnostic, "nonforward_callable_denied")
    if not all(_signature_accepts_call(original, call) for call in calls):
        diagnostic = _diagnostic(
            RunnableErrorCode.SIGNATURE_DRIFT,
            "Inspectable current callable cannot accept the recorded positional arity.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_signature",
            provenance=provenance,
            details=tuple(
                ("call_arity", f"{call.call_id}:{call.num_positional_args}+{call.num_keyword_args}")
                for call in calls
            ),
        )
        return _unavailable_resolution(entry, diagnostic, provenance)
    diagnostics: tuple[RunnableDiagnostic, ...] = ()
    if moved:
        diagnostics = (
            _diagnostic(
                RunnableErrorCode.CALLABLE_MOVED_OR_RENAMED,
                "Recorded callable resolved through an explicit compatibility successor.",
                descriptor=descriptor,
                registry_id=entry.registry_id,
                affected_ops=affected_ops,
                detection_stage="resolver_alias",
                provenance=provenance,
                details=(("resolved_qualname", resolved_qualname),),
            ),
        )
    return _Resolution(
        record=ResolverRecord(
            registry_id=entry.registry_id,
            status=status,
            recorded_key=entry.key,
            resolved_qualname=resolved_qualname,
            provenance=provenance,
            diagnostics=diagnostics,
        ),
        func=original,
    )


def _unavailable_resolution(
    entry: CallableRegistryEntry,
    diagnostic: RunnableDiagnostic,
    provenance: str,
) -> _Resolution:
    """Build one unavailable per-registry resolution result."""

    return _Resolution(
        record=ResolverRecord(
            registry_id=entry.registry_id,
            status=ResolverStatus.UNAVAILABLE,
            recorded_key=entry.key,
            resolved_qualname=None,
            provenance=provenance,
            diagnostics=(diagnostic,),
        ),
        func=None,
    )


def _resolve_exact_key(
    key: FunctionRegistryKey,
    stock_path: str | None,
) -> tuple[Callable[..., Any], str] | None:
    """Resolve one key by getattr on allowlisted roots and namespaces only."""

    property_getter = _safe_tensor_property_getter(key)
    if property_getter is not None:
        return property_getter, f"torch.Tensor.{key.qualname}"
    if key.namespace in _ALLOWED_EXACT_ROOTS:
        # r49 secF_1: resolve through the shared allowlisted reader, which special-cases the
        # top-level ``torch`` root to ``torch_attr`` (no PEP-562 lazy ``torch.__getattr__``
        # submodule import / deprecated ``replacement()``). The prior direct
        # ``getattr(_ALLOWED_EXACT_ROOTS[key.namespace], ...)`` fired that lazy hazard on an
        # attacker qualname (``onnx`` / ``_dynamo``) at plain ``tl.load()`` -- the exact
        # subscript-aliased-root site the AST immunizer could not see.
        func = _getattr_allowlisted(key.namespace, key.qualname)
        if callable(func):
            return cast(Callable[..., Any], func), f"{key.namespace}.{key.qualname}"
    if stock_path is None:
        return None
    namespace, _, qualname = stock_path.rpartition(".")
    if namespace not in _ENUMERATED_TORCH_NAMESPACES:
        return None
    func = _getattr_allowlisted(namespace, qualname)
    if func is None:
        return None
    return func, f"{namespace}.{qualname}"


def _safe_tensor_property_getter(key: FunctionRegistryKey) -> Callable[..., Any] | None:
    """Return a callable getter for an allowlisted tensor property key."""

    if (
        key.namespace != "torch.Tensor"
        or key.dispatch_kind != "method"
        or key.qualname not in _SAFE_TENSOR_PROPERTY_NAMES
    ):
        return None

    def getter(tensor: torch.Tensor) -> torch.Tensor:
        """Return one safe tensor property value."""

        return cast(torch.Tensor, getattr(tensor, key.qualname))

    getter.__name__ = str(key.qualname)
    getter.__qualname__ = f"Tensor.{key.qualname}"
    getter.__module__ = "torch._tensor"
    return getter


def _getattr_allowlisted(namespace: str, qualname: str) -> Callable[..., Any] | None:
    """Read a callable from one explicitly allowlisted in-memory namespace."""

    if namespace in _ALLOWED_EXACT_ROOTS:
        root = _ALLOWED_EXACT_ROOTS[namespace]
    elif namespace in _ENUMERATED_TORCH_NAMESPACES:
        root = _torch_namespace(namespace)
    else:
        return None
    if root is None:
        return None
    # r42 secC_1: the PEP-562 lazy-submodule ``__getattr__`` hazard (unrequested
    # ``_inductor``/``_dynamo``/``_export``/``onnx`` import + raw error leak) lives ONLY on the
    # top-level ``torch`` module. Reading its ``__dict__`` directly never fires ``__getattr__``
    # and still resolves every real ``torch.*`` callable. Class roots (``torch.Tensor`` /
    # ``_VariableFunctions``) and the proxying ``torch._VF`` module expose inherited/proxied
    # callables only through ``getattr`` and carry no lazy-import hazard (their enumerated
    # namespaces are fixed, never attacker-chosen lazy submodules).
    if root is torch:
        value = torch_attr(qualname)
    else:
        value = getattr(root, qualname, None)
    return cast(Callable[..., Any], value) if callable(value) else None


def _torch_namespace(namespace: str) -> Any | None:
    """Traverse an enumerated torch namespace without importing modules."""

    parts = namespace.removeprefix("torch.").split(".")
    # r49 secF_1: the FIRST hop is off the TOP-LEVEL torch module, whose PEP-562
    # ``__getattr__`` can fire a lazy submodule import; resolve it through ``torch_attr``
    # (identifier-only ``torch.__dict__`` read -- proven behavior-preserving over all
    # enumerated first-parts). Later hops are off submodule/class roots that carry no
    # lazy-import hazard, so they stay on ``getattr``.
    current: Any = torch_attr(parts[0])
    for part in parts[1:]:
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def _stock_path_from_key(key: FunctionRegistryKey) -> str | None:
    """Return a normalized stock torch path without importing custom modules."""

    if key.namespace != "custom":
        return f"{key.namespace}.{key.qualname}"
    if not key.import_path:
        return None
    module_name, separator, qualname = key.import_path.partition(":")
    if not separator or not (module_name == "torch" or module_name.startswith("torch.")):
        return None
    return f"{module_name}.{qualname}"


def _is_captured_internal_torch_builtin_key(key: FunctionRegistryKey) -> bool:
    """Return whether ``key`` identifies a recipe-sensitive torch builtin.

    Parameters
    ----------
    key:
        Saved callable registry key.

    Returns
    -------
    bool
        Whether the key was emitted for a canonical
        ``_VariableFunctionsClass`` builtin.
    """

    if key.namespace != "custom" or key.import_path is None:
        return False
    module_name, separator, qualname = key.import_path.partition(":")
    return (
        separator == ":"
        and module_name == "torch._C._VariableFunctionsClass"
        and qualname == key.qualname
    )


def _key_display_path(key: FunctionRegistryKey) -> str:
    """Return a stable display path for one recorded registry key."""

    return (
        key.import_path.replace(":", ".") if key.import_path else f"{key.namespace}.{key.qualname}"
    )


def _is_private_path(path: str) -> bool:
    """Return whether a recorded torch path traverses a private namespace."""

    return any(part.startswith("_") for part in path.split(".")[1:-1])


def _unwrap_decorated(func: Callable[..., Any]) -> Callable[..., Any]:
    """Translate a TorchLens decorated function to its original callable."""

    current = func
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        original = _state._decorated_to_orig.get(id(current))
        if original is None:
            break
        current = original
    return current


@lru_cache(maxsize=1)
def _reverse_index() -> Mapping[str, tuple[_ReverseCandidate, ...]]:
    """Build the cached reverse index over TorchLens's torch wrapper inventory."""

    index: dict[str, list[_ReverseCandidate]] = {}
    seen: set[tuple[str, str]] = set()
    for namespace, qualname in get_orig_torch_funcs(include_torchvision=False):
        marker = (namespace, qualname)
        if marker in seen:
            continue
        seen.add(marker)
        root = _torch_namespace(namespace) if namespace != "torch" else torch
        if root is None:
            continue
        # r49 secF_1: reading a top-level torch qualname via ``getattr`` can fire the PEP-562
        # lazy ``__getattr__``; route the torch root through ``torch_attr`` (proven behavior-
        # preserving over TorchLens's own top-level torch qualname inventory). Submodule roots
        # carry no lazy hazard and stay on ``getattr``.
        value = torch_attr(qualname) if root is torch else getattr(root, qualname, None)
        if not callable(value):
            continue
        candidate = _ReverseCandidate(
            namespace=namespace,
            qualname=qualname,
            func=_unwrap_decorated(cast(Callable[..., Any], value)),
        )
        index.setdefault(qualname, []).append(candidate)
    return {name: tuple(candidates) for name, candidates in index.items()}


def _reverse_candidates(
    qualname: str,
    calls: tuple[RunnableCallDescriptor, ...],
) -> tuple[_ReverseCandidate, ...]:
    """Return name- and arity-compatible reverse-index candidates."""

    func_name = qualname.rsplit(".", maxsplit=1)[-1]
    return tuple(
        candidate
        for candidate in _reverse_index().get(func_name, ())
        if all(_signature_accepts_call(candidate.func, call) for call in calls)
    )


def _best_ranked_candidates(
    candidates: tuple[_ReverseCandidate, ...],
    key: FunctionRegistryKey,
    stock_path: str | None,
) -> tuple[_ReverseCandidate, ...]:
    """Keep only candidates at the best deterministic namespace rank."""

    expected_namespace: str = key.namespace
    if stock_path is not None:
        expected_namespace = stock_path.rpartition(".")[0]
    public_rank = {
        "torch": 1,
        "torch.nn.functional": 2,
        "torch.Tensor": 3,
        "operator": 4,
    }

    def rank(candidate: _ReverseCandidate) -> int:
        """Return one candidate's deterministic namespace rank."""

        if candidate.namespace == expected_namespace:
            return 0
        return public_rank.get(candidate.namespace, 10)

    best = min(rank(candidate) for candidate in candidates)
    ranked = [candidate for candidate in candidates if rank(candidate) == best]
    deduplicated: dict[int, _ReverseCandidate] = {}
    for candidate in ranked:
        deduplicated.setdefault(id(candidate.func), candidate)
    return tuple(deduplicated.values())


def _signature_accepts_call(
    func: Callable[..., Any],
    call: RunnableCallDescriptor,
) -> bool:
    """Return whether an inspectable callable accepts recorded positional arity."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    argument_paths = (
        *(argument.argument_path for argument in call.tensor_arguments),
        *(argument.argument_path for argument in call.literal_arguments),
    )
    # r53 free_1: the placeholder count derives from the VALIDATED positional
    # leaf roots, never directly from the persisted ``num_positional_args``
    # integer -- parse anchoring guarantees the two agree for any loaded
    # descriptor, and an in-memory descriptor that bypassed parsing can no
    # longer scale this allocation with a hostile integer.
    positional_roots = {
        path[1]
        for path in argument_paths
        if len(path) >= 2 and path[0] == "args" and isinstance(path[1], int)
    }
    placeholders = (object(),) * len(positional_roots)
    keyword_names = {
        path[1]
        for path in argument_paths
        if len(path) >= 2 and path[0] == "kwargs" and isinstance(path[1], str)
    }
    keyword_placeholders = {name: object() for name in keyword_names}
    try:
        signature.bind_partial(*placeholders, **keyword_placeholders)
    except TypeError:
        return False
    return True


def _descriptor_version_diagnostics(
    descriptor: SparseRunDescriptor,
) -> tuple[RunnableDiagnostic, ...]:
    """Return structured readiness failures for unsupported descriptor ceilings."""

    diagnostics: list[RunnableDiagnostic] = []
    expected_strings = (
        ("capability", descriptor.capability, RUNNABLE_TLSPEC_SCHEMA_VERSION),
        ("call_recipe", descriptor.call_recipe, RUNNABLE_CALL_RECIPE_VERSION),
        (
            "initializer_policy_version",
            descriptor.initializer_policy_version,
            RUNNABLE_INITIALIZER_POLICY_VERSION,
        ),
        ("state_binding", descriptor.state_binding, "module_path_role_v1"),
        ("input_binding", descriptor.input_binding, "model_site_io_role_v1"),
        (
            "control_witness",
            descriptor.control_witness,
            "scalar_bool_and_arm_entry_v1",
        ),
        (
            "compatibility.descriptor_version",
            descriptor.compatibility.descriptor_version,
            RUNNABLE_TLSPEC_SCHEMA_VERSION,
        ),
        (
            "compatibility.call_recipe_version",
            descriptor.compatibility.call_recipe_version,
            RUNNABLE_CALL_RECIPE_VERSION,
        ),
        (
            "compatibility.initializer_policy_version",
            descriptor.compatibility.initializer_policy_version,
            RUNNABLE_INITIALIZER_POLICY_VERSION,
        ),
    )
    for field_name, actual, expected in expected_strings:
        if actual == expected:
            continue
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                f"Unsupported runnable {field_name}={actual!r}; runtime supports {expected!r}.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(("field", field_name), ("supported", str(expected))),
            )
        )
    activations_layer = descriptor.payload_layers.activations
    if activations_layer.schema != RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                f"Unsupported activation payload schema {activations_layer.schema!r}; "
                f"runtime supports {RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION!r}.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(
                    ("field", "payload_layers.activations.schema"),
                    ("supported", RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION),
                ),
            )
        )
    nonpersistent_layer = descriptor.payload_layers.nonpersistent_buffers
    has_nonpersistent_slots = any(
        slot.role is TensorSlotRole.BUFFER
        and slot.state_binding is not None
        and not slot.state_binding.persistent
        for slot in descriptor.tensor_slots
    )
    if (
        nonpersistent_layer.schema != "runnable_nonpersistent_buffer_v1"
        or nonpersistent_layer.present != has_nonpersistent_slots
    ):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                "Non-persistent buffer payload declaration disagrees with its slots.",
                descriptor=descriptor,
                detection_stage="descriptor_payload_validation",
                details=(
                    ("schema", nonpersistent_layer.schema),
                    ("declared_present", str(nonpersistent_layer.present)),
                    ("has_nonpersistent_slots", str(has_nonpersistent_slots)),
                ),
            )
        )
    if descriptor.callable_ref_schema != RUNNABLE_CALLABLE_REF_SCHEMA_VERSION:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_REF_SCHEMA,
                "Callable reference schema is newer or otherwise unsupported.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(
                    ("recorded", str(descriptor.callable_ref_schema)),
                    ("supported", str(RUNNABLE_CALLABLE_REF_SCHEMA_VERSION)),
                ),
            )
        )
    incompatible_key_ids = tuple(
        entry.registry_id
        for entry in descriptor.callable_registry
        if entry.key.version != RUNNABLE_CALLABLE_REF_SCHEMA_VERSION
    )
    if (
        descriptor.compatibility.callable_ref_schema_version != RUNNABLE_CALLABLE_REF_SCHEMA_VERSION
        or incompatible_key_ids
    ):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_REF_SCHEMA,
                "Compatibility metadata or a registry key uses an unsupported ref schema.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(
                    (
                        "compatibility_recorded",
                        str(descriptor.compatibility.callable_ref_schema_version),
                    ),
                    ("registry_ids", ",".join(incompatible_key_ids)),
                ),
            )
        )
    return tuple(diagnostics)


def _readiness_report(
    descriptor: SparseRunDescriptor,
    *,
    records: tuple[ResolverRecord, ...],
    diagnostics: tuple[RunnableDiagnostic, ...],
    ready: bool,
) -> ReadinessReport:
    """Build the frozen readiness report shape for one sparse descriptor."""

    state_sources = [StateSource.RANDOM_INITIALIZATION]
    if descriptor.payload_layers.weights.present:
        state_sources.insert(0, StateSource.EMBEDDED_CAPTURE_STATE)
    return ReadinessReport(
        status=ReadinessStatus.READY if ready else ReadinessStatus.UNAVAILABLE,
        provider=RunProvider.LOADED_SPARSE,
        backend=descriptor.backend,
        capability=descriptor.capability,
        resolver_records=records,
        state_sources_available=tuple(state_sources),
        # r71 A3: readiness republishes the parser-DERIVED completeness floor (equal
        # to the persisted summary by the parse-time equality check), never a raw
        # summary read (source-scan tripwire).
        witness_completeness=derived_witness_completeness(descriptor.coverage_gaps),
        diagnostics=diagnostics,
    )


def _analysis_only_readiness(trace: Any) -> ReadinessReport:
    """Build unsupported readiness for a loaded analysis-only Trace."""

    backend = str(getattr(trace, "backend", "unknown"))
    diagnostic = _diagnostic(
        RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
        "Loaded analysis artifact has no sparse runnable descriptor.",
        backend=backend,
        detection_stage="descriptor_presence",
    )
    return ReadinessReport(
        status=ReadinessStatus.UNAVAILABLE,
        provider=RunProvider.LOADED_ANALYSIS,
        backend=backend,
        capability=None,
        resolver_records=(),
        state_sources_available=(),
        witness_completeness=None,
        diagnostics=(diagnostic,),
    )


def _store_readiness(
    trace: Any,
    descriptor: SparseRunDescriptor | None,
    report: ReadinessReport,
    attachments: Mapping[str, Callable[..., Any]] | None,
) -> None:
    """Store transient descriptor/readiness state with atomic callable attachment."""

    trace.__dict__["_runnable_descriptor"] = descriptor
    trace.__dict__["_runnable_readiness"] = report
    if attachments is None:
        trace.__dict__.pop("_runnable_callables_by_call_id", None)
    else:
        trace.__dict__["_runnable_callables_by_call_id"] = dict(attachments)


def _diagnostic(
    code: RunnableErrorCode,
    message: str,
    *,
    descriptor: SparseRunDescriptor | None = None,
    backend: str | None = None,
    registry_id: str | None = None,
    affected_ops: tuple[str, ...] = (),
    detection_stage: str,
    provenance: str | None = None,
    details: tuple[tuple[str, str], ...] = (),
) -> RunnableDiagnostic:
    """Build one complete Stage 3 readiness diagnostic record."""

    recorded_runtime = descriptor.compatibility.backend_version if descriptor is not None else None
    current_backend = descriptor.backend if descriptor is not None else backend
    current_runtime = (
        str(torch.__version__) if current_backend == "torch" else platform.python_version()
    )
    return RunnableDiagnostic(
        code=code,
        message=message,
        registry_id=registry_id,
        affected_op_labels=affected_ops,
        recorded_runtime=recorded_runtime,
        current_runtime=current_runtime,
        detection_stage=detection_stage,
        resolver_provenance=provenance,
        analysis_load_available=True,
        details=details,
    )


def _parse_registry_key(value: Mapping[str, Any]) -> FunctionRegistryKey:
    """Parse one FunctionRegistryKey-shaped JSON object."""

    import_path = value.get("import_path")
    if import_path is not None and not isinstance(import_path, str):
        raise TypeError("FunctionRegistryKey.import_path must be a string or null.")
    return FunctionRegistryKey(
        namespace=cast(Any, _string(value, "namespace")),
        qualname=_string(value, "qualname"),
        dispatch_kind=cast(Any, _string(value, "dispatch_kind")),
        version=_integer(value, "version"),
        import_path=import_path,
    )


def _parse_call(value: Mapping[str, Any]) -> RunnableCallDescriptor:
    """Parse one runnable computational call descriptor.

    r53 free_1: the persisted arity integers are anchored to the actual dense
    argument leaves BEFORE the descriptor is constructed, so no downstream
    consumer (`_signature_accepts_call`, `_pre_call_contract_checks`,
    `_execute_sparse_call`) ever scales an allocation by an unvalidated
    manifest integer.
    """

    call_id = _string(value, "call_id")
    num_positional_args = _integer(value, "num_positional_args")
    num_keyword_args = _integer(value, "num_keyword_args")
    tensor_arguments = tuple(
        TensorArgumentRef(
            argument_path=_path(item, "argument_path"),
            slot_id=_string(item, "slot_id"),
        )
        for item in _mapping_sequence(value, "tensor_arguments")
    )
    literal_arguments = tuple(
        LiteralArgumentRef(
            argument_path=_path(item, "argument_path"),
            value=_parse_literal(item["value"]),
        )
        for item in _mapping_sequence(value, "literal_arguments")
    )
    _validate_call_arity(
        call_id, num_positional_args, num_keyword_args, tensor_arguments, literal_arguments
    )
    # r71 A2: the per-call obligation fields are REQUIRED (call-recipe v3); absence is
    # a parse failure (analysis-only load), never a defaulted empty obligation set.
    obligations: list[CallControlObligation] = []
    for item in _mapping_sequence(value, "control_obligations"):
        kind_text = _string(item, "kind")
        if kind_text not in {"scalar_bool", "loop_predicate"}:
            raise ContextFieldInvalidError(
                "calls.control_obligations",
                f"obligation kind {kind_text!r} is outside the closed control vocabulary",
            )
        raw_conditional = item.get("conditional_id")
        obligations.append(
            CallControlObligation(
                kind=ControlWitnessKind(kind_text),
                output_slot_id=_string(item, "output_slot_id"),
                site_label=_string(item, "site_label"),
                conditional_id=(
                    None
                    if raw_conditional is None
                    else _integer_item(raw_conditional, "conditional_id")
                ),
            )
        )
    dependencies: list[ControlDependencyEdge] = []
    for item in _mapping_sequence(value, "control_dependencies"):
        dependencies.append(
            ControlDependencyEdge(
                conditional_id=_integer(item, "conditional_id"),
                arm_kind=_string(item, "arm_kind"),
                parent_op_label=_string(item, "parent_op_label"),
                child_op_label=_string(item, "child_op_label"),
            )
        )
    return RunnableCallDescriptor(
        call_id=call_id,
        op_labels=_string_tuple(value, "op_labels"),
        registry_id=_string(value, "registry_id"),
        dispatch_kind=cast(Any, _string(value, "dispatch_kind")),
        argument_names=_string_tuple(value, "argument_names"),
        num_positional_args=num_positional_args,
        num_keyword_args=num_keyword_args,
        tensor_arguments=tensor_arguments,
        literal_arguments=literal_arguments,
        output_slot_ids=_string_tuple(value, "output_slot_ids"),
        parent_call_ids=_string_tuple(value, "parent_call_ids"),
        is_inplace=_boolean(value, "is_inplace"),
        runtime_fingerprint=_string(value, "runtime_fingerprint"),
        # v2: the per-call execution context is REQUIRED; absence is a parse
        # failure (legacy artifacts are analysis-only), never "assume disabled".
        execution_context=_parse_call_execution_context(_mapping(value, "execution_context")),
        control_obligations=tuple(obligations),
        control_dependencies=tuple(dependencies),
    )


def _parse_slot(value: Mapping[str, Any]) -> TensorSlotDescriptor:
    """Parse one value-free tensor slot descriptor."""

    input_value = value.get("input_binding")
    state_value = value.get("state_binding")
    output_path = value.get("output_path")
    version_of = value.get("version_of")
    producer_slot_id = value.get("producer_slot_id")
    device_index = value.get("device_index")
    slot_id = _string(value, "slot_id")
    shape = tuple(_integer_item(item, "shape") for item in _sequence(value, "shape"))
    rank = _integer(value, "rank")
    # r47 secD_1/secF_1: the tensor-slot dtype is attacker-controlled. Parse-validate it against
    # the live dtype table (closed vocabulary, like the ambient/autocast dtypes) so a hazardous
    # ``"onnx"`` / ``"_dynamo"`` / ``"has_cuda"`` slot-dtype is refused at PARSE -- before the
    # ``.run()`` random-initializer would resolve it -- via ``torch_attr`` (no lazy import).
    dtype = _validated_dtype_literal("tensor_slots.dtype", _string(value, "dtype"))
    # r53 free_1: anchor the persisted shape/rank BEFORE the descriptor exists, so
    # the ``.run()`` random role-initializer's ``torch.empty(slot.shape)`` can never
    # be scaled by a structurally impossible manifest integer.
    _validate_slot_shape(slot_id, shape, rank, dtype)
    # r55 FORK D: the tensor-slot ``device_type`` is attacker-controlled and feeds
    # ``torch.device(...)`` at run-preparation (state alloc + output-slot preflight).
    # Anchor it to the SAME closed torch device-type vocabulary the ambient/autocast
    # device literals use, so a bogus/path-bearing device token is refused at PARSE
    # (analysis-only), never reaching the run-prep device constructor.
    device_type = _string(value, "device_type")
    if device_type not in _DEVICE_TYPE_VOCABULARY:
        raise DescriptorStructuralBoundError(
            RunnableErrorCode.STATE_SHAPE_MISMATCH,
            f"tensor_slots[{slot_id}].device_type",
            f"device type {device_type!r} is outside the closed device vocabulary",
        )
    return TensorSlotDescriptor(
        slot_id=slot_id,
        role=TensorSlotRole(_string(value, "role")),
        use_sites=tuple(
            TensorUseSite(
                call_id=_string(item, "call_id"),
                argument_path=_path(item, "argument_path"),
            )
            for item in _mapping_sequence(value, "use_sites")
        ),
        shape=shape,
        dtype=dtype,
        rank=rank,
        device_type=device_type,
        device_index=None if device_index is None else _integer_item(device_index, "device_index"),
        mutable=_boolean(value, "mutable"),
        version_of=_optional_string(version_of, "version_of"),
        producer_slot_id=_optional_string(producer_slot_id, "producer_slot_id"),
        output_path=None if output_path is None else _path_value(output_path, "output_path"),
        input_binding=(
            None
            if input_value is None
            else _parse_input_binding(_mapping_item(input_value, "input_binding"))
        ),
        state_binding=(
            None
            if state_value is None
            else _parse_state_binding(_mapping_item(state_value, "state_binding"))
        ),
        # r71 A2: REQUIRED owner-record obligations -- absence is a parse failure.
        host_escape=_boolean(value, "host_escape"),
        inert_sink=_boolean(value, "inert_sink"),
    )


def _parse_input_binding(value: Mapping[str, Any]) -> InputSlotBinding:
    """Parse one model-input slot binding."""

    position = value["model_site_position"]
    if isinstance(position, list):
        parsed_position: str | int | tuple[str | int, ...] = _path_value(
            position, "model_site_position"
        )
    elif isinstance(position, (str, int)) and not isinstance(position, bool):
        parsed_position = position
    else:
        raise TypeError("model_site_position must be a string, integer, or path array.")
    return InputSlotBinding(
        io_role=cast(Any, _string(value, "io_role")),
        model_ref=_string(value, "model_ref"),
        model_site_position=parsed_position,
        container_record_id=_integer(value, "container_record_id"),
        container_path=_path(value, "container_path"),
    )


def _parse_state_binding(value: Mapping[str, Any]) -> StateSlotBinding:
    """Parse one named parameter or buffer binding (r71: + totalized declared facts)."""

    disposition = value.get("host_escape_disposition")
    if disposition is not None and disposition not in {"escaped", "inert"}:
        raise ContextFieldInvalidError(
            "tensor_slots.state_binding.host_escape_disposition",
            f"disposition {disposition!r} is outside the closed claim vocabulary",
        )
    captured_grad_fn = _boolean(value, "captured_grad_fn")
    if captured_grad_fn:
        # No staged leaf can carry a grad_fn (r65 F-1): a True fact can never be
        # reproduced by any staged or fresh-oracle state, so an artifact claiming it
        # is refused at parse -- fail closed, analysis-only.
        raise ContextFieldInvalidError(
            "tensor_slots.state_binding.captured_grad_fn",
            "grad_fn presence cannot be reproduced by staged state",
        )
    return StateSlotBinding(
        module_path=_string(value, "module_path"),
        state_dict_name=_string(value, "state_dict_name"),
        semantic_role=StateSlotRole(_string(value, "semantic_role")),
        trainable=_boolean(value, "trainable"),
        persistent=_boolean(value, "persistent"),
        alias_group=_optional_string(value.get("alias_group"), "alias_group"),
        captured_requires_grad=_boolean(value, "captured_requires_grad"),
        captured_grad_fn=captured_grad_fn,
        host_escape_disposition=cast(Any, disposition),
    )


def _parse_witness(value: Mapping[str, Any]) -> ControlWitness:
    """Parse one ordered control-flow witness."""

    return ControlWitness(
        witness_id=_string(value, "witness_id"),
        kind=ControlWitnessKind(_string(value, "kind")),
        order=_integer(value, "order"),
        call_id=_optional_string(value.get("call_id"), "call_id"),
        site_label=_string(value, "site_label"),
        observed_value=_parse_literal(value["observed_value"]),
    )


def _parse_payload_layer(value: Mapping[str, Any]) -> PayloadLayerDescriptor:
    """Parse one optional payload-layer declaration."""

    return PayloadLayerDescriptor(
        present=_boolean(value, "present"),
        schema=_string(value, "schema"),
    )


def _parse_nonpersistent_buffer_payload_layer(
    payload_layers: Mapping[str, Any],
) -> PayloadLayerDescriptor:
    """Parse capture-embedded non-persistent buffers with a legacy default."""

    value = payload_layers.get("nonpersistent_buffers")
    if not isinstance(value, Mapping):
        return PayloadLayerDescriptor(
            present=False,
            schema="runnable_nonpersistent_buffer_v1",
        )
    return _parse_payload_layer(value)


def _parse_activation_payload_layer(
    value: Mapping[str, Any],
) -> PayloadLayerDescriptor | ActivationPayloadLayerDescriptor:
    """Parse selected-activation membership and eligibility digests."""

    if not _boolean(value, "present"):
        return _parse_payload_layer(value)
    return ActivationPayloadLayerDescriptor(
        present=_boolean(value, "present"),
        schema=cast(Any, _string(value, "schema")),
        input_fingerprints=tuple(
            _parse_input_fingerprint(item)
            for item in _mapping_sequence(value, "input_fingerprints")
        ),
        members=tuple(
            ActivationPayloadMember(
                blob_id=_string(item, "blob_id"),
                slot_id=_string(item, "slot_id"),
                call_id=_optional_string(item.get("call_id"), "call_id"),
                op_label=_string(item, "op_label"),
                field=cast(Any, _string(item, "field")),
                byte_digest=_string(item, "byte_digest"),
            )
            for item in _mapping_sequence(value, "members")
        ),
        original_input_digests=tuple(
            SlotByteDigest(
                slot_id=_string(item, "slot_id"),
                byte_digest=_string(item, "byte_digest"),
            )
            for item in _mapping_sequence(value, "original_input_digests")
        ),
        capture_state_digests=tuple(
            StateByteDigest(
                state_dict_name=_string(item, "state_dict_name"),
                byte_digest=_string(item, "byte_digest"),
            )
            for item in _mapping_sequence(value, "capture_state_digests")
        ),
    )


def _parse_diagnostic(value: Mapping[str, Any]) -> RunnableDiagnostic:
    """Parse one persisted producer diagnostic."""

    return RunnableDiagnostic(
        code=RunnableErrorCode(_string(value, "code")),
        message=_string(value, "message"),
        registry_id=_optional_string(value.get("registry_id"), "registry_id"),
        affected_op_labels=_string_tuple(value, "affected_op_labels"),
        recorded_runtime=_optional_string(value.get("recorded_runtime"), "recorded_runtime"),
        current_runtime=_optional_string(value.get("current_runtime"), "current_runtime"),
        detection_stage=_string(value, "detection_stage"),
        resolver_provenance=_optional_string(
            value.get("resolver_provenance"), "resolver_provenance"
        ),
        analysis_load_available=_boolean(value, "analysis_load_available"),
        details=tuple(
            (_string_item(item[0], "details key"), _string_item(item[1], "details value"))
            for item in _sequence(value, "details")
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2
        ),
    )


def _validate_literal_atom_value(kind: "LiteralAtomKind", value: Any) -> None:
    """Enforce that an atom's JSON value matches its declared ``kind``.

    ROBUSTNESS (secC informational note). ``LiteralAtom`` is a *scalar* grammar
    node: downstream witness / decode logic assumes an atom carries the scalar its
    ``kind`` promises (e.g. ``bool(...)`` / ``.get(...)`` on a witness
    ``observed_value``). The encoder (``_io/runnable.py``) only ever emits one JSON
    type per kind, but a hand-edited ``manifest.json`` could tag ``kind="int"`` on a
    list / dict / string, which decodes inertly yet makes scalar-assuming logic
    inconsistent. Reject the type mismatch at parse so the grammar invariant is
    upheld. (A genuine container value has its own ``LiteralSequence`` /
    ``LiteralMapping`` node; it never rides on a scalar atom.) ``bool`` is a subclass
    of ``int`` in Python, so ``INT`` explicitly excludes ``bool`` (and vice versa) to
    match the encoder's distinct ``BOOL`` / ``INT`` kinds; JSON ``float`` never
    parses as ``int``/``bool`` so ``FLOAT`` needs no such exclusion.
    """

    if kind in (LiteralAtomKind.NONE, LiteralAtomKind.ELLIPSIS):
        if value is not None:
            raise ValueError(
                f"Runnable literal atom kind {kind.value!r} requires a null value, "
                f"got {type(value).__name__}."
            )
    elif kind is LiteralAtomKind.BOOL:
        if not isinstance(value, bool):
            raise ValueError(
                f"Runnable literal atom kind 'bool' requires a bool value, "
                f"got {type(value).__name__}."
            )
    elif kind is LiteralAtomKind.INT:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"Runnable literal atom kind 'int' requires an int value, "
                f"got {type(value).__name__}."
            )
        # r55 CLASS 3 (free_1): close the literal/slot magnitude asymmetry. A slot
        # shape is bounded to a signed-64-bit byte product at parse; a literal int
        # (e.g. the ``n`` in a taken-path ``torch.arange(n)``/``zeros(n)``) carried
        # NO magnitude bound, so a self-consistent descriptor could drive an
        # allocation more extreme than any slot dimension is allowed to be. Gate it
        # under the SAME int64 ceiling here; a genuinely-large-but-feasible literal
        # is still bounded further by the run-prep output-slot allocation preflight
        # and the per-call projection preflight, never over-refused at parse.
        if abs(value) > _INT64_MAX:
            raise DescriptorStructuralBoundError(
                RunnableErrorCode.STATE_SHAPE_MISMATCH,
                "literal_arguments",
                f"literal int magnitude {value} exceeds the signed 64-bit ceiling",
            )
    elif kind is LiteralAtomKind.FLOAT:
        if not isinstance(value, float):
            raise ValueError(
                f"Runnable literal atom kind 'float' requires a float value, "
                f"got {type(value).__name__}."
            )
    elif kind in (LiteralAtomKind.STR, LiteralAtomKind.NONFINITE_FLOAT):
        if not isinstance(value, str):
            raise ValueError(
                f"Runnable literal atom kind {kind.value!r} requires a str value, "
                f"got {type(value).__name__}."
            )


def _parse_literal(value: Any, *, _depth: int = 0) -> NonTensorLiteral:
    """Parse one recursively tagged non-tensor literal."""

    if _depth > _MAX_LITERAL_NESTING_DEPTH:
        raise ValueError(
            f"Runnable literal nesting exceeds the maximum depth of {_MAX_LITERAL_NESTING_DEPTH}."
        )
    mapping = _mapping_item(value, "literal")
    keys = set(mapping)
    if keys == {"kind", "value"}:
        kind = LiteralAtomKind(_string(mapping, "kind"))
        atom_value = mapping["value"]
        _validate_literal_atom_value(kind, atom_value)
        return LiteralAtom(
            kind=kind,
            value=cast(Any, atom_value),
        )
    if keys == {"qualname"}:
        qualname = _string(mapping, "qualname")
        _validate_torch_device_literal_shape(qualname)
        return LiteralTorchSymbol(qualname=qualname)
    if keys == {"start", "stop", "step"}:
        return LiteralSlice(
            start=_parse_literal_slice_component(mapping["start"], "start", _depth=_depth + 1),
            stop=_parse_literal_slice_component(mapping["stop"], "stop", _depth=_depth + 1),
            step=_parse_literal_slice_component(mapping["step"], "step", _depth=_depth + 1),
        )
    if keys == {"kind", "items"}:
        return LiteralSequence(
            kind=LiteralSequenceKind(_string(mapping, "kind")),
            items=tuple(
                _parse_literal(item, _depth=_depth + 1) for item in _sequence(mapping, "items")
            ),
        )
    if keys == {"entries"}:
        return LiteralMapping(
            entries=tuple(
                LiteralMappingEntry(
                    key=_parse_literal_key(item["key"], _depth=_depth + 1),
                    value=_parse_literal(item["value"], _depth=_depth + 1),
                )
                for item in _mapping_sequence(mapping, "entries")
            )
        )
    raise ValueError(f"Unsupported tagged runnable literal fields: {sorted(keys)}")


def _validate_torch_device_literal_shape(qualname: str) -> None:
    """Refuse a malformed ``torch.device(...)`` literal payload at descriptor parse (r41 secC).

    Parse-time belt for the run-time typed wrap in
    ``torchlens._runnable_execution._decode_torch_symbol``: only the DEVICE literal's
    payload is validated here -- ``torch.device`` construction is inert (no import, no
    exec; it succeeds for absent backends such as ``cuda:0`` on a CPU-only host, so it
    cannot over-refuse) -- while the symbol ALLOWLIST stays exclusively the runtime
    decoder's job (no duplication). Raising ``ValueError`` routes through
    ``parse_sparse_run_descriptor``'s existing descriptor-parse refusal, degrading the
    load to ANALYSIS-ONLY with the typed diagnostic intact (the corr2_3 disposition),
    never a hard load failure.
    """

    if qualname.startswith("torch.device(") and qualname.endswith(")"):
        try:
            torch.device(qualname[13:-1])
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ValueError(f"Malformed torch.device literal payload {qualname!r}: {exc}") from exc


def _parse_literal_slice_component(value: Any, field_name: str, *, _depth: int = 0) -> LiteralAtom:
    """Parse one ``slice.start``/``.stop``/``.step`` tagged atom.

    A slice component is restricted to the ``NONE`` or ``INT`` atom kinds on the
    encode side (see ``_io/runnable.py::_encode_slice_component``); reject anything
    else here rather than silently widening what a loaded bundle may claim.
    """

    parsed = _parse_literal(value, _depth=_depth)
    if not isinstance(parsed, LiteralAtom) or parsed.kind not in (
        LiteralAtomKind.NONE,
        LiteralAtomKind.INT,
    ):
        raise ValueError(
            f"Runnable literal slice component {field_name!r} must be a NONE or INT atom."
        )
    return parsed


def _parse_literal_key(value: Any, *, _depth: int = 0) -> LiteralAtom | LiteralTupleKey:
    """Parse one safe mapping-key literal."""

    if _depth > _MAX_LITERAL_NESTING_DEPTH:
        raise ValueError(
            f"Runnable literal nesting exceeds the maximum depth of {_MAX_LITERAL_NESTING_DEPTH}."
        )
    mapping = _mapping_item(value, "literal mapping key")
    if set(mapping) == {"kind", "value"}:
        parsed = _parse_literal(mapping, _depth=_depth + 1)
        if isinstance(parsed, LiteralAtom):
            return parsed
    if set(mapping) == {"items"}:
        return LiteralTupleKey(
            items=tuple(
                _parse_literal_key(item, _depth=_depth + 1) for item in _sequence(mapping, "items")
            )
        )
    raise ValueError("Runnable literal mapping key is not an atom or tuple key.")


def _mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    """Return one required mapping field."""

    return _mapping_item(value[field], field)


def _mapping_item(value: Any, label: str) -> Mapping[str, Any]:
    """Validate and return one arbitrary mapping value."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return cast(Mapping[str, Any], value)


def _sequence(value: Mapping[str, Any], field: str) -> Sequence[Any]:
    """Return one required non-string sequence field."""

    item = value[field]
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
        raise TypeError(f"{field} must be an array.")
    return item


def _mapping_sequence(value: Mapping[str, Any], field: str) -> tuple[Mapping[str, Any], ...]:
    """Return one required array of object values."""

    return tuple(_mapping_item(item, field) for item in _sequence(value, field))


def _string(value: Mapping[str, Any], field: str) -> str:
    """Return one required string field."""

    return _string_item(value[field], field)


def _string_item(value: Any, label: str) -> str:
    """Validate and return one arbitrary string value."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    """Validate and return one optional string value."""

    if value is None:
        return None
    return _string_item(value, label)


def _string_tuple(value: Mapping[str, Any], field: str) -> tuple[str, ...]:
    """Return one required array of strings."""

    return tuple(_string_item(item, field) for item in _sequence(value, field))


def _integer(value: Mapping[str, Any], field: str) -> int:
    """Return one required non-boolean integer field."""

    return _integer_item(value[field], field)


def _integer_item(value: Any, label: str) -> int:
    """Validate and return one arbitrary non-boolean integer value."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    return value


def _boolean(value: Mapping[str, Any], field: str) -> bool:
    """Return one required boolean field."""

    item = value[field]
    if not isinstance(item, bool):
        raise TypeError(f"{field} must be a boolean.")
    return item


def _path(value: Mapping[str, Any], field: str) -> tuple[str | int, ...]:
    """Return one required string/integer path field."""

    return _path_value(value[field], field)


def _path_value(value: Any, label: str) -> tuple[str | int, ...]:
    """Validate and return one arbitrary container path array."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an array.")
    result: list[str | int] = []
    for item in value:
        # ``bool`` is a subclass of ``int`` and is accepted as a faithful,
        # round-trippable container-path key (``True`` stays distinct from ``1``
        # as a mapping key), matching the producer's normalized path grammar so a
        # bool-keyed output/input container never advertises runnable then loses
        # it on load.
        if not isinstance(item, (str, int)):
            raise TypeError(f"{label} entries must be strings or integers.")
        result.append(item)
    return tuple(result)


__all__ = [
    "attach_sparse_run_readiness",
    "parse_sparse_run_descriptor",
    "preflight_sparse_run_descriptor",
]
