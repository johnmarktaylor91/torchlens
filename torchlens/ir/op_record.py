"""Decomposed op record model: ``OpRecord = OpCore + typed facets``.

Producer-unification design of record (v4): the torch journal's op record
decomposes into a 13-field required core plus 13 typed optional facets. Every
one of the 54 legacy ``OpEvent`` fields has an explicit disposition here; the
``function`` and ``templates`` facets reuse the existing ``FunctionCallRef``
and ``ArgTemplateRef`` classes, ``source_trace``/``source_trace_id`` travel
via ``IngestExtras`` (they are trace-identity joins, not record facts), and
``grad_fn_handle`` never crosses onto ``OpRecord`` (single ownership: the
journal's ``grad_fn_handles_by_label_raw`` side index).

The strict read protocol exposes the legacy flat names as properties so
migrated consumers read either record shape; an unknown name raises
``OpRecordAttributeError`` — an ``AttributeError`` subclass, so surviving
``getattr(event, name, default)`` sites keep today's default semantics.

The typed amendment lane (nine exact-set families) also lives here: schemas
are structural (keyword-only constructors, no ``**kwargs``, no optional patch
paths), identity fields are unpatchable, and appends/folds runtime-validate
against the registry. Everything in this module is INERT until P3/P4 wire it
into the emit and ingest paths.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any

from .events import ArgTemplateRef, FunctionCallRef, OutputRef, ParentEdge
from .refs import ParamRef
from .semantics import BackendSemantics, CapturePolicy

OP_RECORD_SCHEMA_VERSION = 1


class OpRecordAttributeError(AttributeError):
    """Strict-protocol refusal for an unknown record attribute.

    Subclasses ``AttributeError`` deliberately: consumer sites relying on
    ``getattr(record, name, default)`` (for example the raw-shape hash's
    ``modules``/``func_name`` fallbacks) keep their exact default semantics
    instead of crashing.
    """


# ---------------------------------------------------------------------------
# Core + facets
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpCore:
    """The 13 required fields every journal op record carries."""

    seq: int
    kind: str  # "op" | "source" | "synthetic_output"
    label_raw: str
    layer_label_raw: str
    layer_type: str
    raw_index: int
    type_index: int
    step_index: int
    pass_index: int
    parents: tuple[ParentEdge, ...]
    output: OutputRef
    is_bottom_level: bool
    func_call_id: int | None


@dataclass(frozen=True, slots=True)
class GraphFacet:
    """Raw graph-topology metadata beyond the core parent edges."""

    parent_arg_positions: dict[str, dict[Any, str]]
    edge_uses: tuple[object, ...] = ()
    unattributed_tensor_args: tuple[str, ...] = ()
    dropped_edge_tensor_args: tuple[str, ...] = ()
    is_output_parent: bool = False
    input_was_parameter: bool = False
    equivalence_class: str | None = None


@dataclass(frozen=True, slots=True)
class ModulesFacet:
    """Module containment snapshots taken at op creation."""

    module_stack: tuple[object, ...] = ()
    modules: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class AncestryFacet:
    """Exhaustive-leg ancestry unions."""

    input_ancestors: frozenset[str] = frozenset()
    internal_source_ancestors: frozenset[str] = frozenset()
    root_ancestors: frozenset[str] = frozenset()
    has_internal_source_ancestor: bool = False


@dataclass(frozen=True, slots=True)
class AutogradFacet:
    """Portable autograd metadata (never the live handle)."""

    grad_fn_class_qualname: str | None = None


@dataclass(frozen=True, slots=True)
class TransformFacet:
    """torch.func / transform boundary metadata.

    ``transform_config`` here is the CLEAN user mapping: the two legacy
    smuggled channels are first-class fields (``fn_code_location``) or live
    on the annotations facet (``_tl_annotations``); the generated scatter
    re-injects both into the persisted ``transform_config`` cell for
    byte-identical parity.
    """

    is_transform: bool = False
    transform_kind: str | None = None
    transform_chain: tuple[str, ...] = ()
    transform_config: dict[str, object] = field(default_factory=dict)
    transform_fn_name: str | None = None
    transform_fn_qualname: str | None = None
    transform_fn_source: object | None = None
    fn_code_location: object | None = None


@dataclass(frozen=True, slots=True)
class ControlFacet:
    """Scalar-bool control-flow observations."""

    is_scalar_bool: bool | None = None
    bool_value: bool | None = None


@dataclass(frozen=True, slots=True)
class ParamsFacet:
    """Parameter participation."""

    params: tuple[ParamRef, ...] = ()
    parent_params: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationsFacet:
    """First-class annotation payload (absorbs both `_tl_annotations` channels)."""

    annotations: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyFacet:
    """Capture policy, backend semantics, and stream state flags."""

    backend_semantics: BackendSemantics | None = None
    policy: CapturePolicy | None = None
    predicate_matched: bool = False
    tracing_finished: bool = False
    construction_done: bool = False


@dataclass(frozen=True, slots=True)
class RecordingFacet:
    """Predicate-recording context (sparse producer, boundary retention)."""

    record_context: object | None = None
    capture_spec: object | None = None


@dataclass(frozen=True, slots=True)
class InterventionFacet:
    """Interventions that touched this op.

    ``intervention_template_ref`` is deliberately ABSENT: verified dead in P2
    (every producer writes None; no consumer reads it), it stays on the compat
    ``OpEvent`` only and dies with it in S15.
    """

    intervention_fired: bool = False
    intervention_replaced: bool = False
    fire_results: tuple[object, ...] = ()


# Facet name -> class; ``function``/``templates`` reuse the journal ref types.
FACET_CLASSES: dict[str, type] = {
    "function": FunctionCallRef,
    "templates": ArgTemplateRef,
    "graph": GraphFacet,
    "modules": ModulesFacet,
    "ancestry": AncestryFacet,
    "autograd": AutogradFacet,
    "transform": TransformFacet,
    "control": ControlFacet,
    "params": ParamsFacet,
    "annotations": AnnotationsFacet,
    "policy": PolicyFacet,
    "recording": RecordingFacet,
    "intervention": InterventionFacet,
}


def _default_graph_facet() -> GraphFacet:
    """Build the checked-in absent-``graph``-facet default."""

    # The parent_arg_positions default is the sparse producer's literal
    # (consumers index the two domains unguarded).
    return GraphFacet(parent_arg_positions={"args": {}, "kwargs": {}})


# Checked-in absent-facet defaults: ingest materializes an absent facet from
# THIS table (S5: absent facet != fabricated empty facet — the reducer and the
# scatter both read these, and a patch against an absent facet without a
# defaults entry is a planted-battery violation).
FACET_DEFAULTS: dict[str, Any] = {
    "graph": _default_graph_facet,
    "modules": ModulesFacet,
    "ancestry": AncestryFacet,
    "autograd": AutogradFacet,
    "transform": TransformFacet,
    "control": ControlFacet,
    "params": ParamsFacet,
    "annotations": AnnotationsFacet,
    "policy": PolicyFacet,
    "recording": RecordingFacet,
    "intervention": InterventionFacet,
    # function/templates have no absent default: function is REQUIRED for
    # kind="op" (SHELL supplies the minimal name-only ref), and templates is
    # ALWAYS PRESENT on torch (its two supply gates control the VALUES).
}


@dataclass(frozen=True, slots=True)
class IngestExtras:
    """Record-adjacent values that are joins, not record facts."""

    source_trace: object | None = None
    source_trace_id: str | None = None
    # Compat channel until torch decomposition deletes the ingest fallback:
    # the live handle is owned by the journal's side index, never the record.
    grad_fn_handle: object | None = None


@dataclass(frozen=True, slots=True)
class OpRecord:
    """One decomposed journal op record: required core + optional facets."""

    core: OpCore
    function: FunctionCallRef | None = None
    templates: ArgTemplateRef | None = None
    graph: GraphFacet | None = None
    modules_facet: ModulesFacet | None = None
    ancestry: AncestryFacet | None = None
    autograd: AutogradFacet | None = None
    transform: TransformFacet | None = None
    control: ControlFacet | None = None
    params_facet: ParamsFacet | None = None
    annotations_facet: AnnotationsFacet | None = None
    policy_facet: PolicyFacet | None = None
    recording: RecordingFacet | None = None
    intervention: InterventionFacet | None = None

    # ---- strict read protocol (legacy flat names) --------------------------

    def __getattr__(self, name: str) -> Any:
        raise OpRecordAttributeError(
            f"OpRecord has no attribute {name!r} (strict protocol; "
            "use the facet paths or the legacy property names)"
        )

    def _facet_or_default(self, facet_name: str) -> Any:
        """Return facet ``facet_name``, materializing the checked-in default if absent.

        Raises
        ------
        OpRecordAttributeError
            If the facet is absent and has no ``FACET_DEFAULTS`` entry, which makes
            it required rather than defaultable.
        """

        attribute = _FACET_ATTRIBUTES[facet_name]
        value = object.__getattribute__(self, attribute)
        if value is not None:
            return value
        default_factory = FACET_DEFAULTS.get(facet_name)
        if default_factory is None:
            raise OpRecordAttributeError(
                f"required facet {facet_name!r} absent on {self.core.label_raw!r}"
            )
        return default_factory()

    # core passthroughs
    @property
    def seq(self) -> int:
        return self.core.seq

    @property
    def kind(self) -> str:
        return self.core.kind

    @property
    def label_raw(self) -> str:
        return self.core.label_raw

    @property
    def layer_label_raw(self) -> str:
        return self.core.layer_label_raw

    @property
    def layer_type(self) -> str:
        return self.core.layer_type

    @property
    def raw_index(self) -> int:
        return self.core.raw_index

    @property
    def type_index(self) -> int:
        return self.core.type_index

    @property
    def step_index(self) -> int:
        return self.core.step_index

    @property
    def pass_index(self) -> int:
        return self.core.pass_index

    @property
    def parents(self) -> tuple[ParentEdge, ...]:
        return self.core.parents

    @property
    def output(self) -> OutputRef:
        return self.core.output

    @property
    def is_bottom_level(self) -> bool:
        return self.core.is_bottom_level

    @property
    def func_call_id(self) -> int | None:
        return self.core.func_call_id

    # facet-backed legacy names
    @property
    def parent_arg_positions(self) -> dict[str, dict[Any, str]]:
        """Mirror ``graph.parent_arg_positions`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").parent_arg_positions

    @property
    def _edge_uses(self) -> tuple[object, ...]:
        """Mirror ``graph.edge_uses`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").edge_uses

    @property
    def unattributed_tensor_args(self) -> tuple[str, ...]:
        """Mirror ``graph.unattributed_tensor_args`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").unattributed_tensor_args

    @property
    def dropped_edge_tensor_args(self) -> tuple[str, ...]:
        """Mirror ``graph.dropped_edge_tensor_args`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").dropped_edge_tensor_args

    @property
    def is_output_parent(self) -> bool:
        """Mirror ``graph.is_output_parent`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").is_output_parent

    @property
    def input_was_parameter(self) -> bool:
        """Mirror ``graph.input_was_parameter`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").input_was_parameter

    @property
    def equivalence_class(self) -> str | None:
        """Mirror ``graph.equivalence_class`` through the facet default (legacy flat name)."""

        return self._facet_or_default("graph").equivalence_class

    @property
    def module_stack(self) -> tuple[object, ...]:
        """Mirror ``modules.module_stack`` through the facet default (legacy flat name)."""

        return self._facet_or_default("modules").module_stack

    @property
    def modules(self) -> tuple[tuple[str, int], ...]:
        """Mirror ``modules.modules`` through the facet default (legacy flat name)."""

        return self._facet_or_default("modules").modules

    @property
    def input_ancestors(self) -> frozenset[str]:
        """Mirror ``ancestry.input_ancestors`` through the facet default (legacy flat name)."""

        return self._facet_or_default("ancestry").input_ancestors

    @property
    def internal_source_ancestors(self) -> frozenset[str]:
        """Mirror ``ancestry.internal_source_ancestors`` through the facet default (legacy flat name)."""

        return self._facet_or_default("ancestry").internal_source_ancestors

    @property
    def root_ancestors(self) -> frozenset[str]:
        """Mirror ``ancestry.root_ancestors`` through the facet default (legacy flat name)."""

        return self._facet_or_default("ancestry").root_ancestors

    @property
    def has_internal_source_ancestor(self) -> bool:
        """Mirror ``ancestry.has_internal_source_ancestor`` through the facet default (legacy flat name)."""

        return self._facet_or_default("ancestry").has_internal_source_ancestor

    @property
    def grad_fn_class_qualname(self) -> str | None:
        """Mirror ``autograd.grad_fn_class_qualname`` through the facet default (legacy flat name)."""

        return self._facet_or_default("autograd").grad_fn_class_qualname

    @property
    def is_transform(self) -> bool:
        """Mirror ``transform.is_transform`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").is_transform

    @property
    def transform_kind(self) -> str | None:
        """Mirror ``transform.transform_kind`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").transform_kind

    @property
    def transform_chain(self) -> tuple[str, ...]:
        """Mirror ``transform.transform_chain`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").transform_chain

    @property
    def transform_config(self) -> dict[str, object]:
        """Mirror ``transform.transform_config`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").transform_config

    @property
    def transform_fn_name(self) -> str | None:
        """Mirror ``transform.transform_fn_name`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").transform_fn_name

    @property
    def transform_fn_qualname(self) -> str | None:
        """Mirror ``transform.transform_fn_qualname`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").transform_fn_qualname

    @property
    def transform_fn_source(self) -> object | None:
        """Mirror ``transform.transform_fn_source`` through the facet default (legacy flat name)."""

        return self._facet_or_default("transform").transform_fn_source

    @property
    def is_scalar_bool(self) -> bool | None:
        """Mirror ``control.is_scalar_bool`` through the facet default (legacy flat name)."""

        return self._facet_or_default("control").is_scalar_bool

    @property
    def bool_value(self) -> bool | None:
        """Mirror ``control.bool_value`` through the facet default (legacy flat name)."""

        return self._facet_or_default("control").bool_value

    @property
    def params(self) -> tuple[ParamRef, ...]:
        """Mirror ``params.params`` through the facet default (legacy flat name)."""

        return self._facet_or_default("params").params

    @property
    def parent_params(self) -> tuple[object, ...]:
        """Mirror ``params.parent_params`` through the facet default (legacy flat name)."""

        return self._facet_or_default("params").parent_params

    @property
    def backend_semantics(self) -> BackendSemantics | None:
        """Mirror ``policy.backend_semantics`` through the facet default (legacy flat name)."""

        return self._facet_or_default("policy").backend_semantics

    @property
    def policy(self) -> CapturePolicy | None:
        """Mirror ``policy.policy`` through the facet default (legacy flat name)."""

        return self._facet_or_default("policy").policy

    @property
    def predicate_matched(self) -> bool:
        """Mirror ``policy.predicate_matched`` through the facet default (legacy flat name)."""

        return self._facet_or_default("policy").predicate_matched

    @property
    def tracing_finished(self) -> bool:
        """Mirror ``policy.tracing_finished`` through the facet default (legacy flat name)."""

        return self._facet_or_default("policy").tracing_finished

    @property
    def construction_done(self) -> bool:
        """Mirror ``policy.construction_done`` through the facet default (legacy flat name)."""

        return self._facet_or_default("policy").construction_done

    @property
    def record_context(self) -> object | None:
        """Mirror ``recording.record_context`` through the facet default (legacy flat name)."""

        return self._facet_or_default("recording").record_context

    @property
    def capture_spec(self) -> object | None:
        """Mirror ``recording.capture_spec`` through the facet default (legacy flat name)."""

        return self._facet_or_default("recording").capture_spec

    @property
    def intervention_fired(self) -> bool:
        """Mirror ``intervention.intervention_fired`` through the facet default (legacy flat name)."""

        return self._facet_or_default("intervention").intervention_fired

    @property
    def intervention_replaced(self) -> bool:
        """Mirror ``intervention.intervention_replaced`` through the facet default (legacy flat name)."""

        return self._facet_or_default("intervention").intervention_replaced

    @property
    def fire_results(self) -> tuple[object, ...]:
        """Mirror ``intervention.fire_results`` through the facet default (legacy flat name)."""

        return self._facet_or_default("intervention").fire_results

    @property
    def intervention_template_ref(self) -> None:
        # compat-only field, verified dead in P2; always None on OpRecord
        return None

    # Trace-identity joins travel via IngestExtras, never the record. The
    # torch producer has stamped ``source_trace=None`` on every event since
    # the backref removal, so the compat read is a constant; live consumers
    # (`LiveOpView`) resolve ``event.source_trace or trace`` unchanged.
    @property
    def source_trace(self) -> None:
        return None

    @property
    def source_trace_id(self) -> None:
        return None


_FACET_ATTRIBUTES: dict[str, str] = {
    "function": "function",
    "templates": "templates",
    "graph": "graph",
    "modules": "modules_facet",
    "ancestry": "ancestry",
    "autograd": "autograd",
    "transform": "transform",
    "control": "control",
    "params": "params_facet",
    "annotations": "annotations_facet",
    "policy": "policy_facet",
    "recording": "recording",
    "intervention": "intervention",
}


# ---------------------------------------------------------------------------
# The typed amendment lane (nine exact-set families) — inert until P4
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpAmendment:
    """One typed post-commit knowledge record targeting a committed op."""

    seq: int  # writer-stamped at append into the journal's global domain
    run_nonce: int  # matches the owning stream's run nonce at append
    target_seq: int
    target_label_raw: str  # integrity anchor; must agree at fold (multi-pass
    # fastlog journals have no single seq domain, so label is the resolver
    # and target_seq is cross-checked only within single-domain journals)
    family: str
    patch: tuple[tuple[str, Any], ...]  # ordered (path, value) pairs


# Exact ORDERED patch-path sets per family; required == allowed. Value types
# are runtime-validated at append AND fold. ``object`` admits any non-None
# opaque value (record_context / capture_spec carriers).
AMENDMENT_FAMILIES: dict[str, tuple[tuple[str, tuple[type, ...]], ...]] = {
    "lookback_retention": (
        ("core.output", (OutputRef,)),
        ("policy.predicate_matched", (bool,)),
    ),
    "graph_edge_insertion": (
        ("core.parents", (tuple,)),
        ("graph.parent_arg_positions", (dict,)),
    ),
    "raw_hook_intervention": (("intervention.intervention_replaced", (bool,)),),
    "module_exit_intervention": (
        ("intervention.intervention_fired", (bool,)),
        ("intervention.intervention_replaced", (bool,)),
        ("intervention.fire_results", (tuple,)),
    ),
    "module_boundary_retention": (
        ("core.output", (OutputRef,)),
        ("policy.policy", (CapturePolicy,)),
        ("policy.predicate_matched", (bool,)),
        ("recording.capture_spec", (object,)),
        ("recording.record_context", (object,)),
    ),
    # documented legacy quirk preserved: record_context is computed by the
    # promotion helper but NOT forwarded by the legacy call — the committed
    # record never receives it, so the family schema excludes it.
    "output_parent_promotion": (
        ("graph.is_output_parent", (bool,)),
        ("core.output", (OutputRef,)),
        ("policy.policy", (CapturePolicy,)),
        ("policy.predicate_matched", (bool,)),
        ("recording.capture_spec", (object,)),
    ),
    "late_buffer_output_parent": (("graph.is_output_parent", (bool,)),),
    # the six preview promotion sites (review C6): two families because the
    # registry is exact-set — jax/tinygrad also rebind the output.
    "preview_output_parent_mark": (("graph.is_output_parent", (bool,)),),
    "preview_output_parent_rebind": (
        ("graph.is_output_parent", (bool,)),
        ("core.output", (OutputRef,)),
    ),
}

# 1:1 legacy fold table: registry path -> flat OpEvent field name. Covers the
# exact union of family paths; outlives P7 only as the preview-journal fold
# guard and dies in S15 with OpEvent.
PATH_TO_FLAT: dict[str, str] = {
    "core.output": "output",
    "core.parents": "parents",
    "graph.parent_arg_positions": "parent_arg_positions",
    "graph.is_output_parent": "is_output_parent",
    "policy.policy": "policy",
    "policy.predicate_matched": "predicate_matched",
    "recording.capture_spec": "capture_spec",
    "recording.record_context": "record_context",
    "intervention.intervention_fired": "intervention_fired",
    "intervention.intervention_replaced": "intervention_replaced",
    "intervention.fire_results": "fire_results",
}


def record_with_path_updates(record: OpRecord, items: Iterable[tuple[str, Any]]) -> OpRecord:
    """Return a new record with ordered ``(facet path, value)`` pairs applied.

    Each path names its owning core field or facet slot directly; an absent
    facet materializes from ``FACET_DEFAULTS`` before the patch. This is the
    ONE decomposed-leg fold primitive: the amendment reducer applies patches
    through it, and the legacy flat spelling below translates onto it.
    """

    from dataclasses import replace as dataclass_replace

    core_updates: dict[str, Any] = {}
    facet_updates: dict[str, dict[str, Any]] = {}
    for path, value in items:
        owner, _, field_name = path.partition(".")
        if owner == "core":
            core_updates[field_name] = value
        else:
            facet_updates.setdefault(owner, {})[field_name] = value
    record_changes: dict[str, Any] = {}
    if core_updates:
        record_changes["core"] = dataclass_replace(record.core, **core_updates)
    for facet_name, kwargs in facet_updates.items():
        current = record._facet_or_default(facet_name)
        record_changes[_FACET_ATTRIBUTES[facet_name]] = dataclass_replace(current, **kwargs)
    return dataclass_replace(record, **record_changes)


def apply_patch_items(event: Any, items: Iterable[tuple[str, Any]]) -> Any:
    """Apply ordered ``(facet path, value)`` pairs to either journal shape.

    The dual-leg amendment fold primitive (DoR 4.4): decomposed ``OpRecord``s
    fold via facet ``dataclasses.replace`` (:func:`record_with_path_updates`);
    compat ``OpEvent``s fold via the 1:1 ``PATH_TO_FLAT`` table. The table
    outlives P7 only as the preview-journal fold guard and dies in S15.
    """

    from dataclasses import replace as dataclass_replace

    if isinstance(event, OpRecord):
        return record_with_path_updates(event, items)
    return dataclass_replace(event, **{PATH_TO_FLAT[path]: value for path, value in items})


# Identity fields are structurally unpatchable: outside every schema AND
# refused by validation even on a forged raw OpAmendment.
_UNPATCHABLE_PREFIXES: tuple[str, ...] = (
    "core.seq",
    "core.kind",
    "core.label_raw",
    "core.layer_label_raw",
    "core.raw_index",
    "core.type_index",
    "core.func_call_id",
)


class AmendmentValidationError(ValueError):
    """A typed amendment violates its family's exact-set schema."""


def validate_amendment(amendment: OpAmendment) -> None:
    """Runtime-validate one amendment against the closed registry.

    Runs at append AND at fold, so a direct ``OpAmendment(...)`` construction
    that bypasses a typed constructor is caught at the journal boundary.
    """

    schema = AMENDMENT_FAMILIES.get(amendment.family)
    if schema is None:
        raise AmendmentValidationError(f"unregistered amendment family {amendment.family!r}")
    patch_paths = tuple(path for path, _ in amendment.patch)
    if len(set(patch_paths)) != len(patch_paths):
        raise AmendmentValidationError(
            f"duplicate patch paths in {amendment.family!r}: {patch_paths}"
        )
    schema_paths = tuple(path for path, _ in schema)
    if patch_paths != schema_paths:
        raise AmendmentValidationError(
            f"family {amendment.family!r} patch paths {patch_paths} != exact "
            f"registry set {schema_paths} (exact-set, ordered)"
        )
    for (path, value), (_, allowed_types) in zip(amendment.patch, schema, strict=True):
        if any(path.startswith(prefix) for prefix in _UNPATCHABLE_PREFIXES):
            raise AmendmentValidationError(f"identity field {path!r} is unpatchable")
        if allowed_types == (object,):
            continue
        if not isinstance(value, allowed_types):
            raise AmendmentValidationError(
                f"{amendment.family!r} path {path!r}: value type "
                f"{type(value).__qualname__} not in {allowed_types}"
            )
    if not amendment.target_label_raw:
        raise AmendmentValidationError("amendment requires a target_label_raw anchor")


def _amendment(family: str, target_seq: int, target_label_raw: str, *values: Any) -> OpAmendment:
    """Build one validated ``OpAmendment`` of ``family`` for the named target op."""

    schema = AMENDMENT_FAMILIES[family]
    amendment = OpAmendment(
        seq=0,
        run_nonce=0,
        target_seq=target_seq,
        target_label_raw=target_label_raw,
        family=family,
        patch=tuple((path, value) for (path, _), value in zip(schema, values, strict=True)),
    )
    validate_amendment(amendment)
    return amendment


# Typed keyword-only constructors: the exact path set is structural (no
# **kwargs, no optional patch paths, duplicates impossible).


def amend_lookback_retention(
    target_seq: int, target_label_raw: str, *, output: OutputRef, predicate_matched: bool
) -> OpAmendment:
    """Lookback retention replaces the committed output payload."""

    return _amendment("lookback_retention", target_seq, target_label_raw, output, predicate_matched)


def amend_graph_edge_insertion(
    target_seq: int,
    target_label_raw: str,
    *,
    parents: tuple[ParentEdge, ...],
    parent_arg_positions: dict[str, dict[Any, str]],
) -> OpAmendment:
    """Public ``register_tensor_connection`` manual edge insertion."""

    return _amendment(
        "graph_edge_insertion", target_seq, target_label_raw, parents, parent_arg_positions
    )


def amend_raw_hook_intervention(
    target_seq: int, target_label_raw: str, *, intervention_replaced: bool
) -> OpAmendment:
    """A raw forward hook replaced the module output."""

    return _amendment("raw_hook_intervention", target_seq, target_label_raw, intervention_replaced)


def amend_module_exit_intervention(
    target_seq: int,
    target_label_raw: str,
    *,
    intervention_fired: bool,
    intervention_replaced: bool,
    fire_results: tuple[object, ...],
) -> OpAmendment:
    """A live-fire intervention observed at module exit."""

    return _amendment(
        "module_exit_intervention",
        target_seq,
        target_label_raw,
        intervention_fired,
        intervention_replaced,
        fire_results,
    )


def amend_module_boundary_retention(
    target_seq: int,
    target_label_raw: str,
    *,
    output: OutputRef,
    policy: CapturePolicy,
    predicate_matched: bool,
    capture_spec: object,
    record_context: object,
) -> OpAmendment:
    """Module-boundary retention re-binds payload + policy on the boundary op."""

    return _amendment(
        "module_boundary_retention",
        target_seq,
        target_label_raw,
        output,
        policy,
        predicate_matched,
        capture_spec,
        record_context,
    )


def amend_output_parent_promotion(
    target_seq: int,
    target_label_raw: str,
    *,
    is_output_parent: bool,
    output: OutputRef,
    policy: CapturePolicy,
    predicate_matched: bool,
    capture_spec: object,
) -> OpAmendment:
    """Capture-end output-parent promotion (torch backend)."""

    return _amendment(
        "output_parent_promotion",
        target_seq,
        target_label_raw,
        is_output_parent,
        output,
        policy,
        predicate_matched,
        capture_spec,
    )


def amend_late_buffer_output_parent(
    target_seq: int, target_label_raw: str, *, is_output_parent: bool
) -> OpAmendment:
    """Pre-0 late-buffer output-parent marking."""

    return _amendment("late_buffer_output_parent", target_seq, target_label_raw, is_output_parent)


def amend_preview_output_parent_mark(
    target_seq: int, target_label_raw: str, *, is_output_parent: bool
) -> OpAmendment:
    """Preview-backend output-parent promotion (mark only)."""

    return _amendment("preview_output_parent_mark", target_seq, target_label_raw, is_output_parent)


def amend_preview_output_parent_rebind(
    target_seq: int, target_label_raw: str, *, is_output_parent: bool, output: OutputRef
) -> OpAmendment:
    """Preview-backend output-parent promotion that also rebinds the output."""

    return _amendment(
        "preview_output_parent_rebind", target_seq, target_label_raw, is_output_parent, output
    )


TYPED_CONSTRUCTORS: dict[str, Any] = {
    "lookback_retention": amend_lookback_retention,
    "graph_edge_insertion": amend_graph_edge_insertion,
    "raw_hook_intervention": amend_raw_hook_intervention,
    "module_exit_intervention": amend_module_exit_intervention,
    "module_boundary_retention": amend_module_boundary_retention,
    "output_parent_promotion": amend_output_parent_promotion,
    "late_buffer_output_parent": amend_late_buffer_output_parent,
    "preview_output_parent_mark": amend_preview_output_parent_mark,
    "preview_output_parent_rebind": amend_preview_output_parent_rebind,
}


# ---------------------------------------------------------------------------
# Ingest adaptation: compat OpEvent -> (OpRecord, IngestExtras)
# ---------------------------------------------------------------------------


def op_record_from_event(event: Any) -> tuple[OpRecord, IngestExtras]:
    """Adapt one compat ``OpEvent`` at the single ingest boundary.

    A facet is constructed only when its values differ from the checked-in
    defaults, so the adapter round-trips absent facets through the defaults
    table (the property the scatter parity gate proves cell-for-cell).
    """

    raw_config = dict(event.transform_config)
    annotations_payload = raw_config.pop("_tl_annotations", None)
    fn_code_location = raw_config.pop("fn_code_location", None)

    core = OpCore(
        seq=event.seq,
        kind=event.kind,
        label_raw=event.label_raw,
        layer_label_raw=event.layer_label_raw,
        layer_type=event.layer_type,
        raw_index=event.raw_index,
        type_index=event.type_index,
        step_index=event.step_index,
        pass_index=event.pass_index,
        parents=tuple(event.parents),
        output=event.output,
        is_bottom_level=event.is_bottom_level,
        func_call_id=event.func_call_id,
    )

    graph = GraphFacet(
        parent_arg_positions=event.parent_arg_positions,
        edge_uses=tuple(event._edge_uses),
        unattributed_tensor_args=tuple(event.unattributed_tensor_args),
        dropped_edge_tensor_args=tuple(event.dropped_edge_tensor_args),
        is_output_parent=event.is_output_parent,
        input_was_parameter=event.input_was_parameter,
        equivalence_class=event.equivalence_class,
    )
    modules_facet = (
        ModulesFacet(module_stack=tuple(event.module_stack), modules=tuple(event.modules))
        if event.module_stack or event.modules
        else None
    )
    ancestry = (
        AncestryFacet(
            input_ancestors=frozenset(event.input_ancestors),
            internal_source_ancestors=frozenset(event.internal_source_ancestors),
            root_ancestors=frozenset(event.root_ancestors),
            has_internal_source_ancestor=event.has_internal_source_ancestor,
        )
        if (
            event.input_ancestors
            or event.internal_source_ancestors
            or event.root_ancestors
            or event.has_internal_source_ancestor
        )
        else None
    )
    autograd = (
        AutogradFacet(grad_fn_class_qualname=event.grad_fn_class_qualname)
        if event.grad_fn_class_qualname is not None
        else None
    )
    transform = (
        TransformFacet(
            is_transform=event.is_transform,
            transform_kind=event.transform_kind,
            transform_chain=tuple(event.transform_chain),
            transform_config=raw_config,
            transform_fn_name=event.transform_fn_name,
            transform_fn_qualname=event.transform_fn_qualname,
            transform_fn_source=event.transform_fn_source,
            fn_code_location=fn_code_location,
        )
        if (
            event.is_transform
            or event.transform_kind is not None
            or event.transform_chain
            or raw_config
            or event.transform_fn_name is not None
            or event.transform_fn_qualname is not None
            or event.transform_fn_source is not None
            or fn_code_location is not None
        )
        else None
    )
    control = (
        ControlFacet(is_scalar_bool=event.is_scalar_bool, bool_value=event.bool_value)
        if event.is_scalar_bool is not None or event.bool_value is not None
        else None
    )
    params_facet = (
        ParamsFacet(params=tuple(event.params), parent_params=tuple(event.parent_params))
        if event.params or event.parent_params
        else None
    )
    annotations_facet = (
        AnnotationsFacet(annotations=dict(annotations_payload))
        if isinstance(annotations_payload, dict) and annotations_payload
        else None
    )
    policy_facet = PolicyFacet(
        backend_semantics=event.backend_semantics,
        policy=event.policy,
        predicate_matched=event.predicate_matched,
        tracing_finished=event.tracing_finished,
        construction_done=event.construction_done,
    )
    recording = (
        RecordingFacet(record_context=event.record_context, capture_spec=event.capture_spec)
        if event.record_context is not None or event.capture_spec is not None
        else None
    )
    intervention = (
        InterventionFacet(
            intervention_fired=event.intervention_fired,
            intervention_replaced=event.intervention_replaced,
            fire_results=tuple(event.fire_results),
        )
        if (event.intervention_fired or event.intervention_replaced or event.fire_results)
        else None
    )

    record = OpRecord(
        core=core,
        function=event.function,
        templates=event.templates,
        graph=graph,
        modules_facet=modules_facet,
        ancestry=ancestry,
        autograd=autograd,
        transform=transform,
        control=control,
        params_facet=params_facet,
        annotations_facet=annotations_facet,
        policy_facet=policy_facet,
        recording=recording,
        intervention=intervention,
    )
    extras = IngestExtras(
        source_trace=event.source_trace,
        source_trace_id=event.source_trace_id,
        grad_fn_handle=event.grad_fn_handle,
    )
    return record, extras


def record_facet_field_names() -> dict[str, tuple[str, ...]]:
    """Facet name -> field names (Tier walker + manifest generator input)."""

    return {
        name: tuple(f.name for f in dataclass_fields(cls)) for name, cls in FACET_CLASSES.items()
    }
