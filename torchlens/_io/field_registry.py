"""Persisted-field registry: the one queryable index of the tlspec envelope.

The v9 envelope (megaplan P05 / ecosystem MEMO 3.1, 3.3) needs one place that
can answer, per record contract: which field names are DECLARED, which of
them PERSIST (``FieldPolicy`` other than ``DROP``), what evidence TIER each
persisted field carries, and which legacy alias spellings a reader still
accepts. Everything here is DERIVED from the existing single sources of
truth -- the per-class ``FIELD_POLICY`` tables (built from ``*_FIELD_ORDER``
+ ``PORTABLE_STATE_SPEC``) and the raw ``PORTABLE_STATE_SPEC`` dicts on
nested value classes -- so the registry can never drift from the code that
actually serializes state. It declares NO new fields and moves NONE: field-set
changes land only at the coordinated schema write (C07), through the
``sprint/field_intent.tsv`` intake.

Tier partition (WALKTHROUGH section 3 field triage; DIGEST-AUDIT 4b):

- ``FACT``   -- captured truth. Cannot be recomputed from other persisted
  state; losing it loses evidence (the ``dropped_edge_tensor_args`` lesson:
  the only real unknown field in torchlens history was a fact whose loss
  also disabled a metadata invariant).
- ``DERIVED`` -- recomputable projections of facts (ancestry closures,
  shortened labels, lookup keys). Persisting them is a speed/ergonomics
  choice, not an evidence obligation.
- ``ENRICHMENT`` -- decoration whose absence degrades convenience, never
  analysis integrity; the lazification candidates for the tlspec 8->9 bump.

The partition below is the ENVELOPE default: every persisted field is FACT
unless the census evidence already classifies it, and C07 amends tiers
through the field-intent intake, never by editing consumers.
"""

from __future__ import annotations

from enum import Enum
from functools import cache
from importlib import import_module
from typing import Any

from . import FieldPolicy


class FieldTier(str, Enum):
    """Evidence tier of one persisted field (closed vocabulary)."""

    FACT = "fact"
    DERIVED = "derived"
    ENRICHMENT = "enrichment"


#: The record contracts: every class that owns a ``*_FIELD_ORDER`` catalog
#: (mirrors the lockstep Catalog registry in tests/test_schema_lockstep.py;
#: the lockstep test there guards the catalog list, the registry test here
#: guards this mapping against it).
RECORD_CONTRACT_CLASSES: dict[str, tuple[str, str]] = {
    "trace": ("torchlens.data_classes.trace", "Trace"),
    "op": ("torchlens.data_classes.op", "Op"),
    "layer": ("torchlens.data_classes.layer", "Layer"),
    "param": ("torchlens.data_classes.param", "Param"),
    "buffer": ("torchlens.data_classes.buffer", "Buffer"),
    "grad_fn": ("torchlens.data_classes.grad_fn", "GradFn"),
    "grad_fn_call": ("torchlens.data_classes.grad_fn_call", "GradFnCall"),
    "module_call": ("torchlens.data_classes.module", "ModuleCall"),
    "module": ("torchlens.data_classes.module", "Module"),
    "backward_pass": ("torchlens.data_classes.backward_pass", "BackwardPass"),
    "func_call_location": ("torchlens.data_classes.func_call_location", "FuncCallLocation"),
    "aten_op": ("torchlens.data_classes.aten_op", "AtenOp"),
}

#: Modules that declare nested value-object ``PORTABLE_STATE_SPEC`` tables
#: (component contracts riding inside record state). Guarded by the source
#: scan in tests/test_tlspec_envelope_registry.py: a new spec-declaring file
#: not listed here fails that test, so the writer contract can never silently
#: miss a component grammar.
COMPONENT_SPEC_MODULES: tuple[str, ...] = (
    "torchlens.data_classes.aten_op",
    "torchlens.data_classes.backward_pass",
    "torchlens.data_classes.buffer",
    "torchlens.data_classes.derived_grad",
    "torchlens.data_classes.func_call_location",
    "torchlens.data_classes.grad_fn",
    "torchlens.data_classes.grad_fn_call",
    "torchlens.data_classes.layer",
    "torchlens.data_classes.module",
    "torchlens.data_classes.op",
    "torchlens.data_classes.prehook",
    "torchlens.data_classes.trace",
    "torchlens.intervention.types",
    "torchlens.ir.container",
    "torchlens.ir.container_registry",
    "torchlens.ir.events",
    "torchlens.kernel_telemetry",
)

#: Tier overrides for persisted fields with census evidence (DIGEST-AUDIT 4b,
#: r2-corrected). Everything not listed is FACT by conservative default:
#: treating a derived field as fact never loses truth, while the reverse
#: launders evidence into "recomputable". PROVISIONAL pending the C07 census;
#: amendments route through sprint/field_intent.tsv, never ad-hoc edits.
_FIELD_TIER_OVERRIDES: dict[str, dict[str, FieldTier]] = {
    "op": {
        # P4a ancestry closures: pure folds over the recorded edge table.
        "root_ancestors": FieldTier.DERIVED,
        "input_ancestors": FieldTier.DERIVED,
        "internal_source_ancestors": FieldTier.DERIVED,
        "output_descendants": FieldTier.DERIVED,
        # P4c label projections (derivation caveats r2 C7 -- still derived).
        "label_short": FieldTier.DERIVED,
        "layer_label_short": FieldTier.DERIVED,
        "lookup_keys": FieldTier.DERIVED,
        # E2 P6: never a real value on any backend; P7 KEEP-OPT candidates.
        "bytes_delta_at_call": FieldTier.ENRICHMENT,
        "bytes_peak_at_call": FieldTier.ENRICHMENT,
        "func_autocast_state": FieldTier.ENRICHMENT,
        "func_config": FieldTier.ENRICHMENT,
        "code_context": FieldTier.ENRICHMENT,
    },
    "trace": {
        "code_context": FieldTier.ENRICHMENT,
    },
}


def _resolve_record_class(record_key: str) -> type:
    """Import and return the record class owning ``record_key``'s contract."""

    module_name, class_name = RECORD_CONTRACT_CLASSES[record_key]
    return getattr(import_module(module_name), class_name)


@cache
def record_field_policies(record_key: str) -> dict[str, FieldPolicy]:
    """Return the declared ``name -> FieldPolicy`` table for one record contract.

    Parameters
    ----------
    record_key:
        Key in :data:`RECORD_CONTRACT_CLASSES`.

    Returns
    -------
    dict[str, FieldPolicy]
        Every declared field (user-facing FIELD_ORDER rows and runtime-only
        policy rows) with its portable scrub policy.
    """

    cls = _resolve_record_class(record_key)
    field_policy = getattr(cls, "FIELD_POLICY")  # noqa: B009 -- typed as bare `type`
    return {name: row.portable_policy for name, row in field_policy.items()}


@cache
def persisted_field_names(record_key: str) -> frozenset[str]:
    """Names that PERSIST for one record contract (policy other than DROP)."""

    return frozenset(
        name
        for name, policy in record_field_policies(record_key).items()
        if policy is not FieldPolicy.DROP
    )


@cache
def field_tiers(record_key: str) -> dict[str, FieldTier]:
    """Tier per persisted field for one record contract (default FACT)."""

    overrides = _FIELD_TIER_OVERRIDES.get(record_key, {})
    return {
        name: overrides.get(name, FieldTier.FACT)
        for name in sorted(persisted_field_names(record_key))
    }


def component_spec_tables() -> dict[str, dict[str, str]]:
    """Collect every nested value-object ``PORTABLE_STATE_SPEC`` table.

    Returns
    -------
    dict[str, dict[str, str]]
        ``"module.Class" -> {field_name: policy_value}`` for every class in
        :data:`COMPONENT_SPEC_MODULES` that declares its own spec (inherited
        specs are not re-counted).
    """

    tables: dict[str, dict[str, str]] = {}
    for module_name in COMPONENT_SPEC_MODULES:
        module = import_module(module_name)
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not isinstance(obj, type):
                continue
            spec = obj.__dict__.get("PORTABLE_STATE_SPEC")
            if not isinstance(spec, dict) or not spec:
                continue
            qualname = f"{obj.__module__}.{obj.__qualname__}"
            tables[qualname] = {name: policy.value for name, policy in sorted(spec.items())}
    return tables


def known_state_keys(cls: type, *, extra: frozenset[str] = frozenset()) -> frozenset[str]:
    """The full set of state keys a governed reader KNOWS for ``cls``.

    Used by the load-time known/unknown partition (state contract, MEMO 3.3):
    declared policy rows, default-fill spellings, class-declared legacy
    aliases (``PORTABLE_STATE_ALIASES``), the version envelope, plus any
    call-site extras (keys the owning ``__setstate__`` consumes and pops
    before installing state).

    Parameters
    ----------
    cls:
        Record class being restored.
    extra:
        Additional caller-known keys.

    Returns
    -------
    frozenset[str]
        Every key that is NOT an unknown field for this contract.
    """

    known: set[str] = {"tlspec_version", "_tlspec_prerelease"}
    field_policy = getattr(cls, "FIELD_POLICY", None)
    if field_policy:
        known.update(field_policy.keys())
    spec = getattr(cls, "PORTABLE_STATE_SPEC", None)
    if isinstance(spec, dict):
        known.update(spec.keys())
    defaults = getattr(cls, "DEFAULT_FILL_STATE", None)
    if isinstance(defaults, dict):
        known.update(defaults.keys())
    aliases = getattr(cls, "PORTABLE_STATE_ALIASES", None)
    if aliases:
        known.update(aliases)
    mirror = getattr(cls, "PORTABLE_STATE_MIRROR_CONTRACT", None)
    if mirror is not None:
        # Aggregate facades (Layer over Op) pickle the mirrored record's
        # field surface; fold the mirrored contract into the known set.
        mirror_cls = getattr(import_module(mirror[0]), mirror[1])
        known.update(known_state_keys(mirror_cls))
    known.update(extra)
    return frozenset(known)


def registry_snapshot() -> dict[str, Any]:
    """One JSON-serializable snapshot of the whole registry.

    The canonical input to the writer contract digest and the committed
    contract-of-record golden (``torchlens/schemas/writer_contract_v8.json``).
    """

    records: dict[str, Any] = {}
    for record_key in sorted(RECORD_CONTRACT_CLASSES):
        policies = record_field_policies(record_key)
        tiers = field_tiers(record_key)
        cls = _resolve_record_class(record_key)
        aliases = sorted(getattr(cls, "PORTABLE_STATE_ALIASES", ()) or ())
        records[record_key] = {
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "field_policies": {name: policy.value for name, policy in sorted(policies.items())},
            "persisted_field_names": sorted(persisted_field_names(record_key)),
            "field_tiers": {name: tier.value for name, tier in tiers.items()},
            "aliases": aliases,
        }
    return {
        "record_contracts": records,
        "component_specs": component_spec_tables(),
    }
