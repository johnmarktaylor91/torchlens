"""Record columns and declarative family grammars for the 3.0 saved format.

Columns derive from storage bindings; FLAG_FIELDS gates optional payloads.
The manifest stamps the column-schema digest. FAMILIES binds nested grammar
nodes to columns for the shared decoder; cross-record checks stay procedural.
Design: ``docs/maintainers/saved_format_3_design.md``, sections 3, 4 and 9.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from importlib import import_module
from types import MappingProxyType
from typing import Any

from ..._data_classes._accessor_base import Accessor
from ..._data_classes._schema_bindings import STORAGE_BINDINGS
from ..._data_classes.field_policy import StorageKind
from ..field_registry import RECORD_CONTRACT_CLASSES, persisted_field_names
from .contract import JSON_TYPE_BY_STORAGE_KIND, RECORD_INDEX

#: Reusable grammar leaves. All nodes are data; decoding never calls a validator
#: named by an artifact. ``optional`` lists keys that may be absent, ``open``
#: admits user-owned keys, and ``nullable`` admits an explicit None.
_ANY: dict[str, Any] = {"kind": "scalar"}
_STR = {"kind": "scalar", "type": "str"}
_TEXT = {**_STR, "min_length": 1}
_INT = {"kind": "scalar", "type": "int"}
_COUNT = {**_INT, "minimum": 0}
_POSITIVE = {**_INT, "minimum": 1}
_BOOL = {"kind": "scalar", "type": "bool"}
_DIGEST = {**_STR, "pattern": r"[0-9a-f]{64}", "pattern_method": "fullmatch"}
_STRINGS = {"kind": "list", "items": _STR}
_RELATION = {**_STR, "vocab": ("exact", "upper_bound", "lower_bound", "unknown")}

#: Nested-family sub-schemas by family name. A node with ``record`` and ``column`` binds to
#: that cell; ``path`` selects an optional annotation key; nonbinding nodes describe nested rows.
_CORE_FAMILIES: dict[str, dict[str, Any]] = {
    "distributed_scope": {
        "record": "trace",
        "column": "_provenance_distributed_scope",
        "kind": "scalar",
        "vocab": (None, "rank_local_shard"),
        "code": "artifact_distributed_scope_invalid",
    },
    "timing_provenance": {
        "record": "trace",
        "column": "grad_fn_timing_provenance",
        "kind": "scalar",
        "vocab": (None, "unmeasured", "perf_counter"),
        "code": "artifact_grad_fn_timing_provenance_invalid",
    },
    "checkpoint_token": {
        "kind": "object",
        "keys": {
            "pack_count": _COUNT,
            "unpack_evidence_count": _COUNT,
            "site_key_candidates": {**_STRINGS, "sorted_unique": True},
            # Windows and retained-op site membership are relational checks.
            "window_evidence": _ANY,
        },
    },
    "checkpoint_witness": {
        "record": "trace",
        "column": "checkpoint_invocation_witness",
        "kind": "object",
        "nullable": True,
        "code": "artifact_checkpoint_witness_invalid",
        "keys": {
            "token_count": _COUNT,
            "tokens": {"kind": "object", "values": {"family": "checkpoint_token"}},
            "degrade_flags": {
                **_STRINGS,
                "sorted_unique": True,
                "items": {
                    **_STR,
                    "vocab": (
                        "classifier_unavailable",
                        "patch_unavailable",
                        "exotic_subclass",
                        "unmatched_backward_warn",
                        "reentrant_node_discovered",
                        "unwitnessed_checkpoint_enter",
                    ),
                },
            },
            "verdict": {
                **_STR,
                "vocab": (
                    "checkpoint_invocations_observed",
                    "evidence_incomplete",
                    "no_checkpoint_invocation_observed",
                ),
            },
        },
        "ordinal_keys": ("tokens", "token_count"),
        "rules": (
            {
                "when": {"degrade_flags": "truthy"},
                "keys": {"verdict": {"vocab": ("evidence_incomplete",)}},
            },
            {
                "when": {"degrade_flags": "falsy", "token_count": "truthy"},
                "keys": {"verdict": {"vocab": ("checkpoint_invocations_observed",)}},
            },
            {
                "when": {"degrade_flags": "falsy", "token_count": "falsy"},
                "keys": {"verdict": {"vocab": ("no_checkpoint_invocation_observed",)}},
            },
        ),
    },
    "capture_advisories": {
        "record": "trace",
        "column": "annotations",
        "path": ("capture_advisories",),
        "kind": "list",
        "code": "artifact_capture_advisories_invalid",
        "items": {
            "kind": "object",
            "keys": {
                "kind": _TEXT,
                "count": _POSITIVE,
                "first_location": {**_STR, "nullable": True},
                "message": _STR,
            },
        },
    },
    "kernel_launch": {
        "kind": "object",
        "attributes": True,
        "keys": {
            "attribution_status": {
                **_STR,
                "vocab": ("ambiguous", "attributed", "unattributed", "unavailable"),
            },
            "launch_name": {**_STR, "nullable": True},
            "device": {**_STR, "nullable": True},
            "stream": {"kind": "scalar", "type": ("str", "int"), "nullable": True},
            "runtime_correlation": {"kind": "scalar", "type": ("str", "int"), "nullable": True},
            "duration": {
                "kind": "scalar",
                "type": "number",
                "minimum": 0,
                "finite": True,
                "nullable": True,
            },
        },
    },
    "kernel_telemetry": {
        "record": "trace",
        "column": "annotations",
        "path": ("_kernel_telemetry",),
        "kind": "object",
        "code": "artifact_kernel_telemetry_invalid",
        "keys": {
            "_available": _BOOL,
            "_launches": {"kind": "list", "items": {"family": "kernel_launch"}},
            "_relations": {"kind": "list"},
        },
        "rules": (
            {
                "when": {"_available": "falsy"},
                "keys": {
                    "_launches": {
                        "kind": "list",
                        "length": 1,
                        "items": {
                            "kind": "object",
                            "attributes": True,
                            "keys": {
                                "attribution_status": {"vocab": ("unavailable",)},
                                **{
                                    key: {"vocab": (None,)}
                                    for key in (
                                        "device",
                                        "duration",
                                        "launch_name",
                                        "runtime_correlation",
                                        "stream",
                                    )
                                },
                            },
                        },
                    },
                },
            },
        ),
    },
    "sidecar_namespace": {
        "record": "trace",
        "column": "annotations",
        "path": ("sidecar",),
        "kind": "object",
        "code": "artifact_sidecar_invalid",
        "key": {**_STR, "pattern": r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"},
        "values": {
            "kind": "object",
            "keys": {
                "schema_id": _TEXT,
                "version": _POSITIVE,
                "owner": _TEXT,
                "payload": _ANY,
            },
        },
    },
    "source_snapshots": {
        "record": "trace",
        "column": "_source_snapshots",
        "kind": "list",
        "nullable": True,
        "code": "artifact_source_snapshots_invalid",
        "unique_by": ("path", "digest"),
        "items": {
            "kind": "object",
            "keys": {
                "path": _TEXT,
                "digest": _DIGEST,
                "text": {**_STR, "nullable": True},
            },
        },
    },
}


# Closed TorchLens annotation keys coexist with arbitrary user annotation keys.
_COLLECTIVE_KEYS = dict.fromkeys(
    ("schema", "kind", "func", "correlation", "group", "events", "roles", "witness"), _ANY
)
_LABEL = {**_STR, "max_length": 200}
_FLOPS_ROW: dict[str, Any] = {
    "kind": "tuple",
    "length": 5,
    "positions": {
        0: _LABEL,
        1: _POSITIVE,
        2: {**_LABEL, "nullable": True},
        3: {**_COUNT, "nullable": True},
        4: {**_COUNT, "nullable": True},
    },
}
_COLLECTIVE_FAMILIES: dict[str, dict[str, Any]] = {
    "collective_payload": {
        "kind": "object",
        "discriminator": "schema",
        "cases": {
            "collective_boundary_v1": {
                "kind": "object",
                "open": True,
                "keys": _COLLECTIVE_KEYS,
            },
            "functional_collective_boundary_v0": {
                "kind": "object",
                "open": True,
                "keys": {
                    **_COLLECTIVE_KEYS,
                    "events": {
                        "kind": "object",
                        "open": True,
                        "keys": {
                            "event_model": {"vocab": ("funcol_issue_is_launch",)},
                        },
                    },
                    "disclosures": {"kind": "list"},
                },
            },
        },
    },
    "annotation_families": {
        "record": "trace",
        "column": "annotations",
        "kind": "object",
        "nullable": True,
        "code": "artifact_annotations_invalid",
        "open": True,
        "optional": ("logged_values", "distributed"),
        "keys": {
            "logged_values": {"kind": "object", "nullable": True, "key": _STR},
            "distributed": {
                "kind": "object",
                "nullable": True,
                "open": True,
                "optional": ("boundaries",),
                "keys": {
                    "boundaries": {
                        "kind": "list",
                        "nullable": True,
                        "items": {"family": "collective_payload"},
                    },
                },
            },
        },
    },
    "op_annotations": {
        "record": "op",
        "column": "annotations",
        "kind": "object",
        "nullable": True,
        "code": "artifact_annotations_invalid",
        "open": True,
        "optional": (
            "collective",
            "save_mode",
            "saved_out_version",
            "varying_across_passes",
            "dedup_source_id",
            "dedup_source_version",
            "dedup_reference_label",
        ),
        "keys": {
            "collective": {"family": "collective_payload", "nullable": True},
            "save_mode": {"vocab": (None, "copy", "reference", "view", "cpu_async")},
            "saved_out_version": {**_INT, "nullable": True},
            "varying_across_passes": {"kind": "object", "nullable": True, "key": _STR},
            "dedup_source_id": {**_INT, "nullable": True},
            "dedup_source_version": {**_INT, "nullable": True},
            "dedup_reference_label": {**_STR, "nullable": True},
        },
    },
    "evidence_claims": {
        "record": "trace",
        "column": "_graph_structure_evidence",
        "kind": "object",
        "nullable": True,
        "code": "artifact_structure_evidence_invalid",
        "open": True,
        "keys": {
            "claims": {
                "kind": "object",
                "keys": {
                    "graph_structure": {"vocab": ("observed_canonical", "refuted")},
                    "shapes_dtypes": {"vocab": ("hypothesis", "corroborated", "refuted")},
                    "flops_geometry_bytes": {
                        "vocab": ("hypothesis_estimate", "corroborated", "refuted")
                    },
                    "declared_buffer_mutations": {
                        "vocab": ("hypothesis", "corroborated", "refuted")
                    },
                    "measured_values": {"vocab": ("unavailable",)},
                    "timing_allocator_memory": {"vocab": ("unavailable",)},
                    "declared_branch_assumptions": {"vocab": ("none", "hypothesis", "refuted")},
                },
            },
            "discharge": {"any_of": ({"vocab": ("absent",)}, {"kind": "object"})},
        },
    },
    "episode_step_stamps": {
        "record": "op",
        "column": "episode_step",
        **_COUNT,
        "nullable": True,
        "code": "artifact_episode_step_invalid",
    },
    "forward_peak_memory_reason": {
        "record": "trace",
        "column": "_cost_forward_peak_memory_reason",
        **_TEXT,
        "nullable": True,
        "code": "artifact_forward_peak_memory_invalid",
    },
    "backward_peak_memory_reason": {
        "record": "trace",
        "column": "backward_peak_memory_reason",
        **_TEXT,
        "nullable": True,
        "code": "artifact_backward_peak_memory_invalid",
    },
    "executed_flops_records": {
        "record": "backward_pass",
        "column": "_executed_flops_record",
        "kind": "object",
        "nullable": True,
        "code": "artifact_executed_flops_record_invalid",
        "keys": {
            "convention": _LABEL,
            "executed_flops": {**_COUNT, "nullable": True},
            "rows": {"kind": "tuple", "items": _FLOPS_ROW},
            "uncounted_ops": {
                "kind": "tuple",
                "items": {
                    "kind": "tuple",
                    "length": 2,
                    "positions": {0: _LABEL, 1: _POSITIVE},
                },
            },
            "unattributed": {
                "kind": "tuple",
                "items": {
                    "kind": "tuple",
                    "length": 2,
                    "positions": {0: _LABEL, 1: {**_COUNT, "nullable": True}},
                },
            },
        },
        "rules": (
            {
                "when": {"executed_flops": "present"},
                "keys": {
                    "rows": {
                        "kind": "tuple",
                        "items": {
                            **_FLOPS_ROW,
                            "positions": {**_FLOPS_ROW["positions"], 3: _COUNT},
                        },
                    },
                },
            },
        ),
        "sum": {"total": "executed_flops", "terms": (("rows", 3), ("unattributed", 1))},
    },
}


# Audit target rows retain their independent cross-record anchor checks.
_AUDIT_COMMON = {"edit": _STR, "kind": _STR, "resolve_digest": _DIGEST, "selection_repr": _STR}
_REPR_TUPLE = {**_STR, "max_length": 4096, "literal_tuple": True}
_OCCURRENCE = {
    "kind": "object",
    "keys": {
        "consumer": _STR,
        "edge_address": _REPR_TUPLE,
        "value_digest": _DIGEST,
    },
}
_EVENT_KEYS = {
    "kind": {"vocab": ("EVENT",)},
    "schema": {"vocab": ("intervention_event_v2",)},
    **dict.fromkeys(("event_id", "transaction_id", "door", "selection_repr"), _TEXT),
    "parent_event_id": {**_STR, "nullable": True},
    "lane": {"vocab": ("replay", "rerun", "capture", "live_hook", "set_only", "bind")},
    "status": {"vocab": ("fired", "no_fire", "error", "armed")},
    "fire_count": _COUNT,
    **dict.fromkeys(("edit_names", "site_keys", "zero_fire_rule_ids"), _STRINGS),
    "rules": {"kind": "list", "items": {"kind": "object"}},
    "error": {**_STR, "nullable": True},
    "event_digest": _DIGEST,
    "seq": _ANY,
    "prev_event_digest": _ANY,
}
_AUDIT_FAMILIES: dict[str, dict[str, Any]] = {
    "intervention_audit": {
        "record": "trace",
        "column": "intervention_audit",
        "kind": "list",
        "nullable": True,
        "code": "artifact_intervention_audit_invalid",
        "items": {
            "kind": "object",
            "discriminator": "kind",
            "cases": {
                "ACT": {
                    "kind": "object",
                    "optional": ("patch_source",),
                    "keys": {
                        **_AUDIT_COMMON,
                        "patch_source": {
                            "kind": "object",
                            "keys": dict.fromkeys(
                                (
                                    "source_model_class",
                                    "source_object_id",
                                    "source_trace_label",
                                ),
                                _STR,
                            ),
                        },
                        "sites": _ANY,
                    },
                },
                "EDGE": {
                    "kind": "object",
                    "keys": {
                        **_AUDIT_COMMON,
                        "edges": _ANY,
                    },
                },
                "PARAM": {
                    "kind": "object",
                    "keys": {
                        **_AUDIT_COMMON,
                        "disclosure": _STR,
                        "params": _ANY,
                    },
                },
                "EVENT": {
                    "kind": "object",
                    "keys": _EVENT_KEYS,
                    "optional": ("seq", "prev_event_digest"),
                },
            },
        },
    },
    "param_occurrence": {
        "record": "trace",
        "column": "intervention_audit",
        "path": ("*", "params", "*", "occurrences", "*"),
        "code": "artifact_intervention_audit_invalid",
        **_OCCURRENCE,
    },
    "selection_recipe": {
        "any_of": (
            {
                "kind": "object",
                "keys": {
                    "edge_address": _REPR_TUPLE,
                    "resolve_digest": _DIGEST,
                },
            },
            {
                "kind": "object",
                "keys": {
                    "edge_address": _REPR_TUPLE,
                    "resolve_digest": _DIGEST,
                    "note": _STR,
                    "param_address": _TEXT,
                },
            },
            {
                "kind": "object",
                "keys": {
                    "relation": _RELATION,
                    "resolve_digest": _DIGEST,
                    "selected": _COUNT,
                    "site_key": _REPR_TUPLE,
                    "source": _STR,
                },
            },
        ),
    },
    "selection_recipes": {
        "requires": ("trace", "intervention_audit", "present"),
        "record": "op",
        "column": "edits",
        "path": ("*", "@edit"),
        "kind": "object",
        "attributes": True,
        "open": True,
        "when_attribute_present": "selection_recipe",
        "code": "artifact_intervention_audit_invalid",
        "keys": {
            "name": _STR,
            "selection_recipe": {"family": "selection_recipe"},
        },
    },
}

#: The closed family table: the three declaration blocks above, read-only after import.
FAMILIES: Mapping[str, dict[str, Any]] = MappingProxyType(
    {**_CORE_FAMILIES, **_COLLECTIVE_FAMILIES, **_AUDIT_FAMILIES}
)

#: Payload include flags and the stored keys each one gates (the scrub's policy
#: resolver; mirrored by ``tools/docs/persisted_fields.py`` for the page).
FLAG_FIELDS: dict[str, tuple[str, ...]] = {
    "include_outs": ("out", "transformed_out"),
    "include_grads": ("grad", "transformed_grad"),
    "include_saved_args": (
        "forward_args",
        "forward_kwargs",
        "out_versions_by_child",
        "saved_args",
        "saved_kwargs",
    ),
    "include_rng_states": ("func_rng_states",),
    "include_buffer_values": ("_buffer_initial_values",),
    "include_custom_attributes": ("custom_attributes",),
}
#: The bindings key where it differs from the record contract key.
_BINDING_KEY_ALIASES = {"aten_op": "primitive_op"}
_FLAG_BY_FIELD = {field: flag for flag, fields in FLAG_FIELDS.items() for field in fields}


@dataclass(frozen=True)
class Column:
    """Schema of one record column.

    Attributes
    ----------
    name, storage, json_type:
        Stored key, binding StorageKind and contract JSON cell type.
    flag, annotation:
        Optional payload gate (None means required) and declared annotation text.
    """

    name: str
    storage: str
    json_type: str
    flag: str | None
    annotation: str | None


@cache
def record_schema(record: str) -> dict[str, Column]:
    """Return the typed schema of one record kind (``op``, ``trace``, ...).

    Returns
    -------
    dict[str, Column]
        Persisted columns in bindings order; unbound keys follow alphabetically
        with the ``value`` type. Record kinds come from ``contract.RECORD_INDEX``.
    """

    bindings = STORAGE_BINDINGS.get(_BINDING_KEY_ALIASES.get(record, record), {})
    persisted = persisted_field_names(record)
    ordered = [name for name in bindings if name in persisted]
    ordered.extend(sorted(persisted.difference(bindings)))
    columns: dict[str, Column] = {}
    for name in ordered:
        binding = bindings.get(name)
        kind = binding.kind if binding is not None else StorageKind.SCALAR
        if kind is StorageKind.RUNTIME:
            continue
        columns[name] = Column(
            name=name,
            storage=kind.value,
            json_type=JSON_TYPE_BY_STORAGE_KIND[kind],
            flag=_FLAG_BY_FIELD.get(name),
            annotation=None if binding is None else binding.annotation,
        )
    return columns


def level_columns(record: str, payload_policy: Mapping[str, bool]) -> list[str]:
    """Return the columns one record file carries under a payload policy.

    Returns
    -------
    list[str]
        Column names in schema order. Missing payload flags count as off.
    """

    return [
        name
        for name, column in record_schema(record).items()
        if column.flag is None or bool(payload_policy.get(column.flag, False))
    ]


@cache
def schema_digest() -> str:
    """Return the manifest's SHA-256 of the canonical JSON of all record schemas."""

    payload = {
        record: [asdict(column) for column in record_schema(record).values()]
        for record, _, _ in RECORD_INDEX
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: Aggregates require the live trace link; scrubbed copies have no link.
_LINK_AGGREGATES = (
    "forward_args_template",
    "forward_kwargs_template",
    "param_memory",
    "has_trainable_params",
    "has_frozen_params",
    "backward_duration",
    "func_calls_duration",
    "num_descendant_calls",
    "max_descendant_depth",
)
LIVE_COMPUTED_COLUMNS: dict[str, frozenset[str]] = {
    "module": frozenset(
        {
            *_LINK_AGGREGATES,
            "num_param_tensors_trainable",
            "num_param_tensors_frozen",
            "recursive_params",
            "num_recursive_params",
            "num_recursive_param_tensors",
            "num_recursive_param_tensors_trainable",
            "num_recursive_param_tensors_frozen",
            "forward_duration",
            "total_forward_duration",
            "total_backward_duration",
            "total_func_calls_duration",
            "total_output_activation_memory",
            "total_internal_activation_memory",
            "total_output_gradient_memory",
            "total_internal_gradient_memory",
            "total_autograd_memory",
        }
    ),
    "module_call": frozenset(
        {
            *_LINK_AGGREGATES,
            "module",
            "num_param_tensors",
            "num_param_tensors_trainable",
            "num_param_tensors_frozen",
            "output_activation_memory",
            "internal_activation_memory",
            "output_gradient_memory",
            "internal_gradient_memory",
            "autograd_memory",
        }
    ),
}
#: Computed columns resolved from source text: read live only when the save
#: keeps source (``include_source``); otherwise the scrubbed copy's value stands.
SOURCE_COMPUTED_COLUMNS: dict[str, frozenset[str]] = {"op": frozenset({"arg_expressions"})}
#: Capture outcomes retain the live value object instead of the scrub's string.
LIVE_TRACE_FIELDS = frozenset({"_capture_outcome"})
_COMPUTED_ERRORS = (AttributeError, LookupError, RuntimeError, TypeError, ValueError)


@cache
def _record_kinds_by_class() -> dict[type, str]:
    """Return record class to record kind for the twelve record contracts."""

    return {
        getattr(import_module(module), name): kind
        for kind, (module, name) in RECORD_CONTRACT_CLASSES.items()
    }


def record_kind_of(value: Any) -> str | None:
    """Return the record kind of a record object, or ``None`` for any other value."""

    kinds = _record_kinds_by_class()
    kind = kinds.get(type(value))
    if kind is not None:
        return kind
    for cls in type(value).__mro__[1:]:
        if cls in kinds:
            return kinds[cls]
    return None


class LiveCells:
    """Resolve the cell values the writer encodes, reading live where the copy cannot.

    Parameters
    ----------
    pairs:
        Scrubbed-copy/live-original pairs, including the trace state mapping.
        Nested live records map to copies; accessor members become row edges.
    include_source:
        Whether source-derived computed columns read from the live record.
    """

    def __init__(
        self, pairs: Sequence[tuple[Any, Any]] = (), *, include_source: bool = True
    ) -> None:
        # Objects the walk kept as they were pair with themselves; drop them.
        self._pairs = [
            (copy, live) for copy, live in pairs if copy is not None and copy is not live
        ]
        self._live = {id(copy): live for copy, live in self._pairs}
        self._copy = {id(live): copy for copy, live in self._pairs}
        self._include_source = include_source

    def cell(self, kind: str, record: Any, state: Mapping[str, Any], column: str) -> Any:
        """Return the value of one cell of ``record`` (``state`` is its stored state)."""

        live = self._live.get(id(record))
        if kind == "trace":
            from_live = column in LIVE_TRACE_FIELDS and live is not None
            value = getattr(live, column, None) if from_live else state.get(column)
        elif column in state:
            value = state[column]
        elif record_schema(kind)[column].storage != StorageKind.COMPUTED.value:
            value = None
        else:
            reads_live = column in LIVE_COMPUTED_COLUMNS.get(kind, ()) or (
                self._include_source and column in SOURCE_COMPUTED_COLUMNS.get(kind, ())
            )
            source = live if reads_live and live is not None else record
            try:
                value = getattr(source, column)
            except _COMPUTED_ERRORS:
                value = None
        return self._to_copies(_edge_members(value))

    def _to_copies(self, value: Any) -> Any:
        """Replace live records inside ``value`` by their scrubbed copies."""

        if id(value) in self._copy:
            return self._copy[id(value)]
        if isinstance(value, list):
            return [self._to_copies(item) for item in value]
        if isinstance(value, tuple) and not hasattr(value, "_fields"):
            return tuple(self._to_copies(item) for item in value)
        if isinstance(value, dict):
            return {key: self._to_copies(item) for key, item in value.items()}
        return value


def _edge_members(value: Any) -> Any:
    """Return accessor members, whose own keys preserve their lookup identities."""

    if isinstance(value, Accessor):
        members = list(value._list)
        if all(record_kind_of(member) is not None for member in members):
            return members
    return value
