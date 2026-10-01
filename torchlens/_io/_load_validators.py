"""Load-boundary structural validators for the op graph, edge carriers, and annotations.

AUD-CODE 3.0 (a)/(b)/(d) (W051-IO): three claim families the portable loader
never checked before -- the retained op family's label/pass/relation
coherence (``artifact_graph_structure_invalid``), the L6 tier-(ii) edge/param
substitution carriers corroborated against the intervention audit and the
``region_do`` operation stream (``artifact_edge_substitutions_invalid``), and
the TorchLens-owned annotation families (``artifact_annotations_invalid``).
Split from ``_io/forgery_validation.py`` (R43 size discipline, T102); the one
consumer is ``validate_persisted_forgery_surfaces``, and every refusal routes
through that module's shared teaching ``_refuse``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

from .forgery_validation import _is_digest, _is_int, _is_sequence, _refuse, _trace_ops

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

_GRAPH_REMEDY = (
    "re-save the artifact from its source capture with one current TorchLens "
    "version; do not hand-edit op rows, labels, pass stamps, or relation lists"
)


def _graph_invalid(message: str, field: str, reason: str) -> NoReturn:
    """Raise the graph-structure refusal (AUD-CODE 3.0b)."""

    _refuse(
        message,
        code="artifact_graph_structure_invalid",
        field=field,
        reason=reason,
        remedy=_GRAPH_REMEDY,
    )


def _validate_graph_structure(trace: Trace) -> None:
    """Validate the retained op family's label/pass/relation coherence.

    Every retained op carries a unique string ``label``; ``pass_index`` is a
    1-based int within ``num_passes >= 1``; every ``parents`` / ``children``
    entry resolves to a retained op (by pass-qualified label or bare layer
    label). A dropped row, a grafted duplicate, a dangling relation, or an
    out-of-range pass stamp all used to load silently.
    """

    ops = _trace_ops(trace)
    if not ops:
        return
    known = _known_relation_targets(ops)
    for op in ops:
        label = str(getattr(op, "label", "<unknown>"))
        pass_index = getattr(op, "pass_index", None)
        num_passes = getattr(op, "num_passes", None)
        if not (
            _is_int(pass_index)
            and _is_int(num_passes)
            and num_passes >= 1
            and 1 <= pass_index <= num_passes
        ):
            _graph_invalid(
                f"op {label!r} carries pass_index={pass_index!r} / num_passes={num_passes!r}; "
                "pass stamps are 1-based ints with 1 <= pass_index <= num_passes",
                "Op.pass_index",
                "pass_stamp_range",
            )
        for relation in ("parents", "children"):
            for other in getattr(op, relation, None) or ():
                if other not in known:
                    _graph_invalid(
                        f"op {label!r} lists {relation} entry {other!r}, which names no "
                        "retained op",
                        f"Op.{relation}",
                        "dangling_relation",
                    )


def _known_relation_targets(ops: Sequence[Any]) -> set[str]:
    """Census the unique pass-qualified labels (plus bare layer labels) relations may name.

    Refuses a non-string / empty label or a duplicate label (a grafted row).
    """

    seen: set[str] = set()
    for op in ops:
        label = getattr(op, "label", None)
        if not isinstance(label, str) or not label:
            _graph_invalid(f"op row carries a non-string label {label!r}", "Op.label", "label_type")
        if label in seen:
            _graph_invalid(
                f"op label {label!r} occurs more than once", "Op.label", "duplicate_label"
            )
        seen.add(label)
    known = set(seen)
    for op in ops:
        layer_label = getattr(op, "layer_label", None)
        if isinstance(layer_label, str):
            known.add(layer_label)
    return known


_EDGE_REMEDY = (
    "re-apply the intervention on a fresh capture and re-save; do not hand-edit "
    "Op.edge_substitutions / Op.edge_replacement_stamps or the intervention audit"
)
#: The plain edge writer's entry shape; ``substitution_kind`` ("region" /
#: "param") is stamped only by the region and param-substitution writers.
_EDGE_ENTRY_REQUIRED = frozenset({"value", "parent_label", "resolve_digest", "helper_name"})
_EDGE_STAMP_REQUIRED = frozenset({"verdict", "value_digest", "resolve_digest"})


def _edge_invalid(message: str, reason: str) -> NoReturn:
    """Raise the tier-(ii) edge-carrier refusal (AUD-CODE 3.0a)."""

    _refuse(
        message,
        code="artifact_edge_substitutions_invalid",
        field="Op.edge_substitutions",
        reason=reason,
        remedy=_EDGE_REMEDY,
    )


def _region_anchor_index(trace: Trace) -> dict[str, set[str]]:
    """Map each persisted ``region_do`` digest to the exit addresses it applied.

    Region edits (F43) ride the free-form ``state_history`` operation stream,
    not the closed ``intervention_audit`` grammar (a new audit-row KIND is a
    C07-owned amendment), so region-kind carriers corroborate against that
    stream: the entry's ``resolve_digest`` is the region digest and its
    occurrence address must be among the recorded exits.
    """

    anchors: dict[str, set[str]] = {}
    for record in getattr(trace, "state_history", None) or ():
        if not isinstance(record, Mapping) or record.get("op") != "region_do":
            continue
        digest = record.get("region_digest")
        if not isinstance(digest, str):
            continue
        addresses = anchors.setdefault(digest, set())
        for exit_row in record.get("exits") or ():
            if isinstance(exit_row, Mapping) and isinstance(exit_row.get("edge_address"), str):
                addresses.add(exit_row["edge_address"])
    return anchors


def _audit_anchor_index(trace: Trace) -> tuple[set[str], set[str], set[str]]:
    """Collect EDGE addresses, PARAM occurrence addresses, and resolve digests."""

    edge_addresses: set[str] = set()
    param_addresses: set[str] = set()
    digests: set[str] = set()
    for row in getattr(trace, "intervention_audit", None) or ():
        if not isinstance(row, Mapping):
            continue
        digest = row.get("resolve_digest")
        if isinstance(digest, str):
            digests.add(digest)
        for edge in row.get("edges") or ():
            if isinstance(edge, Mapping) and isinstance(edge.get("edge_address"), str):
                edge_addresses.add(edge["edge_address"])
        for param in row.get("params") or ():
            if not isinstance(param, Mapping):
                continue
            for occurrence in param.get("occurrences") or ():
                if isinstance(occurrence, Mapping) and isinstance(
                    occurrence.get("edge_address"), str
                ):
                    param_addresses.add(occurrence["edge_address"])
    return edge_addresses, param_addresses, digests


class _EdgeAnchors(NamedTuple):
    """The independently persisted anchors a tier-(ii) carrier entry corroborates against."""

    edge_addresses: set[str]
    param_addresses: set[str]
    digests: set[str]
    region_exits: dict[str, set[str]]


def _validate_edge_carriers(trace: Trace) -> None:
    """Corroborate every persisted tier-(ii) edge/param substitution carrier.

    ``Op.edge_substitutions`` / ``Op.edge_replacement_stamps`` are the L6
    stage-3 replacement stores. Post-load, forward validation cannot re-run
    them, so a forged carrier used to load with zero corroboration. Each
    entry must ride a canonical ``(arg_kind, arg_path)`` key mirrored by a
    stamp, name a resolve digest present in the intervention audit, and land
    on an occurrence address the audit's EDGE (or PARAM occurrence) rows
    record for THIS op.
    """

    carriers = [
        (
            op,
            getattr(op, "edge_substitutions", None),
            getattr(op, "edge_replacement_stamps", None),
        )
        for op in _trace_ops(trace)
    ]
    carriers = [entry for entry in carriers if entry[1] is not None or entry[2] is not None]
    if not carriers:
        return
    anchors = _EdgeAnchors(*_audit_anchor_index(trace), _region_anchor_index(trace))
    for op, store, stamps in carriers:
        label = str(getattr(op, "label", "<unknown>"))
        if not isinstance(store, Mapping) or not isinstance(stamps, Mapping):
            _edge_invalid(
                f"op {label!r} carries edge_substitutions/edge_replacement_stamps that are "
                "not both mappings",
                "carrier_type",
            )
        if set(store) != set(stamps):
            _edge_invalid(
                f"op {label!r} edge_substitutions keys {sorted(map(repr, store))!r} do not "
                f"match its edge_replacement_stamps keys {sorted(map(repr, stamps))!r}",
                "stamp_key_mismatch",
            )
        for key, entry in store.items():
            _validate_edge_entry(op, key, entry, stamps[key], anchors)


def _validate_edge_entry(op: Any, key: Any, entry: Any, stamp: Any, anchors: _EdgeAnchors) -> None:
    """Corroborate ONE carrier entry of ``op`` (address shape, digest, occurrence) and its stamp."""

    label = str(getattr(op, "label", "<unknown>"))
    func_call_id = getattr(op, "func_call_id", None)
    if not (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], str)
        and isinstance(key[1], tuple)
    ):
        _edge_invalid(
            f"op {label!r} edge_substitutions key {key!r} is not a canonical "
            "(arg_kind, arg_path) occurrence address",
            "address_shape",
        )
    if not isinstance(entry, Mapping) or not set(entry) >= _EDGE_ENTRY_REQUIRED:
        _edge_invalid(
            f"op {label!r} edge_substitutions entry at {key!r} lacks the required "
            f"fields {sorted(_EDGE_ENTRY_REQUIRED)!r}",
            "entry_schema",
        )
    digest = entry["resolve_digest"]
    kind = entry.get("substitution_kind")
    address = repr((func_call_id, *key))
    if kind == "region":
        # Region digests are the F43 ``region-<hex>`` identity, anchored on
        # the persisted ``region_do`` operation rows, never on the audit.
        addresses = anchors.region_exits.get(digest) if isinstance(digest, str) else None
        if addresses is None:
            _edge_invalid(
                f"op {label!r} region edge_substitutions entry at {key!r} names region "
                f"digest {digest!r}, which no persisted region_do operation records",
                "uncorroborated_digest",
            )
    else:
        if not _is_digest(digest) or digest not in anchors.digests:
            _edge_invalid(
                f"op {label!r} edge_substitutions entry at {key!r} names resolve_digest "
                f"{digest!r}, which no intervention audit row records",
                "uncorroborated_digest",
            )
        addresses = anchors.param_addresses if kind == "param" else anchors.edge_addresses
    if address not in addresses:
        _edge_invalid(
            f"op {label!r} edge_substitutions entry at {key!r} (occurrence {address}) "
            "is recorded by no intervention audit EDGE/PARAM occurrence row or "
            "region_do operation row",
            "uncorroborated_address",
        )
    if not isinstance(stamp, Mapping) or not set(stamp) >= _EDGE_STAMP_REQUIRED:
        _edge_invalid(
            f"op {label!r} edge_replacement_stamps entry at {key!r} lacks the required "
            f"fields {sorted(_EDGE_STAMP_REQUIRED)!r}",
            "stamp_schema",
        )
    if (
        not isinstance(stamp["verdict"], bool)
        or not isinstance(stamp["value_digest"], str)
        or stamp["resolve_digest"] != digest
    ):
        _edge_invalid(
            f"op {label!r} edge_replacement_stamps entry at {key!r} is off-shape or "
            "names a different resolve digest than its substitution entry",
            "stamp_relation",
        )


_ANNOTATIONS_REMEDY = (
    "re-save the artifact from its source capture with one current TorchLens "
    "version; TorchLens-owned annotation families are written by capture, never "
    "hand-edited"
)
_SAVE_MODES = frozenset({"copy", "reference", "view", "cpu_async"})
_COLLECTIVE_SCHEMA = "collective_boundary_v1"
_COLLECTIVE_REQUIRED = frozenset(
    {"schema", "kind", "func", "correlation", "group", "events", "roles", "witness"}
)


def _annotations_invalid(message: str, field: str, reason: str) -> NoReturn:
    """Raise the annotation-family refusal (AUD-CODE 3.0d)."""

    _refuse(
        message,
        code="artifact_annotations_invalid",
        field=field,
        reason=reason,
        remedy=_ANNOTATIONS_REMEDY,
    )


def _validate_annotation_families(trace: Trace) -> None:
    """Validate the TorchLens-owned annotation families' shapes at load.

    User annotation keys stay OPEN (any key, plain data); the families
    TorchLens itself writes are closed: ``logged_values`` (str-keyed mapping),
    ``distributed`` (mapping; ``boundaries`` rows are ``collective_boundary_v1``
    payloads), per-op ``collective`` (the same payload), ``save_mode`` (the
    closed ``SaveMode`` vocabulary), ``saved_out_version`` (int/None),
    ``varying_across_passes`` (str-keyed mapping), and the dedup trio.
    """

    annotations = getattr(trace, "annotations", None)
    if annotations is not None and not isinstance(annotations, Mapping):
        _annotations_invalid(
            f"Trace.annotations must be a mapping, got {type(annotations).__name__}",
            "Trace.annotations",
            "type",
        )
    if isinstance(annotations, Mapping):
        logged = annotations.get("logged_values")
        if logged is not None and (
            not isinstance(logged, Mapping) or any(not isinstance(k, str) for k in logged)
        ):
            _annotations_invalid(
                "logged_values must be a str-keyed mapping of observer values",
                'Trace.annotations["logged_values"]',
                "logged_values_shape",
            )
        distributed = annotations.get("distributed")
        if distributed is not None:
            if not isinstance(distributed, Mapping):
                _annotations_invalid(
                    f"distributed journal must be a mapping, got {type(distributed).__name__}",
                    'Trace.annotations["distributed"]',
                    "distributed_type",
                )
            boundaries = distributed.get("boundaries")
            if boundaries is not None and (
                not _is_sequence(boundaries)
                or any(not _is_collective_payload(row) for row in boundaries)
            ):
                _annotations_invalid(
                    "distributed.boundaries rows must be collective_boundary_v1 payloads",
                    'Trace.annotations["distributed"]',
                    "boundaries_shape",
                )
    for op in _trace_ops(trace):
        _validate_op_annotations(op)


def _is_collective_payload(row: Any) -> bool:
    """True iff ``row`` is a ``collective_boundary_v1`` payload mapping."""

    return (
        isinstance(row, Mapping)
        and row.get("schema") == _COLLECTIVE_SCHEMA
        and set(row) >= _COLLECTIVE_REQUIRED
    )


def _validate_op_annotations(op: Any) -> None:
    """Validate one op's TorchLens-owned annotation keys (AUD-CODE 3.0d)."""

    label = str(getattr(op, "label", "<unknown>"))
    annotations = getattr(op, "annotations", None)
    if annotations is None:
        return
    if not isinstance(annotations, Mapping):
        _annotations_invalid(
            f"op {label!r} annotations must be a mapping, got {type(annotations).__name__}",
            "Op.annotations",
            "type",
        )
    collective = annotations.get("collective")
    if collective is not None and not _is_collective_payload(collective):
        _annotations_invalid(
            f"op {label!r} collective annotation is not a collective_boundary_v1 payload",
            'Op.annotations["collective"]',
            "collective_shape",
        )
    save_mode = annotations.get("save_mode")
    if save_mode is not None and save_mode not in _SAVE_MODES:
        _annotations_invalid(
            f"op {label!r} save_mode {save_mode!r} is outside {sorted(_SAVE_MODES)!r}",
            'Op.annotations["save_mode"]',
            "save_mode_vocabulary",
        )
    saved_version = annotations.get("saved_out_version")
    if saved_version is not None and not _is_int(saved_version):
        _annotations_invalid(
            f"op {label!r} saved_out_version {saved_version!r} must be an int or None",
            'Op.annotations["saved_out_version"]',
            "saved_out_version_type",
        )
    varying = annotations.get("varying_across_passes")
    if varying is not None and (
        not isinstance(varying, Mapping) or any(not isinstance(k, str) for k in varying)
    ):
        _annotations_invalid(
            f"op {label!r} varying_across_passes must be a str-keyed mapping",
            'Op.annotations["varying_across_passes"]',
            "varying_shape",
        )
    for key in ("dedup_source_id", "dedup_source_version"):
        value = annotations.get(key)
        if value is not None and not _is_int(value):
            _annotations_invalid(
                f"op {label!r} {key} {value!r} must be an int",
                f'Op.annotations["{key}"]',
                "dedup_type",
            )
    reference = annotations.get("dedup_reference_label")
    if reference is not None and not isinstance(reference, str):
        _annotations_invalid(
            f"op {label!r} dedup_reference_label {reference!r} must be a string",
            'Op.annotations["dedup_reference_label"]',
            "dedup_type",
        )
