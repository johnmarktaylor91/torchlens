"""Portable state scrubbing for TorchLens model logs.
This module converts a live ``Trace`` object graph into portable metadata
plus a list of tensor blob specs. It is the save-side counterpart to
rehydration: every class-specific ``_PORTABLE_STATE_SPEC`` is applied here so
tensor payloads become ``BlobRef`` placeholders and non-portable live objects
are dropped or stringified before writing ``metadata.pkl``.
"""

from __future__ import annotations

import copy
import functools
import inspect
import pickle
import re
import warnings
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .._constants import MODEL_LOG_FIELD_ORDER
from .._data_classes._state_adapter import state_items, state_new, state_restore
from .._data_classes.trace import Trace, _scrubbed_transform_repr
from .._utils._display_bounded import RecordTuple, short_dataclass_repr
from ..errors._base import TorchLensWarning
from . import TLSPEC_VERSION, ArtifactError, BlobRef, FieldPolicy, prerelease as _prerelease
from ._small_raw import _small_raw_value
from ._source_privacy import (
    _apply_advisory_source_policy,
    _apply_conditional_source_policy,
    _apply_frame_source_policy,
    _apply_source_metadata_policy,
    _apply_trace_blob_policy,
)
from .payload_codec import PayloadCodec, get_payload_codec

# Replay-safe literals that must round-trip BYTE-EXACT through scrub/save/load.
# ``bytes`` and ``slice`` are declared replay-safe output/argument literals
# (see ``_literal_value_supported``); without keeping them here they fall through
# to ``_stringify_value`` and a ``bytes`` output leaf becomes ``"<scrubbed:bytes>"``
# while the loaded run still reports VERIFIED -- a silent honesty violation. Both
# are immutable and natively serializable by the portable (pickle) codec.
_SIMPLE_KEEP_TYPES = (str, int, float, bool, type(None), torch.dtype, torch.device, bytes, slice)
# Canonical remapped param barcode token (``param_000001``), for R21-1's
# trace-level equivalence-key ordering.
_EQUIV_PARAM_TOKEN = re.compile(r"param_\d{6}")

#: One P-independent scan for the exact live-barcode shape: 8 chars of the
#: barcode alphabet, bounded by non-alphanumerics (barcodes are '_'-joined in
#: identity strings, and '_' is outside the alphabet). See the R29 note at
#: the remap construction site.
_BARCODE_TOKEN_PATTERN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Za-z]{8}(?![0-9A-Za-z])")
_BARCODE_TOKEN_FULLMATCH = re.compile(r"[0-9A-Za-z]{8}").fullmatch
_PORTABLE_WALK_MAX_DEPTH = 200
_SCRUB_IN_PROGRESS = object()


def _pin_in_memo(memo: dict[int, Any], value: Any) -> None:
    """Keep ``value`` alive for the memo's lifetime so its ``id`` cannot recycle.
    The portable-walk memos key rebuilt results by ``id(original)``. When an
    original container is a temporary (dropped once its owner's field is
    replaced), CPython may reuse its address for a later object, which would
    then falsely hit the memo and receive an unrelated rebuilt value. Pinning
    every memoized original under a reserved key (``id(memo)``, the same
    device ``copy.deepcopy`` uses) makes the id-keyed lookup sound.
    Parameters
    ----------
    memo:
        Identity-keyed walk memo whose lifetime bounds the pin.
    value:
        Original object being memoized.
    """

    memo.setdefault(id(memo), []).append(value)


@dataclass(frozen=True)
class BlobSpec:
    """A payload selected for portable blob persistence.
    Parameters
    ----------
    blob_id:
        Opaque zero-padded blob identifier.
    value:
        Logical backend payload to persist.
    kind:
        Logical payload kind, for example ``"out"``.
    label:
        Human-readable TorchLens layer label.
    logical_backend:
        Backend name associated with ``value``.
    """

    blob_id: str
    value: Any
    kind: str
    label: str
    logical_backend: str

    def __iter__(self) -> Iterator[Any]:
        """Yield legacy tuple fields for conservative call-site migration."""

        yield self.blob_id
        yield self.value
        yield self.kind
        yield self.label

    def __getitem__(self, index: int) -> Any:
        """Return a legacy tuple field by position."""

        return (self.blob_id, self.value, self.kind, self.label)[index]


@dataclass
class _ScrubOptions:
    """Flags controlling which optional tensor payloads are preserved."""

    include_outs: bool
    include_grads: bool
    include_saved_args: bool
    include_rng_states: bool
    include_source: bool = True
    include_custom_attributes: bool = True
    include_buffer_values: bool = True
    sparse_runnable: bool = False
    # W3 (weightsfree): the scrubbed trace carries the structure-only marker,
    # so buffer-op value payloads and pre-forward buffer values strip to
    # declared geometry on EVERY scrub caller (unified save + streamed
    # finalize), never reaching an artifact the M-C2 load gate would refuse.
    structure_only: bool = False
    backend_name: str = "torch"
    payload_materialization: bool = True
    payload_codec: PayloadCodec = field(default_factory=lambda: get_payload_codec("torch"))
    unsupported_tensor_records: list[dict[str, str]] = field(default_factory=list)
    # Once-per-type ledger of save-side container downgrades (foreign tuple
    # subclasses flattened, foreign defaultdict factories dropped), keyed by
    # qualified type name so a metadata tree full of one type warns once.
    container_portability_disclosures: dict[str, str] = field(default_factory=dict)


def scrub_for_save(
    trace: Trace,
    *,
    include_outs: bool = True,
    include_grads: bool = True,
    include_saved_args: bool = False,
    include_rng_states: bool = False,
    include_source: bool = True,
    include_custom_attributes: bool = True,
    include_buffer_values: bool = True,
    backend_name: str | None = None,
    payload_materialization: bool = True,
    sparse_runnable: bool = False,
    blob_id_floor: int = 0,
) -> tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]:
    """Scrub a ``Trace`` into portable metadata plus tensor blob specs.

    Parameters
    ----------
    trace:
        Live model log to scrub.
    include_outs:
        Whether saved outs should be replaced with blob references.
    include_grads:
        Whether grads should be replaced with blob references.
    include_saved_args:
        Whether captured args/kwargs and related tensor payloads should be
        preserved via blob references.
    include_rng_states:
        Whether captured RNG state tensors should be preserved via blob
        references.
    include_source:
        Whether captured model source text (the ``_source_code_blob`` class /
        ``__init__`` / ``forward`` source, per-call ``code_context`` source
        lines, and captured docstrings) should be embedded. Absolute source
        paths are always reduced to a bare basename regardless of this flag,
        so no ``$HOME`` / username / filesystem layout ever reaches the bundle.
        With ``include_source=False`` the source text, source-file references,
        and docstrings are dropped entirely.
    include_custom_attributes:
        Whether harvested public module instance attributes
        (``Module.custom_attributes``) are persisted. These are arbitrary
        user values (config scalars, but also anything a module holds as a
        public attribute), so ``False`` drops the whole channel from the
        artifact. Values are NEVER rewritten or partially scrubbed: the
        channel ships verbatim or not at all.
    include_buffer_values:
        Whether captured pre-forward buffer values
        (``Trace._buffer_initial_values``: the value each registered buffer
        held before the forward overwrote it) are persisted. These are
        training-data-derived state (running statistics, counters, caches),
        so ``False`` drops the whole channel; values are never rewritten.
    backend_name:
        Backend identifier for payload audit records. Defaults to
        ``trace.provenance.backend`` when present.
    payload_materialization:
        Whether this backend can serialize materialized tensor payloads.
    sparse_runnable:
        Whether to apply the runnable sparse-core payload exclusion policy.
    blob_id_floor:
        Ids start at ``blob_id_floor + 1``; a streaming finalize passes the
        writer's issued count so no scrub id repeats a streamed id.

    Returns
    -------
    tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]
        Scrubbed top-level state dict plus blob specs and unsupported tensor
        audit records.
    """

    options = _ScrubOptions(
        include_outs=include_outs,
        include_grads=include_grads,
        include_saved_args=include_saved_args,
        include_rng_states=include_rng_states,
        include_source=include_source,
        include_custom_attributes=include_custom_attributes,
        include_buffer_values=include_buffer_values,
        sparse_runnable=sparse_runnable,
        structure_only=bool(isinstance(trace, Trace) and trace._options_structure_only),
        backend_name=str(backend_name or trace._provenance_backend),
        payload_materialization=payload_materialization,
    )
    options.payload_codec = get_payload_codec(options.backend_name)
    memo: dict[int, Any] = {}
    blob_specs: list[BlobSpec] = []
    blob_counter = [blob_id_floor]

    scrubbed_model = _scrub_value(trace, options, memo, blob_specs, blob_counter)
    if not isinstance(scrubbed_model, Trace):
        raise ArtifactError("Portable scrub expected a scrubbed Trace instance.")

    scrubbed_state = {k: v for k, v in state_items(scrubbed_model) if k != "_layer_accessor_cache"}
    scrubbed_state["tlspec_version"] = TLSPEC_VERSION

    module_accessor = getattr(trace, "_module_logs", None)
    scrubbed_state["_io_module_accessor_state"] = (
        _scrub_value(module_accessor, options, memo, blob_specs, blob_counter)
        if module_accessor is not None
        else None
    )
    _stamp_replacement_evidence(trace, scrubbed_state)
    # AUD-CODE 3.11g: annotation tensors become plain detached tensors on the
    # scrubbed copy (the live trace keeps its session-time objects).
    from ._portability_preflight import sanitize_state_annotations

    sanitize_state_annotations(scrubbed_state)
    _scrub_nondeterministic_identities(scrubbed_state)
    detach_conditional_trace_backrefs(scrubbed_state)
    return scrubbed_state, blob_specs, options.unsupported_tensor_records


def _stamp_replacement_evidence(trace: Trace, state: dict[str, Any]) -> None:
    """Stamp the live replacement-corroboration verdict into persisted op state.

    Journal edit records (``InterventionAppliedEvent``) are live-capture
    runtime facts that never serialize, so a loaded artifact cannot re-derive
    whether a ``func_name="intervention_replacement"`` op was a GENUINE
    observed replacement or a plain-capture placeholder -- the loaded-artifact
    validation arm used to fail OPEN, laundering placeholders through a
    save/load round trip (the locked 2026-06-02 rule requires a plain-capture
    placeholder to STILL fail). Save runs while the live authority is intact,
    so the verdict is evaluated here with the full live evidence chain
    (journal causal binding, or the push/rerun FireRecord + armed-spec
    fallback) and stamped into the op's portable ``annotations`` under
    ``replacement_evidence_v1``. Re-saving a loaded trace re-derives the
    verdict from the stamp itself, so the verdict is preserved, never
    upgraded.

    Parameters
    ----------
    trace:
        Live source trace being saved (full evidence authority).
    state:
        Scrubbed top-level trace state, mutated before it is persisted.
    """

    scrubbed_ops = [
        op
        for op in (state.get("_layer_list") or ())
        if getattr(op, "func_name", None) == "intervention_replacement"
        or getattr(op, "is_edited", False)
    ]
    if not scrubbed_ops:
        return
    from ..validation._invariants_backward_flow import op_has_genuine_replacement_evidence

    live_ops_by_label = {getattr(op, "label", None): op for op in (trace._layer_list or ())}
    for scrubbed_op in scrubbed_ops:
        live_op = live_ops_by_label.get(getattr(scrubbed_op, "label", None))
        corroborated = bool(
            live_op is not None and op_has_genuine_replacement_evidence(live_op, trace)
        )
        annotations = dict(getattr(scrubbed_op, "annotations", None) or {})
        annotations["replacement_evidence_v1"] = {
            "corroborated": corroborated,
            "origin": "live_capture_save_stamp",
        }
        scrubbed_op.annotations = annotations


def _scrub_nondeterministic_identities(state: dict[str, Any]) -> None:
    """Remap process-local identity tokens to deterministic trace-local ordinals.

    Parameters
    ----------
    state:
        Scrubbed top-level trace state, mutated before it is persisted.

    Notes
    -----
    Capture-time parameter barcodes and CPython ``id()`` values are useful while
    constructing a trace, but neither is a portable identity.  The remap keeps all
    within-artifact joins intact while making equivalent captures serialize the same
    logical identifiers.
    """

    ops = list(state.get("_layer_list") or ())
    layers = list((state.get("_layer_logs") or {}).values())
    params = sorted(
        (state.get("_param_logs") or {}).values(),
        key=lambda param: (
            str(getattr(param, "address", "")),
            str(getattr(param, "name", "")),
        ),
    )

    barcode_map: dict[str, str] = {}

    def register_barcode(value: Any) -> None:
        """Register one live barcode in deterministic encounter order."""

        if isinstance(value, str) and value not in barcode_map:
            barcode_map[value] = f"param_{len(barcode_map) + 1:06d}"

    for param in params:
        register_barcode(getattr(param, "barcode", None))
    for record in (*ops, *layers):
        for barcode in getattr(record, "_param_barcodes", ()) or ():
            register_barcode(barcode)

    # R29 (b4, 4th round): the all-P alternation regex trialed every one of P
    # branches at essentially every position of every param-FREE record's
    # guaranteed-miss key -- O(V_paramfree x P), ~40s of pure regex misses per
    # portable save at 100k ops / 2k params. Live barcodes are exactly 8
    # chars of ``[0-9A-Za-z]`` (utils/hashing barcode alphabet), so one
    # P-independent bounded-token scan plus a dict probe does the same work
    # in O(L) per string (~300x measured at P=4000). The alternation remains
    # as the fallback for any registered token violating the 8-char
    # invariant (legacy/exotic captures), keeping remap coverage identical.
    fast_barcode_scan = bool(barcode_map) and all(
        _BARCODE_TOKEN_FULLMATCH(barcode) is not None for barcode in barcode_map
    )
    if fast_barcode_scan:
        barcode_pattern = _BARCODE_TOKEN_PATTERN
    elif barcode_map:
        barcode_pattern = re.compile(
            "|".join(re.escape(barcode) for barcode in sorted(barcode_map, key=len, reverse=True))
        )
    else:
        barcode_pattern = None

    def remap_barcode_text(value: Any) -> Any:
        """Replace registered barcodes in a scalar identity string."""

        if not isinstance(value, str):
            return value
        if value in barcode_map:
            return barcode_map[value]
        if barcode_pattern is None:
            return value
        if fast_barcode_scan:
            return barcode_pattern.sub(
                lambda match: barcode_map.get(match.group(0), match.group(0)), value
            )
        return barcode_pattern.sub(lambda match: barcode_map[match.group(0)], value)

    def canonical_equivalence_key(value: Any) -> Any:
        """Remap AND canonically order the ``param_NNNNNN`` run in an equiv key.

        R21-1: an ``op_equivalence_classes`` key is
        ``f"{layer_type}_{'_'.join(sorted(raw_barcodes))}"`` -- but the raw barcodes
        are per-capture RANDOM, so weight-vs-bias order in the key was a coin flip
        per capture. The op-level ``equivalence_class`` field is re-sorted into
        canonical order at save, but the trace-level dict keys fell through to the
        ORDER-PRESERVING substring remapper and stayed random. Remapping barcodes to
        their canonical ``param_NNNNNN`` ids and then sorting that trailing run makes
        the persisted key byte-reproducible across processes.
        """

        remapped = remap_barcode_text(value)
        if not isinstance(remapped, str):
            return remapped
        matches = list(_EQUIV_PARAM_TOKEN.finditer(remapped))
        if len(matches) <= 1:
            return remapped
        tokens = [match.group(0) for match in matches]
        run_start = matches[0].start()
        run_end = matches[-1].end()
        # Sort ONLY the contiguous param-token run; the producer legally appends
        # `_outindex{N}` (multi-output param ops) after it, and dropping that tail
        # collides all N keys into one, silently losing equivalence groups.
        if remapped[run_start:run_end] != "_".join(tokens):
            return remapped
        return remapped[:run_start] + "_".join(sorted(tokens)) + remapped[run_end:]

    equivalence_class_map: dict[str, str] = {}
    for param in params:
        param.barcode = remap_barcode_text(getattr(param, "barcode", None))
    for record in (*ops, *layers):
        original_barcodes = list(getattr(record, "_param_barcodes", ()) or ())
        original_group = "_".join(sorted(original_barcodes))
        remapped_barcodes = [remap_barcode_text(barcode) for barcode in original_barcodes]
        remapped_group = "_".join(sorted(remapped_barcodes))
        record._param_barcodes = remapped_barcodes
        equivalence_class = getattr(record, "equivalence_class", None)
        if isinstance(equivalence_class, str) and original_group:
            record.equivalence_class = equivalence_class.replace(original_group, remapped_group)
        else:
            record.equivalence_class = remap_barcode_text(equivalence_class)
        if isinstance(equivalence_class, str) and isinstance(record.equivalence_class, str):
            equivalence_class_map[equivalence_class] = record.equivalence_class
        # Spec gate FIRST: Layer delegates `parent_param_ops` per pass and its
        # multi-pass accessor raises InvalidArgumentError (a ValueError, which
        # getattr does not swallow), so probing before the gate broke every
        # save of a multi-pass trace. Layer's _PORTABLE_STATE_SPEC has no
        # `parent_param_ops` row; only Op-side records reach the getattr.
        record_spec = getattr(type(record), "_PORTABLE_STATE_SPEC", {})
        parent_param_ops = (
            getattr(record, "parent_param_ops", None) if "parent_param_ops" in record_spec else None
        )
        if isinstance(parent_param_ops, dict):
            record.parent_param_ops = {
                remap_barcode_text(key): value for key, value in parent_param_ops.items()
            }

    equivalence_groups = state.get("_graph_op_equivalence_classes")
    if isinstance(equivalence_groups, dict):
        rebuilt_groups = {
            equivalence_class_map.get(key) or canonical_equivalence_key(key): value
            for key, value in equivalence_groups.items()
        }
        # Annotated as the common `dict` supertype: the two branches below
        # construct a `defaultdict` and a plain `dict` respectively (a
        # `defaultdict` is a `dict`, not the reverse), so the variable needs
        # the wider declared type rather than inferring it from whichever
        # branch happens to assign first.
        remapped_groups: dict[Any, Any]
        if isinstance(equivalence_groups, defaultdict):
            # `type(groups)(generator)` treats the first positional arg as
            # `default_factory`, which must be callable or None: a generator
            # of (key, value) pairs raised `TypeError: first argument must be
            # callable or None`. Preserve the factory explicitly instead.
            remapped_groups = type(equivalence_groups)(equivalence_groups.default_factory)
            remapped_groups.update(rebuilt_groups)
        else:
            remapped_groups = type(equivalence_groups)(rebuilt_groups)
        state["_graph_op_equivalence_classes"] = remapped_groups

    grad_fn_order = list(state.get("_autograd_grad_fn_order") or ())
    grad_fn_logs = state.get("_grad_fn_logs") or {}
    grad_id_map: dict[int, int] = {}

    def register_grad_id(value: Any) -> None:
        """Register one autograd identity in deterministic discovery order."""

        if isinstance(value, int) and not isinstance(value, bool) and value not in grad_id_map:
            grad_id_map[value] = len(grad_id_map) + 1

    for grad_id in grad_fn_order:
        register_grad_id(grad_id)
    for grad_id in grad_fn_logs:
        register_grad_id(grad_id)
    for grad_fn in grad_fn_logs.values():
        register_grad_id(getattr(grad_fn, "grad_fn_object_id", None))
        register_grad_id(getattr(grad_fn, "creator_object_id", None))
        for next_id in getattr(grad_fn, "next_grad_fn_ids", ()) or ():
            register_grad_id(next_id)
    for record in (*ops, *layers):
        register_grad_id(getattr(record, "grad_fn_object_id", None))
    # grind-r5 b3 R21-1: the BACKWARD-PASS rows carry the same autograd
    # identities in ``root_grad_fn_ids``; the a169e886 dense-ordinal fix never
    # reached them, so a raw grad_fn memory address persisted verbatim into
    # ``.tlspec`` (process-dependent artifact bytes + a dangling address in a
    # portable file). Register + remap them through the same trace-local map.
    backward_pass_logs = state.get("_backward_pass_logs")
    pass_records = (
        tuple(backward_pass_logs.values()) if isinstance(backward_pass_logs, dict) else ()
    )
    for pass_record in pass_records:
        for root_id in getattr(pass_record, "root_grad_fn_ids", None) or ():
            register_grad_id(root_id)

    def remap_grad_id(value: Any) -> Any:
        """Return the trace-local ordinal for one autograd identity."""

        return grad_id_map.get(value, value)

    if isinstance(grad_fn_logs, dict):
        remapped_logs = type(grad_fn_logs)()
        for grad_id, grad_fn in grad_fn_logs.items():
            grad_fn.grad_fn_object_id = remap_grad_id(grad_fn.grad_fn_object_id)
            grad_fn.creator_object_id = remap_grad_id(grad_fn.creator_object_id)
            grad_fn.next_grad_fn_ids = [
                remap_grad_id(next_id) for next_id in grad_fn.next_grad_fn_ids
            ]
            remapped_logs[remap_grad_id(grad_id)] = grad_fn
        state["_grad_fn_logs"] = remapped_logs
    state["_autograd_grad_fn_order"] = [remap_grad_id(grad_id) for grad_id in grad_fn_order]
    roots = state.get("_autograd_backward_root_grad_fn_object_ids") or ()
    state["_autograd_backward_root_grad_fn_object_ids"] = [remap_grad_id(g) for g in roots]
    for record in (*ops, *layers):
        record.grad_fn_object_id = remap_grad_id(getattr(record, "grad_fn_object_id", None))
    for pass_record in pass_records:
        root_ids = getattr(pass_record, "root_grad_fn_ids", None)
        if root_ids:
            pass_record.root_grad_fn_ids = [remap_grad_id(root_id) for root_id in root_ids]

    for object_id_key in ("_provenance_model_object_id", "_provenance_input_object_id"):
        state[object_id_key] = 1 if state.get(object_id_key) is not None else None


def detach_conditional_trace_backrefs(value: Any) -> None:
    """Detach runtime-only conditional-arm trace bindings from scrubbed metadata.

    ``ConditionalAccessor`` predates the portable-state policy system, so recursive
    scrubbing treats it as an inert metadata value. Each arm nevertheless carries a
    private ``_trace`` back-reference used by live convenience accessors. Leaving that
    reference attached serializes the entire live trace a second time and can embed raw
    tensor storages in ``metadata.pkl`` outside the manifest/blob boundary.

    The scrub product receives shallow copies of the accessor records with only the
    runtime binding removed. The live trace is never mutated, and load rebinds the copied
    arms to the rehydrated trace.

    Parameters
    ----------
    value:
        Scrubbed top-level metadata mapping to update in place.
    """

    from .._data_classes.trace import Conditional, ConditionalAccessor

    if not isinstance(value, MutableMapping):
        return
    accessor = value.get("_graph_conditionals")
    if not isinstance(accessor, ConditionalAccessor):
        return
    detached: list[Conditional] = []
    for conditional in accessor.values():
        detached_arms = []
        for arm in conditional.arms:
            arm_copy = copy.copy(arm)
            arm_copy._trace = None
            detached_arms.append(arm_copy)
        conditional_copy = copy.copy(conditional)
        conditional_copy.arms = detached_arms
        detached.append(conditional_copy)
    value["_graph_conditionals"] = ConditionalAccessor(detached)


def _mapping_key_payload_reason(key: Any, options: _ScrubOptions) -> str | None:
    """Return why ``key`` embeds a tensor payload, or ``None`` if it is safe.

    Mapping VALUES are recursively scrubbed and blobified, so any tensor they
    contain is lifted into a manifest-indexed :class:`BlobRef` blob file. Mapping
    KEYS, by contrast, are copied verbatim and pickled straight into
    ``metadata.pkl``. A tensor hidden in a key therefore never crosses the
    tensor-policy decision, never gets a ``tensors`` / ``body_index`` manifest
    row, and never produces a blob file. That yields either a bundle whose
    manifest silently disagrees with its metadata payload (a dense tensor key that
    still loads) or a bundle that :func:`~torchlens.validation.validate_tlspec`
    accepts yet the restricted loader refuses (a policy-rejected tensor key the
    safe unpickler will not reconstruct). Because the validator cannot discover a
    pickle payload hidden inside a key from the manifest relations, the producer
    must refuse it up front so ``validate_tlspec`` and ``tl.load`` always agree.

    ``payload_codec.can_encode`` is the same predicate the value path
    (:func:`_blobify_recursive_value`, line ~1041) uses to decide what becomes a
    blob, so it is exactly the set of keys that would otherwise bypass the blob
    belt; hashable composite keys (``tuple`` / ``frozenset``) are walked so a
    tensor nested inside a composite key is caught too.
    """

    stack: list[Any] = [key]
    while stack:
        item = stack.pop()
        if options.payload_codec.can_encode(item):
            return f"a {type(item).__name__} payload"
        if isinstance(item, (tuple, frozenset, list, set)):
            stack.extend(item)
    return None


def _reject_payload_mapping_keys(mapping: Mapping[Any, Any], options: _ScrubOptions) -> None:
    """Refuse a mapping whose keys embed tensor payloads.

    See :func:`_mapping_key_payload_reason` for why key-embedded tensors are a
    save/validate/load asymmetry and must be rejected producer-side.
    """

    for key in mapping:
        reason = _mapping_key_payload_reason(key, options)
        if reason is not None:
            raise ArtifactError(
                f"Cannot save a mapping that uses {reason} as (or inside) a key. "
                "Tensor payloads are portable only as mapping VALUES, which are "
                "indexed into the bundle blob manifest; a tensor embedded in a key "
                "bypasses the tensor policy and blob inventory, producing a bundle "
                "that validates but cannot be loaded. Move the tensor into the "
                "mapping value."
            )


_SCRUB_SIMPLE = 0
_SCRUB_SIZE = 1
_SCRUB_BLOBREF = 2
_SCRUB_LIST = 3
_SCRUB_TUPLE = 4
_SCRUB_SET = 5
_SCRUB_FROZENSET = 6
_SCRUB_OBJECT = 7
_SCRUB_ORDERED_DICT = 8
_SCRUB_MAPPING = 9
# A ``_PORTABLE_STATE_SPEC`` owner that is not one of the containers above; the
# spec itself is read at scrub time (tests edit spec rows in place).
_SCRUB_SPEC_OBJECT = 10

_SCRUB_VALUE_KINDS: weakref.WeakKeyDictionary[type, int] = weakref.WeakKeyDictionary()


#: Container classification order for :func:`_scrub_value_kind`: exactly the
#: ``isinstance`` chain it replaces. ``torch.Size`` precedes ``tuple`` (it is a
#: tuple subclass) and ``OrderedDict``/``defaultdict`` precede plain ``dict`` so
#: their observable ordering and default-factory behavior survive a round trip.
_SCRUB_KIND_ORDER: tuple[tuple[type | tuple[type, ...], int], ...] = (
    (_SIMPLE_KEEP_TYPES, _SCRUB_SIMPLE),
    (torch.Size, _SCRUB_SIZE),
    (BlobRef, _SCRUB_BLOBREF),
    (list, _SCRUB_LIST),
    (tuple, _SCRUB_TUPLE),
    (set, _SCRUB_SET),
    (frozenset, _SCRUB_FROZENSET),
    (OrderedDict, _SCRUB_ORDERED_DICT),
    (dict, _SCRUB_MAPPING),
)


def _scrub_value_kind(value_type: type) -> int:
    """Classify one node type for :func:`_scrub_value`, memoized per type.

    Containers are matched in :data:`_SCRUB_KIND_ORDER`; a spec owner is only
    recognized after every container kind, as the old chain did, and anything
    else is a spec-less leaf.
    """

    kind = _SCRUB_OBJECT
    for base, candidate in _SCRUB_KIND_ORDER:
        if issubclass(value_type, base):
            kind = candidate
            break
    else:
        if getattr(value_type, "_PORTABLE_STATE_SPEC", None) is not None:
            kind = _SCRUB_SPEC_OBJECT
    _SCRUB_VALUE_KINDS[value_type] = kind
    return kind


def _type_is_load_reconstructible(value_type: type) -> bool:
    """Return whether the safe unpickler can rebuild instances of this type.

    Consults the loader's ACTUAL type authority rather than a namespace prefix
    test (R10-4): the safe unpickler admits a ``torchlens`` type ONLY if its
    exact ``(module, qualname)`` is on the vetted-inert ``_SAFE_TORCHLENS_TYPES``
    allowlist and it is not an extras-gated appliance module. The prefix test
    preserved off-allowlist / appliance torchlens types by TYPE into
    ``metadata.pkl`` that the loader then REFUSED -- a save-succeeds /
    load-refuses trap that made the whole bundle unloadable.
    """

    module = getattr(value_type, "__module__", "") or ""
    root = module.split(".", 1)[0]
    if root == "torchlens":
        from ._safe_unpickle import _SAFE_TORCHLENS_TYPES, _is_torchlens_appliance_module

        name = getattr(value_type, "__qualname__", None) or getattr(value_type, "__name__", "")
        return (module, name) in _SAFE_TORCHLENS_TYPES and not _is_torchlens_appliance_module(
            module
        )
    return root == "torch"


def _factory_is_load_reconstructible(factory: Any) -> bool:
    """Return whether the safe unpickler admits ``factory`` as a bare global.

    The loader admits a ``builtins`` / ``collections`` default_factory ONLY if
    its exact ``(module, name)`` is on ``_SAFE_EXPLICIT_GLOBALS`` (the pure-data
    constructors: ``list``/``dict``/``int``/``OrderedDict``/...), never every
    ``builtins`` global -- so a ``builtins.eval`` factory the prefix test
    preserved was a load-refuses trap. A ``torchlens`` factory must additionally
    pass ``is_inert_first_party_callable``.
    """

    module = getattr(factory, "__module__", "") or ""
    root = module.split(".", 1)[0]
    if root == "torch":
        return True
    if root == "torchlens":
        from .._utils._callable_safety import is_inert_first_party_callable
        from ._safe_unpickle import _is_torchlens_owned

        return _is_torchlens_owned(factory) and is_inert_first_party_callable(factory)
    from ._safe_unpickle import _SAFE_EXPLICIT_GLOBALS

    name = getattr(factory, "__qualname__", None) or getattr(factory, "__name__", "")
    return (module, name) in _SAFE_EXPLICIT_GLOBALS


def _disclose_container_downgrade(
    options: _ScrubOptions | None,
    subject: Any,
    message: str,
) -> None:
    """Warn ONCE per downgraded container/factory type per scrub pass."""

    module = getattr(subject, "__module__", "") or type(subject).__module__
    qualname = getattr(subject, "__qualname__", None) or type(subject).__qualname__
    key = f"{module}.{qualname}"
    if options is not None:
        if key in options.container_portability_disclosures:
            return
        options.container_portability_disclosures[key] = message
    warnings.warn(
        f"{message.format(name=key)} The saved bundle stays loadable; only the "
        "container's original type is not preserved.",
        TorchLensWarning,
        stacklevel=2,
    )


def _rebuild_tuple_value(
    value: tuple[Any, ...],
    items: Iterable[Any],
    options: _ScrubOptions | None = None,
) -> tuple[Any, ...]:
    """Rebuild a tuple-like container, preserving only load-reconstructible types.

    A USER tuple subclass (e.g. a caller-module namedtuple) used to be
    preserved by TYPE into ``metadata.pkl``; the default-deny safe unpickler
    then refused the foreign class at load, so the bundle SAVED fine and the
    whole bundle REFUSED to load. Foreign types are now flattened to a plain
    tuple at save time with a disclosure warning -- and a preserved type whose
    constructor rejects the rebuilt items is disclosed too, never silently
    downgraded.

    Parameters
    ----------
    value:
        Source tuple-like container.
    items:
        Scrubbed child values.
    options:
        Active scrub options carrying the once-per-type disclosure ledger.

    Returns
    -------
    tuple[Any, ...]
        Rebuilt tuple of the preserved type, or a plain tuple with disclosure.
    """

    materialized = tuple(items)
    if isinstance(value, torch.Size):
        return torch.Size(materialized)
    value_type = type(value)
    if value_type is tuple or value_type is RecordTuple:
        return materialized
    if not _type_is_load_reconstructible(value_type):
        _disclose_container_downgrade(
            options,
            value_type,
            "Flattening tuple subclass {name} to a plain tuple in portable "
            "metadata: the type is not resolvable by the default-deny bundle "
            "loader, so preserving it would make the bundle unloadable.",
        )
        return materialized
    maker = getattr(value_type, "_make", None)
    if callable(maker):
        try:
            return maker(materialized)
        except (TypeError, ValueError):
            _disclose_container_downgrade(
                options,
                value_type,
                "Flattening tuple subclass {name} to a plain tuple in portable "
                "metadata: its _make constructor rejected the scrubbed items.",
            )
            return materialized
    try:
        return value_type(materialized)
    except (TypeError, ValueError):
        _disclose_container_downgrade(
            options,
            value_type,
            "Flattening tuple subclass {name} to a plain tuple in portable "
            "metadata: its constructor does not accept a single iterable.",
        )
        return materialized


def _portable_default_factory(
    value: defaultdict[Any, Any],
    options: _ScrubOptions | None,
) -> Any:
    """Return a load-safe ``default_factory`` for a scrubbed defaultdict.

    A factory from a USER module rehydrates as an inert foreign-callable
    placeholder under the default-deny loader, turning the first missing-key
    read into an ``UnpicklingError`` mid-analysis. Such a factory is dropped
    at save time with disclosure; the loaded mapping then behaves as a plain
    dict (``KeyError`` on missing keys). Builtins and torch/torchlens-owned
    factories stay.
    """

    factory = value.default_factory
    if factory is None:
        return None
    if _factory_is_load_reconstructible(factory):
        return factory
    _disclose_container_downgrade(
        options,
        factory,
        "Dropping non-portable defaultdict default_factory {name} in portable "
        "metadata: the default-deny bundle loader would rehydrate it as a "
        "placeholder that raises on the first missing-key read.",
    )
    return None


@dataclass(frozen=True, slots=True, eq=False)
class _ScrubWalk:
    """Per-pass state shared by every node of one portable scrub walk.

    ``payload_owner`` is ``None`` on the METADATA walk (:func:`_scrub_value`)
    and the ``(owner, field_name)`` pair of a ``BLOB_RECURSIVE`` field on the
    PAYLOAD walk (:func:`_blobify_recursive_value`). Both walks share every
    container rule in :data:`_SCRUB_NODE_HANDLERS` and differ only at the
    leaves (:func:`_scrub_object_node`) and in their cycle/depth wording.
    """

    __repr__ = short_dataclass_repr

    options: _ScrubOptions
    memo: dict[int, Any]
    blob_specs: list[BlobSpec]
    blob_counter: list[int]
    stringify_unknown: bool = False
    payload_owner: tuple[Any, str] | None = None

    @property
    def noun(self) -> str:
        """Return the wording of this walk's depth and cycle refusals."""

        return "metadata" if self.payload_owner is None else "payload"

    def for_fields(self) -> _ScrubWalk:
        """Return the metadata walk that scrubs a spec object's fields.

        Neither ``stringify_unknown`` nor a payload owner crosses into a
        ``_PORTABLE_STATE_SPEC`` object: its fields follow their declared
        policies on a plain metadata walk sharing this walk's memo and blobs.
        """

        if self.payload_owner is None and not self.stringify_unknown:
            return self
        return _ScrubWalk(self.options, self.memo, self.blob_specs, self.blob_counter)


def _scrub_value(
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
    stringify_unknown: bool = False,
    _depth: int = 0,
) -> Any:
    """Recursively scrub a value while preserving shared object identity.

    Parameters
    ----------
    stringify_unknown:
        When ``True``, a value with neither a container type handled here
        nor a ``_PORTABLE_STATE_SPEC`` falls back to
        :func:`_stringify_value` instead of being returned unchanged. Used
        by the ``save_raw_input``/``save_transformed_output`` ``True`` policy
        (:func:`_scrub_raw_value_for_save`) so live, un-picklable objects
        (generators, locks, open file handles, sockets, ...) nested inside
        a user's raw input/output never reach ``pickle.dump`` unmodified.
        Default ``False`` preserves the general-purpose scrub pass used for
        the rest of the ``Trace`` object graph.
    """

    walk = _ScrubWalk(options, memo, blob_specs, blob_counter, stringify_unknown=stringify_unknown)
    return _scrub_node(value, walk, _depth)


def _scrub_node(value: Any, walk: _ScrubWalk, depth: int) -> Any:
    """Scrub one node by dispatching on its memoized value kind.

    One cached type lookup replaces the up-to-ten ``isinstance`` chain the
    old branch table re-ran for every one of the quarter-million nodes a ResNet scrub
    visits; :func:`_scrub_value_kind` resolves the kinds in the identical
    order, so the selected rule is unchanged.
    """

    if depth > _PORTABLE_WALK_MAX_DEPTH:
        raise ArtifactError(
            f"Portable {walk.noun} exceeds the maximum depth of {_PORTABLE_WALK_MAX_DEPTH}."
        )
    kind = _SCRUB_VALUE_KINDS.get(type(value))
    if kind is None:
        kind = _scrub_value_kind(type(value))
    return _SCRUB_NODE_HANDLERS[kind](value, walk, depth)


def _keep_node(value: Any, walk: _ScrubWalk, depth: int) -> Any:
    """Keep a replay-safe literal or an already-blobified ``BlobRef`` as is."""

    return value


def _scrub_size_node(value: Any, walk: _ScrubWalk, depth: int) -> Any:
    """Rebuild a ``torch.Size`` (an int tuple, never memoized or walked)."""

    return torch.Size(value)


def _scrub_mutable_collection(
    value: Any, walk: _ScrubWalk, depth: int, builtin: type[list[Any]] | type[set[Any]]
) -> Any:
    """Rebuild a list or set under the shared memo so shared identity survives.

    The empty rebuilt container is memoized BEFORE its items are scrubbed, so
    a self-referencing list/set rebuilds as a self-referencing copy instead
    of recursing until the depth cap. Only the metadata walk discloses a
    flattened subclass (R10-9 symmetry with tuples); the payload walk never
    did, and this keeps its warning surface unchanged.

    Items are appended in a plain loop rather than through a generator: a
    generator is one more Python frame per nesting level, and the depth cap
    must fire as a typed ``ArtifactError`` before the interpreter's recursion
    limit does.
    """

    memo = walk.memo
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]
    if type(value) is not builtin and walk.payload_owner is None:
        name = builtin.__name__
        _disclose_container_downgrade(
            walk.options,
            value,
            f"Dropping non-portable {name} subclass {{name}} in portable metadata; "
            f"it rebuilds as a plain {name}.",
        )
    rebuilt = builtin()
    _pin_in_memo(memo, value)
    memo[obj_id] = rebuilt
    add = rebuilt.append if isinstance(rebuilt, list) else rebuilt.add
    for item in value:
        add(_scrub_node(item, walk, depth + 1))
    return rebuilt


def _rebuild_frozenset_value(
    value: Any, items: Iterable[Any], options: _ScrubOptions | None
) -> frozenset[Any]:
    """Rebuild a frozenset from its scrubbed items (subclass type not kept)."""

    return frozenset(items)


@dataclass(frozen=True, slots=True)
class _ImmutableRule:
    """How :func:`_scrub_immutable_collection` rebuilds one container kind."""

    rebuild: Callable[[Any, Iterable[Any], _ScrubOptions | None], Any]
    noun: str
    disclose_subclass: bool


def _scrub_immutable_collection(
    value: Any, walk: _ScrubWalk, depth: int, rule: _ImmutableRule
) -> Any:
    """Rebuild an immutable container whose identity exists only once built.

    A tuple or frozenset cannot be memoized before its items are scrubbed
    (there is no empty instance to fill in), so a re-entry while it is in
    progress is a true cycle and is refused rather than looped on. A
    frozenset subclass flattens to a plain frozenset, disclosed once on the
    metadata walk (``disclose_subclass``); tuple subclasses are disclosed by
    :func:`_rebuild_tuple_value` itself, which preserves the loadable ones.
    """

    memo = walk.memo
    obj_id = id(value)
    cached = memo.get(obj_id)
    if cached is _SCRUB_IN_PROGRESS:
        raise ArtifactError(f"Portable {walk.noun} contains a cycle through a {rule.noun}.")
    if cached is not None:
        return cached
    if rule.disclose_subclass and type(value) is not frozenset and walk.payload_owner is None:
        _disclose_container_downgrade(
            walk.options,
            value,
            "Dropping non-portable frozenset subclass {name} in portable "
            "metadata; it rebuilds as a plain frozenset.",
        )
    _pin_in_memo(memo, value)
    memo[obj_id] = _SCRUB_IN_PROGRESS
    try:
        items: list[Any] = []
        for item in value:
            items.append(_scrub_node(item, walk, depth + 1))
        rebuilt = rule.rebuild(value, items, walk.options)
    except BaseException:
        memo.pop(obj_id, None)
        raise
    memo[obj_id] = rebuilt
    return rebuilt


def _empty_mapping_like(
    value: Mapping[Any, Any], options: _ScrubOptions
) -> MutableMapping[Any, Any]:
    """Return the empty portable mapping a scrubbed ``value`` rebuilds into.

    ``defaultdict`` keeps a load-reconstructible factory
    (:func:`_portable_default_factory`), ``OrderedDict`` keeps its ordering
    type, and every other mapping (plain ``dict`` and its subclasses)
    rebuilds as a plain ``dict``.
    """

    if isinstance(value, defaultdict):
        return defaultdict(_portable_default_factory(value, options))
    if isinstance(value, OrderedDict):
        return OrderedDict()
    return {}


def _scrub_mapping_node(value: Any, walk: _ScrubWalk, depth: int) -> Any:
    """Rebuild any mapping kind after refusing tensor-payload keys.

    The key refusal runs for EVERY mapping flavour before the rebuild so a
    payload cannot slip into ``metadata.pkl`` unscrubbed and un-inventoried
    (see :func:`_mapping_key_payload_reason`). Values are scrubbed; keys are
    copied verbatim.
    """

    _reject_payload_mapping_keys(value, walk.options)
    memo = walk.memo
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]
    _pin_in_memo(memo, value)
    rebuilt = _empty_mapping_like(value, walk.options)
    memo[obj_id] = rebuilt
    for key, item in value.items():
        rebuilt[key] = _scrub_node(item, walk, depth + 1)
    return rebuilt


def _scrub_object_node(value: Any, walk: _ScrubWalk, depth: int) -> Any:
    """Scrub a spec-less non-container node: tensor payload or opaque leaf.

    The payload walk blobifies anything its codec recognizes and stringifies
    every other spec-less object (the ``saved_args``/``custom_attributes``
    contract). The metadata walk keeps a spec-less object unless
    ``stringify_unknown`` is set AND it cannot be pickled, in which case it
    becomes a bounded ``<scrubbed:TypeName>`` placeholder.
    """

    owner = walk.payload_owner
    if owner is not None and walk.options.payload_codec.can_encode(value):
        return _blobify_tensor_field(
            owner[0], owner[1], value, walk.blob_specs, walk.blob_counter, walk.options
        )
    if owner is not None:
        return _stringify_value(value)
    if walk.stringify_unknown and not _is_safely_picklable(value):
        return _stringify_value(value)
    return value


def _scrub_spec_object(value: Any, walk: _ScrubWalk, depth: int) -> Any:
    """Scrub an object that declares a ``_PORTABLE_STATE_SPEC``, field by field.

    On the payload walk an encodable spec object is blobbed whole; otherwise
    its fields follow their declared policies on a plain metadata walk
    (:meth:`_ScrubWalk.for_fields`). The object is memoized before its fields
    are scrubbed so self-references resolve to the scrubbed copy.
    """

    owner = walk.payload_owner
    if owner is not None:
        if walk.options.payload_codec.can_encode(value):
            return _blobify_tensor_field(
                owner[0], owner[1], value, walk.blob_specs, walk.blob_counter, walk.options
            )
        walk = walk.for_fields()
    elif walk.stringify_unknown:
        walk = walk.for_fields()
    memo = walk.memo
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]
    scrubbed_obj = state_new(type(value))
    _pin_in_memo(memo, value)
    memo[obj_id] = scrubbed_obj
    owner_is_trace = isinstance(value, Trace)
    scrubbed_state = _scrub_spec_fields(value, walk, depth, owner_is_trace=owner_is_trace)
    if owner_is_trace:
        _stamp_trace_portable_rows(value, scrubbed_state, walk.options)
    else:
        _apply_owner_source_policies(value, scrubbed_state, walk.options)
    return state_restore(scrubbed_obj, scrubbed_state)


def _scrub_spec_fields(
    value: Any, walk: _ScrubWalk, depth: int, *, owner_is_trace: bool
) -> dict[str, Any]:
    """Return the scrubbed state of a ``_PORTABLE_STATE_SPEC`` owner.

    An undeclared live field is refused (:func:`_undeclared_field_error`)
    unless it is a ``functools.cached_property`` cell (derived state, skipped).
    DROP/WEAKREF_STRIP and plain KEEP are resolved inline, in ``_scrub_field``'s
    own branch order, because this loop scrubs hundreds of thousands of fields
    per ResNet save and one Python call per field was measurable.
    """

    spec: Mapping[str, FieldPolicy] = type(value)._PORTABLE_STATE_SPEC
    raw_fields = _TRACE_RAW_FIELDS if owner_is_trace else frozenset[str]()
    scrubbed_state: dict[str, Any] = {}
    for field_name, field_value in _state_items_for_scrub(value, spec):
        if owner_is_trace and _is_runtime_only_trace_field(field_name):
            continue
        declared = spec.get(field_name)
        if declared is None:
            _refuse_undeclared_field(value, field_name, owner_is_trace)
            continue
        if field_name in _NORMALIZED_FIELD_NAMES:
            field_value = _normalize_field_value(field_name, field_value)
        policy = _effective_policy(value, field_name, declared, walk.options)
        if policy in _NULLED_POLICIES:
            scrubbed_state[field_name] = None
        elif policy is FieldPolicy.KEEP and field_name not in raw_fields:
            scrubbed_state[field_name] = _scrub_node(field_value, walk, depth + 1)
        else:
            scrubbed_state[field_name] = _scrub_field(
                owner=value,
                field_name=field_name,
                field_value=field_value,
                policy=policy,
                walk=walk,
                depth=depth + 1,
            )
    return scrubbed_state


#: Per-kind scrub rules, indexed by the :func:`_scrub_value_kind` constants.
#: Container rules are bound with ``functools.partial`` (a C-level call, no
#: extra Python frame) so every nesting level costs the same two frames the
#: old branch chain did; see :func:`_scrub_mutable_collection`.
_SCRUB_NODE_HANDLERS: tuple[Callable[[Any, _ScrubWalk, int], Any], ...] = (
    _keep_node,  # _SCRUB_SIMPLE
    _scrub_size_node,  # _SCRUB_SIZE
    _keep_node,  # _SCRUB_BLOBREF
    functools.partial(_scrub_mutable_collection, builtin=list),  # _SCRUB_LIST
    functools.partial(  # _SCRUB_TUPLE
        _scrub_immutable_collection,
        rule=_ImmutableRule(_rebuild_tuple_value, "tuple", disclose_subclass=False),
    ),
    functools.partial(_scrub_mutable_collection, builtin=set),  # _SCRUB_SET
    functools.partial(  # _SCRUB_FROZENSET
        _scrub_immutable_collection,
        rule=_ImmutableRule(_rebuild_frozenset_value, "frozenset", disclose_subclass=True),
    ),
    _scrub_object_node,  # _SCRUB_OBJECT
    _scrub_mapping_node,  # _SCRUB_ORDERED_DICT
    _scrub_mapping_node,  # _SCRUB_MAPPING
    _scrub_spec_object,  # _SCRUB_SPEC_OBJECT
)

#: Trace fields whose KEEP policy still routes through the raw-value save policy.
_TRACE_RAW_FIELDS = frozenset({"input", "transformed_output"})
#: Policies whose field is written back as ``None`` without a walk.
_NULLED_POLICIES = frozenset({FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP})

#: Fields whose live value is normalized before its policy applies.
_NORMALIZED_FIELD_NAMES = frozenset({"_is_in_conditional_body", "_capture_outcome"})


def _normalize_field_value(field_name: str, field_value: Any) -> Any:
    """Normalize the two fields whose live value is not their portable value."""

    if field_name == "_is_in_conditional_body" and field_value is None:
        return False
    if field_name == "_capture_outcome" and field_value is not None:
        # The settled capture outcome persists as its STRING-ONLY payload
        # (tlspec v7): the safe-unpickle allowlist never needs the record
        # class, and loads parse the payload against closed vocabularies.
        return field_value.to_payload() if hasattr(field_value, "to_payload") else None
    return field_value


def _refuse_undeclared_field(owner: Any, field_name: str, owner_is_trace: bool) -> None:
    """Refuse a live field missing from the spec unless it is a cached_property cell."""

    if not _is_cached_property_cell(type(owner), field_name):
        raise _undeclared_field_error(owner, field_name, owner_is_trace)


def _is_cached_property_cell(owner_type: type, field_name: str) -> bool:
    """Return whether an undeclared field is a ``functools.cached_property`` cell.

    A ``cached_property`` read caches its value in the instance ``__dict__``
    under the property's own name (e.g. the public ``Trace.recipe`` accessor).
    Those cells are DERIVED state that rebuilds on access, never portable
    fields, and they appear only after a read -- refusing them made a
    read-only public property poison every later ``Trace.save`` (B3R4-R10-1).
    The check runs only on the refusal path, so the hot field loop pays nothing.
    """

    return isinstance(
        inspect.getattr_static(owner_type, field_name, None), functools.cached_property
    )


def _undeclared_field_hint(value: Any, field_name: str, owner_is_trace: bool) -> str:
    """Name the writer that populated an undeclared field, when one is known.

    An undeclared ``_<name>_cache`` cell is almost always DERIVED state poked
    into ``__dict__`` by a lazily-caching public accessor read (the
    ModuleCall/Module ``.facets`` incident): name that accessor when it
    exists. A LEDGERED-but-undeclared Trace transient (the draw()
    ``_last_encoding_state`` incident) quotes its ledger row so the message
    names the writer. Either way the message points at the read that
    poisoned the save, not just an internal field the user never heard of.
    """

    if field_name.startswith("_") and field_name.endswith("_cache"):
        accessor_name = field_name[1 : -len("_cache")]
        if isinstance(inspect.getattr_static(type(value), accessor_name, None), property):
            return (
                f" This cell was populated by reading the public "
                f"`.{accessor_name}` accessor; a lazy derived cache "
                f"must be declared FieldPolicy.DROP in "
                f"{type(value).__name__}._PORTABLE_STATE_SPEC so a "
                f"read-only access cannot poison a later save."
            )
    if owner_is_trace:
        from .._data_classes._trace_components import TRACE_EXTERNAL_WRITE_EXEMPTIONS

        ledger_reason = TRACE_EXTERNAL_WRITE_EXEMPTIONS.get(field_name)
        if ledger_reason is not None:
            return (
                f" This field is ledgered in "
                f"TRACE_EXTERNAL_WRITE_EXEMPTIONS as: {ledger_reason!r}."
                f" The ledger documents the write; it is not a scrub "
                f"policy. Enroll the runtime-only transient in the "
                f"scrub's runtime-only set (or give it a "
                f"FieldPolicy.DROP row) so populating it cannot "
                f"poison a later save."
            )
    return ""


def _undeclared_field_error(value: Any, field_name: str, owner_is_trace: bool) -> ArtifactError:
    """Build the refusal for a live state field with no portability policy."""

    return ArtifactError(
        f"{type(value).__name__}.{field_name} is missing from "
        f"_PORTABLE_STATE_SPEC. Every live state field needs an "
        f"explicit portability policy before it can be saved."
        f"{_undeclared_field_hint(value, field_name, owner_is_trace)}"
    )


def _stamp_trace_portable_rows(
    trace: Trace, scrubbed_state: dict[str, Any], options: _ScrubOptions
) -> None:
    """Add the Trace-only portable rows and apply the Trace source policies."""

    # B8-20: a functools.partial repr embeds its bound argument VALUES, so the
    # scrubbed persistence repr redacts them.
    scrubbed_state["_activation_transform_repr"] = _scrubbed_transform_repr(
        trace._options_activation_transform
    )
    # P7/R10: a capture TorchLens itself refused to bless must not round-trip
    # into "no claim". The NEGATIVE disclosure persists as a string-only row
    # (mirroring _capture_outcome's treatment); True/None stay session-time,
    # so a loaded artifact can never CLAIM verification -- the row can only
    # ever worsen a verdict, preserving the monotonicity the runnable side
    # enforces structurally. rescue_rerun stays session-time as documented.
    if trace._provenance_verified is False:
        reason = trace._provenance_verification_reason
        scrubbed_state["_capture_verification"] = {
            "verified": False,
            "reason": str(reason) if reason is not None else None,
        }
    scrubbed_state["tlspec_version"] = TLSPEC_VERSION
    _apply_prerelease_rows(scrubbed_state)
    _apply_advisory_source_policy(scrubbed_state, options)
    _apply_source_metadata_policy(scrubbed_state, options)
    _apply_trace_blob_policy(scrubbed_state, options)


def _apply_prerelease_rows(scrubbed_state: dict[str, Any]) -> None:
    """Mark a switch-on write, or strip gated annotation keys from a real one."""

    if _prerelease._ACTIVE:
        # EVERY switch-on write is marked, whether or not a gated field
        # actually persisted in this state -- no per-field accounting can
        # omit the marker, so switched and real-version artifacts are
        # never indistinguishable (loads refuse the marker typed unless
        # the switch is active; see torchlens._io.prerelease).
        scrubbed_state[_prerelease.PRERELEASE_STATE_KEY] = _prerelease.prerelease_marker_payload()
        return
    # Registered pre-release ANNOTATIONS sub-keys are new persistence
    # write paths inside the already-persisting annotations mapping
    # (e.g. the episode ledger): they never ride a real artifact of
    # the frozen tlspec version. The pop acts on the scrubbed copy
    # only -- the live trace keeps its session-time annotations.
    gated_annotation_keys = _prerelease.gated_annotations_keys()
    if not gated_annotation_keys:
        return
    annotations_state = scrubbed_state.get("annotations")
    if isinstance(annotations_state, dict):
        for gated_key in gated_annotation_keys:
            annotations_state.pop(gated_key, None)


def _apply_owner_source_policies(
    value: Any, scrubbed_state: dict[str, Any], options: _ScrubOptions
) -> None:
    """Apply the source-privacy policy of a non-Trace spec object, by shape.

    Owners are matched by name or field signature rather than ``isinstance``
    to avoid an import cycle between ``_io`` and ``data_classes``.
    """

    # ``FuncCallLocation`` is a unique internal class with no subclasses.
    if type(value).__name__ == "FuncCallLocation":
        _apply_frame_source_policy(scrubbed_state, options)
    # ``Module``/``GradFn`` logs carry class/init/forward(/backward) source metadata
    # under public field names (matched by signature); the Trace's private
    # ``_source_*`` spellings never reach here (the Trace applies the policy itself).
    elif "class_docstring" in scrubbed_state:
        _apply_source_metadata_policy(scrubbed_state, options)
    # ``ConditionalEvent`` records (in ``conditional_records``) carry the absolute
    # path of the user's forward-defining module in ``source_file``; only the
    # persisted, spec-bearing record is relativized/dropped.
    elif "source_file" in scrubbed_state and "branch_ranges" in scrubbed_state:
        _apply_conditional_source_policy(scrubbed_state, options, drop_value="")
    # ``Conditional`` records (in the public ``conditionals`` accessor) carry the
    # same absolute path in an OPTIONAL ``source_file``; dropped to ``None``.
    elif "source_file" in scrubbed_state and "arms" in scrubbed_state:
        _apply_conditional_source_policy(scrubbed_state, options, drop_value=None)


def _state_items_for_scrub(
    value: Any, spec: Mapping[str, FieldPolicy]
) -> Iterable[tuple[str, Any]]:
    """Yield scrub fields in the portable on-disk order for ``value``.

    Parameters
    ----------
    value:
        Portable-state object being scrubbed.
    spec:
        Portable state policy mapping for ``value``.

    Returns
    -------
    Iterable[tuple[str, Any]]
        Field/value pairs in deterministic scrub order. Only Trace receives
        portable serialization ordering; other classes keep adapter order,
        including slot order for slotted objects.
    """

    if isinstance(value, Trace):
        return _trace_state_items_for_scrub(value, spec)
    return state_items(value)


def _trace_state_items_for_scrub(
    trace: Trace, spec: Mapping[str, FieldPolicy]
) -> Iterable[tuple[str, Any]]:
    """Yield Trace fields in deterministic portable serialization order.

    Parameters
    ----------
    trace:
        Trace whose live state should be scrubbed.
    spec:
        Trace portable state policy mapping.

    Yields
    ------
    tuple[str, Any]
        Live Trace field/value pairs: first fields present in
        ``MODEL_LOG_FIELD_ORDER`` order, then remaining present fields in
        ``_PORTABLE_STATE_SPEC`` order, then any still-unseen fields so the
        existing unknown-field guard can raise.
    """

    live_state = dict(state_items(trace))
    yielded_fields: set[str] = set()

    for field_name in MODEL_LOG_FIELD_ORDER:
        if field_name in live_state:
            yielded_fields.add(field_name)
            yield field_name, live_state[field_name]

    for field_name in spec:
        if field_name in live_state and field_name not in yielded_fields:
            yielded_fields.add(field_name)
            yield field_name, live_state[field_name]

    for field_name, field_value in live_state.items():
        if field_name not in yielded_fields:
            yield field_name, field_value


def _is_runtime_only_trace_field(field_name: str) -> bool:
    """Return whether a Trace field is intentionally omitted from portable state.

    Parameters
    ----------
    field_name
        Trace ``__dict__`` key under scrub consideration.

    Returns
    -------
    bool
        True when the field stores runtime-only interpreter or visualization state.
    """

    return field_name in {
        "_had_unattributed_tensor_args",
        "_module_boundary_adoptions",
        "_module_boundary_outside_sources",
        "_orphan_unattributed_tensor_args",
        "_plain_scalar_escape_labels",
        "_plain_direct_escape_gaps",
        "_foreign_departures",
        "_unrecorded_operator_mutations",
        "_last_sibling_ordering_decision",
        # Sibling render diagnostic (same _render_dot write site as the row
        # above); left unenrolled, ONE draw() poisoned every later Trace.save.
        "_last_encoding_state",
        # Layout-execution geometry record (vizmech D24), same write site class.
        "_last_render_geometry",
        "_pending_container_collapse_nodes",
        "_defer_streaming_bundle_finalization",
        "_capture_producer_policy",
        "_capture_config",
        "_stop_directive",
        "_keep_outs_in_memory",
        "_capture_container_structure",
        "_capture_output_structure",
        "_out_sink",
        "_out_writer",
        "_container_ordinals_by_output_op_label",
        "_container_ordinals_by_input_func_call_id",
        "_validation_replay_status",
        "_capture_parent_edge_truth",
        "_orphan_pruned_func_call_ids",
        # B1-17: `_tl_predicate_intervention_{spec,target}_keys`,
        # `_source_bundle_{path,manifest_sha256}` and the output-parent retention
        # flag (now inside `_rerun_save_request`) moved out of this allowance into
        # declared `Trace._PORTABLE_STATE_SPEC` DROP rows. This allowance is consulted
        # BEFORE the spec, so an entry here makes the declared policy dead code.
        "jax_equation_captures",
        "jax_ordered_captures",
        "jax_region_captures",
        "_jax_capture_index_to_raw_op_label",
        "jax_capture_index_to_final_op_label",
        "jax_outvar_key_to_capture_index",
        "_selective_save_hidden_payloads",
        # B1-02: the four semantic-output scratch names moved OUT of this
        # allowance and into `Trace._PORTABLE_STATE_SPEC` as declared
        # `FieldPolicy.DROP` rows. This allowance is consulted BEFORE the spec,
        # so an entry here makes the declared policy dead code -- a policy
        # flipped to KEEP would still be silently dropped. Declared rows keep
        # one authority.
        "tinygrad_payload_policy",
        "tinygrad_uop_captures",
        "_mlx_op_captures",
        "_mlx_replay_inventory",
    }


def _structure_only_buffer_payload_field(owner: Any, field_name: str) -> bool:
    """Whether ``field_name`` is buffer-value state a weights-free save drops (W3)."""

    if field_name == "_buffer_initial_values":
        return True
    return field_name in {
        "out",
        "transformed_out",
        "grad",
        "transformed_grad",
        "saved_args",
        "saved_kwargs",
    } and bool(getattr(owner, "is_buffer", False))


def _effective_policy(
    owner: Any,
    field_name: str,
    base_policy: FieldPolicy,
    options: _ScrubOptions,
) -> FieldPolicy:
    """Resolve the runtime policy for a field after include-flag overrides."""

    if options.sparse_runnable and field_name in {
        "_annotation_blobs",
        "_args",
        "_buffer_initial_values",
        "_derived_grad_payload",
        "_kwargs",
        "custom_attributes",
        "forward_args",
        "forward_kwargs",
        "func_config",
        "func_rng_states",
        "grad",
        "grad_inputs",
        "grad_outputs",
        "input_activations",
        "_orphan_records",
        "out",
        "out_versions_by_child",
        "parent_params",
        "payload",
        "input",
        "transformed_output",
        "saved_args",
        "saved_kwargs",
        "transformed_grad",
        "transformed_out",
    }:
        return FieldPolicy.DROP

    if options.structure_only and _structure_only_buffer_payload_field(owner, field_name):
        # W3 (weightsfree L6): buffer registration writes buffer VALUE
        # payloads under the structure-only marker (BatchNorm running stats --
        # training-derived state a weights-free artifact must not carry), so
        # every structure-only save of a buffer-holding model wrote an
        # artifact the M-C2 load coherence gate refused (save-then-cannot-
        # load). Strip buffer payloads to declared geometry (shape/dtype
        # metadata fields are untouched); M-C2/M-C3 stay strict at load.
        return FieldPolicy.DROP
    if field_name == "_buffer_initial_values" and not options.include_buffer_values:
        # R62 buffer extension: pre-forward buffer values shipped at EVERY
        # save level (audit included) with no opt-out. Same whole-channel
        # shape as custom_attributes below: drop entirely, never rewrite.
        return FieldPolicy.DROP
    if field_name == "custom_attributes" and not options.include_custom_attributes:
        # Ungated the harvested module instance attributes were a silent
        # portable-privacy channel (disputed-r2 b8/R62). The gate is
        # whole-channel: drop entirely, never rewrite user values.
        return FieldPolicy.DROP
    if field_name in {"out", "transformed_out"} and not options.include_outs:
        return FieldPolicy.DROP
    if field_name in {"grad", "transformed_grad"} and not options.include_grads:
        return FieldPolicy.DROP
    if field_name in {
        "saved_args",
        "saved_kwargs",
        "out_versions_by_child",
        "forward_args",
        "forward_kwargs",
    }:
        return FieldPolicy.BLOB_RECURSIVE if options.include_saved_args else FieldPolicy.DROP
    if field_name == "func_rng_states":
        return FieldPolicy.BLOB_RECURSIVE if options.include_rng_states else FieldPolicy.DROP
    if _prerelease._ACTIVE:
        # Sprint-gated fields (declared DROP under the current tlspec version)
        # persist with their intended policy ONLY beneath the test-only
        # activation switch; the write then carries the pre-release marker
        # stamped below, so it can never pass as a real artifact.
        override = _prerelease.persisted_policy_override(type(owner), field_name)
        if override is not None:
            return override
    return base_policy


def _scrub_field(
    *,
    owner: Any,
    field_name: str,
    field_value: Any,
    policy: FieldPolicy,
    walk: _ScrubWalk,
    depth: int,
) -> Any:
    """Scrub one object field according to its effective field policy.

    r69 E: an effective ``DROP``/``WEAKREF_STRIP`` wins BEFORE every field-specific
    serializer. The sparse-runnable effective policy lists ``raw_input``/``raw_output``
    in its DROP set, so a runnable save always scrubs both to ``None`` -- including
    when the ordinary ``save_raw_input``/``save_transformed_output`` policy is ``True`` or
    ``"small"``. Pre-r69 the Trace raw-value special case ran first and the generic
    small-value policy retained a nested tensor inside the sparse core, firing the
    (unchanged) ``assert_sparse_core_has_no_tensor_payload`` tripwire on a fully
    witnessed nested-string capture (r68 hon1-F5). Ordinary analysis saves are
    unaffected: their effective raw-field policy is never ``DROP`` here.
    """

    if policy in {FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP}:
        return None
    if isinstance(owner, Trace) and field_name in {"input", "transformed_output"}:
        return _scrub_raw_value_for_save(owner, field_name, field_value, walk)
    if policy == FieldPolicy.STRINGIFY:
        return _stringify_value(field_value)
    if policy == FieldPolicy.BLOB:
        return _blobify_tensor_field(
            owner, field_name, field_value, walk.blob_specs, walk.blob_counter, walk.options
        )
    if policy == FieldPolicy.BLOB_RECURSIVE:
        return _blobify_recursive_value(
            owner=owner,
            field_name=field_name,
            value=field_value,
            options=walk.options,
            memo=walk.memo,
            blob_specs=walk.blob_specs,
            blob_counter=walk.blob_counter,
            depth=depth,
        )
    return _scrub_node(field_value, walk, depth)


_RAW_SAVE_POLICY_ATTR = {
    "input": "_options_save_raw_input",
    "transformed_output": "_options_save_transformed_output",
}


def _scrub_raw_value_for_save(
    owner: Trace,
    field_name: str,
    value: Any,
    walk: _ScrubWalk,
) -> Any:
    """Apply a raw-value save policy before portable metadata serialization.

    Parameters
    ----------
    owner:
        Trace carrying the raw-value save policy.
    field_name:
        Raw-value field being serialized.
    value:
        Raw user input or output metadata to serialize.
    walk:
        Active scrub walk (options, identity memo, blob specs and counter).

    Returns
    -------
    Any
        Scrubbed raw value, a bounded placeholder, or ``None``.
    """

    policy_name = f"save_{field_name}"
    policy = getattr(owner, _RAW_SAVE_POLICY_ATTR.get(field_name, policy_name), "small")
    if policy is False:
        return None
    if policy is True:
        # Scrub with ``stringify_unknown=True`` (not the default
        # passthrough) so live, un-picklable objects nested inside a raw
        # input/output (generators, locks, open file handles, sockets, ...)
        # fall through to ``_stringify_value``'s bounded placeholder instead
        # of reaching ``pickle.dump`` unmodified. This mirrors the
        # ``saved_args``/``forward_args``/``custom_attributes`` policy,
        # which already uses a stringify fallback (via
        # ``_blobify_recursive_value``) for exactly this class of object.
        return _scrub_value(
            value,
            walk.options,
            walk.memo,
            walk.blob_specs,
            walk.blob_counter,
            stringify_unknown=True,
        )
    if policy != "small":
        raise ArtifactError(f"{policy_name} must be 'small', True, or False.")
    return _small_raw_value(value, field_name=field_name)


def _blobify_tensor_field(
    owner: Any,
    field_name: str,
    field_value: Any,
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
    options: _ScrubOptions,
) -> Any:
    """Replace a tensor field with a ``BlobRef`` and record its blob spec."""

    if field_value is None:
        return None
    if not options.payload_materialization and not isinstance(field_value, torch.Tensor):
        return _audit_null_tensor_field(
            owner,
            field_name,
            options,
            reason=f"{options.backend_name}_array_audit_null",
        )
    if not options.payload_codec.can_encode(field_value):
        raise ArtifactError(
            f"{type(owner).__name__}.{field_name} expected a codec-encodable payload for "
            "portable blobification, "
            f"got {type(field_value).__name__}."
        )
    blob_id = _next_blob_id(blob_counter)
    kind = _blob_kind_for_field(owner, field_name)
    label = _blob_label_for_owner(owner)
    tensor_payload = (
        field_value.detach() if isinstance(field_value, torch.nn.Parameter) else field_value
    )
    blob_specs.append(
        BlobSpec(
            blob_id=blob_id,
            value=tensor_payload,
            kind=kind,
            label=label,
            logical_backend=options.backend_name,
        )
    )
    return BlobRef(blob_id=blob_id, kind=kind)


def _audit_null_tensor_field(
    owner: Any,
    field_name: str,
    options: _ScrubOptions,
    *,
    reason: str,
) -> None:
    """Record an audit-only tensor payload that cannot be blobified.

    Parameters
    ----------
    owner:
        Object carrying the payload field.
    field_name:
        Name of the payload field.
    options:
        Active scrub options that collect unsupported tensor records.
    reason:
        Stable manifest reason code.

    Returns
    -------
    None
        The live payload is replaced by ``None`` in scrubbed state.
    """

    options.unsupported_tensor_records.append(
        {
            "owner_type": type(owner).__name__,
            "owner_label": _blob_label_for_owner(owner),
            "field": field_name,
            "kind": _blob_kind_for_field(owner, field_name),
            "reason": reason,
        }
    )
    return None


def _blobify_recursive_value(
    *,
    owner: Any,
    field_name: str,
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
    depth: int = 0,
) -> Any:
    """Blobify tensors recursively inside nested containers.

    This is the PAYLOAD walk: the same per-kind container rules as
    :func:`_scrub_value` (one memo, one cycle discipline, one tensor-key
    refusal), with leaves that blobify every value the payload codec
    recognizes and stringify every other spec-less object
    (:func:`_scrub_object_node`).
    """

    walk = _ScrubWalk(options, memo, blob_specs, blob_counter, payload_owner=(owner, field_name))
    return _scrub_node(value, walk, depth)


def _stringify_value(value: Any) -> str:
    """Convert a non-portable object into a stable placeholder string."""

    return f"<scrubbed:{type(value).__name__}>"


_KNOWN_SAFELY_PICKLABLE_TYPES = (torch.Tensor,)


def _is_safely_picklable(value: Any) -> bool:
    """Return whether ``value`` can be pickled without raising.

    Used by :func:`_scrub_value`'s ``stringify_unknown`` path to decide
    whether an unspecced leaf value (one with no
    ``_PORTABLE_STATE_SPEC``) reached via ``save_raw_input``/
    ``save_transformed_output``'s ``True`` policy should be kept as-is (full
    retention, per those options' docs) or degraded to a bounded
    :func:`_stringify_value` placeholder. Probing picklability once, here,
    at scrub time -- rather than letting the real ``pickle.dump()`` over
    the full scrubbed state discover the problem later -- is what keeps
    live-resource objects (generators, locks, open file handles, sockets,
    ...) from ever reaching that later call, where a failure mid-write can
    otherwise cost an already-saved bundle (see ``bundle.py``'s ``save()``
    atomic-write/backup-restore contract).

    ``torch.Tensor`` and ``numpy.ndarray`` values whose dtype holds no
    object references (numeric, bool, string, and structured dtypes with
    no object-typed field) are exempted from the probe: they always
    support the standard pickle protocol via their own ``__reduce_ex__``
    and hold no live OS resources that could make pickling fail, so
    probing them would only pay a full extra ``pickle.dumps()`` pass --
    doubling CPU and transiently doubling peak memory -- for a result
    that is always ``True``. Any ``numpy.ndarray`` whose dtype reports
    ``hasobject`` (a plain ``object``-dtype array, or a structured/
    compound/nested dtype with an object-typed field), however, is NOT
    exempted: it may hold arbitrary live Python objects (generators,
    locks, or other unpicklable values) in any of its fields, just like a
    plain ``list`` or ``dict`` can, so exempting it would reintroduce
    exactly the hard ``save()`` failure this probe exists to prevent.
    ``dtype.hasobject`` is used rather than a top-level
    ``dtype != np.dtype("object")`` identity check because the latter is
    only true for a bare object dtype -- a structured dtype whose fields
    include ``object`` is never itself equal to ``np.dtype("object")`` and
    would wrongly slip through an identity comparison even though it
    genuinely embeds live object references. Skipping the probe only for
    provably-safe types keeps the live-resource detection intact for
    every value it actually protects.

    Parameters
    ----------
    value:
        Candidate value to probe.

    Returns
    -------
    bool
        ``True`` if ``pickle.dumps(value)`` succeeds, ``False`` otherwise.
    """

    if isinstance(value, _KNOWN_SAFELY_PICKLABLE_TYPES):
        return True
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return True

    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return False
    return True


def _next_blob_id(blob_counter: list[int]) -> str:
    """Allocate the next monotonically increasing zero-padded blob id."""

    blob_counter[0] += 1
    return f"{blob_counter[0]:010d}"


def _blob_kind_for_field(owner: Any, field_name: str) -> str:
    """Map an object field name to the portable manifest tensor kind."""

    if field_name == "out":
        return "out"
    if field_name == "transformed_out":
        return "transformed_out"
    if field_name == "grad":
        return "grad"
    if field_name == "transformed_grad":
        return "transformed_grad"
    if field_name in {"saved_args", "saved_kwargs"}:
        return "captured_arg"
    if field_name == "out_versions_by_child":
        return "child_version"
    if field_name == "func_rng_states":
        return "rng_state"
    if field_name in {"forward_args", "forward_kwargs"}:
        return "module_arg"
    if field_name in {"_args", "_kwargs", "payload"} and type(owner).__name__ in {
        "ModuleInputSnapshot",
        "TensorInputObservation",
    }:
        return "pre_hook_input"
    if field_name == "func_config":
        return "func_config"
    if field_name == "custom_attributes":
        return "module_meta"
    if field_name in {"grad_inputs", "grad_outputs"}:
        return "grad_fn_grad"
    if field_name == "_buffer_initial_values":
        return "buffer_initial_value"
    if field_name == "_annotation_blobs":
        return "annotation_blob"
    if field_name == "_orphan_records":
        return "orphan_payload"
    if field_name == "edge_substitutions":
        # L6 stage 3: tier-(ii) occurrence-granular substituted-value payloads
        # (BLOB_RECURSIVE under the pre-release switch / from the wave-3 bump).
        return "edge_substitution"
    raise ArtifactError(f"No blob kind mapping defined for {type(owner).__name__}.{field_name}.")


def _blob_label_for_owner(owner: Any) -> str:
    """Return the human-readable label stored alongside a blob spec."""

    if hasattr(owner, "label") and getattr(owner, "label") is not None:
        return str(getattr(owner, "label"))
    if hasattr(owner, "call_label") and getattr(owner, "call_label") is not None:
        return str(getattr(owner, "call_label"))
    if hasattr(owner, "address") and getattr(owner, "address") is not None:
        return str(getattr(owner, "address"))
    return type(owner).__name__
