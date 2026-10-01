"""Portable state scrubbing for TorchLens model logs.

This module converts a live ``Trace`` object graph into portable metadata
plus a list of tensor blob specs. It is the save-side counterpart to
rehydration: every class-specific ``PORTABLE_STATE_SPEC`` is applied here so
tensor payloads become ``BlobRef`` placeholders and non-portable live objects
are dropped or stringified before writing ``metadata.pkl``.
"""

from __future__ import annotations

import copy
import functools
import inspect
import logging
import pickle
import re
import warnings
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import torch

from ..constants import MODEL_LOG_FIELD_ORDER
from ..data_classes._state_adapter import state_items, state_new, state_restore
from ..data_classes.trace import Trace, _scrubbed_transform_repr
from ..errors._base import TorchLensWarning
from . import TLSPEC_VERSION, BlobRef, FieldPolicy, TorchLensIOError, prerelease as _prerelease
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
_RAW_INPUT_TEXT_LIMIT = 10_000
_RAW_INPUT_TENSOR_BYTES_LIMIT = 1_000_000
_RAW_OUTPUT_TEXT_LIMIT = _RAW_INPUT_TEXT_LIMIT
_RAW_OUTPUT_TENSOR_BYTES_LIMIT = _RAW_INPUT_TENSOR_BYTES_LIMIT
_RAW_CONTAINER_ITEM_LIMIT = 20
_RAW_INPUT_IMAGE_MAX_EDGE = 256
_RAW_INPUT_IMAGE_BYTES_LIMIT = 256_000
_RAW_IMAGE_SENTINEL = "__torchlens_small_raw_image__"
_PORTABLE_WALK_MAX_DEPTH = 200
_SCRUB_IN_PROGRESS = object()
_LOGGER = logging.getLogger(__name__)


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
        ``trace.backend`` when present.
    payload_materialization:
        Whether this backend can serialize materialized tensor payloads.
    sparse_runnable:
        Whether to apply the runnable sparse-core payload exclusion policy.

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
        structure_only=bool(getattr(trace, "structure_only", False)),
        backend_name=str(backend_name or getattr(trace, "backend", "torch")),
        payload_materialization=payload_materialization,
    )
    options.payload_codec = get_payload_codec(options.backend_name)
    memo: dict[int, Any] = {}
    blob_specs: list[BlobSpec] = []
    blob_counter = [0]

    scrubbed_model = _scrub_value(trace, options, memo, blob_specs, blob_counter)
    if not isinstance(scrubbed_model, Trace):
        raise TorchLensIOError("Portable scrub expected a scrubbed Trace instance.")

    scrubbed_state = dict(state_items(scrubbed_model))
    scrubbed_state["tlspec_version"] = TLSPEC_VERSION

    module_accessor = getattr(trace, "_module_logs", None)
    if module_accessor is not None:
        scrubbed_state["_io_module_accessor_state"] = _scrub_value(
            module_accessor,
            options,
            memo,
            blob_specs,
            blob_counter,
        )
    else:
        scrubbed_state["_io_module_accessor_state"] = None
    _stamp_replacement_evidence(trace, scrubbed_state)
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
        for op in (state.get("layer_list") or ())
        if getattr(op, "func_name", None) == "intervention_replacement"
        or getattr(op, "intervention_replaced", False)
    ]
    if not scrubbed_ops:
        return
    from ..validation._invariants_backward_flow import op_has_genuine_replacement_evidence

    live_ops_by_label = {
        getattr(op, "label", None): op for op in (getattr(trace, "layer_list", None) or ())
    }
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

    ops = list(state.get("layer_list") or ())
    layers = list((state.get("layer_logs") or {}).values())
    params = sorted(
        (state.get("param_logs") or {}).values(),
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
        # save of a multi-pass trace. Layer's PORTABLE_STATE_SPEC has no
        # `parent_param_ops` row; only Op-side records reach the getattr.
        record_spec = getattr(type(record), "PORTABLE_STATE_SPEC", {})
        parent_param_ops = (
            getattr(record, "parent_param_ops", None) if "parent_param_ops" in record_spec else None
        )
        if isinstance(parent_param_ops, dict):
            record.parent_param_ops = {
                remap_barcode_text(key): value for key, value in parent_param_ops.items()
            }

    equivalence_groups = state.get("op_equivalence_classes")
    if isinstance(equivalence_groups, dict):
        state["op_equivalence_classes"] = type(equivalence_groups)(
            (equivalence_class_map.get(key) or canonical_equivalence_key(key), value)
            for key, value in equivalence_groups.items()
        )

    grad_fn_order = list(state.get("grad_fn_order") or ())
    grad_fn_logs = state.get("grad_fn_logs") or {}
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
    backward_pass_logs = state.get("backward_pass_logs")
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
        state["grad_fn_logs"] = remapped_logs
    state["grad_fn_order"] = [remap_grad_id(grad_id) for grad_id in grad_fn_order]
    state["backward_root_grad_fn_object_ids"] = [
        remap_grad_id(grad_id) for grad_id in (state.get("backward_root_grad_fn_object_ids") or ())
    ]
    for record in (*ops, *layers):
        record.grad_fn_object_id = remap_grad_id(getattr(record, "grad_fn_object_id", None))
    for pass_record in pass_records:
        root_ids = getattr(pass_record, "root_grad_fn_ids", None)
        if root_ids:
            pass_record.root_grad_fn_ids = [remap_grad_id(root_id) for root_id in root_ids]

    state["model_object_id"] = 1 if state.get("model_object_id") is not None else None
    state["input_object_id"] = 1 if state.get("input_object_id") is not None else None


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

    from ..data_classes.trace import Conditional, ConditionalAccessor

    if not isinstance(value, MutableMapping):
        return
    accessor = value.get("conditionals")
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
    value["conditionals"] = ConditionalAccessor(detached)


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
            raise TorchLensIOError(
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
# The two mapping kinds sort ABOVE _SCRUB_OBJECT so one ``>=`` test selects "is a
# mapping" (which shares the key-payload refusal) before splitting on flavour.
_SCRUB_ORDERED_DICT = 8
_SCRUB_MAPPING = 9

_SCRUB_VALUE_KINDS: weakref.WeakKeyDictionary[type, int] = weakref.WeakKeyDictionary()


def _scrub_value_kind(value_type: type) -> int:
    """Classify one node type for :func:`_scrub_value`, memoized per type.

    The branch order below is exactly the ``isinstance`` chain it replaces:
    ``torch.Size`` before ``tuple`` (it is a tuple subclass), and
    ``OrderedDict``/``defaultdict`` before plain ``dict`` so their observable
    ordering and default-factory behavior survive a round trip.
    """

    if issubclass(value_type, _SIMPLE_KEEP_TYPES):
        kind = _SCRUB_SIMPLE
    elif issubclass(value_type, torch.Size):
        kind = _SCRUB_SIZE
    elif issubclass(value_type, BlobRef):
        kind = _SCRUB_BLOBREF
    elif issubclass(value_type, list):
        kind = _SCRUB_LIST
    elif issubclass(value_type, tuple):
        kind = _SCRUB_TUPLE
    elif issubclass(value_type, set):
        kind = _SCRUB_SET
    elif issubclass(value_type, frozenset):
        kind = _SCRUB_FROZENSET
    elif issubclass(value_type, OrderedDict):
        kind = _SCRUB_ORDERED_DICT
    elif issubclass(value_type, dict):
        kind = _SCRUB_MAPPING
    else:
        kind = _SCRUB_OBJECT
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
        from ..utils._callable_safety import is_inert_first_party_callable
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
    if value_type is tuple:
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
        nor a ``PORTABLE_STATE_SPEC`` falls back to
        :func:`_stringify_value` instead of being returned unchanged. Used
        by the ``save_raw_input``/``save_raw_output`` ``True`` policy
        (:func:`_scrub_raw_value_for_save`) so live, un-picklable objects
        (generators, locks, open file handles, sockets, ...) nested inside
        a user's raw input/output never reach ``pickle.dump`` unmodified.
        Default ``False`` preserves the general-purpose scrub pass used for
        the rest of the ``Trace`` object graph.
    """

    if _depth > _PORTABLE_WALK_MAX_DEPTH:
        raise TorchLensIOError(
            f"Portable metadata exceeds the maximum depth of {_PORTABLE_WALK_MAX_DEPTH}."
        )

    # One cached type lookup replaces the up-to-ten ``isinstance`` chain this
    # branch table used to re-run for every one of the ~225k nodes a ResNet scrub
    # visits. :func:`_scrub_value_kind` resolves the branches in the identical
    # order, so the selected branch is unchanged.
    kind = _SCRUB_VALUE_KINDS.get(type(value))
    if kind is None:
        kind = _scrub_value_kind(type(value))
    if kind == _SCRUB_SIMPLE:
        return value
    if kind == _SCRUB_SIZE:
        return torch.Size(value)
    if kind == _SCRUB_BLOBREF:
        return value
    if kind == _SCRUB_LIST:
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        if type(value) is not list:
            # Symmetry with the tuple-subclass downgrade disclosure (R10-9): a
            # list/set/frozenset subclass rebuilds as a plain builtin, silently
            # losing its type; disclose it once per type like tuples do.
            _disclose_container_downgrade(
                options,
                value,
                "Dropping non-portable list subclass {name} in portable metadata; "
                "it rebuilds as a plain list.",
            )
        rebuilt_list: list[Any] = []
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt_list
        rebuilt_list.extend(
            _scrub_value(
                item,
                options,
                memo,
                blob_specs,
                blob_counter,
                stringify_unknown,
                _depth + 1,
            )
            for item in value
        )
        return rebuilt_list
    if kind == _SCRUB_TUPLE:
        obj_id = id(value)
        cached = memo.get(obj_id)
        if cached is _SCRUB_IN_PROGRESS:
            raise TorchLensIOError("Portable metadata contains a cycle through a tuple.")
        if cached is not None:
            return cached
        _pin_in_memo(memo, value)
        memo[obj_id] = _SCRUB_IN_PROGRESS
        try:
            rebuilt_tuple = _rebuild_tuple_value(
                value,
                (
                    _scrub_value(
                        item,
                        options,
                        memo,
                        blob_specs,
                        blob_counter,
                        stringify_unknown,
                        _depth + 1,
                    )
                    for item in value
                ),
                options,
            )
        except BaseException:
            memo.pop(obj_id, None)
            raise
        memo[obj_id] = rebuilt_tuple
        return rebuilt_tuple
    if kind == _SCRUB_SET:
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        if type(value) is not set:
            _disclose_container_downgrade(
                options,
                value,
                "Dropping non-portable set subclass {name} in portable metadata; "
                "it rebuilds as a plain set.",
            )
        rebuilt_set: set[Any] = set()
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt_set
        rebuilt_set.update(
            _scrub_value(
                item,
                options,
                memo,
                blob_specs,
                blob_counter,
                stringify_unknown,
                _depth + 1,
            )
            for item in value
        )
        return rebuilt_set
    if kind == _SCRUB_FROZENSET:
        obj_id = id(value)
        cached = memo.get(obj_id)
        if cached is _SCRUB_IN_PROGRESS:
            raise TorchLensIOError("Portable metadata contains a cycle through a frozenset.")
        if cached is not None:
            return cached
        if type(value) is not frozenset:
            _disclose_container_downgrade(
                options,
                value,
                "Dropping non-portable frozenset subclass {name} in portable "
                "metadata; it rebuilds as a plain frozenset.",
            )
        _pin_in_memo(memo, value)
        memo[obj_id] = _SCRUB_IN_PROGRESS
        try:
            rebuilt_frozenset = frozenset(
                _scrub_value(
                    item,
                    options,
                    memo,
                    blob_specs,
                    blob_counter,
                    stringify_unknown,
                    _depth + 1,
                )
                for item in value
            )
        except BaseException:
            memo.pop(obj_id, None)
            raise
        memo[obj_id] = rebuilt_frozenset
        return rebuilt_frozenset
    if kind >= _SCRUB_ORDERED_DICT:
        # Every mapping kind: refuse tensor-payload keys before the type-specific
        # branches rebuild the mapping so a payload cannot slip into
        # ``metadata.pkl`` unscrubbed and un-inventoried.
        _reject_payload_mapping_keys(value, options)
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        _pin_in_memo(memo, value)
        if isinstance(value, defaultdict):
            rebuilt: defaultdict[Any, Any] = defaultdict(_portable_default_factory(value, options))
            memo[obj_id] = rebuilt
            for key, item in value.items():
                rebuilt[key] = _scrub_value(
                    item,
                    options,
                    memo,
                    blob_specs,
                    blob_counter,
                    stringify_unknown,
                    _depth + 1,
                )
            return rebuilt
        if kind == _SCRUB_ORDERED_DICT:
            rebuilt_ordered: OrderedDict[Any, Any] = OrderedDict()
            memo[obj_id] = rebuilt_ordered
            for key, item in value.items():
                rebuilt_ordered[key] = _scrub_value(
                    item,
                    options,
                    memo,
                    blob_specs,
                    blob_counter,
                    stringify_unknown,
                    _depth + 1,
                )
            return rebuilt_ordered
        rebuilt_mapping: dict[Any, Any] = {}
        memo[obj_id] = rebuilt_mapping
        for key, item in value.items():
            rebuilt_mapping[key] = _scrub_value(
                item,
                options,
                memo,
                blob_specs,
                blob_counter,
                stringify_unknown,
                _depth + 1,
            )
        return rebuilt_mapping

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is None:
        if stringify_unknown and not _is_safely_picklable(value):
            return _stringify_value(value)
        return value
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]

    scrubbed_obj = state_new(type(value))
    _pin_in_memo(memo, value)
    memo[obj_id] = scrubbed_obj
    scrubbed_state: dict[str, Any] = {}
    owner_is_trace = isinstance(value, Trace)
    for field_name, field_value in _state_items_for_scrub(value, spec):
        if owner_is_trace and _is_runtime_only_trace_field(field_name):
            continue
        if field_name not in spec:
            # A ``functools.cached_property`` read caches its value in the
            # instance ``__dict__`` under the property's own name (e.g. the
            # public ``Trace.intervention_spec`` accessor). Those cells are
            # DERIVED state that rebuilds on access, never portable fields,
            # and they appear only after a read -- refusing them here made a
            # read-only public property poison every later ``tl.save``
            # (B3R4-R10-1). Skip the whole class structurally; the check runs
            # only on the refusal path, so the hot field loop pays nothing.
            if isinstance(
                inspect.getattr_static(type(value), field_name, None),
                functools.cached_property,
            ):
                continue
            # Teach at the point of failure: an undeclared ``_<name>_cache``
            # cell is almost always DERIVED state poked into ``__dict__`` by a
            # lazily-caching public accessor read (the ModuleCall/Module
            # ``.facets`` incident) -- name that accessor when it exists so the
            # message points at the read that poisoned the save, not just an
            # internal field the user never heard of.
            accessor_hint = ""
            if field_name.startswith("_") and field_name.endswith("_cache"):
                accessor_name = field_name[1 : -len("_cache")]
                if isinstance(
                    inspect.getattr_static(type(value), accessor_name, None),
                    property,
                ):
                    accessor_hint = (
                        f" This cell was populated by reading the public "
                        f"`.{accessor_name}` accessor; a lazy derived cache "
                        f"must be declared FieldPolicy.DROP in "
                        f"{type(value).__name__}.PORTABLE_STATE_SPEC so a "
                        f"read-only access cannot poison a later save."
                    )
            # Same lesson for LEDGERED-but-undeclared Trace transients (the
            # draw() `_last_encoding_state` incident: ledgered as
            # "scrub-declared runtime-only" while no declaration existed) --
            # quote the ledger row so the message names the writer.
            if owner_is_trace and not accessor_hint:
                from ..data_classes._trace_components import (
                    TRACE_EXTERNAL_WRITE_EXEMPTIONS,
                )

                ledger_reason = TRACE_EXTERNAL_WRITE_EXEMPTIONS.get(field_name)
                if ledger_reason is not None:
                    accessor_hint = (
                        f" This field is ledgered in "
                        f"TRACE_EXTERNAL_WRITE_EXEMPTIONS as: {ledger_reason!r}."
                        f" The ledger documents the write; it is not a scrub "
                        f"policy. Enroll the runtime-only transient in the "
                        f"scrub's runtime-only set (or give it a "
                        f"FieldPolicy.DROP row) so populating it cannot "
                        f"poison a later save."
                    )
            raise TorchLensIOError(
                f"{type(value).__name__}.{field_name} is missing from "
                f"PORTABLE_STATE_SPEC. Every live state field needs an "
                f"explicit portability policy before it can be saved."
                f"{accessor_hint}"
            )
        if field_name == "_is_in_conditional_body" and field_value is None:
            field_value = False
        if field_name == "_capture_outcome" and field_value is not None:
            # The settled capture outcome persists as its STRING-ONLY payload
            # (tlspec v7): the safe-unpickle allowlist never needs the record
            # class, and loads parse the payload against closed vocabularies.
            field_value = field_value.to_payload() if hasattr(field_value, "to_payload") else None
        policy = _effective_policy(value, field_name, spec[field_name], options)
        # The two overwhelmingly common policies are resolved here instead of
        # through :func:`_scrub_field`, which is one Python call per field on a walk
        # that scrubs ~160k fields per ResNet save. Both shortcuts reproduce
        # ``_scrub_field``'s own branch order exactly: its DROP/WEAKREF_STRIP test
        # runs first, and KEEP falls through every field-specific serializer to the
        # generic ``_scrub_value`` -- except for a Trace raw input/output field,
        # which keeps routing through the full function.
        if policy is FieldPolicy.DROP or policy is FieldPolicy.WEAKREF_STRIP:
            scrubbed_state[field_name] = None
            continue
        if policy is FieldPolicy.KEEP and not (
            owner_is_trace and (field_name == "raw_input" or field_name == "raw_output")
        ):
            scrubbed_state[field_name] = _scrub_value(
                field_value,
                options,
                memo,
                blob_specs,
                blob_counter,
                _depth=_depth + 1,
            )
            continue
        scrubbed_state[field_name] = _scrub_field(
            owner=value,
            field_name=field_name,
            field_value=field_value,
            policy=policy,
            options=options,
            memo=memo,
            blob_specs=blob_specs,
            blob_counter=blob_counter,
            depth=_depth + 1,
        )

    if isinstance(value, Trace):
        # B8-20: a functools.partial repr embeds its bound argument VALUES, so the
        # scrubbed persistence repr redacts them.
        scrubbed_state["_activation_transform_repr"] = _scrubbed_transform_repr(
            value.activation_transform
        )
        # P7/R10: a capture TorchLens itself refused to bless must not round-trip
        # into "no claim". The NEGATIVE disclosure persists as a string-only row
        # (mirroring _capture_outcome's treatment); True/None stay session-time,
        # so a loaded artifact can never CLAIM verification -- the row can only
        # ever worsen a verdict, preserving the monotonicity the runnable side
        # enforces structurally. rescue_rerun stays session-time as documented.
        if getattr(value, "capture_verified", None) is False:
            reason = getattr(value, "capture_verification_reason", None)
            scrubbed_state["_capture_verification"] = {
                "verified": False,
                "reason": str(reason) if reason is not None else None,
            }
        scrubbed_state["tlspec_version"] = TLSPEC_VERSION
        if _prerelease._ACTIVE:
            # EVERY switch-on write is marked, whether or not a gated field
            # actually persisted in this state -- no per-field accounting can
            # omit the marker, so switched and real-version artifacts are
            # never indistinguishable (loads refuse the marker typed unless
            # the switch is active; see torchlens._io.prerelease).
            scrubbed_state[_prerelease.PRERELEASE_STATE_KEY] = (
                _prerelease.prerelease_marker_payload()
            )
        else:
            # Registered pre-release ANNOTATIONS sub-keys are new persistence
            # write paths inside the already-persisting annotations mapping
            # (e.g. the episode ledger): they never ride a real artifact of
            # the frozen tlspec version. The pop acts on the scrubbed copy
            # only -- the live trace keeps its session-time annotations.
            gated_annotation_keys = _prerelease.gated_annotations_keys()
            if gated_annotation_keys:
                annotations_state = scrubbed_state.get("annotations")
                if isinstance(annotations_state, dict):
                    for gated_key in gated_annotation_keys:
                        annotations_state.pop(gated_key, None)
        _apply_advisory_source_policy(scrubbed_state, options)
        _apply_source_metadata_policy(scrubbed_state, options)
        _apply_trace_blob_policy(scrubbed_state, options)
    # ``FuncCallLocation`` is matched by name rather than ``isinstance`` to avoid
    # an import cycle between ``_io`` and ``data_classes``; it is a unique
    # internal class with no subclasses.
    elif type(value).__name__ == "FuncCallLocation":
        _apply_frame_source_policy(scrubbed_state, options)
    # ``Module`` logs carry the same per-module class/init/forward source
    # metadata (docstrings + absolute source files) as the Trace. Dispatch on the
    # field signature rather than the class so any source-metadata owner is
    # covered without an ``_io`` -> ``data_classes`` import cycle.
    elif "class_docstring" in scrubbed_state:
        _apply_source_metadata_policy(scrubbed_state, options)
    # ``ConditionalEvent`` records (in ``conditional_records``) carry the absolute
    # path of the user's forward-defining module in ``source_file``. Dispatch on the
    # field signature (the ``ConditionalEvent`` name is shared with a capture-time
    # event class) so only the persisted, spec-bearing record is relativized/dropped.
    elif "source_file" in scrubbed_state and "branch_ranges" in scrubbed_state:
        _apply_conditional_source_policy(scrubbed_state, options, drop_value="")
    # ``Conditional`` records (in the public ``conditionals`` accessor) carry the
    # same absolute path in an OPTIONAL ``source_file``; dropped to ``None``.
    elif "source_file" in scrubbed_state and "arms" in scrubbed_state:
        _apply_conditional_source_policy(scrubbed_state, options, drop_value=None)

    return state_restore(scrubbed_obj, scrubbed_state)


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
        ``PORTABLE_STATE_SPEC`` order, then any still-unseen fields so the
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
        "_module_entry_adoptions",
        "_last_sibling_ordering_decision",
        # Sibling render diagnostic (same _render_dot write site as the row
        # above); left unenrolled, ONE draw() poisoned every later tl.save.
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
        # `_source_bundle_{path,manifest_sha256}` and
        # `_retain_layers_to_save_output_parents` moved out of this allowance
        # into declared `Trace.PORTABLE_STATE_SPEC` DROP rows. This allowance is
        # consulted BEFORE the spec, so an entry here makes the declared policy
        # dead code.
        "jax_closed_jaxpr",
        "jax_equation_captures",
        "jax_ordered_captures",
        "jax_region_captures",
        "_jax_capture_index_to_raw_op_label",
        "jax_capture_index_to_final_op_label",
        "jax_inlined_call_primitives",
        "jax_outvar_key_to_capture_index",
        "jax_static_argnums",
        "_selective_save_hidden_payloads",
        # B1-02: the four semantic-output scratch names moved OUT of this
        # allowance and into `Trace.PORTABLE_STATE_SPEC` as declared
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
        "orphan_records",
        "out",
        "out_versions_by_child",
        "parent_params",
        "payload",
        "raw_input",
        "raw_output",
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
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
    depth: int,
) -> Any:
    """Scrub one object field according to its effective field policy.

    r69 E: an effective ``DROP``/``WEAKREF_STRIP`` wins BEFORE every field-specific
    serializer. The sparse-runnable effective policy lists ``raw_input``/``raw_output``
    in its DROP set, so a runnable save always scrubs both to ``None`` -- including
    when the ordinary ``save_raw_input``/``save_raw_output`` policy is ``True`` or
    ``"small"``. Pre-r69 the Trace raw-value special case ran first and the generic
    small-value policy retained a nested tensor inside the sparse core, firing the
    (unchanged) ``assert_sparse_core_has_no_tensor_payload`` tripwire on a fully
    witnessed nested-string capture (r68 hon1-F5). Ordinary analysis saves are
    unaffected: their effective raw-field policy is never ``DROP`` here.
    """

    if policy in {FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP}:
        return None
    if isinstance(owner, Trace) and field_name in {"raw_input", "raw_output"}:
        return _scrub_raw_value_for_save(
            owner,
            field_name,
            field_value,
            options,
            memo,
            blob_specs,
            blob_counter,
        )
    if policy == FieldPolicy.STRINGIFY:
        return _stringify_value(field_value)
    if policy == FieldPolicy.BLOB:
        return _blobify_tensor_field(
            owner, field_name, field_value, blob_specs, blob_counter, options
        )
    if policy == FieldPolicy.BLOB_RECURSIVE:
        return _blobify_recursive_value(
            owner=owner,
            field_name=field_name,
            value=field_value,
            options=options,
            memo=memo,
            blob_specs=blob_specs,
            blob_counter=blob_counter,
            depth=depth,
        )
    return _scrub_value(
        field_value,
        options,
        memo,
        blob_specs,
        blob_counter,
        _depth=depth,
    )


def _scrub_raw_value_for_save(
    owner: Trace,
    field_name: str,
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
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
    options:
        Active scrub options.
    memo:
        Object-identity memo for recursive scrubbing.
    blob_specs:
        Accumulated tensor blob specs.
    blob_counter:
        Mutable blob id counter.

    Returns
    -------
    Any
        Scrubbed raw value, a bounded placeholder, or ``None``.
    """

    policy_name = f"save_{field_name}"
    policy = getattr(owner, policy_name, "small")
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
        return _scrub_value(value, options, memo, blob_specs, blob_counter, stringify_unknown=True)
    if policy != "small":
        raise TorchLensIOError(f"{policy_name} must be 'small', True, or False.")
    return _small_raw_value(value, field_name=field_name)


def _small_raw_value(value: Any, *, field_name: str) -> Any:
    """Return a bounded portable representation of a raw value.

    Parameters
    ----------
    value:
        Raw user input or output metadata.
    field_name:
        Raw-value field being serialized.

    Returns
    -------
    Any
        Truncated string, small tensor, bounded image bytes, recursively
        bounded container, or ``None`` when the value is too large or
        unsupported. PIL images under ``save_raw_input="small"`` are copied,
        downsampled to at most 256 px on the longest edge, encoded as PNG or
        JPEG bytes, and dropped if the encoded payload exceeds 256 KB.
    """

    if value is None:
        return None
    if isinstance(value, str):
        text_limit = _RAW_OUTPUT_TEXT_LIMIT if field_name == "raw_output" else _RAW_INPUT_TEXT_LIMIT
        return value[:text_limit]
    if isinstance(value, torch.Tensor):
        tensor_bytes = value.nelement() * value.element_size()
        tensor_limit = (
            _RAW_OUTPUT_TENSOR_BYTES_LIMIT
            if field_name == "raw_output"
            else _RAW_INPUT_TENSOR_BYTES_LIMIT
        )
        if tensor_bytes <= tensor_limit:
            return value
        _LOGGER.debug(
            "Dropping %s tensor over small-policy cap: %s bytes", field_name, tensor_bytes
        )
        return None
    if field_name == "raw_input":
        small_image = _small_raw_image_value(value)
        if small_image is not None:
            return small_image
    if isinstance(value, list):
        return [
            _small_raw_value(item, field_name=field_name)
            for item in value[:_RAW_CONTAINER_ITEM_LIMIT]
        ]
    if isinstance(value, tuple):
        return tuple(
            _small_raw_value(item, field_name=field_name)
            for item in value[:_RAW_CONTAINER_ITEM_LIMIT]
        )
    if isinstance(value, dict):
        return {
            key: _small_raw_value(item, field_name=field_name)
            for key, item in list(value.items())[:_RAW_CONTAINER_ITEM_LIMIT]
        }
    _LOGGER.debug(
        "Dropping unsupported %s type under small policy: %s",
        field_name,
        type(value).__name__,
    )
    return None


def _small_raw_image_value(value: Any) -> dict[str, Any] | None:
    """Return bounded bytes for a PIL image under the small raw-input policy.

    Parameters
    ----------
    value:
        Candidate raw input value.

    Returns
    -------
    dict[str, Any] | None
        Sentinel dictionary containing encoded image bytes and metadata, or
        ``None`` when ``value`` is not a PIL image or cannot fit the cap.
    """

    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return None
    if not isinstance(value, PILImage):
        return None

    image = value.copy()
    original_size = tuple(image.size)
    image.thumbnail((_RAW_INPUT_IMAGE_MAX_EDGE, _RAW_INPUT_IMAGE_MAX_EDGE))
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    encoded = _encode_small_raw_image(image, fmt="PNG")
    image_format = "PNG"
    if encoded is None and image.mode != "RGBA":
        encoded = _encode_small_raw_image(image.convert("RGB"), fmt="JPEG")
        image_format = "JPEG"
    if encoded is None:
        _LOGGER.debug("Dropping raw_input PIL image over small-policy cap.")
        return None
    return {
        _RAW_IMAGE_SENTINEL: True,
        "format": image_format,
        "mode": image.mode,
        "size": tuple(image.size),
        "original_size": original_size,
        "max_edge": _RAW_INPUT_IMAGE_MAX_EDGE,
        "bytes_limit": _RAW_INPUT_IMAGE_BYTES_LIMIT,
        "data": encoded,
    }


def _encode_small_raw_image(image: Any, *, fmt: str) -> bytes | None:
    """Encode a PIL image and enforce the small-policy byte cap.

    Parameters
    ----------
    image:
        PIL image to encode.
    fmt:
        Pillow format name.

    Returns
    -------
    bytes | None
        Encoded bytes when they fit the cap, otherwise ``None``.
    """

    buffer = BytesIO()
    save_kwargs: dict[str, Any] = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs.update({"quality": 85, "optimize": True})
    image.save(buffer, **save_kwargs)
    data = buffer.getvalue()
    if len(data) > _RAW_INPUT_IMAGE_BYTES_LIMIT:
        return None
    return data


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
        raise TorchLensIOError(
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
    """Blobify tensors recursively inside nested containers."""

    if depth > _PORTABLE_WALK_MAX_DEPTH:
        raise TorchLensIOError(
            f"Portable payload exceeds the maximum depth of {_PORTABLE_WALK_MAX_DEPTH}."
        )

    def recurse(item: Any) -> Any:
        """Blobify one child at the next portable-walk depth."""

        return _blobify_recursive_value(
            owner=owner,
            field_name=field_name,
            value=item,
            options=options,
            memo=memo,
            blob_specs=blob_specs,
            blob_counter=blob_counter,
            depth=depth + 1,
        )

    if isinstance(value, _SIMPLE_KEEP_TYPES):
        return value
    if isinstance(value, torch.Size):
        return torch.Size(value)
    if isinstance(value, BlobRef):
        return value
    if options.payload_codec.can_encode(value):
        return _blobify_tensor_field(owner, field_name, value, blob_specs, blob_counter, options)
    if isinstance(value, list):
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        rebuilt_list: list[Any] = []
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt_list
        rebuilt_list.extend(recurse(item) for item in value)
        return rebuilt_list
    if isinstance(value, tuple):
        obj_id = id(value)
        cached = memo.get(obj_id)
        if cached is _SCRUB_IN_PROGRESS:
            raise TorchLensIOError("Portable payload contains a cycle through a tuple.")
        if cached is not None:
            return cached
        _pin_in_memo(memo, value)
        memo[obj_id] = _SCRUB_IN_PROGRESS
        try:
            rebuilt_tuple = _rebuild_tuple_value(value, (recurse(item) for item in value), options)
        except BaseException:
            memo.pop(obj_id, None)
            raise
        memo[obj_id] = rebuilt_tuple
        return rebuilt_tuple
    if isinstance(value, dict):
        # ``dict`` covers ``OrderedDict`` / ``defaultdict``; refuse tensor-payload
        # keys here (the value path below blobifies tensor VALUES, so a tensor KEY
        # would otherwise bypass the blob manifest and body index entirely).
        _reject_payload_mapping_keys(value, options)
    if isinstance(value, OrderedDict):
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        rebuilt_ordered: OrderedDict[Any, Any] = OrderedDict()
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt_ordered
        for key, item in value.items():
            rebuilt_ordered[key] = recurse(item)
        return rebuilt_ordered
    if isinstance(value, defaultdict):
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        rebuilt: defaultdict[Any, Any] = defaultdict(_portable_default_factory(value, options))
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt
        for key, item in value.items():
            rebuilt[key] = recurse(item)
        return rebuilt
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        rebuilt_mapping: dict[Any, Any] = {}
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt_mapping
        for key, item in value.items():
            rebuilt_mapping[key] = recurse(item)
        return rebuilt_mapping
    if isinstance(value, set):
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        rebuilt_set: set[Any] = set()
        _pin_in_memo(memo, value)
        memo[obj_id] = rebuilt_set
        rebuilt_set.update(recurse(item) for item in value)
        return rebuilt_set
    if isinstance(value, frozenset):
        obj_id = id(value)
        cached = memo.get(obj_id)
        if cached is _SCRUB_IN_PROGRESS:
            raise TorchLensIOError("Portable payload contains a cycle through a frozenset.")
        if cached is not None:
            return cached
        _pin_in_memo(memo, value)
        memo[obj_id] = _SCRUB_IN_PROGRESS
        try:
            rebuilt_frozenset = frozenset(recurse(item) for item in value)
        except BaseException:
            memo.pop(obj_id, None)
            raise
        memo[obj_id] = rebuilt_frozenset
        return rebuilt_frozenset

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is not None:
        return _scrub_value(
            value,
            options,
            memo,
            blob_specs,
            blob_counter,
            _depth=depth,
        )
    return _stringify_value(value)


def _stringify_value(value: Any) -> str:
    """Convert a non-portable object into a stable placeholder string."""

    return f"<scrubbed:{type(value).__name__}>"


_KNOWN_SAFELY_PICKLABLE_TYPES = (torch.Tensor,)


def _is_safely_picklable(value: Any) -> bool:
    """Return whether ``value`` can be pickled without raising.

    Used by :func:`_scrub_value`'s ``stringify_unknown`` path to decide
    whether an unspecced leaf value (one with no
    ``PORTABLE_STATE_SPEC``) reached via ``save_raw_input``/
    ``save_raw_output``'s ``True`` policy should be kept as-is (full
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
    if field_name == "orphan_records":
        return "orphan_payload"
    if field_name == "edge_substitutions":
        # L6 stage 3: tier-(ii) occurrence-granular substituted-value payloads
        # (BLOB_RECURSIVE under the pre-release switch / from the wave-3 bump).
        return "edge_substitution"
    raise TorchLensIOError(f"No blob kind mapping defined for {type(owner).__name__}.{field_name}.")


def _blob_label_for_owner(owner: Any) -> str:
    """Return the human-readable label stored alongside a blob spec."""

    if hasattr(owner, "label") and getattr(owner, "label") is not None:
        return str(getattr(owner, "label"))
    if hasattr(owner, "call_label") and getattr(owner, "call_label") is not None:
        return str(getattr(owner, "call_label"))
    if hasattr(owner, "address") and getattr(owner, "address") is not None:
        return str(getattr(owner, "address"))
    return type(owner).__name__
