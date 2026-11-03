"""Canonical identities and record joins for portable scrub metadata."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

# Canonical remapped param barcode token (``param_000001``), for R21-1's
# trace-level equivalence-key ordering.
_EQUIV_PARAM_TOKEN = re.compile(r"param_\d{6}")

#: One P-independent scan for the exact live-barcode shape: 8 chars of the
#: barcode alphabet, bounded by non-alphanumerics (barcodes are '_'-joined in
#: identity strings, and '_' is outside the alphabet). See the R29 note at
#: the remap construction site.
_BARCODE_TOKEN_PATTERN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Za-z]{8}(?![0-9A-Za-z])")
_BARCODE_TOKEN_FULLMATCH = re.compile(r"[0-9A-Za-z]{8}").fullmatch


# Essential complexity: one identity pass preserves coupled record, equivalence and autograd joins.
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
    _canonicalize_param_links(params, (*ops, *layers), remap_barcode_text)
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


def _canonicalize_param_links(
    params: list[Any], records: tuple[Any, ...], remap: Callable[[Any], Any]
) -> None:
    """Bind copied parameter edges to their authoritative record before barcodes change.

    Capture operations can retain separate copies of a Param with the same capture-time
    barcode. They are edges to the trace's parameter row, not independent parameter rows.
    An unknown barcode stays distinct and still reaches the writer's strict refusal.
    """

    canonical = {param.barcode: param for param in params if isinstance(param.barcode, str)}
    for record in records:
        references = getattr(record, "_param_logs", None)
        if isinstance(references, (list, tuple)):
            members = [
                canonical.get(getattr(param, "barcode", None), param) for param in references
            ]
            record._param_logs = tuple(members) if isinstance(references, tuple) else members
    for param in params:
        param.barcode = remap(getattr(param, "barcode", None))
