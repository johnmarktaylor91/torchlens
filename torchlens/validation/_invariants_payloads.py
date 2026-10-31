"""Backend-neutral edges and payload metadata invariants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _dtype_values_match,
        _resolve_trace_label,
        op_has_genuine_replacement_evidence,
    )

__all__ = (
    "_check_backend_neutral_graph_topology",
    "_check_edge_use_parent_arg_invariants",
    "_check_op_log_fields",
    "_check_payload_metadata_invariants",
    "_live_payload_value",
    "_check_live_payload_metadata",
    "_payload_shape",
    "_payload_dtype",
    "_payload_memory",
)


def _check_backend_neutral_graph_topology(ml: Trace) -> None:
    """Check parent/child symmetry for non-torch traces where fields exist.

    Parameters
    ----------
    ml:
        Postprocessed non-torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a populated parent or child list references a missing layer or lacks
        the reciprocal edge.
    """

    name = "backend_neutral_graph_topology"
    labels = {
        getattr(layer, "label", getattr(layer, "layer_label", ""))
        for layer in getattr(ml, "_layer_list", ())
    } | {
        getattr(layer, "layer_label", getattr(layer, "label", ""))
        for layer in getattr(ml, "_layer_list", ())
    }
    for layer in getattr(ml, "_layer_list", ()):
        label = getattr(layer, "layer_label", getattr(layer, "label", type(layer).__name__))
        parents = list(getattr(layer, "parents", ()) or ())
        children = list(getattr(layer, "children", ()) or ())
        for parent_label in parents:
            if parent_label not in labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} has parent {parent_label!r} outside trace labels",
                )
            parent = ml[parent_label]
            parent_children = set(getattr(parent, "children", ()) or ())
            if (
                label not in parent_children
                and getattr(layer, "label", label) not in parent_children
            ):
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {parent_label!r} as parent, but reciprocal child is missing",
                )
        for child_label in children:
            if child_label not in labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} has child {child_label!r} outside trace labels",
                )
            child = ml[child_label]
            child_parents = set(getattr(child, "parents", ()) or ())
            if label not in child_parents and getattr(layer, "label", label) not in child_parents:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {child_label!r} as child, but reciprocal parent is missing",
                )


def _check_edge_use_parent_arg_invariants(ml: Trace) -> None:
    """Check existing edge-use records and parent-arg references.

    Precondition contract: edge-use metadata is optional on torch graph edges.
    The torch eager builder emits ``_edge_uses`` only for args/kwargs-derived
    parent entries. Buffer-source, output, control, module, and
    intervention-injected edges may legitimately have no edge-use record. When
    an ``_edge_uses`` record exists, its kind must be in ``EdgeUseKind`` and
    its parent/child labels must resolve. When a ``parent_arg_positions`` entry
    exists, its referenced parent label must resolve. This invariant never
    asserts that every parent edge has a corresponding edge-use record.

    Edge-occurrence MULTIPLICITY witness (b9-opus R75-1): on live frozen
    torch traces the canonical CSR dataflow edge-occurrence table
    (``_trace_core/relations.py``) is additionally cross-checked per
    (child, parent) against the independently stored roots -- the row's
    ``parents`` entries plus its per-position ``parent_arg_positions`` map.
    The correspondence is EXACT EQUALITY by construction of the freeze
    (``relation_views.py`` pass 2 emits, for every parents-list entry that
    resolves, one occurrence per attributed arg position naming that parent,
    or exactly one when no position names it), and it was verified
    empirically on plain MLPs, genuine double consumption (``h + h``, which
    legitimately records parallel occurrences), same-tensor multi-position
    ops, recurrent loops (pass-qualified parent labels are unresolved on
    BOTH sides, so they cancel), and buffer-carrying models. Genuine
    parallel edges therefore stay green, while DUPLICATING one occurrence
    in the edge table -- invisible to every deduped label view and to the
    per-position arg map -- breaks the count equality and fails here.
    Loaded, preview, and detached traces carry no CSR edge table
    (``dataflow_edges is None``), so the multiplicity witness is vacuous
    there by precondition, exactly like the freeze it mirrors.

    Parameters
    ----------
    ml:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If populated edge-use or parent-arg-position metadata is malformed,
        references labels that do not resolve, or (live frozen traces) the
        canonical edge table's per-(child, parent) occurrence counts disagree
        with the parents/parent-arg-position roots.
    """

    name = "edge_use_parent_arg_consistency"
    valid_edge_uses = {"arg", "kwarg", "container", "module", "buffer", "output", "control"}
    valid_arg_kinds = {"positional", "keyword"}
    for layer in ml._layer_list:
        layer_label = getattr(layer, "layer_label", type(layer).__name__)
        for record in getattr(layer, "_edge_uses", ()) or ():
            edge_use = getattr(record, "edge_use", None)
            if edge_use not in valid_edge_uses:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has invalid edge_use kind {edge_use!r}",
                )
            arg_kind = getattr(record, "arg_kind", None)
            if arg_kind not in valid_arg_kinds:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has invalid edge arg_kind {arg_kind!r}",
                )
            parent_label = getattr(record, "parent_label", None)
            if not isinstance(parent_label, str) or _resolve_trace_label(ml, parent_label) is None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has edge-use record with unresolved parent "
                    f"{parent_label!r}",
                )
            child_label = getattr(record, "child_label", None)
            if not isinstance(child_label, str) or _resolve_trace_label(ml, child_label) is None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has edge-use record with unresolved child "
                    f"{child_label!r}",
                )

        parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
        if not isinstance(parent_arg_positions, Mapping):
            raise MetadataInvariantError(
                name,
                f"Layer '{layer_label}' has non-mapping parent_arg_positions",
            )
        for arg_domain in ("args", "kwargs"):
            entries = parent_arg_positions.get(arg_domain, {}) or {}
            if not isinstance(entries, Mapping):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}] is not a mapping",
                )
            for position, parent_label in entries.items():
                if not isinstance(parent_label, str):
                    raise MetadataInvariantError(
                        name,
                        f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}]"
                        f"[{position!r}] is not a label string",
                    )
                if _resolve_trace_label(ml, parent_label) is None:
                    raise MetadataInvariantError(
                        name,
                        f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}]"
                        f"[{position!r}] references missing parent {parent_label!r}",
                    )

    # Edge-occurrence MULTIPLICITY witness (see docstring). The label views
    # dedup and the arg map is per-position, so a duplicated occurrence in
    # the CSR table changes NEITHER recorded root -- only a per-(child,
    # parent) COUNT comparison against those roots can see it.
    core = getattr(ml, "_trace_core", None)
    store = getattr(core, "ops", None) if core is not None else None
    edges = getattr(store, "dataflow_edges", None) if store is not None else None
    ref_labels = getattr(store, "ref_labels", None) if store is not None else None
    label_rows = getattr(core, "label_rows", None) if core is not None else None
    if edges is None or ref_labels is None or not label_rows:
        return
    # Iterate PER-PASS op records, never the layer aggregate: on a
    # multi-pass layer the bare label resolves to ONE pass's row while the
    # layer facade mirrors a different pass's parents, so pairing
    # ``label_rows`` keys with ``layer_dict_all_keys`` entries compared
    # pass-1 edges against pass-3 roots and false-failed honest recurrent
    # traces (caught by the validation-decision golden's TinyRecurrent).
    # Each op record carries its own row and its own pass-exact
    # parents/parent_arg_positions staging.
    for layer in getattr(ml, "_layer_list", ()) or ():
        for op in getattr(layer, "ops", ()) or ():
            row = getattr(op, "_row", None)
            if row is None:
                continue
            parent_arg_positions = getattr(op, "parent_arg_positions", None) or {}
            position_counts: Counter = Counter()
            if isinstance(parent_arg_positions, Mapping):
                for arg_domain in ("args", "kwargs"):
                    domain_map = parent_arg_positions.get(arg_domain) or {}
                    if isinstance(domain_map, Mapping):
                        for parent_label in domain_map.values():
                            if isinstance(parent_label, str):
                                position_counts[parent_label] += 1
            # One expected occurrence per parents-list entry per attributed
            # arg position (or exactly one when unattributed), aggregated by
            # resolved source row -- verbatim the freeze's pass-2 emission
            # rule, including its skip of labels ``label_rows`` cannot
            # resolve (cross-pass recurrent parents such as
            # ``grucell_1_3:1`` and boundary spellings never emit edges).
            expected: Counter = Counter()
            for parent_label in getattr(op, "parents", ()) or ():
                source_row = label_rows.get(parent_label)
                if source_row is None:
                    continue
                positions = position_counts.get(parent_label, 0)
                expected[source_row] += positions if positions else 1
            # Child-direction MULTIPLICITY witness (b3-opus R05): the frozen
            # children view is DEDUPED by construction (genuine double
            # consumption records ONE children entry; multiplicity lives in
            # the parent-side arg positions checked above), so a duplicated
            # entry in a children sequence is producer/shadow corruption the
            # name-based symmetry checks cannot see -- the twice-fixed
            # duplicated-child-edge producer bug had no tripwire. Unresolved
            # (cross-pass) spellings are counted too: dedup is a property of
            # the label sequence itself, not of resolution.
            child_counts: Counter = Counter(
                child_label
                for child_label in getattr(op, "children", ()) or ()
                if isinstance(child_label, str)
            )
            duplicated = next((label for label, count in child_counts.items() if count > 1), None)
            if duplicated is not None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{getattr(op, 'label', '<unknown>')}' children names "
                    f"'{duplicated}' {child_counts[duplicated]} times -- the frozen "
                    f"children view is deduped by construction, so a repeated entry "
                    f"is a duplicated child edge",
                )
            actual = Counter(edge.source for edge in edges.in_edges(row))
            if actual != expected:
                offender = next(
                    source for source in {*actual, *expected} if actual[source] != expected[source]
                )
                offender_label = ref_labels.get(offender, f"<row {offender}>")
                raise MetadataInvariantError(
                    name,
                    f"Layer '{getattr(op, 'label', '<unknown>')}' edge-occurrence "
                    f"multiplicity mismatch for parent '{offender_label}': the "
                    f"canonical edge table records {actual[offender]} occurrence(s) "
                    f"but the parents/parent_arg_positions roots imply "
                    f"{expected[offender]}",
                )


def _check_op_log_fields(ml: Trace) -> None:
    """Check D: per-layer field consistency (shape, dtype, pass numbering, func, nesting).

    Validates:
    - Saved tensor shape/dtype match actual out (when saved).
    - Pass numbering: pass_index >= 1, num_passes >= pass_index.
    - Computational layers have callable func and non-empty func_name.
    - step_index >= 1 for non-input/non-buffer layers.
    - module_call_depth matches len(modules).
    - Label format: pass-qualified label has ':' iff multi-pass; no-pass label never has ':'.
    """
    name = "op_log_fields"

    for lpl in ml._layer_list:
        label = lpl.layer_label

        # Tensor shape/dtype consistency when outs are saved
        if lpl.has_saved_activation and lpl.out is not None:
            actual_shape = tuple(lpl.out.shape)
            if lpl.shape != actual_shape:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: shape={lpl.shape} != actual shape={actual_shape}",
                )
            if lpl.dtype != lpl.out.dtype:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: dtype={lpl.dtype} != actual dtype={lpl.out.dtype}",
                )

        # Pass numbering
        if lpl.pass_index < 1:
            raise MetadataInvariantError(name, f"Layer {label}: pass_index={lpl.pass_index} < 1")
        if lpl.num_passes < lpl.pass_index:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: num_passes={lpl.num_passes} < pass_index={lpl.pass_index}",
            )

        # A GENUINE raw-forward-hook output replacement is legitimately
        # functionless: the user substituted an opaque tensor for a module's
        # output, so there is no torch function to validate. This exemption is
        # deliberately narrow -- it must NOT cover auto-synthesized placeholders
        # during plain capture (a previous band-aid widened it to silence the
        # vmap-built attention mask, disarming this tripwire). Round-26 W3-2
        # hardening: the per-op attributes below are written by the placeholder
        # synthesizer itself, so the exemption additionally requires the
        # trace-level replacement-event ledger to corroborate the claim; a
        # placeholder stamped during PLAIN capture (no recorded replacement
        # event) now fails this invariant, per the 2026-06-02 lesson.
        is_functionless_replacement = (
            lpl.func_name == "intervention_replacement"
            and getattr(lpl, "is_edited", False)
            and not getattr(lpl, "is_internal_source", False)
            and op_has_genuine_replacement_evidence(lpl, ml)
        )

        # An internally generated *source* tensor whose construction TorchLens
        # could not trace (e.g. an attention mask built inside torch.vmap) is a
        # genuine functionless graph source, exactly like a buffer: func is None
        # and func_name is "none". Traced ops that merely have an internal-source
        # ancestor still carry a real callable func and are NOT exempted here.
        is_functionless_internal_source = (
            getattr(lpl, "is_internal_source", False) and lpl.func is None
        )

        # Function applied (non-input, non-buffer, non-output, non-source,
        # non-hook-replacement layers).
        if not (
            lpl.is_input
            or lpl.is_buffer
            or lpl.is_output
            or is_functionless_internal_source
            or is_functionless_replacement
        ):
            if bool(getattr(ml, "_loaded_from_bundle", False)):
                # Loaded-artifact arm (R10-3): ``Op.func`` is FieldPolicy.DROP
                # and ``__getstate__`` hard-nulls it, so ``callable(func)``
                # false-fired on EVERY loaded torch trace. The loaded arm
                # asserts the known load-time shape POSITIVELY -- a non-None
                # func on a loaded artifact is tampering or a schema change
                # this invariant must see -- while the func_name checks below
                # stay enforced in both arms. The live arm is untouched.
                if lpl.func is not None:
                    raise MetadataInvariantError(
                        name,
                        f"Layer {label}: loaded artifact carries a non-None func "
                        "(func is FieldPolicy.DROP and never persists)",
                    )
            elif not callable(lpl.func):
                raise MetadataInvariantError(name, f"Layer {label}: func is not callable")
            if not lpl.func_name:
                raise MetadataInvariantError(name, f"Layer {label}: func_name is empty")
            if str(lpl.func_name).lower() == "none":
                # The functionless sentinel on an op that carries a real
                # callable is name corruption (deephunt M2): "none" is only
                # minted for output bookkeeping nodes and functionless
                # internal sources, both excluded from this block above.
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: computational op carries the functionless "
                    "sentinel func_name 'none'",
                )

        # Operation numbering (input/buffer/output bookkeeping layers have step_index=0)
        if not (lpl.is_input or lpl.is_buffer or lpl.is_output):
            if lpl.step_index is not None and lpl.step_index < 1:
                raise MetadataInvariantError(
                    name, f"Layer {label}: step_index={lpl.step_index} < 1"
                )
        if lpl.raw_index < 1:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: raw_index={lpl.raw_index} < 1",
            )

        # Module attribution coherence. The historical check compared
        # ``module_call_depth`` against ``len(modules)`` -- but ``module_call_depth`` IS
        # ``return len(self.modules)``, so it was a TAUTOLOGY that could never fire and
        # the "module nesting depth" tripwire provided zero coverage.
        #
        # The real cross-field constraint is between the two INDEPENDENTLY stored
        # attribution surfaces: ``module`` is derived as ``modules[-1] if modules else
        # None``, so `module` non-None with an EMPTY roster is a state an honest capture
        # cannot produce -- and it is exactly what a lost module-enter/exit event, a pop
        # without a push, or an op emitted after a frame pop produces. That direction was
        # entirely unguarded (``_check_module_containment_logic`` early-continues on an
        # empty roster), so the whole "TorchLens lost an op's module attribution" class
        # was invisible.
        if not lpl.modules and lpl.module is not None:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: modules=() but module={lpl.module!r}; `module` is "
                f"derived as modules[-1], so an empty roster with a named module means "
                f"the module attribution was DROPPED",
            )
        # B3R7-R05-1: ``module_call_stack`` is the root-first ModuleCall
        # labels ACTIVE for this op (the glossary sentence), i.e. exactly the
        # containment fact ``modules`` carries -- the historical tri-fact
        # (fed-call stack for module inputs, containment addresses for module
        # outputs, empty otherwise) is gone. The three facts are tied here so
        # the class cannot regress: the stack must equal ``modules`` entry for
        # entry, and ``module_call_depth`` must be its length. Membership of
        # the entries themselves is ``module_containment_logic``'s check (via
        # ``modules``); equality makes it cover this field too.
        if tuple(lpl.module_call_stack) != tuple(lpl.modules):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_call_stack={tuple(lpl.module_call_stack)!r} "
                f"!= modules={tuple(lpl.modules)!r}; the stack must be the "
                f"root-first ModuleCall labels active for this op (a fed-call "
                f"or address-form value is the retired tri-fact bug)",
            )
        if lpl.module_call_depth != len(lpl.module_call_stack):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_call_depth={lpl.module_call_depth} != "
                f"len(module_call_stack)={len(lpl.module_call_stack)}",
            )

        # Label format: pass-qualified label has ":" iff multi-pass
        if lpl.num_passes > 1 and ":" not in lpl.label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: multi-pass but label='{lpl.label}' has no ':'",
            )
        if ":" in lpl.layer_label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: layer_label='{lpl.layer_label}' contains ':'",
            )


def _check_payload_metadata_invariants(ml: Trace) -> None:
    """Check saved and transformed live payload metadata.

    Precondition contract: tensor payload fields may be legitimately absent
    because of selective save, loaded traces, detached/audit-only metadata,
    disk-only storage, streaming finalization, or gradient eviction. This check
    compares shape, dtype, and memory only when a live payload object is
    present. Presence of a live raw or transformed activation requires
    ``has_saved_activation=True``; presence of a live raw or transformed
    gradient requires ``has_grad=True``. Missing payloads never imply
    corruption by themselves.

    Parameters
    ----------
    ml:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a present payload disagrees with its recorded metadata.
    """

    name = "payload_metadata_invariants"
    for op in ml._layer_list:
        label = getattr(op, "label", getattr(op, "layer_label", type(op).__name__))
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "out"),
            shape=getattr(op, "shape", None),
            dtype=getattr(op, "dtype", None),
            memory=getattr(op, "activation_memory", None),
            presence_flag=getattr(op, "has_saved_activation", False),
            presence_flag_name="has_saved_activation",
            payload_name="out",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "transformed_out"),
            shape=getattr(op, "transformed_out_shape", None),
            dtype=getattr(op, "transformed_out_dtype", None),
            memory=getattr(op, "transformed_activation_memory", None),
            presence_flag=getattr(op, "has_saved_activation", False),
            presence_flag_name="has_saved_activation",
            payload_name="transformed_out",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "grad"),
            shape=getattr(op, "grad_shape", None),
            dtype=getattr(op, "grad_dtype", None),
            memory=getattr(op, "gradient_memory", None),
            presence_flag=getattr(op, "has_grad", False),
            presence_flag_name="has_grad",
            payload_name="grad",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "transformed_grad"),
            shape=getattr(op, "transformed_grad_shape", None),
            dtype=getattr(op, "transformed_grad_dtype", None),
            memory=getattr(op, "transformed_gradient_memory", None),
            presence_flag=getattr(op, "has_grad", False),
            presence_flag_name="has_grad",
            payload_name="transformed_grad",
        )
        for record in getattr(op, "_grad_records", ()) or ():
            record_label = f"{label}.grad_record[{getattr(record, 'backward_pass_index', '?')}]"
            _check_live_payload_metadata(
                name,
                record_label,
                payload=_live_payload_value(record, "grad"),
                shape=getattr(record, "shape", None),
                dtype=getattr(record, "dtype", None),
                memory=getattr(record, "memory", None),
                presence_flag=getattr(record, "is_saved", False),
                presence_flag_name="is_saved",
                payload_name="grad",
            )
            _check_live_payload_metadata(
                name,
                record_label,
                payload=_live_payload_value(record, "transformed_grad"),
                shape=getattr(record, "transformed_grad_shape", None),
                dtype=getattr(record, "transformed_grad_dtype", None),
                memory=getattr(record, "transformed_gradient_memory", None),
                presence_flag=getattr(record, "is_saved", False),
                presence_flag_name="is_saved",
                payload_name="transformed_grad",
            )


def _live_payload_value(owner: object, payload_name: str) -> object | None:
    """Return an already-live payload without invoking guarded payload accessors.

    Parameters
    ----------
    owner:
        Object that owns the payload field.
    payload_name:
        Name of the payload field to inspect.

    Returns
    -------
    object or None
        The live payload object, or ``None`` when no payload is currently attached.
    """

    slot_getter = getattr(owner, "_slot", None)
    if callable(slot_getter):
        return slot_getter(payload_name, None)
    return getattr(owner, payload_name, None)


def _check_live_payload_metadata(
    name: str,
    label: str,
    *,
    payload: object | None,
    shape: object,
    dtype: object,
    memory: object,
    presence_flag: bool,
    presence_flag_name: str,
    payload_name: str,
) -> None:
    """Check metadata for one live tensor-like payload.

    Parameters
    ----------
    name:
        Invariant name used in raised errors.
    label:
        Owner label for diagnostics.
    payload:
        Live payload object, or ``None`` when absent.
    shape:
        Recorded shape metadata.
    dtype:
        Recorded dtype metadata.
    memory:
        Recorded memory metadata.
    presence_flag:
        Boolean metadata that should be true when payload is present.
    presence_flag_name:
        Name of ``presence_flag`` for diagnostics.
    payload_name:
        Payload field name for diagnostics.

    Raises
    ------
    MetadataInvariantError
        If present payload metadata disagrees with the payload.
    """

    if payload is None:
        return
    if not presence_flag:
        raise MetadataInvariantError(
            name,
            f"{label} has live {payload_name} payload but {presence_flag_name}=False",
        )
    actual_shape = _payload_shape(payload)
    if actual_shape is not None and shape != actual_shape:
        raise MetadataInvariantError(
            name,
            f"{label} {payload_name} shape metadata {shape!r} != payload shape {actual_shape!r}",
        )
    actual_dtype = _payload_dtype(payload)
    if (
        actual_dtype is not None
        and dtype is not None
        and not _dtype_values_match(dtype, actual_dtype)
    ):
        raise MetadataInvariantError(
            name,
            f"{label} {payload_name} dtype metadata {dtype!r} != payload dtype {actual_dtype!r}",
        )
    actual_memory = _payload_memory(payload)
    if actual_memory is not None and memory is not None:
        if not isinstance(memory, int):
            raise MetadataInvariantError(
                name,
                f"{label} {payload_name} memory metadata {memory!r} is not an integer",
            )
        if memory != actual_memory:
            raise MetadataInvariantError(
                name,
                f"{label} {payload_name} memory metadata {memory!r} != payload memory "
                f"{actual_memory!r}",
            )


def _payload_shape(payload: object) -> tuple[int, ...] | None:
    """Return a tuple shape for a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple when available.
    """

    shape = getattr(payload, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _payload_dtype(payload: object) -> object | None:
    """Return dtype metadata from a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    object | None
        Payload dtype when available.
    """

    return getattr(payload, "dtype", None)


def _payload_memory(payload: object) -> int | None:
    """Return byte memory for a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    int | None
        Number of bytes when ``nelement`` and ``element_size`` are available.
    """

    nelement = getattr(payload, "nelement", None)
    element_size = getattr(payload, "element_size", None)
    if not callable(nelement) or not callable(element_size):
        return None
    return int(nelement() * element_size())
