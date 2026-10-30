"""Equivalence, ordering, and loop-detection invariants."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .invariants import (
        _EQUIVALENT_OPS_UNAVAILABLE,
        _RAW_LABEL_BEARING_DICT_FIELDS,
        _RAW_LABEL_BEARING_LIST_FIELDS,
        _RAW_LABEL_BEARING_SCALAR_FIELDS,
        _RAW_LABEL_PATTERN,
        MetadataInvariantError,
        _retained_orphan_op_labels,
    )

__all__ = (
    "_canonical_equivalent_ops",
    "_check_equivalence_symmetry",
    "_check_graph_ordering",
    "_check_loop_detection_invariants",
)


def _canonical_equivalent_ops(owner: object) -> object:
    """Return the shared ``equivalent_ops`` object, skipping the copy-on-read copy.

    ``Op.equivalent_ops`` (through ``Op.__getattribute__``) and
    ``Layer.equivalent_ops`` (through its property) both hand back a fresh
    ``set(...)`` copy on every read, because ONE canonical set object backs every
    Op *and* Layer of an equivalence class and a shared mutable set must never be
    alias-corruptible by a single holder. That barrier is correct, but it makes
    each read O(group size).

    The object returned here is used ONLY as an identity token for the
    verified-key memos in ``_check_equivalence_symmetry``: it is never iterated,
    never mutated, and never handed to a caller, so the copy-on-read guarantee is
    untouched.

    Fails CLOSED. ``Op`` is ``__slots__``-based and exposes its raw slot through
    ``_slot``; ``Layer`` stores the field in its instance ``__dict__`` (as its
    property documents). An owner matching neither shape yields
    ``_EQUIVALENT_OPS_UNAVAILABLE``, which disables memoization for that owner
    instead of collapsing distinct groups onto one shared token -- a shared token
    would let a corrupted group inherit a clean group's verdict and silently
    disarm this check.

    Parameters
    ----------
    owner:
        ``Op`` or ``Layer`` record holding an ``equivalent_ops`` group.

    Returns
    -------
    object
        The canonical stored object, or ``_EQUIVALENT_OPS_UNAVAILABLE``.
    """

    slot_reader = getattr(owner, "_slot", None)
    if callable(slot_reader):
        return slot_reader("equivalent_ops", _EQUIVALENT_OPS_UNAVAILABLE)
    instance_dict = getattr(owner, "__dict__", None)
    if isinstance(instance_dict, dict):
        return instance_dict.get("equivalent_ops", _EQUIVALENT_OPS_UNAVAILABLE)
    return _EQUIVALENT_OPS_UNAVAILABLE


def _check_equivalence_symmetry(ml: Trace) -> None:
    """Check L: op_equivalence_classes groups reference valid Op labels.

    Validates:
    - Each equivalence set value is actually a set.
    - All labels in equivalence sets exist in op_labels.
    - Per-Op and per-Layer equivalent_ops views match the trace-level groups
      and stay in the final op-label namespace.
    """
    name = "equivalence_symmetry"
    call_site_grouped = _is_call_site_grouped(ml)
    label_set = set(ml.op_labels) | _retained_orphan_op_labels(ml)

    # op_equivalence_classes is keyed by equivalence type descriptors (not Op labels),
    # with values being sets of Op labels in that equivalence group.
    for eq_type, equiv_set in ml.op_equivalence_classes.items():
        if not isinstance(equiv_set, set):
            raise MetadataInvariantError(
                name,
                f"op_equivalence_classes['{eq_type}'] is not a set",
            )
        for label in equiv_set:
            if label not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"op_equivalence_classes['{eq_type}'] contains '{label}' not in op_labels",
                )

    # Each Op that appears in any equivalence group should exist.
    all_equiv_labels = set()
    for equiv_set in ml.op_equivalence_classes.values():
        all_equiv_labels.update(equiv_set)
    extra = all_equiv_labels - label_set
    if extra:
        raise MetadataInvariantError(
            name,
            f"op_equivalence_classes contains labels not in op_labels: {extra}",
        )

    # Per-Op verdict is a pure function of the (canonical `equivalent_ops` object,
    # expected-group object) PAIR plus the loop-invariant `label_set`. Ops of one
    # equivalence class all share a single canonical object, so re-running the body
    # for each of them re-paid an O(group size) copy-on-read, an O(group size)
    # label scan, and an O(group size) set comparison -- quadratic on any trace with
    # one large class (an N-step chain, a many-layer transformer).
    #
    # Memoizing the pair is verdict-identical: iteration order is unchanged, so a
    # failing pair still raises at the FIRST Op carrying it, and skipping later Ops
    # with a pair already proven clean cannot surface an error the full body would
    # have raised. Both objects are retained in the keepalive list so no `id()` can
    # be recycled onto a different object mid-check.
    verified_op_pairs: set[tuple[int, int]] = set()
    verified_keepalive: list[object] = []
    # Membership index: per-op ``equivalence_class`` carries the module suffix
    # appended at op creation (plus loop-detection ``_argsig`` splits), while
    # the trace dict keys are pre-suffix, so a key-based lookup misses for
    # EVERY parameterized in-module layer and the group comparison below
    # silently skipped -- symmetric corruption of ``equivalent_ops`` on all
    # passes of a shared in-module layer passed the whole suite. Resolving the
    # expected group by MEMBERSHIP (the same reconciliation the recurrent-group
    # consistency check at the bottom of this module already uses) makes the
    # invariant actually run for suffixed classes; the key lookup stays first
    # so a label scrubbed OUT of its group still resolves the group and fails.
    group_by_member: dict[str, set] = {}
    for group in ml.op_equivalence_classes.values():
        if isinstance(group, set):
            for member_label in group:
                group_by_member[member_label] = group
    for op in ml.layer_list:
        equivalence_class = getattr(op, "equivalence_class", None)
        expected_group = (
            ml.op_equivalence_classes.get(equivalence_class)
            if isinstance(equivalence_class, str)
            else None
        )
        if expected_group is None:
            expected_group = group_by_member.get(op.label)
        canonical = _canonical_equivalent_ops(op)
        op_pair_key: tuple[int, int] | None = None
        if canonical is not _EQUIVALENT_OPS_UNAVAILABLE:
            op_pair_key = (id(canonical), id(expected_group))
            if op_pair_key in verified_op_pairs:
                continue
        equivalent_ops = getattr(op, "equivalent_ops", None)
        if not isinstance(equivalent_ops, (set, frozenset)):
            raise MetadataInvariantError(
                name,
                f"{op.label}.equivalent_ops is not a set",
            )
        for label in equivalent_ops:
            if label not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"{op.label}.equivalent_ops contains '{label}' not in op_labels",
                )
        if expected_group is not None and equivalent_ops != expected_group:
            raise MetadataInvariantError(
                name,
                f"{op.label}.equivalent_ops={sorted(equivalent_ops)} != expected "
                f"{sorted(expected_group)}",
            )
        if op_pair_key is not None:
            verified_op_pairs.add(op_pair_key)
            verified_keepalive.append((canonical, expected_group))

    # Same argument for the per-Layer body, whose verdict is a pure function of the
    # layer's canonical object and the ordered canonical objects of its passes.
    verified_layer_keys: set[tuple[int, tuple[int, ...]]] = set()
    for layer in ml.layer_logs.values():
        layer_canonical = _canonical_equivalent_ops(layer)
        pass_canonicals = [_canonical_equivalent_ops(op) for op in layer.ops.values()]
        layer_key: tuple[int, tuple[int, ...]] | None = None
        if layer_canonical is not _EQUIVALENT_OPS_UNAVAILABLE and all(
            item is not _EQUIVALENT_OPS_UNAVAILABLE for item in pass_canonicals
        ):
            layer_key = (
                id(layer_canonical),
                tuple(id(item) for item in pass_canonicals),
            )
            if layer_key in verified_layer_keys:
                continue
        equivalent_ops = getattr(layer, "equivalent_ops", None)
        if not isinstance(equivalent_ops, (set, frozenset)):
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label}.equivalent_ops is not a set",
            )
        for label in equivalent_ops:
            if label not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer {layer.layer_label}.equivalent_ops contains '{label}' not in op_labels",
                )
        pass_ops = list(layer.ops.values())
        pass_equivalent_ops = {frozenset(getattr(op, "equivalent_ops", set())) for op in pass_ops}
        if len(pass_equivalent_ops) > 1:
            # Under callsite_v1 the passes of one layer may sit in different
            # equivalence sets exactly when their classes differ only in what
            # the grouping rule says is not evidence (argument/shape hash rows
            # 49 and 50, parameter identity row 51, module path rows 30 and 31);
            # the layer then serves one of its passes' sets.
            narrowed = call_site_grouped and all(
                _equivalence_classes_agree(
                    pass_ops[0], pass_ops[0].equivalence_class, op, op.equivalence_class, True
                )
                for op in pass_ops
            )
            if not narrowed:
                raise MetadataInvariantError(
                    name,
                    f"Layer {layer.layer_label} ops disagree on equivalent_ops: "
                    f"{sorted(sorted(group) for group in pass_equivalent_ops)}",
                )
            if frozenset(equivalent_ops) not in pass_equivalent_ops:
                raise MetadataInvariantError(
                    name,
                    f"Layer {layer.layer_label}.equivalent_ops={sorted(equivalent_ops)} is "
                    "none of its passes' equivalent_ops",
                )
        elif pass_equivalent_ops and equivalent_ops != set(next(iter(pass_equivalent_ops))):
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label}.equivalent_ops={sorted(equivalent_ops)} != "
                f"pass equivalent_ops={sorted(next(iter(pass_equivalent_ops)))}",
            )
        if layer_key is not None:
            verified_layer_keys.add(layer_key)
            verified_keepalive.append((layer_canonical, pass_canonicals))


def _check_graph_ordering(ml: Trace) -> None:
    """Check M: graph ordering invariants.

    Validates:
    - raw_index is unique across all layers and monotonically
      increasing in layer_list order.
    - step_index is unique among computational layers (non-input, non-buffer,
      non-output).
    - Topological order: every parent's raw_index < child's.
    - No raw labels (``l_\\d+``) survive postprocessing.
    """
    name = "graph_ordering"

    # raw_index uniqueness and monotonicity
    seen_rt_nums: dict[int, str] = {}
    prev_rt = -1
    for lpl in ml.layer_list:
        rt = lpl.raw_index
        if rt in seen_rt_nums:
            raise MetadataInvariantError(
                name,
                f"Duplicate raw_index={rt}: '{seen_rt_nums[rt]}' and '{lpl.layer_label}'",
            )
        seen_rt_nums[rt] = lpl.layer_label
        if rt <= prev_rt:
            raise MetadataInvariantError(
                name,
                f"raw_index not monotonically increasing: "
                f"{prev_rt} then {rt} at '{lpl.layer_label}'",
            )
        prev_rt = rt

    # step_index uniqueness among computational layers
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)
    output_set = set(ml.output_layers)
    seen_op_nums: dict[int, str] = {}
    for lpl in ml.layer_list:
        label = lpl.layer_label
        if label in input_set or label in buffer_set or label in output_set:
            continue
        op = lpl.step_index
        if op is not None:
            if op in seen_op_nums:
                raise MetadataInvariantError(
                    name,
                    f"Duplicate step_index={op}: '{seen_op_nums[op]}' and '{label}'",
                )
            seen_op_nums[op] = label

    # Topological order: parent.raw_index < child.raw_index
    rt_map = {lpl.layer_label: lpl.raw_index for lpl in ml.layer_list}
    rt_map.update({lpl.label: lpl.raw_index for lpl in ml.layer_list})
    for lpl in ml.layer_list:
        for p in lpl.parents:
            if rt_map.get(p, -1) >= lpl.raw_index:
                raise MetadataInvariantError(
                    name,
                    f"Topological violation: parent '{p}' (rt={rt_map.get(p)}) "
                    f">= child '{lpl.layer_label}' (rt={lpl.raw_index})",
                )

    # No raw labels survive postprocessing -- on ANY label-bearing surface, not just
    # ``layer_labels``. A raw label surviving in a relation set or in the scalar
    # ``buffer_source`` field is an unresolvable lookup key handed to users and
    # persisted into portable artifacts, which is exactly what this check exists to stop.
    for label in ml.layer_labels:
        if _RAW_LABEL_PATTERN.search(label):
            raise MetadataInvariantError(name, f"Raw label '{label}' survived postprocessing")
    for lpl in ml.layer_list:
        for field in _RAW_LABEL_BEARING_LIST_FIELDS:
            for item in getattr(lpl, field, ()) or ():
                if isinstance(item, str) and _RAW_LABEL_PATTERN.search(item):
                    raise MetadataInvariantError(
                        name,
                        f"Raw label {item!r} survived postprocessing in {lpl.layer_label}.{field}",
                    )
        for field in _RAW_LABEL_BEARING_SCALAR_FIELDS:
            item = getattr(lpl, field, None)
            if isinstance(item, str) and _RAW_LABEL_PATTERN.search(item):
                raise MetadataInvariantError(
                    name,
                    f"Raw label {item!r} survived postprocessing in {lpl.layer_label}.{field}",
                )
        for field in _RAW_LABEL_BEARING_DICT_FIELDS:
            mapping = getattr(lpl, field, None) or {}
            for key, values in mapping.items():
                candidates = [key]
                if isinstance(values, str):
                    candidates.append(values)
                elif isinstance(values, (list, tuple, set, frozenset)):
                    candidates.extend(values)
                for item in candidates:
                    if isinstance(item, str) and _RAW_LABEL_PATTERN.search(item):
                        raise MetadataInvariantError(
                            name,
                            f"Raw label {item!r} survived postprocessing in "
                            f"{lpl.layer_label}.{field}",
                        )


def _check_loop_detection_invariants(ml: Trace) -> None:
    """Check N: loop detection / recurrent_ops invariants.

    Validates per-layer:
    - recurrent_ops is non-empty and includes self.
    - Symmetry: all members agree on the same group.
    - All members share: layer_label, equivalence_class,
      func_name (for computational layers).
    - num_passes == len(recurrent_ops).
    - Pass numbering within group is contiguous {1..N}.

    Validates cross-layer:
    - Parameter sharing rule: layers with same (func_name,
      sorted(_param_barcodes)) must share layer_label.
    - Equivalence group consistency: all members of a recurrent_ops
      group belong to the same Trace.op_equivalence_classes set.

    Call-site grouping (``callsite_v1``, grouping rule section 3) replaces two
    of these, and only for traces stamped with that detector:

    - Old: layers sharing (func_name, param barcodes, equivalence_class) are
      one layer. New: two different layers never share a call-site key
      (contract (ii)), and every pass of a layer has one func_name and one
      call-site key (i), and every recurrent group one key (iii). Shared
      weights are not evidence (rows 18, 19, 25, 59).
    - Old: every pass of a layer has one equivalence_class and one
      op_equivalence_classes set. New: the same, except that the class may
      differ in exactly the components the rule says are not evidence: the
      argument/shape hash (rows 49, 50), parameter identity (row 51) and the
      module path (rows 30, 31). Layer type, in-place marker and output index
      must still match (:func:`_equivalence_classes_agree`).

    The call-site key is internal and never persisted, so (i) to (iii) are
    re-checked here only for a live capture that kept its side table
    (``postprocess._call_site_contract``); otherwise they are reported as not
    checked (validation diagnostic ``call_site_contract_unchecked``), never
    passed. Step 7 checks them on every capture.

    Note: subgraph-level adjacency (Rule 3 from loop_detection.py) cannot
    be verified post-hoc from metadata alone.
    """
    name = "loop_detection"
    call_site_grouped = _is_call_site_grouped(ml)
    label_set = set(ml.op_labels)

    # O(1) member-label resolution: inline the first steps of the
    # ``Trace.__getitem__`` string cascade (layer_logs -> ambiguous ->
    # layer_dict_all_keys) and delegate anything else to the full lookup, so
    # every resolution (including ambiguous-key errors) matches ``ml[label]``.
    _MISS = object()
    layer_logs = ml.layer_logs
    all_keys = ml.layer_dict_all_keys
    ambiguous_keys = getattr(ml, "_ambiguous_lookup_keys", {})

    def _resolve(member_label: str) -> Op:
        """Look up one group-member label, matching ``ml[member_label]`` exactly.

        The two unambiguous dict hits are inlined for O(1) resolution; anything
        else (ambiguous keys, substring/ordinal forms, misses) falls through to
        the full ``Trace.__getitem__`` cascade so errors stay identical.
        """

        hit = layer_logs.get(member_label, _MISS)
        if hit is not _MISS:
            return cast("Op", hit)
        if member_label not in ambiguous_keys:
            hit = all_keys.get(member_label, _MISS)
            if hit is not _MISS:
                return cast("Op", hit)
        return ml[member_label]

    # Build same-layer groups from the authoritative recurrent_ops lists
    # Key: frozenset of labels, Value: list of OpLogs in the group
    groups_seen: dict[frozenset[str], list[str]] = {}
    # Group-level checks run once per recurrence group, at its first
    # layer_list encounter.  The symmetry check there proves every member
    # resolves to the same label set, so a later member whose object IS the
    # resolved op (identity-guarded below) would pass the identical checks;
    # re-running them per member is what made this check quadratic.
    member_group: dict[str, frozenset[str]] = {}
    # The shared-func_name check only runs at computational anchors, so a
    # group first encountered via an input/buffer/output member still owes it
    # at its first computational member (matching the original scan order).
    group_func_checked: dict[frozenset[str], bool] = {}

    for lpl in ml.layer_list:
        slo = lpl.recurrent_ops

        group_key = member_group.get(lpl.label)
        if group_key is not None and _resolve(lpl.label) is lpl:
            # This op was already validated as a member of its group; only
            # the genuinely per-layer checks remain.
            if not group_func_checked[group_key] and not (
                lpl.is_input or lpl.is_buffer or lpl.is_output
            ):
                for member_label in slo:
                    member = _resolve(member_label)
                    if member.func_name != lpl.func_name:
                        raise MetadataInvariantError(
                            name,
                            f"recurrent_ops func mismatch: '{lpl.layer_label}' "
                            f"func='{lpl.func_name}' vs '{member_label}' "
                            f"func='{member.func_name}'",
                        )
                group_func_checked[group_key] = True

            # num_passes == len(recurrent_ops)
            if lpl.num_passes != len(slo):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': num_passes={lpl.num_passes} "
                    f"!= len(recurrent_ops)={len(slo)}",
                )
            continue

        if not slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' has empty recurrent_ops",
            )

        # All members in recurrent_ops must exist
        for member in slo:
            if member not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' recurrent_ops contains '{member}' not in op_labels",
                )

        # Self-inclusion
        if lpl.label not in slo:
            raise MetadataInvariantError(
                name,
                f"Op '{lpl.label}' not in its own recurrent_ops",
            )

        # Symmetry: all members agree on the group
        slo_set = set(slo)
        members: list[tuple[str, Op]] = []
        for member_label in slo:
            member = _resolve(member_label)
            members.append((member_label, member))
            if set(member.recurrent_ops) != slo_set:
                raise MetadataInvariantError(
                    name,
                    f"Asymmetric recurrent_ops: '{lpl.layer_label}' has "
                    f"{sorted(slo)} but '{member_label}' has "
                    f"{sorted(member.recurrent_ops)}",
                )

        # All members share layer_label
        for member_label, member in members:
            if member.layer_label != lpl.layer_label:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_ops inconsistency: '{lpl.layer_label}' "
                    f"(no_pass='{lpl.layer_label}') and '{member_label}' "
                    f"(no_pass='{member.layer_label}') differ",
                )

        # All members share equivalence_class (per component under callsite_v1)
        for member_label, member in members:
            if not _equivalence_classes_agree(
                lpl, lpl.equivalence_class, member, member.equivalence_class, call_site_grouped
            ):
                raise MetadataInvariantError(
                    name,
                    f"recurrent_ops type mismatch: '{lpl.layer_label}' "
                    f"type='{lpl.equivalence_class}' vs '{member_label}' "
                    f"type='{member.equivalence_class}'",
                )

        # All members share func_name (for computational layers)
        anchor_computational = not (lpl.is_input or lpl.is_buffer or lpl.is_output)
        if anchor_computational:
            for member_label, member in members:
                if member.func_name != lpl.func_name:
                    raise MetadataInvariantError(
                        name,
                        f"recurrent_ops func mismatch: '{lpl.layer_label}' "
                        f"func='{lpl.func_name}' vs '{member_label}' "
                        f"func='{member.func_name}'",
                    )

        # num_passes == len(recurrent_ops)
        if lpl.num_passes != len(slo):
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}': num_passes={lpl.num_passes} "
                f"!= len(recurrent_ops)={len(slo)}",
            )

        # Pass numbering: unique {1..N}
        group_key = frozenset(slo)
        if group_key not in groups_seen:
            pass_indices = []
            for member_label, member in members:
                pass_indices.append(member.pass_index)
            expected = set(range(1, len(slo) + 1))
            actual = set(pass_indices)
            if actual != expected:
                raise MetadataInvariantError(
                    name,
                    f"Pass numbering for group {sorted(slo)}: expected {expected}, got {actual}",
                )
            groups_seen[group_key] = slo

        for member_label in slo_set:
            member_group[member_label] = group_key
        if group_key not in group_func_checked or anchor_computational:
            group_func_checked[group_key] = anchor_computational

    # Rule 1: Parameter sharing invariant.
    # Layers with the same func_name, identical sorted(_param_barcodes), and the
    # same output-specific equivalence class must share layer_label. The
    # equivalence class keeps distinct outputs of a multi-output parameterized op
    # from being collapsed into one logical layer.
    # Under callsite_v1 call-site contract (ii) replaces it (shared weights are
    # not evidence, rows 18, 19, 25, 59); see the docstring.
    param_groups: dict[tuple[str, tuple[str, ...], str], list[Op]] = defaultdict(list)
    for lpl in [] if call_site_grouped else ml.layer_list:
        if lpl.uses_params and lpl._param_barcodes:
            key = (lpl.func_name, tuple(sorted(lpl._param_barcodes)), lpl.equivalence_class)
            param_groups[key].append(lpl)

    for param_key, layers in param_groups.items():
        if len(layers) > 1:
            no_call_labels = {lpl.layer_label for lpl in layers}
            if len(no_call_labels) > 1:
                raise MetadataInvariantError(
                    name,
                    f"Param sharing violation: layers with same param barcodes "
                    f"{param_key} have different layer_label: {no_call_labels}",
                )

    # Equivalence group ↔ same_layer consistency: all members of a
    # recurrent_ops group must belong to the same equivalence set.
    # Note: Trace.op_equivalence_classes keys use the pre-module-suffix type
    # (from loop_detection), while per-layer equivalence_class has
    # a module suffix appended by control_flow.py. So we check group membership
    # consistency, not exact key matching.
    op_label_to_equiv_key: dict[str, str] = {}
    for eq_type, equiv_set in ml.op_equivalence_classes.items():
        for label in equiv_set:
            op_label_to_equiv_key[label] = eq_type

    for group_key in groups_seen:
        slo = list(group_key)
        if len(slo) <= 1:
            continue
        # All members of a same-layer group should be in the same equivalence set
        equiv_keys = set()
        keyed_members: list[tuple[Op, str]] = []
        for member_label in slo:
            member = _resolve(member_label)
            if member.label in op_label_to_equiv_key:
                equiv_keys.add(op_label_to_equiv_key[member.label])
                keyed_members.append((member, op_label_to_equiv_key[member.label]))
        equiv_stems = {re.sub(r"_outindex\d+$", "", equiv_key) for equiv_key in equiv_keys}
        if call_site_grouped and keyed_members:
            anchor, anchor_key = keyed_members[0]
            spans = not all(
                _equivalence_classes_agree(anchor, anchor_key, member, key, True)
                for member, key in keyed_members
            )
        else:
            spans = len(equiv_keys) > 1 and len(equiv_stems) > 1
        if spans:
            raise MetadataInvariantError(
                name,
                f"recurrent_ops group {sorted(slo)} spans multiple equivalence types: {equiv_keys}",
            )
    if call_site_grouped:
        _check_call_site_contract_post_hoc(ml, groups_seen, member_group, _resolve)


_OUTINDEX_TOKEN = re.compile(r"_outindex(\d+)")


def _is_call_site_grouped(trace: Trace) -> bool:
    """Whether step 7 grouped this trace by call site (stamp ``detector``)."""

    stamp = getattr(trace, "grouping_policy", None)
    return isinstance(stamp, dict) and stamp.get("detector") == "callsite_v1"


def _equivalence_class_components(op: Op, equivalence_class: object) -> tuple[object, ...]:
    """The components of an equivalence class that call-site passes must share.

    Layer type, the in-place marker and the output-index tokens. The rest of
    the class (argument/shape hash, parameter barcodes, module path) is what
    the grouping rule says is not evidence (rows 30, 31, 49, 50, 51).
    """

    layer_type = str(getattr(op, "layer_type", ""))
    text = str(equivalence_class)
    return (
        layer_type,
        text.startswith(f"{layer_type}_"),
        text.startswith(f"{layer_type}_inplace_"),
        tuple(_OUTINDEX_TOKEN.findall(text)),
    )


def _equivalence_classes_agree(
    anchor: Op, anchor_class: object, member: Op, member_class: object, call_site_grouped: bool
) -> bool:
    """Whether two passes' equivalence classes agree under the trace's grouper.

    Exact equality under ``grouper_v1``. Under ``callsite_v1`` equal classes
    agree, and different classes agree only when they differ in nothing but
    the argument/shape hash, parameter identity and module path: layer type,
    in-place marker and output index must match, and each class must carry
    its op's own layer type.
    """

    if anchor_class == member_class:
        return True
    if not call_site_grouped:
        return False
    anchor_parts = _equivalence_class_components(anchor, anchor_class)
    member_parts = _equivalence_class_components(member, member_class)
    return anchor_parts == member_parts and bool(anchor_parts[1])


def _check_call_site_contract_post_hoc(
    trace: Trace,
    groups_seen: dict[frozenset[str], list[str]],
    member_group: dict[str, frozenset[str]],
    resolve: Any,
) -> None:
    """Re-check call-site contract (i) to (iii), or report it as not checked.

    The call-site key is internal and never persisted (grouping rule section
    3), so only a live capture that kept its side table can be re-checked; a
    loaded trace or a fork records the validation diagnostic
    ``call_site_contract_unchecked`` instead of passing silently.
    """

    from ..postprocess._call_site_contract import (
        CallSiteEntry,
        call_site_classes_for,
        call_site_contract_violation,
    )
    from .invariants import MetadataInvariantError

    classes = call_site_classes_for(trace)
    if classes is None:
        from .diagnostics import ValidationDiagnostic, record_validation_diagnostic

        record_validation_diagnostic(
            trace,
            ValidationDiagnostic(
                check="call_site_contract_unchecked",
                message=(
                    "Call-site grouping contract (i) to (iii) not checked: this trace "
                    "carries no call-site side table (the key is never persisted; step 7 "
                    "checked it at capture)."
                ),
                extra={"detector": "callsite_v1"},
            ),
        )
        return
    group_raw: dict[frozenset[str], tuple[str, ...]] = {
        key: tuple(resolve(label)._label_raw for label in labels)
        for key, labels in groups_seen.items()
    }
    entries = [
        CallSiteEntry(
            raw_label=op._label_raw,
            layer=op.layer_label,
            func_name=op.func_name,
            group=group_raw.get(member_group.get(op.label, frozenset()), (op._label_raw,)),
        )
        for op in trace.layer_list
    ]
    violation = call_site_contract_violation(entries, classes)
    if violation is not None:
        raise MetadataInvariantError("loop_detection", violation)
