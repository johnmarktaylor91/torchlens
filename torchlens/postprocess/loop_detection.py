"""Step 7: Adapt Trace state to the backend-neutral recurrence grouper.

Production loop detection lives in :mod:`loop_grouping_adapter`. This module
retains only the Trace adapter, assignment application, and the lightweight
shared-parameter grouping used when full recurrence detection is disabled.
"""

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ..data_classes._site_key import SiteKeyMinter
from ._grouping_stamp import build_grouping_policy_stamp
from .loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

# Value types whose repr is deterministic and content-derived; safe to include in the
# structural argument signature. Anything else risks an object-identity repr
# (``<Foo object at 0x...>``) that would differ across genuinely identical passes.
_SIGNATURE_VALUE_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    slice,
    range,
    type(...),
)

# (module, qualname) pairs of known torch value types whose ``str()`` is deterministic
# and content-derived (``torch.float32``, ``cpu``, ``torch.channels_last``, ...).
_SIGNATURE_SAFE_TORCH_TYPES = {
    ("torch", "dtype"),
    ("torch", "device"),
    ("torch", "memory_format"),
    ("torch", "layout"),
}


def _append_signature_tokens(arg: Any, prefix: str, tokens: list[str], depth: int = 0) -> None:
    """Append deterministic structural tokens for one non-tensor argument.

    Tensor-like arguments (anything not a recognized primitive, container, or safe
    torch value type) contribute a class-name token only -- never a value, shape, or
    object repr -- so genuine variable-length recurrence (a loop whose tensors shrink
    each step) keeps one signature across passes, and object-identity reprs cannot
    make two identical calls differ.

    Parameters
    ----------
    arg:
        Argument value to fingerprint.
    prefix:
        Position/key path prefix.
    tokens:
        Accumulator receiving fingerprint tokens.
    depth:
        Recursion depth guard.
    """

    if depth > 6:
        tokens.append(f"{prefix}=deep")
        return
    if isinstance(arg, _SIGNATURE_VALUE_TYPES):
        tokens.append(f"{prefix}={type(arg).__name__}:{arg!r}")
        return
    if isinstance(arg, dict):
        # Keys get the same identity-repr guard as values: emitting ``key!r``
        # verbatim (and sorting by ``repr``) leaks ``<Foo object at 0x...>``
        # addresses into the signature, so a fresh non-primitive key per call
        # (``cfg={SomeObject(): 1}``) gave every pass of a genuine recurrence
        # a different, ASLR-varying signature and silently ungrouped it.
        entries = sorted(
            ((_signature_key_token(key, depth + 1), key) for key in arg),
            key=lambda entry: entry[0],
        )
        for key_token, key in entries:
            _append_signature_tokens(arg[key], f"{prefix}.k{key_token}", tokens, depth + 1)
        return
    if isinstance(arg, (list, tuple, set, frozenset)):
        elements = (
            sorted(arg, key=lambda element: _signature_element_sort_key(element, depth + 1))
            if isinstance(arg, (set, frozenset))
            else arg
        )
        tokens.append(f"{prefix}={type(arg).__name__}[{len(arg)}]")
        for index, element in enumerate(elements):
            _append_signature_tokens(element, f"{prefix}.{index}", tokens, depth + 1)
        return
    arg_type = type(arg)
    type_key = (getattr(arg_type, "__module__", ""), getattr(arg_type, "__qualname__", ""))
    if type_key in _SIGNATURE_SAFE_TORCH_TYPES:
        tokens.append(f"{prefix}={type_key[1]}:{arg!s}")
        return
    tokens.append(f"{prefix}=<{type_key[0]}.{type_key[1]}>")


def _signature_key_token(key: Any, depth: int) -> str:
    """Return a deterministic, address-free signature token for one dict key.

    Mirrors the value-side policy of :func:`_append_signature_tokens`: primitive
    and safe torch value types contribute type plus content, hashable containers
    recurse element-wise, and anything else contributes its class name only, so
    an object-identity ``repr`` can never make two identical calls differ.

    Parameters
    ----------
    key:
        Dict key to fingerprint.
    depth:
        Recursion depth guard inherited from the signature walk.

    Returns
    -------
    str
        Deterministic key token.
    """

    if depth > 6:
        return "deep"
    if isinstance(key, _SIGNATURE_VALUE_TYPES):
        return f"{type(key).__name__}:{key!r}"
    if isinstance(key, tuple):
        inner = ",".join(_signature_key_token(element, depth + 1) for element in key)
        return f"tuple[{inner}]"
    if isinstance(key, frozenset):
        inner = ",".join(sorted(_signature_key_token(element, depth + 1) for element in key))
        return f"frozenset[{inner}]"
    key_type = type(key)
    type_key = (getattr(key_type, "__module__", ""), getattr(key_type, "__qualname__", ""))
    if type_key in _SIGNATURE_SAFE_TORCH_TYPES:
        return f"{type_key[1]}:{key!s}"
    return f"<{type_key[0]}.{type_key[1]}>"


def _signature_element_sort_key(arg: Any, depth: int) -> tuple[str, ...]:
    """Return the exact address-free tokens used to order one set element.

    Parameters
    ----------
    arg:
        Unordered container element.
    depth:
        Recursion depth inherited from the parent signature walk.

    Returns
    -------
    tuple[str, ...]
        Emitted signature tokens under a neutral prefix.
    """

    tokens: list[str] = []
    _append_signature_tokens(arg, "element", tokens, depth)
    return tuple(tokens)


def _structural_arg_signature(op: Any) -> str:
    """Return the non-tensor structural argument fingerprint for one op.

    Built from the captured ``non_tensor_pos_args``/``non_tensor_kwargs`` so it covers
    exactly the call semantics that parameter identity ignores (``padding``,
    ``stride``, ``dilation``, ``groups``, flags, ...). Two ops must share this
    signature to be grouped as recurrent passes of one parameterized layer.

    Parameters
    ----------
    op:
        Retained Op entry.

    Returns
    -------
    str
        Short deterministic fingerprint.
    """

    tokens: list[str] = []
    for index, arg in enumerate(getattr(op, "non_tensor_pos_args", ()) or ()):
        _append_signature_tokens(arg, f"pos{index}", tokens)
    kwargs = getattr(op, "non_tensor_kwargs", None) or {}
    kwarg_items = kwargs.items() if isinstance(kwargs, dict) else kwargs
    for key, arg in sorted(kwarg_items, key=lambda item: str(item[0])):
        _append_signature_tokens(arg, f"kw_{key}", tokens)
    if not tokens:
        return "no_args"
    return hashlib.md5("|".join(tokens).encode("utf-8", "replace")).hexdigest()[:12]


def _module_site(op: Any) -> tuple[str, ...]:
    """Return the op's module ADDRESS stack (addresses only, no pass numbers).

    Parameters
    ----------
    op:
        Retained Op entry.

    Returns
    -------
    tuple[str, ...]
        Containing-module addresses ordered outer-to-inner; empty when the op runs
        directly in the root forward.
    """

    return tuple(module_pass[0] for module_pass in getattr(op, "modules", None) or ())


def _differentiated_param_equivalence_classes(self: "Trace") -> dict[str, str]:
    """Split parameterized equivalence classes whose members differ in call structure.

    Capture-time parameterized equivalence keys are ``func + parameter barcodes``
    (plus output slot and module suffix) and deliberately omit non-tensor arguments.
    When one class therefore spans calls with DIFFERENT non-tensor structure (one
    kernel applied with ``padding=0`` and ``padding=1``), the members are different
    operations, never recurrent passes of one layer. Mirroring how the module suffix
    already splits tied distinct-module sites, such ambiguous cohorts get an
    ``_argsig`` suffix so every equivalence-class consumer (grouping, the
    loop-detection param-sharing invariant, motif matching) sees distinct sites.
    Cohorts with one uniform signature -- every genuine recurrence, including
    variable-length loops whose tensors change shape -- are left byte-identical.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.

    Returns
    -------
    dict[str, str]
        Effective equivalence class keyed by raw label; only differentiated members
        appear.
    """

    cohorts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for label in self._raw_graph_ws.raw_layer_labels_list:
        node = self[label]
        if getattr(node, "is_orphan", False):
            continue
        if node.uses_params and node._param_barcodes:
            cohorts[node.equivalence_class].append((label, _structural_arg_signature(node)))

    effective: dict[str, str] = {}
    for eq_class, members in cohorts.items():
        if len({signature for _, signature in members}) <= 1:
            continue
        for label, signature in members:
            effective[label] = f"{eq_class}_argsig{signature}"
    return effective


def _mint_site_keys(self: "Trace") -> None:
    """Mint the policy-independent ``site_key_v1`` on every retained op.

    Runs identically on the full-detection and degraded
    (``recurrence_detection=False``) paths -- site keys are structural facts
    computed from raw records BEFORE any grouping verdict, so both paths
    produce byte-identical key multisets (property P4). Orphan ops consume
    no ordinals and keep ``site_key=None`` (the SF-63 ruling).

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.
    """

    minter = SiteKeyMinter()
    for label in self._raw_graph_ws.raw_layer_labels_list:
        node = self[label]
        if getattr(node, "is_orphan", False):
            continue
        node.site_key = minter.mint(
            getattr(node, "modules", None) or (),
            node.layer_type,
            getattr(node, "multi_output_index", None),
        )


def _group_by_shared_params(self: "Trace") -> None:
    """Group repeated uses of the same parameterized function.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.

    Notes
    -----
    Operations without parameters remain individual single-pass layers. The
    helper sets ``_layer_label_raw``, ``recurrent_ops``, ``pass_index``, and
    ``num_passes`` on every retained op, plus the policy-independent
    ``site_key`` (identical to the full-detection path's keys, P4).
    """

    _mint_site_keys(self)
    effective_equivalence = _differentiated_param_equivalence_classes(self)
    for label, effective_class in effective_equivalence.items():
        self[label].equivalence_class = effective_class

    param_barcode_groups: dict[
        tuple[str, tuple[str, ...], int | None, tuple[str, ...], str], list[str]
    ] = defaultdict(list)
    for label in self._raw_graph_ws.raw_layer_labels_list:
        node = self[label]
        if getattr(node, "is_orphan", False):
            continue
        if node.uses_params and node._param_barcodes:
            # Key on the FULL call identity, not bare parameter identity. Output
            # slot: co-outputs of one multi-output call (h and c of an LSTMCell)
            # share function and parameters but are distinct layers, never
            # sequential passes of each other. Module address and non-tensor arg
            # signature: distinct tied modules (encoder/decoder sharing a weight)
            # and one kernel applied with different padding/stride are different
            # semantic sites, never recurrent passes of one layer.
            key = (
                node.func_name,
                tuple(sorted(node._param_barcodes)),
                getattr(node, "multi_output_index", None),
                _module_site(node),
                _structural_arg_signature(node),
            )
            param_barcode_groups[key].append(label)

    for members in param_barcode_groups.values():
        if len(members) <= 1:
            continue
        leader = min(members, key=lambda label: self[label].raw_index)
        leader_raw = self[leader]._layer_label_raw
        for label in members:
            self[label]._layer_label_raw = leader_raw

    _rebuild_pass_assignments(self)
    self.grouping_policy = build_grouping_policy_stamp(
        ran_recurrence_grouping=False,
        requested=getattr(self, "grouping", "structural"),
    )


def _detect_and_label_loops(self: "Trace") -> None:
    """Delegate recurrence grouping to the backend-neutral implementation.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.
    """

    _mint_site_keys(self)
    grouping_graph = _build_recurrence_grouping_graph(self)
    assignments = group_recurrent_nodes(grouping_graph)
    _apply_recurrence_assignments(self, assignments)
    self.grouping_policy = build_grouping_policy_stamp(
        ran_recurrence_grouping=True,
        requested=getattr(self, "grouping", "structural"),
    )


def _build_recurrence_grouping_graph(self: "Trace") -> RecurrenceGroupingGraph:
    """Build the backend-neutral recurrence graph from Trace postprocess state.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.

    Returns
    -------
    RecurrenceGroupingGraph
        Neutral graph containing only the fields the shared grouper needs.
    """

    nodes: dict[str, RecurrenceNode] = {}
    eligible_labels: list[str] = []
    raw_labels = tuple(self._raw_graph_ws.raw_layer_labels_list)
    raw_label_set = set(raw_labels)
    effective_equivalence = _differentiated_param_equivalence_classes(self)
    equivalent_labels_memo: dict[tuple[int, str], tuple[Any, tuple[str, ...]]] = {}
    recurrent_labels_memo: dict[int, tuple[Any, tuple[str, ...]]] = {}

    for label in raw_labels:
        node = self[label]
        effective_key = effective_equivalence.get(label, node.equivalence_class)
        raw_equivalent_labels = node._slot("equivalent_ops")
        # ``equivalent_labels`` must agree with ``equivalence_key``: the argsig
        # split subdivides a capture-time equivalence class, and passing the
        # UNSPLIT membership let every split key seed isomorphic expansion from
        # ALL original class members -- foreign-argsig subgraphs contributed
        # adjacency/param evidence and each split key re-ran a full expansion
        # over the same mixed seed set (deep-hunt L2). Members of one
        # capture-time class share ``equivalence_class`` by construction, so a
        # member's effective key defaults to this node's raw class.
        memo_key = (id(raw_equivalent_labels), effective_key)
        equivalent_labels_cached = equivalent_labels_memo.get(memo_key)
        if equivalent_labels_cached is None:
            equivalent_labels = tuple(
                member
                for member in raw_equivalent_labels
                if effective_equivalence.get(member, node.equivalence_class) == effective_key
            )
            equivalent_labels_memo[memo_key] = (
                raw_equivalent_labels,
                equivalent_labels,
            )
        else:
            equivalent_labels = equivalent_labels_cached[1]
        raw_recurrent_labels = node._slot("recurrent_ops")
        recurrent_labels_cached = recurrent_labels_memo.get(id(raw_recurrent_labels))
        if recurrent_labels_cached is None:
            recurrent_labels = tuple(raw_recurrent_labels)
            recurrent_labels_memo[id(raw_recurrent_labels)] = (
                raw_recurrent_labels,
                recurrent_labels,
            )
        else:
            recurrent_labels = recurrent_labels_cached[1]
        is_pruned = bool(getattr(node, "is_orphan", False))
        retain = not is_pruned
        if retain:
            eligible_labels.append(label)
        nodes[label] = RecurrenceNode(
            label=label,
            raw_order=node.raw_index,
            equivalence_key=effective_key,
            equivalent_labels=equivalent_labels,
            data_parents=tuple(parent for parent in node.parents if parent in raw_label_set),
            data_children=tuple(child for child in node.children if child in raw_label_set),
            layer_label=node._layer_label_raw,
            recurrent_labels=recurrent_labels,
            uses_params=bool(node.uses_params),
            func_name=node.func_name,
            param_barcodes=tuple(node._param_barcodes),
            output_slot=getattr(node, "multi_output_index", None),
            retain=retain,
            pruned=is_pruned,
            recurrence_anchored=(
                bool(getattr(node, "modules", None)) or bool(getattr(node, "is_buffer", False))
            ),
            module_site=_module_site(node),
            arg_signature=_structural_arg_signature(node) if node.uses_params else None,
            site_key=getattr(node, "site_key", None),
        )

    return RecurrenceGroupingGraph(
        nodes=nodes,
        raw_labels=raw_labels,
        source_labels=tuple(self.input_layers + self.internal_source_ops),
        eligible_labels=tuple(eligible_labels),
    )


def _apply_recurrence_assignments(
    self: "Trace",
    assignments: dict[str, RecurrenceAssignment],
) -> None:
    """Apply neutral recurrence assignments back onto Trace ops.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.
    assignments:
        Neutral assignments returned by ``group_recurrent_nodes``.
    """

    recurrent_labels_memo: dict[int, tuple[tuple[str, ...], list[str]]] = {}
    for label, assignment in assignments.items():
        node = self[label]
        recurrent_labels_cached = recurrent_labels_memo.get(id(assignment.recurrent_labels))
        if recurrent_labels_cached is None:
            recurrent_labels = list(assignment.recurrent_labels)
            recurrent_labels_memo[id(assignment.recurrent_labels)] = (
                assignment.recurrent_labels,
                recurrent_labels,
            )
        else:
            recurrent_labels = recurrent_labels_cached[1]
        node._layer_label_raw = assignment.layer_label
        node.recurrent_ops = recurrent_labels
        node.pass_index = assignment.pass_index
        node.num_passes = assignment.num_passes
        node.equivalence_class = assignment.equivalence_key


def _rebuild_pass_assignments(self: "Trace") -> None:
    """Rebuild recurrence membership and pass numbers from authoritative labels.

    Parameters
    ----------
    self:
        Trace whose ``_layer_label_raw`` values define recurrence groups.
    """

    groups: dict[str, list[str]] = defaultdict(list)
    for entry in self:
        if getattr(entry, "is_orphan", False):
            continue
        groups[entry._layer_label_raw].append(entry._label_raw)

    for members in groups.values():
        members_sorted = sorted(members, key=lambda label: self[label].raw_index)
        for pass_index, member_label in enumerate(members_sorted, start=1):
            member = self[member_label]
            member.recurrent_ops = members_sorted
            member.pass_index = pass_index
            member.num_passes = len(members_sorted)
