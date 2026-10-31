"""Single-sourced model-input boundary traversal (r65 Cluster Y, the normative dispatch).

Every walker of the MODEL-INPUT boundary tree routes through
:func:`walk_input_boundary`:

* W1 -- capture literal-leaf walker
  (``torchlens.capture.trace._record_runnable_input_literal_leaves``),
* W2 -- capture metadata-read-site walker
  (``torchlens.capture.trace._record_runnable_input_tensor_sites``),
* W3 -- runtime literal-path walker
  (``torchlens._runnable_execution._runtime_nontensor_leaf_paths``).

One shared container dispatch is an HONESTY invariant, not a refactor: the walkers
witness the same physical tree for different fact families (literal values vs
metadata-read sites), so a container kind handled by one walker but missed by another
silently drops a whole fact family for every leaf beneath it. That is exactly the r64
Finding-1 false-VERIFIED: the metadata-site walker did not descend dataclass
containers, so ``box.x.is_contiguous()`` on a dataclass input field recorded no
metadata witness and a same-value non-contiguous twin replayed the captured branch as
``verified``. With the dispatch single-sourced here, a future container kind MUST be
added to this module and is picked up by every walker in lockstep; a private container
branch inside a walker body is forbidden by a source-scan meta-test
(``tests/test_tlspec_runnable_r65_walker_parity.py``).

Container-kind vocabulary (CLOSED SET -- the supported model-input container set):

    ``tensor`` (leaf) / ``empty`` (zero-length dict-list-tuple-namedtuple) /
    ``namedtuple`` / ``dataclass`` / ``mapping`` / ``sequence`` (list-tuple) /
    ``leaf`` (anything else, including genuinely-opaque objects)

Dispatch ORDER is fixed and load-bearing: tensor, empty container, namedtuple,
dataclass, mapping, sequence, leaf. Empty precedes namedtuple so a zero-field
namedtuple is witnessed as an EMPTY container by KIND; namedtuple precedes dataclass
and mapping so hybrid types keep their historical classification.

Dual mapping-key vocabulary (DECLARED here, residual R6)
--------------------------------------------------------
Two mapping-key path-component vocabularies coexist, both defined in THIS module and
only here:

* :func:`tagged_mapping_key_component` -- the persisted LITERAL vocabulary (W1/W3,
  decoded at run time by ``_value_at_path``): keys are gated by the frozen literal
  grammar (a non-representable key routes its whole child subtree to
  ``on_opaque_key_subtree``, downgrading witness coverage instead of silently dropping
  the subtree) and a bool key is tagged ``(BOOL_KEY_PATH_TAG, key)`` so ``{True: x}``
  stays type-distinct from ``{1: x}`` in the persisted leaf-path set (r29-C2 / F6).
* :func:`raw_mapping_key_component` -- the metadata-fact-site vocabulary (W2, and the
  historical binding/W4 path family): the raw key object, every key accepted. Raw
  bool/int keys CONFLATE (``{True: x}`` and ``{1: x}`` hash equal), which is honest
  today because (a) fact-site paths are in-memory attribution keys resolved through a
  mapping that conflates the two identically at capture and at run, and (b) the
  bool-vs-int input-tree swap itself diverges through the r33 ``_type_strict_path``
  symmetric belt on the tensor-leaf set contract. Re-vocabularying W2 to tagged keys
  would change persisted fact paths for zero honesty gain. Full single-vocabulary
  unification is residual R6 -- a deliberate future persisted-path migration round,
  never a silent drift.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "COMPOSITE_LITERAL_COMPONENT_POLICY",
    "INPUT_CONTAINER_KINDS",
    "InstanceStateInspection",
    "classify_input_container",
    "classify_scalar",
    "inspect_instance_state",
    "instance_state_names",
    "raw_mapping_key_component",
    "reserved_input_path_components",
    "slice_semantic_component",
    "tagged_mapping_key_component",
    "walk_input_boundary",
]


INPUT_CONTAINER_KINDS: frozenset[str] = frozenset(
    {"tensor", "empty", "namedtuple", "dataclass", "mapping", "sequence", "registered", "leaf"}
)
"""The CLOSED container-kind vocabulary of the model-input boundary (see module doc).

r67 C2 adds ``registered`` (a ``tl.types.register_container`` type, descended through its own
``flatten`` with indexed children) and extends ``empty`` to zero-field dataclasses.
"""


MAX_INPUT_CONTAINER_DEPTH: int = 200

INPUT_TREE_MAX_DEPTH: int = MAX_INPUT_CONTAINER_DEPTH
"""The shared model-input boundary nesting ceiling (r-b4 R27-1).

Mirrors the artifact-side literal-decode ceiling
(``_runnable_execution._MAX_DECODE_NESTING_DEPTH`` / the ``_io`` parse quartet):
200 sits far above any real input nesting and well below the interpreter's
default recursion crash depth (the live walkers burn ~2-3 frames per level, so
an unbounded walk died with a raw ``RecursionError`` at ~350 user levels).
Every capture-entry input walker (``walk_input_boundary``,
``snapshot_input_boundary``, ``backends.default_specs._simple_leaves``,
``utils.arg_handling.copy_arg_tree``) enforces this ONE ceiling with a typed
refusal instead of an untyped stdlib crash.
"""


def raise_input_tree_depth_refusal(*, depth: int) -> None:
    """Raise the typed over-depth input-boundary refusal (r-b4 R27-1).

    Parameters
    ----------
    depth:
        Nesting depth at which the ceiling was crossed.
    """

    from torchlens._errors import InvalidArgumentError

    raise InvalidArgumentError(
        "Model-input tree nesting exceeds the supported input-boundary depth "
        f"ceiling ({INPUT_TREE_MAX_DEPTH}).",
        code="input_tree_depth_exceeded",
        remedy=(
            "Flatten the nested input containers (or unwrap the deep wrapper "
            "object) before tracing; the ceiling matches the portable-artifact "
            "nesting bound."
        ),
        depth=depth,
    )


def raise_input_tree_stack_refusal(cause: BaseException | None = None) -> None:
    """Raise the typed stack-budget input-boundary refusal (grind-p3 T11.4).

    The nesting ceiling is a DEPTH bound, but the actual failure class is
    STACK BUDGET: the live walkers burn ~2-3 interpreter frames per level, so
    a LEGAL input (nesting <= the ceiling) still died in a raw
    ``RecursionError`` whenever the caller entered capture with most of the
    stack already consumed (deep user recursion, constrained
    ``sys.setrecursionlimit``). Every walker entry converts that exhaustion
    into this typed refusal instead of an untyped stdlib crash.

    Parameters
    ----------
    cause:
        The caught ``RecursionError``, chained as ``__cause__``.
    """

    from torchlens._errors import InvalidArgumentError

    error = InvalidArgumentError(
        "Walking the model-input tree exhausted the Python stack budget before "
        f"reaching the depth ceiling ({INPUT_TREE_MAX_DEPTH}): capture was "
        "entered with most of the interpreter stack already consumed.",
        code="input_tree_stack_exhausted",
        remedy=(
            "Call the capture entry point from a shallower call stack, or raise "
            "sys.setrecursionlimit() to leave headroom for the bounded input walk."
        ),
    )
    raise error from cause


def raise_input_tree_cycle_refusal(*, kind: str) -> None:
    """Raise the typed cyclic-container input-boundary refusal (r-b4 R27-1).

    Parameters
    ----------
    kind:
        Container kind (closed vocabulary) at which the cycle closed.
    """

    from torchlens._errors import InvalidArgumentError

    raise InvalidArgumentError(
        f"Model-input tree contains a self-referential {kind} container "
        "(a container reachable from itself).",
        code="input_tree_cycle",
        remedy="Remove the container reference cycle from the model input.",
        kind=kind,
    )


def raise_input_tree_namedtuple_refusal(*, declared: int, physical: int) -> None:
    """Raise the typed non-total-namedtuple-schema input refusal (B3R4-R12-1).

    A tuple subclass DECLARING a ``_fields`` schema that does not account for
    the physical tuple (a malformed non-tuple-of-str ``_fields``, or a declared
    arity differing from the physical arity) used to descend only the declared
    names -- NOTHING for a malformed schema -- so every remaining tensor leaf
    silently vanished from the capture-side walkers: the trace had no input
    node and the gap was misattributed to a stale-reference escape.

    Parameters
    ----------
    declared:
        Number of usable declared field names (0 for a malformed schema).
    physical:
        Concrete builtin tuple arity.
    """

    from torchlens._errors import InvalidArgumentError

    raise InvalidArgumentError(
        "Model-input tree contains a tuple subclass declaring a namedtuple "
        "`_fields` schema that does not account for the physical tuple "
        f"({declared} usable declared field(s) vs {physical} physical "
        "element(s); a non-tuple-of-str `_fields` declaration counts as 0).",
        code="input_namedtuple_schema_not_total",
        remedy=(
            "Fix the container's `_fields` declaration (a tuple of one str per "
            "positional element) or pass a plain tuple/list instead."
        ),
        declared_fields=declared,
        physical_arity=physical,
    )


"""Declared nesting bound for the input-boundary walk, mirroring the DECODE direction.

``_runnable_execution._MAX_DECODE_NESTING_DEPTH`` bounds the symmetric decode direction
precisely so a hostile artifact cannot produce an "uncaught ``RecursionError``"; the walk
direction had no equivalent, so a self-referential input container (``d["self"] = d`` --
trivially produced by any graph/tree structure or back-reference) or a legitimately deep
nest ended the capture with a raw ``RecursionError`` from library internals, at a depth
that also depended on how much stack the caller had already consumed. ``RecursionError``
is neither ``_UnsupportedLiteralError`` nor a ``RunnableDiagnostic``, so no honest
degradation was possible.
"""

_UNSET_FIELD = object()
"""Sentinel for a DECLARED dataclass field that the instance never set.

``dataclasses.fields`` enumerates a ``field(init=False)`` entry with no default even
though ``getattr`` raises for it, so both walkers' raw ``getattr`` killed every
intervention-ready capture of a legal dataclass input (a lazily-populated cache/scratch
field) with an untyped ``AttributeError`` from library internals.
"""


def _declared_field_value(value: Any, name: str) -> Any:
    """Read one declared dataclass field, returning :data:`_UNSET_FIELD` if unset."""

    try:
        return getattr(value, name)
    except AttributeError:
        return _UNSET_FIELD


def unset_declared_fields(value: Any) -> tuple[str, ...]:
    """Return the DECLARED dataclass field names the instance does not carry."""

    if not (dataclasses.is_dataclass(value) and not isinstance(value, type)):
        return ()
    return tuple(
        field.name
        for field in dataclasses.fields(value)
        if _declared_field_value(value, field.name) is _UNSET_FIELD
    )


def declares_namedtuple_fields(value: Any) -> bool:
    """Return whether ``value`` is a ``tuple`` subclass DECLARING a ``_fields`` schema.

    Resolved through the raw MRO (:func:`_raw_mro_attr`), never through the instance.
    The historical ``hasattr(value, "_fields")`` spelling ran the instance's
    ``__getattribute__``/``__getattr__`` -- the exact untrusted hook the r71 C contract
    promises never executes during ``snapshot_input_boundary`` -- and let the hook STEER
    classification: a genuine namedtuple subclass whose ``__getattribute__`` raised
    ``AttributeError`` for ``"_fields"`` classified as a plain ``sequence``, so neither
    ``_declared_schema_uninspectable`` nor :func:`undeclared_instance_state` was ever
    consulted and its hidden instance state escaped judgment entirely.

    Presence is judged separately from VALIDITY: a declared-but-malformed ``_fields``
    (a list, a property) still classifies as ``namedtuple`` so the schema-totality
    refusal fires, instead of silently degrading to a zero-field container.
    """

    return isinstance(value, tuple) and _raw_mro_attr(value, "_fields") is not None


def physical_sequence_len(value: Any) -> int:
    """Return the CONCRETE builtin arity of a tuple/list/dict-backed container.

    ``len(value)`` dispatches to an overridable instance ``__len__``: hostile user
    code could both steer the container KIND and forge the recorded physical
    arity (a zero-field namedtuple subclass with ``__len__() == 0`` physically
    carrying ``(tensor, "steer")`` classified ``empty`` and dropped its children
    -- tensors included -- from every walker, with the SAME wrong value computed
    on the capture and runtime snapshots, i.e. a false-VERIFIED shape). Arity
    facts therefore read the concrete builtin slot, consistent with how this
    module already resolves ``_fields``/``__dict__`` through the raw MRO.
    Containers not backed by a builtin (custom ``Mapping`` implementations) have
    no physical storage distinct from their methods and keep the instance
    protocol; their hidden-state honesty is owned by the instance-state proofs.
    """

    if isinstance(value, tuple):
        return tuple.__len__(value)
    if isinstance(value, list):
        return list.__len__(value)
    if isinstance(value, dict):
        return dict.__len__(value)
    return len(value)


def iter_physical_sequence(value: Any) -> Iterator[tuple[int, Any]]:
    """Yield ``(index, child)`` through the concrete builtin slots (inert descent).

    ``enumerate(value)`` dispatches to an overridable ``__iter__`` -- the same
    forgery lane as a lying ``__len__`` -- so tuple/list-backed sequences descend
    by concrete indexed access over :func:`physical_sequence_len`.
    """

    if isinstance(value, tuple):
        for index in range(tuple.__len__(value)):
            yield index, tuple.__getitem__(value, index)
        return
    if isinstance(value, list):
        for index in range(list.__len__(value)):
            yield index, list.__getitem__(value, index)
        return
    yield from enumerate(value)


def empty_input_container_kind(value: Any) -> str | None:
    """Return the KIND of an EMPTY non-tensor container, else ``None`` (inert; r29-C2).

    The canonical implementation, resolving a namedtuple's declared schema through
    :func:`declares_namedtuple_fields`/:func:`_instance_fields` rather than a live
    ``value._fields`` read. Two behavioural fixes over the historical spelling:

    * the namedtuple arm also requires ``len(value) == 0``. A zero-field namedtuple
      physically carrying ``(tensor, "steer")`` used to classify ``empty``, which
      dropped its children -- INCLUDING TENSORS -- from every walker with no refusal
      and no witness gap, i.e. a false VERIFIED shape;
    * a declared-but-malformed ``_fields`` is not treated as "zero fields".
    """

    if declares_namedtuple_fields(value):
        return (
            "namedtuple"
            if len(_instance_fields(value)) == 0 and physical_sequence_len(value) == 0
            else None
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "dataclass" if len(dataclasses.fields(value)) == 0 else None
    if isinstance(value, Mapping):
        return "mapping" if physical_sequence_len(value) == 0 else None
    if isinstance(value, (list, tuple)):
        return "sequence" if physical_sequence_len(value) == 0 else None
    return None


def namedtuple_arity_mismatch(value: Any) -> bool:
    """Return whether a namedtuple-kind container's declared schema is not TOTAL.

    ``True`` when the declared ``_fields`` schema does not account for the physical
    tuple, i.e. a malformed ``_fields`` (non-tuple, non-str members) or
    ``len(value) != len(_fields)``.

    Physical arity was never witnessed: ``tuple.__new__(P, (t, extra))`` on a one-field
    namedtuple walked only ``('x',)``, so snapshots of instances with DIFFERENT hidden
    payloads compared EQUAL, the extra tensor appeared in no ``input_tensor_sites`` and
    in no leaf-path set, and hidden positional state read via ``box[1]`` steered forward
    control flow with no witness -- the recorded path then replayed VERIFIED on a
    changed payload.
    """

    if not declares_namedtuple_fields(value):
        return False
    fields = _instance_fields(value)
    if not fields:
        # Declared but unusable (``_fields`` is a list / a property / holds non-str
        # members). ``_instance_fields``' documented fail-closed did not fire here: for
        # a tuple subclass the instance ``__dict__`` is empty, so
        # ``undeclared_instance_state`` returned False and the node was recorded as a
        # zero-field namedtuple with every child dropped.
        return True
    return len(fields) != physical_sequence_len(value)


def mapping_protocol_entries_mismatch(
    value: Any, protocol_entries: list[tuple[Any, Any]] | None = None
) -> bool:
    """Return whether a dict-backed mapping's protocol view forges its storage.

    The totality fact used to compare only COUNT (``len(items())`` vs the
    concrete ``dict.__len__``), so a count-PRESERVING substitution -- a lying
    ``items()`` presenting ``[("visible", t), ("decoy", zeros)]`` over physical
    storage ``{"visible": t, "hidden": t_h}`` -- still walked the decoy, put the
    hidden tensor in NO snapshot node, and made two instances with DIFFERENT
    hidden payloads snapshot EQUAL while concrete ``dict.__getitem__`` reads
    steered the forward (R12: the last open member of the forgery family whose
    shrink/pad and namedtuple siblings are already refused).

    The comparison is entry-level and order-insensitive: the protocol view and
    the concrete ``dict.items(value)`` storage must bind the SAME canonical key
    tokens to the SAME child objects (identity, not equality -- an equal-valued
    decoy object is exactly the forgery). Order divergence alone (a sorted-view
    subclass) stays legal: the ordered-key witness is derived from the protocol
    traversal the model actually iterates. A key the codec cannot encode
    compares by object identity, so an opaque key must be THE one physical key
    object on both sides. Non-dict-backed Mappings have no physical storage
    distinct from their methods and keep the instance protocol (their
    hidden-state honesty is owned by the instance-state proofs).
    """

    if not isinstance(value, dict):
        return False
    entries = list(value.items()) if protocol_entries is None else protocol_entries
    concrete = list(dict.items(value))
    if len(entries) != len(concrete):
        return True

    def _view(pairs: list[tuple[Any, Any]]) -> dict[Any, int]:
        """Map each pair's encoded key (identity for opaque keys) to its child id."""

        view: dict[Any, int] = {}
        for key, child in pairs:
            try:
                token: Any = encode_mapping_key(key)
            except ValueError:
                token = ("opaque-key-identity", id(key))
            view[token] = id(child)
        return view

    return _view(entries) != _view(concrete)


def classify_input_container(value: Any) -> str:
    """Classify one boundary value into the closed container-kind vocabulary.

    This is THE single container dispatch (order is load-bearing; see the module
    docstring). :func:`walk_input_boundary` consumes it, so adding a container kind
    here extends every input-boundary walker in lockstep.

    Every branch is INERT: no instance attribute read, so a hostile container's
    ``__getattribute__``/``__getattr__`` cannot run here and cannot steer the kind.
    """

    import torch

    from torchlens._ir.container import get_registered_container

    if isinstance(value, torch.Tensor):
        return "tensor"
    # r67 C2 (corr1-3): a REGISTERED container type is the user's explicit declaration and
    # wins over the generic kinds -- capture and runtime both descend its ``flatten``
    # children, so a registered input can never be admitted at save and then fail
    # traversal at bind (advertise-then-fail).
    if not isinstance(value, type) and get_registered_container(type(value)) is not None:
        return "registered"
    if empty_input_container_kind(value) is not None:
        return "empty"
    if declares_namedtuple_fields(value):
        return "namedtuple"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "dataclass"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return "leaf"


def tagged_mapping_key_component(key: Any) -> Any:
    """Persisted-literal mapping-key vocabulary (W1/W3; decoded by ``_value_at_path``).

    Gates the key through the frozen literal grammar (``_encode_literal_key`` raises
    ``_UnsupportedLiteralError`` for a non-representable key -- enum, object, bytes,
    ... -- which :func:`walk_input_boundary` routes to ``on_opaque_key_subtree``) and
    tags a bool key as ``(BOOL_KEY_PATH_TAG, key)`` so it stays type-distinct from
    the equal-valued int key (r29-C2 / F6).
    """

    from torchlens._io.runnable import _encode_literal_key, input_path_key_component

    _encode_literal_key(key)
    return input_path_key_component(key)


def raw_mapping_key_component(key: Any) -> Any:
    """Metadata-fact-site mapping-key vocabulary (W2): the ONE codec, every key accepted.

    r67 C2 retires the dual vocabulary (residual R6): W2 components now come from the
    SAME type-strict codec as the persisted literal paths, so ``{True: x}`` and
    ``{1: x}`` fact sites stay distinct and every path family speaks one language. A
    NON-grammar key keeps the raw key object as an in-memory-only attribution component
    (W2 must accept every key; the literal walker's OPAQUE leaf independently ceilings
    such a subtree, so a dropped metadata fact under an exotic key can never yield a
    false VERIFIED).
    """

    try:
        return encode_mapping_key(key)
    except ValueError:
        return key


def walk_input_boundary(
    value: Any,
    path: tuple[Any, ...] = (),
    *,
    key_component: Callable[[Any], Any],
    on_tensor: Callable[[Any, tuple[Any, ...]], None] | None = None,
    on_leaf: Callable[[Any, tuple[Any, ...]], None] | None = None,
    on_empty_container: Callable[[str, tuple[Any, ...]], None] | None = None,
    on_opaque_key_subtree: Callable[[Any, tuple[Any, ...]], None] | None = None,
) -> None:
    """Traverse one model-input boundary value with the single container dispatch.

    Parameters
    ----------
    value:
        The boundary value to descend.
    path:
        Container path accumulated so far (empty at a top-level site).
    key_component:
        REQUIRED mapping-key vocabulary hook -- one of
        :func:`tagged_mapping_key_component` / :func:`raw_mapping_key_component`
        (every caller must consciously pick a declared vocabulary). It returns the
        path component for a mapping key, or raises ``_UnsupportedLiteralError`` to
        route the key's child subtree to ``on_opaque_key_subtree``.
    on_tensor:
        Called ``(tensor, path)`` for every tensor leaf. ``None`` skips tensors.
    on_leaf:
        Called ``(value, path)`` for every non-tensor, non-container leaf.
        ``None`` ignores such leaves.
    on_empty_container:
        Called ``(kind, path)`` for every EMPTY container, where ``kind`` is the
        ``empty_input_container_kind`` string. ``None`` ignores empty containers.
    on_opaque_key_subtree:
        Called ``(child, parent_path)`` once per mapping key rejected by
        ``key_component`` (the child subtree is NOT descended). ``None`` skips.
    """

    from torchlens._io.runnable import _UnsupportedLiteralError

    # Identity of every CONTAINER on the current descent chain, plus a declared depth
    # bound: a cycle or an over-deep nest ceilings its subtree through the existing
    # opaque channel (witness coverage downgrades, the runnable save refuses through the
    # symmetric snapshot refusal) instead of raising ``RecursionError`` from internals.
    # Ancestors only, NOT a global seen-set: the same leaf legitimately appears at many
    # paths (a shared tensor in two slots) and every occurrence must still be witnessed.
    active_ids: set[int] = set()

    def _descend(value: Any, path: tuple[Any, ...]) -> None:
        """Dispatch one node through the closed container-kind vocabulary."""

        kind = classify_input_container(value)
        if kind not in {"tensor", "leaf"}:
            if id(value) in active_ids or len(path) > MAX_INPUT_CONTAINER_DEPTH:
                if on_opaque_key_subtree is not None:
                    on_opaque_key_subtree(value, path)
                return
            active_ids.add(id(value))
            try:
                _descend_container(value, kind, path)
            finally:
                active_ids.discard(id(value))
            return
        _descend_container(value, kind, path)

    def _descend_container(value: Any, kind: str, path: tuple[Any, ...]) -> None:
        """Dispatch one already-classified, already-cycle-checked node."""

        if kind == "tensor":
            if on_tensor is not None:
                on_tensor(value, path)
            return
        if kind == "empty":
            if on_empty_container is not None:
                empty_kind = empty_input_container_kind(value)
                assert empty_kind is not None  # classify_input_container said "empty"
                on_empty_container(empty_kind, path)
            return
        if kind == "registered":
            # r67 C2 (corr1-3): descend the registration's OWN flatten children with
            # indexed components; a throwing/nonconforming hook routes the node to the
            # opaque handler (fail closed: witness coverage downgrades / save refuses),
            # never a silent drop and never an advertise-then-fail at bind.
            from torchlens._ir.container import get_registered_container

            registration = get_registered_container(type(value))
            try:
                children = list(registration.flatten(value)[0]) if registration else None
            except Exception:
                children = None
            if children is None:
                if on_opaque_key_subtree is not None:
                    on_opaque_key_subtree(value, path)
                return
            for index, child in enumerate(children):
                _descend(child, (*path, index))
            return
        if kind == "namedtuple":
            if namedtuple_arity_mismatch(value):
                # Fail closed (B3R4-R12-1): descending only the declared names
                # -- NOTHING for a malformed schema -- silently dropped the
                # remaining tensor leaves from every capture-side walker (the
                # trace showed NO input node and misattributed the gap as an
                # "escape" while settling COMPLETE). A typed refusal is the
                # honest outcome for BOTH the malformed-``_fields`` shape and a
                # valid schema with the wrong arity: tracing on with silently
                # unattributed leaves is the dishonesty this tripwire exists to
                # stop. (Supersedes the T11.7-era trace-through, whose own test
                # tolerated the unattributed gap; the snapshot walker records
                # the symmetric ``namedtuple_schema_not_total`` refusal fact.)
                raise_input_tree_namedtuple_refusal(
                    declared=len(_instance_fields(value)),
                    physical=physical_sequence_len(value),
                )
            for name in _instance_fields(value):
                _descend(getattr(value, name), (*path, str(name)))
            return
        if kind == "dataclass":
            for field in dataclasses.fields(value):
                child = _declared_field_value(value, field.name)
                if child is _UNSET_FIELD:
                    # A declared-but-unset ``init=False`` field carries no value to
                    # witness. Skipping it symmetrically on the capture and runtime
                    # walkers keeps their leaf-path sets comparable; the snapshot's
                    # ``unset_declared_field`` refusal is what fails the runnable save.
                    continue
                _descend(child, (*path, field.name))
            return
        if kind == "mapping":
            if mapping_protocol_entries_mismatch(value):
                # A dict-backed mapping whose protocol view disagrees with its
                # concrete storage at the ENTRY level (R12: count-preserving
                # substitution included) routes the WHOLE node to the opaque
                # channel instead of walking the forged view: witness coverage
                # ceilings, and the runnable save refuses through the snapshot
                # walker's symmetric ``mapping_protocol_not_total`` refusal --
                # the same fail-closed channel cycles and over-deep nests use.
                if on_opaque_key_subtree is not None:
                    on_opaque_key_subtree(value, path)
                return
            for key, child in value.items():
                try:
                    component = key_component(key)
                except _UnsupportedLiteralError:
                    if on_opaque_key_subtree is not None:
                        on_opaque_key_subtree(child, path)
                    continue
                _descend(child, (*path, component))
            return
        if kind == "sequence":
            for index, child in iter_physical_sequence(value):
                _descend(child, (*path, index))
            return
        if on_leaf is not None:
            on_leaf(value, path)

    try:
        _descend(value, path)
    except RecursionError as exc:
        # A legal (<= ceiling) tree can still exhaust the stack when the caller
        # entered capture deep in its own recursion; refuse typed (T11.4).
        raise_input_tree_stack_refusal(exc)


def refuse_nontotal_namedtuple_inputs(
    input_args: Iterable[Any], input_kwargs: Mapping[Any, Any]
) -> None:
    """Refuse capture entry on any non-total namedtuple input schema (B3R4-R12-1).

    The runnable walkers (W1/W2/W3) run only for intervention-ready captures,
    so a PLAIN capture never traversed its input boundary through the shared
    dispatch: a malformed-``_fields`` tuple subclass reached the tensor-
    extraction BFS, which cannot see positional slots of tuple subclasses, and
    every tensor leaf under it silently vanished (no input node, parents lost,
    the gap misattributed to a stale-reference escape). One inert traversal per
    input site at capture entry makes the typed schema-totality refusal fire
    for every capture, matching the documented
    :func:`declares_namedtuple_fields` contract.

    Parameters
    ----------
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    for value in (*input_args, *input_kwargs.values()):
        walk_input_boundary(value, (), key_component=raw_mapping_key_component)


# --- r67 C2: the input-boundary SNAPSHOT spine -----------------------------------------------
#
# The r66 C2 findings (free-F2/F3, hon1-F1/F2a/F2b/F2c, corr1-3) share one root: the structure
# witness recorded ROOT kind only, so nested kinds, exact classes, empty-dataclass arity,
# non-str/int grammar keys, hidden instance state, and registered containers could be lost or
# treated inconsistently. One traversal per model-input site now produces an immutable
# per-node record list -- kind, exact (module, qualname) type, declared child schema, ordered
# type-strict-encoded mapping keys, and an instance-state/losslessness proof -- persisted by
# the runnable producer and re-derived at bind time by the SAME function, so capture and
# runtime can never diverge in vocabulary.

_KEY_CODEC_TAG = "\x00tlk:"
"""Reserved prefix of the canonical type-strict mapping-key codec (r67 C2).

Every non-``str``/``int`` grammar key (bool, float, None, safe tuple) -- and any string key
that could collide with an encoding -- is encoded into ONE tagged string component, so the
persisted path/key shape stays ``str | int`` while ``True != 1`` and ``1.0 != 1`` stay
type-distinct (``hash(True) == hash(1) == hash(1.0)`` would otherwise conflate them in path
sets and ordered-key facts).
"""

_RESERVED_NAMESPACE_PREFIX = "\x00"
"""The reserved input-path component namespace (r71 D).

Every TorchLens-synthesized path sentinel lives under the null-byte prefix
(``EMPTY_CONTAINER_PATH_MARKER`` / ``BOOL_KEY_PATH_TAG`` / the codec tag). A REAL dict
key sharing the prefix is escaped by :func:`encode_mapping_key` into the ``r:`` lane so
it can never collide with a sentinel interpreted before mapping lookup.
"""


def reserved_input_path_components() -> frozenset[str]:
    """The closed registry of reserved input-path SENTINELS (r71 D).

    Every string sentinel any runtime consumer interprets POSITIONALLY before a mapping
    lookup (``_value_at_path``) is enumerated here; the r71 D meta-test asserts each such
    sentinel is a member AND that ``encode_mapping_key(sentinel) != sentinel`` (a real
    dict key equal to a sentinel is always escaped, never round-tripped raw).
    """

    from torchlens._io.runnable import BOOL_KEY_PATH_TAG, EMPTY_CONTAINER_PATH_MARKER

    return frozenset({EMPTY_CONTAINER_PATH_MARKER, BOOL_KEY_PATH_TAG, _KEY_CODEC_TAG})


def _stock_numpy_scalar_types() -> frozenset[type]:
    """Every CONCRETE stock NumPy scalar class, enumerated programmatically (r69 B).

    Identity-based membership from ``np.typecodes['All']`` -- never a name/module
    string check, so a user subclass can never enter the set regardless of spoofed
    ``__module__``/``__qualname__``.
    """

    global _STOCK_NP_SCALAR_CACHE
    if _STOCK_NP_SCALAR_CACHE is None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is a hard torchlens dependency
            _STOCK_NP_SCALAR_CACHE = (frozenset(), frozenset())
        else:
            all_types: set[type] = set()
            transparent: set[type] = set()
            for code in np.typecodes["All"]:
                try:
                    dtype = np.dtype(code)
                except TypeError:  # pragma: no cover - stale typecode entry
                    continue
                all_types.add(dtype.type)
                # Transparent VALUE wrappers: numeric/bool kinds whose ``.item()``
                # round-trips exactly (itemsize <= 8 excludes lossy longdouble).
                if dtype.kind in "biuf" and dtype.itemsize <= 8:
                    transparent.add(dtype.type)
            _STOCK_NP_SCALAR_CACHE = (frozenset(all_types), frozenset(transparent))
    return _STOCK_NP_SCALAR_CACHE[0]


def _stock_numpy_transparent_scalar_types() -> frozenset[type]:
    """Exact stock NumPy numeric/bool wrapper classes admitted as VALUE-transparent."""

    _stock_numpy_scalar_types()
    assert _STOCK_NP_SCALAR_CACHE is not None
    return _STOCK_NP_SCALAR_CACHE[1]


_STOCK_NP_SCALAR_CACHE: tuple[frozenset[type], frozenset[type]] | None = None


def classify_scalar(value: Any) -> tuple[str, Any]:
    """THE closed input-boundary scalar type-class lattice (r69 B/D).

    Every scalar type dispatch on the model-input boundary -- snapshot leaf
    disposition, literal encoding (``_encode_literal``), runtime literal comparison
    (``_literal_leaf_equal``), and mapping-key encoding (``encode_mapping_key``) --
    consumes this ONE classifier, so a scalar can never be classified two different
    ways by two different code paths (the r68 B/D conflation root). Classification
    order is load-bearing:

    1. ``("semantic", None)`` -- ``enum.Enum`` FIRST (``IntEnum`` / ``IntFlag`` /
       ``StrEnum`` / float-Enum / plain ``Enum``), before any numeric family, then
       (after the exact/stock checks below) every other ``int``/``float``/``str``
       subclass and every non-stock ``np.generic`` subclass. Type identity is
       application semantics (``isinstance``/``is Mode.FAST`` steering); such a
       VALUE refuses runnable save typed and such a runtime value diverges -- it
       never collapses into the builtin atom lane.
    2. ``("plain", value)`` -- ``type(value)`` is EXACTLY ``bool``/``int``/``float``/
       ``str`` (exact ``type() is``, never ``isinstance``).
    3. ``("singleton", value)`` -- ``None`` / ``Ellipsis`` (their existing grammar).
    4. ``("numpy", value.item())`` -- ``type(value)`` is EXACTLY a stock NumPy
       numeric/bool wrapper class (programmatic identity set; broad
       ``isinstance(value, np.generic)`` is forbidden for this admission) AND
       ``.item()`` lands on an exact builtin numeric/bool atom. The RATIFIED
       transparent-wrapper VALUE lane: ``np.float64(2.0) <-> 2.0`` verifies.
       A stock wrapper whose ``.item()`` is not an exact builtin numeric/bool
       falls to ``semantic`` (fail closed, platform ``longdouble`` edges).
    5. ``("opaque", None)`` -- everything else (bytes, complex, Decimal, datetime64,
       arbitrary objects, containers): the existing opaque/outside-grammar lane.
    """

    import enum as _enum

    if isinstance(value, _enum.Enum):
        return ("semantic", None)
    value_type = type(value)
    if value_type in (bool, int, float, str):
        return ("plain", value)
    if value is None or value is Ellipsis:
        return ("singleton", value)
    if value_type in _stock_numpy_transparent_scalar_types():
        try:
            item = value.item()
        except Exception:  # pragma: no cover - stock wrappers implement item()
            return ("semantic", None)
        if type(item) in (bool, int, float):
            return ("numpy", item)
        return ("semantic", None)
    if isinstance(value, (int, float, str)):
        # A builtin-scalar-family subclass outside the exact/stock sets (custom
        # int/float/str subclasses, np.str_, subclasses of np.float64, ...).
        return ("semantic", None)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard torchlens dependency
        return ("opaque", None)
    if isinstance(value, np.generic) and value_type not in _stock_numpy_scalar_types():
        # A non-stock np.generic subclass is user semantics, never a wrapper.
        return ("semantic", None)
    return ("opaque", None)


COMPOSITE_LITERAL_COMPONENT_POLICY: Mapping[str, tuple[str, ...]] = {
    # r71 B: the ONE shared composite-literal component policy table. Each recursive
    # ``NonTensorLiteral`` composite node kind lists the classifier-backed lane every
    # sub-component MUST route through, so a semantic-typed scalar can never launder
    # through ANY composite edge (secA-F1: slice components were the ungated seam).
    # This table is BOTH implementation authority and the future-kind meta-test: a
    # new composite ``NonTensorLiteral`` without a declared policy REDs
    # ``test_r71b_composite_component_policy_covers_every_edge``.
    "LiteralSlice": ("classify_scalar",),  # start / stop / step (int|None, no semantic)
    "LiteralSequence": ("_encode_literal",),  # items recurse through the classifier-first encoder
    "LiteralMapping": (
        "encode_mapping_key",
        "_encode_literal",
    ),  # keys via codec, values via encoder
    "LiteralTupleKey": ("encode_mapping_key",),  # recursive tuple-key items via the key codec
}
"""Closed policy: every composite ``NonTensorLiteral`` edge -> its classifier lane (r71 B)."""


def slice_semantic_component(value: slice) -> str | None:
    """Return the field name of the first semantic-typed slice component, else ``None``.

    r71 B: applies the ``LiteralSlice`` component policy (``classify_scalar`` first) to
    a runtime/capture ``slice`` leaf. A component classifying ``semantic`` (enum member,
    builtin/np-scalar subclass) is outside the grammar -- the snapshot preflight and the
    runtime comparison both refuse/diverge on it, so a plain-int capture and a same-value
    semantic runtime component can never falsely compare equal through ``slice.__eq__``.
    """

    for field_name in ("start", "stop", "step"):
        component = getattr(value, field_name)
        if component is None:
            continue
        if classify_scalar(component)[0] == "semantic":
            return field_name
    return None


def encode_mapping_key(key: Any) -> str | int:
    """Encode one grammar mapping key into the canonical type-strict component (r69 D).

    THE sole mapping-key identity authority: snapshot ordered-key facts, literal
    leaf paths, tensor binding paths, and runtime lookup all speak this token
    vocabulary, and runtime binding compares canonical tokens -- never decoded
    values. Admission is classifier-gated (exact builtin ``str``/``int``/``bool``/
    ``float``/``None`` and EXACT tuples whose members recursively satisfy the
    grammar); enums, NumPy scalar wrappers, builtin-scalar subclasses, and tuple
    subclasses raise ``ValueError`` and refuse typed at save (``opaque_mapping_key``)
    -- a key is never admitted and then advertised-then-failed later.

    Token codec: ``str`` keys pass through unless they collide with the codec prefix
    (then escaped); ``int`` keys pass through as ints; ``bool``/``None`` use tagged
    strings; finite floats use exact ``float.hex()`` (``-0.0`` distinct from
    ``+0.0``); a NaN key encodes its binary64 BIT PATTERN
    (``f:nan:<16 hex digits>``) so distinct NaN payloads stay distinct and an
    identical NaN key can bind itself. The codec is injective over every admitted
    mapping node (duplicate-token nodes refuse at save as ``ambiguous_mapping_key``).
    """

    kind, _payload = classify_scalar(key)
    if kind == "semantic":
        raise ValueError(
            f"Mapping key {key!r} carries semantic type identity (enum or scalar "
            "subclass) and is outside the input-boundary key grammar."
        )
    if kind == "numpy":
        raise ValueError(
            f"Mapping key {key!r} is a NumPy scalar wrapper; key identity is "
            "structural, so wrapper keys are outside the input-boundary key grammar."
        )
    if isinstance(key, bool):
        return f"{_KEY_CODEC_TAG}b:{'1' if key else '0'}"
    if type(key) is int:
        return key
    if type(key) is str:
        if key.startswith(_KEY_CODEC_TAG):
            # Legacy ``s:`` lane: a real key colliding with the codec prefix itself.
            return f"{_KEY_CODEC_TAG}s:{key[len(_KEY_CODEC_TAG) :]}"
        if key.startswith(_RESERVED_NAMESPACE_PREFIX):
            # r71 D: any OTHER null-byte-prefixed key could collide with a reserved
            # input-path sentinel (``EMPTY_CONTAINER_PATH_MARKER`` / ``BOOL_KEY_PATH_TAG``
            # / any future ``\x00`` marker interpreted positionally before mapping
            # lookup). Escape the WHOLE ``\x00`` namespace into the unambiguous
            # ``r:<UTF-8 hex>`` lane so a real dict key equal to a marker round-trips
            # instead of false-DIVERGING on the unchanged input.
            return f"{_KEY_CODEC_TAG}r:{key.encode('utf-8').hex()}"
        return key
    if key is None:
        return f"{_KEY_CODEC_TAG}n:"
    if type(key) is float:
        if key != key:  # NaN: float.hex() collapses every payload to "nan"
            import struct as _struct

            return f"{_KEY_CODEC_TAG}f:nan:{_struct.pack('>d', key).hex()}"
        return f"{_KEY_CODEC_TAG}f:{key.hex()}"
    if type(key) is tuple:
        parts = []
        for item in key:
            encoded = encode_mapping_key(item)
            text = f"i:{encoded}" if isinstance(encoded, int) else f"c:{encoded}"
            parts.append(text.replace("\\", "\\\\").replace("|", "\\|"))
        return f"{_KEY_CODEC_TAG}t:{'|'.join(parts)}"
    raise ValueError(f"Mapping key {key!r} is outside the input-boundary key grammar.")


def decode_mapping_key(component: str | int) -> Any:
    """Decode one canonical component back to its mapping key (mapping nodes only).

    Retained for legitimate output-dict reconstruction and diagnostics; RUNTIME
    INPUT BINDING never decodes -- ``_value_at_path`` encodes each runtime candidate
    and compares exact canonical tokens (r69 D).
    """

    if isinstance(component, int) or not component.startswith(_KEY_CODEC_TAG):
        return component
    body = component[len(_KEY_CODEC_TAG) :]
    tag, _, payload = body.partition(":")
    if tag == "b":
        return payload == "1"
    if tag == "s":
        return _KEY_CODEC_TAG + payload
    if tag == "r":
        # r71 D: the reserved-namespace escape (any ``\x00``-prefixed real key).
        return bytes.fromhex(payload).decode("utf-8")
    if tag == "n":
        return None
    if tag == "f":
        if payload.startswith("nan:"):
            import struct as _struct

            return _struct.unpack(">d", bytes.fromhex(payload[len("nan:") :]))[0]
        return float.fromhex(payload)
    if tag == "t":
        items = []
        part = ""
        escaped = False
        parts: list[str] = []
        for ch in payload:
            if escaped:
                part += ch
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "|":
                parts.append(part)
                part = ""
            else:
                part += ch
        if part or payload:
            parts.append(part)
        for text in parts:
            kind, _, inner = text.partition(":")
            items.append(int(inner) if kind == "i" else decode_mapping_key(inner))
        return tuple(items)
    raise ValueError(f"Unknown key-codec component {component!r}.")


@dataclass(frozen=True, slots=True)
class InstanceStateInspection:
    """One inert instance-state inspection result (r71 C).

    ``complete`` is ``True`` only when the enumeration is PROVABLY total inertly;
    ``reason`` names the closed fail-closed cause otherwise. ``uncertainty is NEVER
    an empty set``: a blinded inspection reports ``complete=False`` with a non-empty
    reason, never ``names=frozenset()`` masquerading as "no extra state".
    """

    names: frozenset[str]
    complete: bool
    reason: str | None


def _raw_mro_attr(value: Any, attr: str) -> Any:
    """Resolve one attribute through raw class-dict MRO lookup (never ``getattr``).

    Returns the WINNING descriptor/object from ``type(value).__mro__`` without
    invoking any descriptor protocol, ``__getattribute__``, or ``__getattr__`` -- so
    a property/custom-descriptor shadow is OBSERVED, never executed.
    """

    for cls in type(value).__mro__:
        if attr in cls.__dict__:
            return cls.__dict__[attr]
    return None


def inspect_instance_state(value: Any) -> InstanceStateInspection:
    """Inertly enumerate the SET instance-state NAMES on ``value``, fail-closed (r71 C).

    THE single instance-state inspector for declared-schema containers. It NEVER routes the instance ``__dict__`` through ``getattr`` (r70 hon1-F1: a property/descriptor-shadowed
    ``__dict__`` executes user code returning attacker-chosen ``{}``, turning
    "untrustworthy" into "proved empty"). Instead:

    * The instance ``__dict__`` descriptor is resolved by RAW MRO lookup. Absence is
      valid (a genuinely slotted type). If present it must be EXACTLY
      ``types.GetSetDescriptorType`` (the standard instance-dict getset); it is
      invoked DIRECTLY and the result must be ``type(result) is dict``. A property,
      custom/replaced descriptor, dict-subclass/non-dict result, or an exception is
      INCOMPLETE (``instance_state_uninspectable``).
    * The all-MRO genuine ``MemberDescriptorType`` slot scan is unchanged (unset
      slots absent, mangled private names, falsey values counted -- inert).

    PRESENCE is state (a ``None``-valued extra counts). Returns ``complete=True`` with
    the actual storage names only when the enumeration is inertly total.
    """

    items = _inspect_instance_state_items(value)
    if items is None:
        return InstanceStateInspection(frozenset(), False, "instance_state_uninspectable")
    return InstanceStateInspection(frozenset(items), True, None)


def _inspect_instance_state_items(value: Any) -> dict[str, Any] | None:
    """Inertly enumerate SET instance-state ``name -> value`` pairs, or ``None``.

    The value-bearing spine under :func:`inspect_instance_state`: same inert
    channels (raw-MRO ``__dict__`` getset invoked directly, all-MRO genuine
    member-descriptor slots), same fail-closed rules -- ``None`` means the
    enumeration cannot be inertly proven total (a blinded inspection is never
    an empty mapping masquerading as "no extra state").
    """

    import types as _types

    items: dict[str, Any] = {}
    dict_descriptor = _raw_mro_attr(value, "__dict__")
    if dict_descriptor is not None:
        if type(dict_descriptor) is not _types.GetSetDescriptorType:
            # A property or non-standard descriptor shadows __dict__: reading it would
            # execute user code and could report attacker-chosen state. Fail closed.
            return None
        try:
            instance_dict = dict_descriptor.__get__(value, type(value))
        except Exception:
            return None
        if type(instance_dict) is not dict:
            # A dict SUBCLASS (or non-dict) result could carry a custom __iter__ /
            # __contains__ that hides keys: not inertly trustworthy.
            return None
        for key in instance_dict:
            items[str(key)] = instance_dict[key]
    for cls in type(value).__mro__:
        raw_slots = cls.__dict__.get("__slots__")
        if raw_slots is None:
            continue
        slot_names = [raw_slots] if isinstance(raw_slots, str) else [str(s) for s in raw_slots]
        for raw_name in slot_names:
            if raw_name in {"__dict__", "__weakref__"}:
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                # CPython mangles private slot names against the DECLARING class.
                stripped = cls.__name__.lstrip("_")
                if stripped:
                    name = f"_{stripped}{raw_name}"
            descriptor = cls.__dict__.get(name)
            if not isinstance(descriptor, _types.MemberDescriptorType):
                # A shadowed/replaced slot descriptor makes the slot storage
                # unreachable by ordinary attribute access; reading anything else
                # here could execute user code, so the name is skipped (inert).
                continue
            try:
                slot_value = descriptor.__get__(value, cls)
            except AttributeError:
                continue  # unset slot: absent
            items[name] = slot_value
    return items


def _protocol_container_instance_state_fact(value: Any) -> tuple[list[list[str]], bool]:
    """Ordered ``(name, token)`` instance-state facts for protocol-container subclasses.

    The exact-type node fact catches a class SWAP but not changed fields on
    another instance of the SAME class (r66 R1 reopened by r2-B3: a
    ``ModeBox(dict)`` whose ``box.mode`` flipped ``'a' -> 'b'`` between capture
    and replay walked to the same tensor-leaf structure and replayed a wrong
    output VERIFIED). Enumerate the instance state inertly and witness each
    attribute: a type-strict literal token where the literal grammar can carry
    the value, else an opaque type-identity token. A changed literal-valued
    mode flag now diverges structurally, while a well-behaved custom Mapping's
    backing store (``self.data = {...}``) witnesses as a stable opaque ``dict``
    token and keeps working. A same-type changed OPAQUE value remains the
    documented residual (arbitrary object values cannot be compared inertly).

    Returns ``(facts, complete)``; ``complete=False`` means the enumeration is
    blinded and the caller must refuse (``instance_state_uninspectable``),
    never read as "no extra state". Exact builtins carry no instance ``__dict__``
    descriptor, so their nodes are byte-identical to the historical shape.
    """

    items = _inspect_instance_state_items(value)
    if items is None:
        return [], False
    facts: list[list[str]] = []
    for name in sorted(items):
        item_value = items[name]
        try:
            token = f"lit:{encode_mapping_key(item_value)!r}"
        except ValueError:
            token = (
                f"{_RESERVED_NAMESPACE_PREFIX}opaque:"
                f"{type(item_value).__module__}:{type(item_value).__qualname__}"
            )
        facts.append([name, token])
    return facts, True


def instance_state_names(value: Any) -> frozenset[str]:
    """Enumerate the SET instance-state names inertly (r69 C; r71 C fail-closed helper).

    Thin compatibility wrapper over :func:`inspect_instance_state`: an INCOMPLETE
    (uninspectable) inspection returns the empty set here for legacy callers, but the
    declared-schema gate consumes the full :class:`InstanceStateInspection` so a
    blinded inspection fails closed rather than reading as "no extra state".
    """

    return inspect_instance_state(value).names


def _stock_tuplegetter_type() -> type:
    """The stock namedtuple field-descriptor class, resolved from a probe.

    ``collections.namedtuple`` binds every declared field to a
    ``_collections._tuplegetter`` reading the field's own tuple position.
    Resolved by probing rather than importing the private module so a
    hypothetical pure-Python fallback still compares against THE class stock
    namedtuples actually use.
    """

    global _TUPLEGETTER_TYPE_CACHE
    if _TUPLEGETTER_TYPE_CACHE is None:
        import collections as _collections

        _TlDescriptorProbe = _collections.namedtuple("_TlDescriptorProbe", "x")
        _TUPLEGETTER_TYPE_CACHE = type(_TlDescriptorProbe.__dict__["x"])
    return _TUPLEGETTER_TYPE_CACHE


_TUPLEGETTER_TYPE_CACHE: type | None = None


def _namedtuple_field_descriptors_shadowed(value: Any) -> bool:
    """Return whether any declared namedtuple field's descriptor is non-stock (T11.1).

    Mirrors the slots rule: each declared field must resolve (raw MRO, never
    ``getattr``) to the stock ``_tuplegetter`` bound to that field's OWN tuple
    index. A ``property`` (or any other descriptor) shadowing a declared field
    bypassed the declared-schema proof entirely: ``getattr`` in every walker
    read the property's DECOY while the hidden physical slot steered forward
    control flow via ``tuple.__getitem__`` -- physical arity matched, no
    instance state existed, no hook was overridden, so no net fired and the
    recorded path replayed VERIFIED against the decoy. A transposed stock
    getter (bound to a different index) is refused for the same reason: the
    witnessed field order would diverge from the physical layout replay
    reconstructs.
    """

    getter_type = _stock_tuplegetter_type()
    for index, name in enumerate(_instance_fields(value)):
        descriptor = _raw_mro_attr(value, name)
        if type(descriptor) is not getter_type:
            return True
        try:
            bound_index = descriptor.__reduce__()[1][0]
        except Exception:
            return True
        if bound_index != index:
            return True
    return False


def _declared_schema_uninspectable(value: Any) -> bool:
    """Return whether a custom attribute hook blinds the declared-field proof (r71 C).

    A dataclass/namedtuple with a non-standard ``__getattribute__`` or ANY MRO
    ``__getattr__`` can compute or hide declared-field values, so the walker cannot
    inertly prove field completeness WITHOUT invoking the untrusted hook. Raw-MRO
    resolution observes the hooks without executing them; any override short-circuits
    to uninspectable BEFORE any declared-field read. Namedtuples pass by default
    (``tuple`` does not override ``__getattribute__`` and declares no ``__getattr__``)
    but every declared field must additionally resolve to the stock positional
    ``_tuplegetter`` (:func:`_namedtuple_field_descriptors_shadowed`, T11.1) --
    a property-shadowed field reads a decoy no other net can catch.
    """

    import types as _types

    getattribute = _raw_mro_attr(value, "__getattribute__")
    # A USER override is a Python callable (``FunctionType``); a builtin standard
    # ``__getattribute__`` (``object``'s, or the ``dict``/``tuple`` slot wrappers a
    # namedtuple or dict-backed container inherits) is a C-level ``wrapper_descriptor``
    # that cannot run arbitrary user code. Only a non-builtin override blinds the proof.
    if getattribute is not object.__getattribute__:
        if not isinstance(getattribute, _types.WrapperDescriptorType):
            return True
        # ...but a wrapper descriptor GRAFTED from a FOREIGN class is not inert for THIS
        # value: ``Sneak.__getattribute__ = type.__getattribute__`` is a wrapper
        # descriptor too, so the blanket allowlist passed the inertness gate and the
        # subsequent declared-field read raised an untyped ``TypeError`` from internals.
        # Require the descriptor to belong to a class on this value's OWN MRO. Tested
        # through ``type(value)`` rather than ``isinstance(value, objclass)``, because
        # the latter reads ``value.__class__`` -- straight back through the grafted
        # descriptor, which raises before the gate can decide.
        objclass = getattr(getattribute, "__objclass__", None)
        if not isinstance(objclass, type) or objclass not in type(value).__mro__:
            return True
    # ``object`` / ``tuple`` declare NO ``__getattr__``, so any MRO ``__getattr__`` is
    # a user-added hook that can compute or hide declared-field values.
    if any("__getattr__" in cls.__dict__ for cls in type(value).__mro__):
        return True
    # T11.1: a declared namedtuple field whose winning descriptor is not the stock
    # positional ``_tuplegetter`` executes user code on every field read and can
    # present a decoy over hidden positional state -- fail closed before any read.
    return declares_namedtuple_fields(value) and _namedtuple_field_descriptors_shadowed(value)


def undeclared_instance_state(value: Any, kind: str) -> bool:
    """Return whether a container INSTANCE carries state its DECLARED schema drops (r69 C).

    Scope: kinds with a declared FIELD schema. A dataclass or namedtuple subclass
    carrying live instance state outside its declared fields would silently lose that
    state on replay reconstruction -- and it can steer control flow the declared schema
    never witnesses (r66 R1: ``box.mode``; r68 hon1-F2: a ``__slots__`` subclass; r68
    hon1-F3: ``hasattr`` presence on a ``None``-valued extra). The live state set comes
    from the ONE inert inspector :func:`inspect_instance_state` (raw-MRO ``__dict__``
    descriptor + all-MRO set slots, presence counting regardless of value -- NO
    ``None`` carve-out), minus the declared field names.

    r71 C fail-closed: an UNINSPECTABLE inspection (property/descriptor-shadowed
    ``__dict__``, non-dict result, custom ``__getattribute__`` / ``__getattr__`` on
    the declared-schema container) reads as undeclared state PRESENT -- the walker
    can never inertly prove completeness, and it must never invoke the shadowing hook
    to try. Registered containers prove completeness through their explicit
    ``state_complete`` declaration instead (checked by the snapshot's registered
    branch).

    Mapping/sequence SUBCLASSES are NOT judged here: their instance state is not
    REFUSED (a custom Mapping legitimately keeps its backing store in ``__dict__``,
    ``self.data = {...}``, and blanket-refusing it would reject every well-behaved
    custom Mapping input) -- it is WITNESSED instead, by the snapshot's per-node
    ``instance_state`` fact (:func:`_protocol_container_instance_state_fact`):
    literal-valued attributes carry type-strict tokens and diverge structurally
    when changed between capture and replay (the r66 R1 / r2-B3 ``box.mode``
    class), opaque values carry a type-identity token. A same-type changed
    OPAQUE attribute value remains the documented residual; the class swap
    itself still diverges through the exact-type node fact.
    """

    import dataclasses as _dc

    is_dataclass_kind = kind == "dataclass" or (kind == "empty" and _dc.is_dataclass(value))
    is_namedtuple_kind = kind == "namedtuple" or (
        kind == "empty" and declares_namedtuple_fields(value)
    )
    if not (is_dataclass_kind or is_namedtuple_kind):
        return False
    # r71 C: a custom attribute hook on the declared-schema container blinds the
    # field-completeness proof BEFORE any field is read -- fail closed.
    if _declared_schema_uninspectable(value):
        return True
    inspection = inspect_instance_state(value)
    if not inspection.complete:
        return True
    if is_dataclass_kind:
        declared = {field.name for field in _dc.fields(value)}
    else:
        declared = {str(name) for name in _instance_fields(value)}
    return bool(inspection.names - declared)


def _instance_fields(value: Any) -> tuple[str, ...]:
    """Read a namedtuple's ``_fields`` at TYPE level (r71 C; inert, no instance hook).

    Resolves ``_fields`` through the raw MRO so a shadowed instance ``_fields`` cannot
    forge the declared schema; requires a tuple of strings (else empty -> the
    all-state set is treated as undeclared, fail closed at the caller).
    """

    fields = _raw_mro_attr(value, "_fields")
    if isinstance(fields, tuple) and all(isinstance(name, str) for name in fields):
        return fields
    return ()


def snapshot_input_boundary(value: Any) -> dict[str, Any]:
    """Build ONE immutable per-node structure snapshot of a model-input site (r67 C2).

    Returns ``{"nodes": [...], "refusals": [...]}`` where every node record carries
    ``path`` (canonical components), ``kind`` (closed vocabulary + ``registered``),
    ``type`` (exact ``[module, qualname]``), and the declared child schema (``fields``
    for namedtuple/dataclass, ordered codec-encoded ``keys`` for mappings, ``size`` for
    sequences/registered AND namedtuples, ``empty_kind`` for empty nodes, ``aux`` for
    registered containers). Refusals name the path and reason -- opaque keys, undeclared
    instance state, non-total namedtuple schemas, unset declared fields, container cycles
    or over-deep nests, throwing/nonconforming or state-incomplete registrations -- and
    the runnable producer refuses the save on any of them (the existing
    ``input_container_contract_missing``); analysis captures are unaffected.

    ``size`` on a namedtuple node is the PHYSICAL tuple arity. Without it, two instances
    of a one-field namedtuple built as ``tuple.__new__(P, (t, extra))`` with different
    hidden payloads produced byte-identical snapshots, so hidden positional state read
    via ``box[1]`` steered the forward with no witness at all.
    """

    import dataclasses as _dc

    from torchlens._ir.container import get_registered_container

    nodes: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    active_ids: set[int] = set()

    def _type_ref(item: Any) -> list[str]:
        """Exact ``(module, qualname)`` witness for one container's class."""

        cls = type(item)
        return [str(cls.__module__), str(cls.__qualname__)]

    def _descend(item: Any, path: tuple[Any, ...]) -> None:
        """Cycle/depth-fence one node, then record it (see :func:`_record`)."""

        kind = classify_input_container(item)
        if kind in {"tensor", "leaf"}:
            _record(item, kind, path)
            return
        if id(item) in active_ids:
            refusals.append({"path": list(path), "reason": "input_container_cycle"})
            nodes.append({"path": list(path), "kind": kind, "type": _type_ref(item)})
            return
        if len(path) > MAX_INPUT_CONTAINER_DEPTH:
            refusals.append({"path": list(path), "reason": "input_container_too_deep"})
            nodes.append({"path": list(path), "kind": kind, "type": _type_ref(item)})
            return
        active_ids.add(id(item))
        try:
            _record(item, kind, path)
        finally:
            active_ids.discard(id(item))

    def _record(item: Any, kind: str, path: tuple[Any, ...]) -> None:
        """Append one input node's structural record and any type refusals.

        Only CONTAINER nodes carry the exact-class witness: a tensor leaf belongs to
        the admission gate and a scalar leaf to the literal-witness VALUE contract,
        so an exact-type fact on leaves would diverge value-equal inputs the value
        contract admits. A semantic scalar -- including one appearing as a ``slice``
        component -- emits a ``semantic_scalar_type`` refusal here and in the
        symmetric runtime snapshot.
        """

        # r67 heavy-gate fix: only CONTAINER nodes carry the exact-class witness. A
        # tensor leaf's identity is the admission gate's domain, and a scalar LEAF
        # literal's semantics are the literal-witness VALUE contract (numeric equality
        # across float subclasses: a captured ``np.float64(2.0)`` re-run with plain
        # ``2.0`` must verify) -- an exact-type fact on leaves would structurally
        # diverge value-equal inputs the value contract admits.
        node: dict[str, Any] = {"path": list(path), "kind": kind}
        if kind not in {"tensor", "leaf"}:
            node["type"] = _type_ref(item)
        if kind == "tensor":
            nodes.append(node)
            return
        registration = None
        if kind not in {"tensor", "leaf"}:
            registration = get_registered_container(type(item))
        if registration is not None:
            node["kind"] = "registered"
            # r71 C adjacent closure: the registration's ``state_complete`` trust
            # rule reads the instance ``__dict__`` through the ONE inert inspector
            # (never a live ``getattr`` __dict__ read -- a property-shadowed __dict__
            # would blind or forge the state check). An uninspectable registered
            # container refuses fail-closed; the explicit ``state_complete=True``
            # trust is preserved.
            if not registration.state_complete:
                inspection = inspect_instance_state(item)
                if not inspection.complete:
                    refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
                elif inspection.names:
                    refusals.append(
                        {"path": list(path), "reason": "registered_state_not_declared_complete"}
                    )
            try:
                children, aux = registration.flatten(item)
                children = list(children)
            except Exception:
                refusals.append({"path": list(path), "reason": "registered_flatten_failed"})
                nodes.append(node)
                return
            try:
                node["aux"] = _safe_aux(aux)
            except ValueError:
                refusals.append({"path": list(path), "reason": "registered_aux_unsafe"})
                nodes.append(node)
                return
            node["size"] = len(children)
            nodes.append(node)
            for index, child in enumerate(children):
                _descend(child, (*path, index))
            return
        # r71 C: a declared-schema container whose attribute hooks / __dict__ cannot
        # be inspected inertly refuses with the DISTINCT ``instance_state_uninspectable``
        # reason BEFORE any declared field is read, so the shadowing hook never runs.
        if kind in {"dataclass", "namedtuple", "empty"} and _declared_schema_uninspectable(item):
            refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
            nodes.append(node)
            return
        if (
            kind in {"dataclass", "namedtuple", "empty"}
            and not inspect_instance_state(item).complete
        ):
            refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
            nodes.append(node)
            return
        if undeclared_instance_state(item, kind):
            refusals.append({"path": list(path), "reason": "undeclared_instance_state"})
        if kind == "empty":
            empty_kind = empty_input_container_kind(item)
            node["empty_kind"] = str(empty_kind)
            if empty_kind in {"mapping", "sequence"}:
                state_facts, state_complete = _protocol_container_instance_state_fact(item)
                if not state_complete:
                    refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
                elif state_facts:
                    node["instance_state"] = state_facts
            nodes.append(node)
            return
        if kind == "namedtuple":
            fields = _instance_fields(item)
            node["fields"] = [str(name) for name in fields]
            # PHYSICAL arity through the concrete builtin (never the instance
            # ``__len__``), so a hidden positional payload cannot make two snapshots
            # compare equal, and a declared schema that does not account for the whole
            # tuple refuses instead of silently dropping the extras (tensors included).
            node["size"] = physical_sequence_len(item)
            if namedtuple_arity_mismatch(item):
                refusals.append({"path": list(path), "reason": "namedtuple_schema_not_total"})
            nodes.append(node)
            for name in fields:
                _descend(getattr(item, name), (*path, str(name)))
            return
        if kind == "dataclass":
            node["fields"] = [field.name for field in _dc.fields(item)]
            unset = unset_declared_fields(item)
            if unset:
                # A declared ``init=False`` field the instance never set is enumerated by
                # ``dataclasses.fields`` but raises on ``getattr``. The walkers used to
                # let that raw ``AttributeError`` escape and kill the whole capture; it
                # is now an explicit structural fact plus a typed save refusal.
                node["unset_fields"] = list(unset)
                refusals.append({"path": list(path), "reason": "unset_declared_field"})
            nodes.append(node)
            for field in _dc.fields(item):
                child = _declared_field_value(item, field.name)
                if child is _UNSET_FIELD:
                    continue
                _descend(child, (*path, field.name))
            return
        if kind == "mapping":
            # Same-class instance state on a Mapping subclass is part of the
            # structure witness (the exact-type fact alone cannot see changed
            # fields on another instance of the SAME class).
            state_facts, state_complete = _protocol_container_instance_state_fact(item)
            if not state_complete:
                refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
            elif state_facts:
                node["instance_state"] = state_facts
            # Derive the ordered-key fact from the SAME ``items()`` traversal that
            # descends the children. Reading ``keys()`` here while the children came
            # from ``items()`` meant a Mapping whose ``keys()`` is defined independently
            # of ``__iter__`` persisted an "ordered keys" witness in a different order
            # from the one the model actually iterates -- and the order-insensitive
            # child-path set could not detect it.
            entries = list(item.items())
            if mapping_protocol_entries_mismatch(item, entries):
                # The protocol view is not TOTAL over the physical dict storage:
                # a lying ``items()``/``keys()`` shrank, padded, or SUBSTITUTED
                # the witnessed structure identically on the capture and runtime
                # snapshots -- a false-VERIFIED shape, the same forgery lane
                # ``physical_sequence_len`` closes for sequences and namedtuples.
                # Entry-level, not count-only (R12): a count-preserving decoy
                # entry hid a physically stored tensor from every snapshot node.
                # A refusal (not a node fact) keeps persisted snapshot shapes
                # byte-stable for existing artifacts while failing closed on
                # both the save and runtime sides.
                refusals.append({"path": list(path), "reason": "mapping_protocol_not_total"})
            keys: list[Any] = []
            encodable = True
            for key in (entry[0] for entry in entries):
                try:
                    keys.append(encode_mapping_key(key))
                except ValueError:
                    refusals.append({"path": list(path), "reason": "opaque_mapping_key"})
                    encodable = False
                    break
            if encodable and len(set(keys)) != len(keys):
                # r69 D: two distinct key objects encoding to ONE canonical token
                # (e.g. multiple same-bit NaN objects -- hash-equal, eq-False, so
                # separate dict slots) make key identity ambiguous: ordered-key
                # facts self-collide and token binding would be first-match-wins.
                # Refuse typed at save; the symmetric runtime snapshot refusal
                # fails the structure check on a runtime-ambiguous mapping.
                refusals.append({"path": list(path), "reason": "ambiguous_mapping_key"})
                encodable = False
            node["keys"] = keys if encodable else []
            nodes.append(node)
            if encodable:
                for key, child in entries:
                    _descend(child, (*path, encode_mapping_key(key)))
            return
        if kind == "sequence":
            node["size"] = physical_sequence_len(item)
            state_facts, state_complete = _protocol_container_instance_state_fact(item)
            if not state_complete:
                refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
            elif state_facts:
                node["instance_state"] = state_facts
            nodes.append(node)
            for index, child in iter_physical_sequence(item):
                _descend(child, (*path, index))
            return
        # Literal leaf: the VALUE is witnessed by the literal walker, but the scalar
        # TYPE-CLASS disposition is decided here by the ONE classifier (r69 B). A
        # semantic-typed scalar (enum member, builtin/np-scalar subclass) steers host
        # control flow through type identity that no persisted value fact can carry,
        # so it refuses the runnable save typed (``semantic_scalar_type``); the SAME
        # refusal fires in the runtime snapshot, so an admitted builtin/stock-numpy
        # capture replayed with a same-value semantic type diverges structurally.
        if classify_scalar(item)[0] == "semantic":
            refusals.append({"path": list(path), "reason": "semantic_scalar_type"})
        # r71 B: a COMPOSITE literal leaf (a ``slice``) is one opaque snapshot leaf, but
        # its sub-components carry type identity too (secA-F1). Apply the shared
        # component policy: a semantic slice component emits the SAME
        # ``semantic_scalar_type`` refusal at save AND in the symmetric runtime
        # snapshot, so a plain-slice capture replayed with a same-value semantic
        # component diverges structurally.
        if isinstance(item, slice) and slice_semantic_component(item) is not None:
            refusals.append({"path": list(path), "reason": "semantic_scalar_type"})
        nodes.append(node)

    def _safe_aux(aux: Any) -> Any:
        """Return the CANONICAL type-strict registered-container aux witness.

        Aux IS the declared instance-state schema of a registered container, so its
        atoms must carry the same type identity every other input-boundary edge does.
        The historical ``isinstance(aux, (bool, int, float, str))`` admission was
        type-BLIND in both directions: it admitted an ``IntEnum`` (semantic type
        identity no persisted value can carry, only refused far downstream with the
        misattributed ``context_field_invalid``), and the snapshot comparison is
        builtin ``==``, where ``True == 1 == 1.0`` and ``-0.0 == 0.0`` -- so a
        ``mode=True`` capture re-run with the ``mode=1`` twin reported VERIFIED
        end-to-end, and a NaN aux false-DIVERGED byte-identical inputs
        (``[nan] != [nan]``).

        Atoms therefore route through :func:`encode_mapping_key` -- the same canonical
        token authority the ``COMPOSITE_LITERAL_COMPONENT_POLICY`` table names for
        mapping keys and tuple-key components: it is classifier-first (semantic and
        NumPy-wrapper atoms raise), keeps ``True``/``1``/``1.0`` and ``-0.0``/``+0.0``
        distinct, and encodes a NaN's binary64 BIT PATTERN so an identical NaN aux
        compares equal to itself. Sequence nodes carry their exact kind so a
        ``tuple`` aux cannot launder into a ``list`` one.

        Sequence admission is EXACT-TYPE (T11.9): the documented type-strict
        schema accepted ``list``/``tuple`` SUBCLASSES via ``isinstance`` and
        encoded them as their plain base kind, erasing the exact class -- the
        very identity every other input-boundary edge witnesses -- plus any
        instance state the subclass carries (a namedtuple aux flattened to a
        bare ``tuple`` row). A subclass now refuses typed
        (``registered_aux_unsafe``) instead of laundering.

        Raises
        ------
        ValueError
            If the aux tree holds a value outside the canonical atom grammar,
            an exact-``list``/``tuple`` node of those, or a sequence SUBCLASS
            carrying semantic type identity.
        """

        if type(aux) in (list, tuple):
            return ["tuple" if type(aux) is tuple else "list", [_safe_aux(i) for i in aux]]
        if isinstance(aux, (list, tuple)):
            raise ValueError(
                f"Registered-container aux node {aux!r} is a {type(aux).__name__} "
                "(a list/tuple SUBCLASS): its exact type and instance state are "
                "outside the type-strict aux grammar."
            )
        return ["atom", encode_mapping_key(aux)]

    try:
        _descend(value, ())
    except RecursionError as exc:
        # Same stack-budget class as the walk direction: typed, never raw (T11.4).
        raise_input_tree_stack_refusal(exc)
    return {"nodes": nodes, "refusals": refusals}
