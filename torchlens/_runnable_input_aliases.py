"""Input alias topology and non-tensor tree contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from . import _state
from ._runnable import (
    ContractCheck,
    InputAttestationFingerprint,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotDescriptor,
    TensorSlotRole,
)
from ._runnable_state import (
    RunResourceCeiling,
    _guarded_defensive_materialize,
    runnable_tensor_byte_digest,
)
from ._utils.alias_footprint import touched_bytes_relation

if TYPE_CHECKING:
    from ._runnable_execution import (
        _FINGERPRINT_ALIGNMENT_MODULUS,
        _contract_check,
        _model_input_arity_positions,
        _model_input_literal_facts,
        _tensor_leaf_paths,
        _value_at_path,
    )

__all__ = (
    "_input_alias_topology_checks",
    "_clone_state_values",
    "_snapshot_input_byte_digests",
    "_snapshot_input_fingerprints",
    "build_input_attestation_fingerprint",
    "_positions_are_mixed",
    "_split_mixed_inputs",
    "_input_site_value",
    "_type_strict_path",
    "_input_tree_contract_checks",
    "_runtime_nontensor_leaf_paths",
    "_input_nontensor_tree_contract_checks",
)


def _input_alias_topology_checks(
    descriptor: SparseRunDescriptor,
    input_slots: Sequence[TensorSlotDescriptor],
    raw_values: Mapping[str, torch.Tensor],
) -> tuple[tuple[ContractCheck, ...], bool]:
    """Fail closed on runtime input aliasing unreproducible against a de-aliased capture.

    Torch capture clones each model-input leaf before the forward (``safe_copy_args``), and
    the sparse replay likewise binds independent per-slot clones, so the captured DAG and the
    replay both reflect DISTINCT-input semantics -- the captured alias topology is always
    all-distinct. Any runtime aliasing between model-input sites therefore differs from the
    captured topology and cannot be reproduced against a fresh model on those same aliased
    inputs:

    * IDENTITY (``forward(a, b)`` with ``a is b`` -- self/cross-attention ``q is k``): the
      captured / replayed clones are distinct objects, so an ``if a is b`` / ``id()`` identity
      branch takes the OTHER arm than a fresh model on the aliased input would -- a false
      VERIFIED even on the ORIGINAL input (r33 F1). This holds with NO in-place mutation, so
      the check is UNCONDITIONAL.
    * OVERLAPPING STORAGE SPANS (two views whose byte spans overlap, a storage-identity
      ``a.data_ptr() == b.data_ptr()`` branch, or an in-place mutation that propagates between
      overlapping sites): the de-aliased clones neither share storage nor propagate a mutation
      between sites, so a fresh model on the aliased input can diverge.

    Both fail closed (``runtime topology != captured`` per the F1 contract: capture de-aliases,
    so ANY runtime aliasing is unreproducible and a genuinely identity/storage-independent model
    cannot be PROVEN so at this surface -- the honest verdict is fail-closed). DISJOINT spans of
    one base (``base[:2]`` / ``base[2:]`` -- same storage pointer, non-overlapping bytes) are NOT
    aliased: a mutation of one cannot reach the other and they are distinct objects, so they
    never trigger. All-distinct inputs (the common case) never trigger -- zero over-trigger on
    the trivial topology.
    """

    resolved: list[tuple[str, torch.Tensor]] = []
    for slot in input_slots:
        value = raw_values.get(slot.slot_id)
        if isinstance(value, torch.Tensor):
            resolved.append((slot.slot_id, value))
    aliased_pairs: set[tuple[str, str]] = set()

    def _ordered_pair(left: str, right: str) -> tuple[str, str]:
        """Canonicalize a slot-id pair by sort order so it dedupes as a set member.

        Aliasing is symmetric, so ``(a, b)`` and ``(b, a)`` must never be two
        entries.
        """

        return (left, right) if left <= right else (right, left)

    # One pass over the unordered input pairs, under ONE logging pause (the
    # nestable toggle was previously re-entered per pair via the
    # ``_touched_bytes_relation`` adapter -- pure per-pair overhead at high arity):
    # * Identity aliasing: two input sites bound to the SAME tensor object
    #   (``a is b``) -- recorded directly, the byte engine is not consulted.
    # * Storage aliasing (r35 decision D): the three-valued touched-byte engine.
    #   PROVED overlap is an observed contradiction (failed check -> DIVERGED);
    #   PROVED disjointness passes; ``unknown`` is the
    #   ``input_alias_topology_unresolved`` unverifiability ceiling -- never
    #   ``overlap`` by assumption and never VERIFIED.
    unresolved = False
    if len(resolved) > 1:
        with _state.pause_logging():
            for i in range(len(resolved)):
                left_id, left = resolved[i]
                for j in range(i + 1, len(resolved)):
                    right_id, right = resolved[j]
                    if left is right:
                        aliased_pairs.add(_ordered_pair(left_id, right_id))
                        continue
                    relation = touched_bytes_relation(left, right)
                    if relation == "overlap":
                        aliased_pairs.add(_ordered_pair(left_id, right_id))
                    elif relation == "unknown":
                        unresolved = True
    if not aliased_pairs:
        return (), unresolved
    return (
        _contract_check(
            "input_alias_topology",
            False,
            RunnableErrorCode.INPUT_TREE_MISMATCH,
            "Runtime model inputs alias (same object or proven-overlapping touched "
            "bytes); capture de-aliases inputs (independent per-slot clones), so the "
            "recorded taken path cannot reproduce an identity or storage-aliasing "
            "dependence and must not be blessed VERIFIED.",
            details=(("aliased_input_slot_pairs", repr(sorted(aliased_pairs))),),
        ),
    ), unresolved


def _clone_state_values(
    descriptor: SparseRunDescriptor,
    values: Mapping[str, torch.Tensor],
    ceiling: RunResourceCeiling,
) -> dict[str, torch.Tensor]:
    """Clone run-local state while preserving alias groups, device, and trainability.

    r37 R5/R13: this second clone runs AFTER staging normalized every value to its
    slot device, so it must not strip anything staging established -- ``clone()``
    keeps the device; alias groups keep one shared clone (identity-keyed); and the
    RECORDED per-slot ``trainable`` bit is restored on the clone (state semantics
    come from the recorded binding, NOT ``mirror_requires_grad`` -- unlike runtime
    INPUTS whose mirror follows the live leaf, see ``_runtime_mirror_clone``).
    r61 corr_2: each identity's single clone routes through the transaction
    ceiling's byte guard, so an oversized staged value refuses typed at
    ``clone_allocation_preflight`` instead of dying in the allocator.

    r67 C5 (corr1-2): the second state clone is a PRE-EXECUTION defensive
    materialization, so it runs under the neutral ambient -- a caller inside
    ``torch.inference_mode()`` no longer mints inference-mode run-local state that
    trips the staged tripwire's ``is_inference`` dim on TorchLens's own output.
    """

    trainable_by_slot = {
        slot.slot_id: bool(slot.state_binding.trainable)
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
    }
    clones_by_identity: dict[int, torch.Tensor] = {}
    cloned: dict[str, torch.Tensor] = {}
    with _guarded_defensive_materialize():
        for slot_id, value in values.items():
            clone = clones_by_identity.get(id(value))
            if clone is None:
                clone = ceiling.guarded_clone(value, call_id=None, slot_id=slot_id)
                if trainable_by_slot.get(slot_id, False) and not clone.requires_grad:
                    try:
                        clone.requires_grad_(True)
                    except RuntimeError:
                        pass  # non-differentiable dtype cannot carry the trainable bit
                clones_by_identity[id(value)] = clone
            cloned[slot_id] = clone
    return cloned


def _snapshot_input_byte_digests(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Digest cloned model inputs before sparse calls can mutate them in place."""

    return {
        slot.slot_id: runnable_tensor_byte_digest(slot_values[slot.slot_id])
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT and slot.slot_id in slot_values
    }


def _snapshot_input_fingerprints(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    input_byte_digests: Mapping[str, str],
) -> dict[str, InputAttestationFingerprint]:
    """Fingerprint the EXECUTED input clones on the capture-identical basis (hon1_3).

    Runs after admission (I5), before any sparse call. The capture side
    fingerprints the retained input clone that seeded the captured forward; both
    sides therefore compare the same clone basis, so a physical twin (layout /
    stride / offset / alignment-class change) is detected exactly.
    """

    fingerprints: dict[str, InputAttestationFingerprint] = {}
    for slot in descriptor.tensor_slots:
        if slot.role is not TensorSlotRole.MODEL_INPUT or slot.slot_id not in slot_values:
            continue
        fingerprints[slot.slot_id] = build_input_attestation_fingerprint(
            slot.slot_id,
            slot_values[slot.slot_id],
            byte_digest=input_byte_digests.get(slot.slot_id),
        )
    return fingerprints


def build_input_attestation_fingerprint(
    slot_id: str,
    value: torch.Tensor,
    *,
    byte_digest: str | None = None,
) -> InputAttestationFingerprint:
    """Build the physical identity fingerprint of one model-input value (hon1_3 H-a).

    Both sides use the same basis: at save time the LIVE retained in-memory input
    that seeded the captured forward; at run time the executed defensive clone.
    Any physical fact that cannot be read fails toward attestation ineligibility
    (a sentinel value that can never equal a well-formed capture record).

    Parameters
    ----------
    slot_id:
        Descriptor slot id of the model input.
    value:
        Live tensor to fingerprint.
    byte_digest:
        Precomputed logical byte digest, or ``None`` to compute one here.

    Returns
    -------
    InputAttestationFingerprint
        Frozen physical fingerprint record.
    """

    def _safe_bool(getter: Callable[[], Any]) -> bool:
        """Read one optional tensor property as a bool, treating any failure as ``False``.

        A property the runtime refuses to answer must not abort fingerprinting;
        ``False`` is the fail-safe answer because it can only widen the
        eligibility mismatch, never mint a false ``attested``.
        """

        try:
            return bool(getter())
        except (RuntimeError, AttributeError, TypeError, NotImplementedError):
            return False

    try:
        alignment = int(value.data_ptr()) % _FINGERPRINT_ALIGNMENT_MODULUS
    except (RuntimeError, AttributeError, TypeError, NotImplementedError):
        # Unreadable pointer: use an out-of-range sentinel so it can never match
        # a well-formed capture record (fail toward not_applicable, never a
        # false ATTESTED).
        alignment = -1
    if byte_digest is None:
        byte_digest = runnable_tensor_byte_digest(value)
    return InputAttestationFingerprint(
        slot_id=slot_id,
        byte_digest=byte_digest,
        device_type=str(value.device.type),
        device_index=None if value.device.index is None else int(value.device.index),
        layout=str(value.layout),
        sizes=tuple(int(item) for item in value.shape),
        strides=tuple(int(item) for item in value.stride()),
        storage_offset=int(value.storage_offset()),
        is_contiguous=_safe_bool(value.is_contiguous),
        is_channels_last=_safe_bool(lambda: value.is_contiguous(memory_format=torch.channels_last)),
        is_channels_last_3d=_safe_bool(
            lambda: value.is_contiguous(memory_format=torch.channels_last_3d)
        ),
        is_conj=_safe_bool(value.is_conj),
        is_neg=_safe_bool(value.is_neg),
        tensor_class=type(value).__qualname__,
        requires_grad=bool(value.requires_grad),
        is_inference=_safe_bool(value.is_inference),
        alignment_class=alignment,
    )


def _positions_are_mixed(positions: set[Any]) -> bool:
    """Return whether a capture carries both positional and keyword model sites."""

    kinds = {p[0] for p in positions if isinstance(p, tuple) and len(p) == 2}
    return "arg" in kinds and "kwarg" in kinds


def _split_mixed_inputs(inputs: Any) -> tuple[Sequence[Any], Mapping[Any, Any]]:
    """Split a combined mixed-input mapping into positional and keyword parts.

    A capture with BOTH positional tensor sites and keyword leaves (e.g.
    ``forward(x, *, add)``) cannot be rebound from a bare sequence (fails the
    keyword site) or a bare mapping (fails positional binding). The runnable
    executor accepts a single combined ``{"args": [...], "kwargs": {...}}``
    mapping so mixed captures have a representable ``run(inputs=)`` spelling.
    """

    if not isinstance(inputs, Mapping) or not set(inputs).issubset({"args", "kwargs"}):
        raise TypeError(
            "mixed positional+keyword captures require an inputs mapping of the "
            "form {'args': [...], 'kwargs': {...}}"
        )
    args = inputs.get("args", ())
    kwargs = inputs.get("kwargs", {})
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise TypeError("mixed-input 'args' must be a sequence")
    if not isinstance(kwargs, Mapping):
        raise TypeError("mixed-input 'kwargs' must be a mapping")
    return args, kwargs


def _input_site_value(inputs: Any, position: Any, positions: set[Any]) -> Any:
    """Select one top-level argument or keyword site from the public input tree."""

    if isinstance(position, tuple) and len(position) == 2:
        kind, key = position
        if _positions_are_mixed(positions):
            args, kwargs = _split_mixed_inputs(inputs)
            if kind == "arg":
                return args[cast(int, key)]
            if kind == "kwarg":
                return kwargs[key]
        if kind == "arg":
            if len(positions) == 1 and key == 0:
                return inputs
            if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
                raise TypeError("multiple positional model sites require a sequence input")
            return inputs[cast(int, key)]
        if kind == "kwarg":
            if not isinstance(inputs, Mapping):
                raise TypeError("keyword model sites require a mapping input")
            return inputs[key]
    return _value_at_path(inputs, position if isinstance(position, tuple) else (position,))


def _type_strict_path(path: Iterable[Any]) -> tuple[Any, ...]:
    """Type-tag numeric mapping-key path components so ``bool``/``int``/``float`` twins do not
    conflate under Python ``==`` (r33 F6).

    A dict key ``True`` compares equal to ``1`` and ``1.0`` (``True == 1 == 1.0`` with colliding
    hashes), so a raw ``(True,)`` leaf path matches ``(1,)`` / ``(1.0,)`` in the contract path
    SET -- a runtime input whose key TYPE changed then silently passes the input-tree tripwire.
    Tagging each numeric component with its concrete type keeps the three distinct. Applied
    SYMMETRICALLY to the recorded and runtime path sets, so an ordinary same-type structure (the
    common case) is unaffected -- zero over-trigger. Non-numeric components (``str`` keys,
    already-encoded ``(BOOL_KEY_PATH_TAG, ...)`` tuples) pass through unchanged.
    """

    tagged: list[Any] = []
    for component in path:
        if isinstance(component, bool):
            tagged.append(("\x00tl_key_bool", bool(component)))
        elif isinstance(component, int):
            tagged.append(("\x00tl_key_int", int(component)))
        elif isinstance(component, float):
            tagged.append(("\x00tl_key_float", float(component)))
        else:
            tagged.append(component)
    return tuple(tagged)


def _input_tree_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
) -> tuple[ContractCheck, ...]:
    """Compare the runtime tensor-leaf tree with every recorded input site."""

    slots = tuple(
        slot
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT and slot.input_binding is not None
    )
    positions = _model_input_arity_positions(descriptor)
    expected_by_position: dict[Any, set[tuple[Any, ...]]] = {}
    for slot in slots:
        assert slot.input_binding is not None
        expected_by_position.setdefault(slot.input_binding.model_site_position, set()).add(
            _type_strict_path(slot.input_binding.container_path)
        )
    checks: list[ContractCheck] = []
    for position, expected in expected_by_position.items():
        try:
            root = _input_site_value(inputs, position, positions)
            actual = {_type_strict_path(p) for p in _tensor_leaf_paths(root)}
        except (KeyError, IndexError, TypeError):
            actual = set()
        checks.append(
            _contract_check(
                f"input_tree:{position!r}",
                actual == expected,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime input tensor-leaf paths disagree with the recorded input tree.",
                details=(
                    ("model_site_position", repr(position)),
                    ("expected_paths", repr(sorted(expected, key=repr))),
                    ("actual_paths", repr(sorted(actual, key=repr))),
                ),
            )
        )
    return tuple(checks)


def _runtime_nontensor_leaf_paths(root: Any) -> set[tuple[str | int, ...]]:
    """Enumerate runtime non-tensor input leaf paths, mirroring the capture walk.

    Routes through the shared boundary traversal (``torchlens._input_walk``, r65
    Cluster Y) with the SAME tagged-key literal vocabulary as the capture-side
    ``_record_runnable_input_literal_leaves``, so the runtime non-tensor leaf-path SET
    aligns with the recorded fact set BY CONSTRUCTION: tensor leaves are skipped,
    namedtuples and dataclasses descend by field name (r42 corr1_2: a tensor-only
    dataclass input's runtime set matches the recorded empty set exactly), mappings
    descend under grammar-encodable keys (a non-encodable key collapses its whole
    subtree to one opaque leaf at the parent path, exactly as capture does),
    lists/tuples descend by index, empty containers surface as synthetic marker
    paths, and every other value is a scalar leaf recorded at its path.
    """

    from torchlens._input_walk import tagged_mapping_key_component, walk_input_boundary
    from torchlens._io.runnable import EMPTY_CONTAINER_PATH_MARKER

    paths: set[tuple[str | int, ...]] = set()
    walk_input_boundary(
        root,
        (),
        key_component=tagged_mapping_key_component,
        on_leaf=lambda _value, path: paths.add(path),
        on_empty_container=lambda _kind, path: paths.add((*path, EMPTY_CONTAINER_PATH_MARKER)),
        on_opaque_key_subtree=lambda _child, path: paths.add(path),
    )
    return paths


def _input_nontensor_tree_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    positions: set[Any],
) -> tuple[ContractCheck, ...]:
    """Compare the runtime NON-tensor leaf-path tree with every recorded input site.

    The per-leaf value check (:func:`_input_literal_contract_checks`) only visits
    leaves RECORDED at capture, so it catches a CHANGED or MISSING non-tensor leaf but
    is blind to an EXTRA runtime non-tensor leaf (an added dict key the model branches
    on via ``'flag' in d`` / ``d.get('mode')``, or a longer list). An extra leaf can
    steer unwitnessed Python control flow while replay still reports VERIFIED against a
    fresh model on the given inputs. This mirrors the tensor-leaf set-equality
    (:func:`_input_tree_contract_checks`) for non-tensor leaves: EVERY model-input site
    is seeded with its recorded non-tensor leaf-path set (the empty set when capture had
    no non-tensor leaf at that site), and any runtime non-tensor leaf path absent at
    capture -- or a recorded one absent at runtime -- diverges the run.
    """

    expected_by_position: dict[Any, set[tuple[Any, ...]]] = {
        position: set() for position in positions
    }
    for _witness, fact in _model_input_literal_facts(descriptor):
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = _type_strict_path(fact.get("path", ()) or ())
        expected_by_position.setdefault(position, set()).add(path)

    checks: list[ContractCheck] = []
    for position, expected in expected_by_position.items():
        try:
            root = _input_site_value(inputs, position, positions)
            actual = {_type_strict_path(p) for p in _runtime_nontensor_leaf_paths(root)}
        except (KeyError, IndexError, TypeError, AttributeError):
            actual = set()
        checks.append(
            _contract_check(
                f"input_nontensor_tree:{position!r}",
                actual == expected,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime non-tensor input leaf paths disagree with the recorded input "
                "tree; an added/removed non-tensor leaf can steer an unwitnessed path.",
                details=(
                    ("model_site_position", repr(position)),
                    ("expected_paths", repr(sorted(expected, key=repr))),
                    ("actual_paths", repr(sorted(actual, key=repr))),
                ),
            )
        )
    return tuple(checks)
