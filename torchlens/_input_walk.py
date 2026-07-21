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
from collections.abc import Mapping
from typing import Any, Callable

__all__ = [
    "INPUT_CONTAINER_KINDS",
    "classify_input_container",
    "raw_mapping_key_component",
    "tagged_mapping_key_component",
    "walk_input_boundary",
]


INPUT_CONTAINER_KINDS: frozenset[str] = frozenset(
    {"tensor", "empty", "namedtuple", "dataclass", "mapping", "sequence", "leaf"}
)
"""The CLOSED container-kind vocabulary of the model-input boundary (see module doc)."""


def classify_input_container(value: Any) -> str:
    """Classify one boundary value into the closed container-kind vocabulary.

    This is THE single container dispatch (order is load-bearing; see the module
    docstring). :func:`walk_input_boundary` consumes it, so adding a container kind
    here extends every input-boundary walker in lockstep.
    """

    import torch

    from torchlens._io.runnable import empty_container_kind

    if isinstance(value, torch.Tensor):
        return "tensor"
    if empty_container_kind(value) is not None:
        return "empty"
    if isinstance(value, tuple) and hasattr(value, "_fields"):
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
    """Metadata-fact-site mapping-key vocabulary (W2): the raw key, every key accepted.

    See the module docstring (dual mapping-key vocabulary, residual R6) for why W2
    keeps raw keys this round: fact-site paths are attribution keys, not persisted
    decodable literals; the bool/int conflation is shielded by the r33
    ``_type_strict_path`` symmetric belt; and a fact site under a non-representable
    key fails literal encoding at witness time while the literal walker's OPAQUE leaf
    independently ceilings the run -- never a false VERIFIED.
    """

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
        ``empty_container_kind`` string. ``None`` ignores empty containers.
    on_opaque_key_subtree:
        Called ``(child, parent_path)`` once per mapping key rejected by
        ``key_component`` (the child subtree is NOT descended). ``None`` skips.
    """

    from torchlens._io.runnable import _UnsupportedLiteralError, empty_container_kind

    def _descend(value: Any, path: tuple[Any, ...]) -> None:
        """Dispatch one node through the closed container-kind vocabulary."""

        kind = classify_input_container(value)
        if kind == "tensor":
            if on_tensor is not None:
                on_tensor(value, path)
            return
        if kind == "empty":
            if on_empty_container is not None:
                empty_kind = empty_container_kind(value)
                assert empty_kind is not None  # classify_input_container said "empty"
                on_empty_container(empty_kind, path)
            return
        if kind == "namedtuple":
            for name in value._fields:
                _descend(getattr(value, name), (*path, str(name)))
            return
        if kind == "dataclass":
            for field in dataclasses.fields(value):
                _descend(getattr(value, field.name), (*path, field.name))
            return
        if kind == "mapping":
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
            for index, child in enumerate(value):
                _descend(child, (*path, index))
            return
        if on_leaf is not None:
            on_leaf(value, path)

    _descend(value, path)
