"""Layer field descriptors: mirror reads, per-layer shadows, tombstones and grad-fn keys.

Split out of ``layer.py`` (size-capped). A Layer's state dictionary is held
by its trace; these descriptors read and write that dictionary. A ``GradFn``
record assigned to ``Layer.grad_fn`` or ``Layer.grad_fn_handle`` is stored as
a key into the trace's grad-fn table, so the layer state holds no record.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

from .._trace_core.op_store import DetachedOpStore
from .._trace_core.relation_views import (
    LAYER_FROZENSET_VIEW_FIELDS,
    LAYER_TUPLE_VIEW_FIELDS,
)
from ._layer_mirror import _layer_rep_op

if TYPE_CHECKING:
    from .layer import Layer

#: Shadow-miss sentinel local to mirror reads.
_LAYER_UNSET = object()

#: Tombstone marking an explicitly deleted mirror field.
_LAYER_DELETED = object()


#: Layer stored relation fields and their immutable view types (the same
#: universe the relation freeze converts; ``equivalent_ops`` is handled by
#: its dedicated property but shares the normalization).
def _build_layer_view_types() -> dict[str, type]:
    """Map each Layer stored relation field to its immutable view type."""

    view_types: dict[str, type] = dict.fromkeys(LAYER_TUPLE_VIEW_FIELDS, tuple)
    view_types.update(dict.fromkeys(LAYER_FROZENSET_VIEW_FIELDS, frozenset))
    view_types["equivalent_ops"] = frozenset
    return view_types


_LAYER_RELATION_VIEW_TYPES: dict[str, type] = _build_layer_view_types()


def _layer_relations_finished(layer: Layer) -> bool:
    """Return whether this Layer's backing capture is finished.

    Mirrors the Op descriptor's finished predicate through the
    representative op's store — sealed, relation-frozen (preview backends
    convert without sealing), or detached (copy/pickle/fork/loaded) — and
    falls back to the owning trace's core op store while the pass accessor
    is not yet populated. Layers with neither (husked after cleanup, bare
    restore shells) are finished by definition.
    """

    store = None
    rep = _layer_rep_op(layer)
    if rep is not None:
        try:
            store = object.__getattribute__(rep, "_core")
        except AttributeError:
            store = None
    if store is None:
        ref = layer.__dict__.get("_source_trace_ref")
        trace = ref() if ref is not None else None
        core = trace.__dict__.get("_trace_core") if trace is not None else None
        store = core.ops if core is not None else None
    if store is None:
        return True
    return bool(
        getattr(store, "frozen", False)
        or getattr(store, "dataflow_edges", None) is not None
        or store.__class__ is DetachedOpStore
    )


def _layer_normalize_relation_write(layer: Layer, name: str, value: Any) -> Any:
    """Normalize a relation container assigned to a finished Layer.

    Applies exactly the relation freeze's conversion rules (subclass-
    inclusive, unlike the freeze's staging-exact checks): sequences
    normalize to ``tuple`` on tuple-view fields, sets to ``frozenset`` on
    frozenset-view fields. Other value shapes (the dict form of
    ``conditional_branch_stack_ops``, scalars, ``None``) pass through
    unchanged, and building-phase writes stay raw so postprocess aliasing
    is preserved.
    """

    view_type = _LAYER_RELATION_VIEW_TYPES.get(name)
    if view_type is None or value.__class__ is view_type:
        return value
    if view_type is tuple:
        if not isinstance(value, (list, tuple)):
            return value
    elif not isinstance(value, (set, frozenset, list, tuple)):
        return value
    if not _layer_relations_finished(layer):
        return value
    return view_type(value)


class _LayerMirrorField:
    """Data descriptor for one Layer field mirrored from the first-pass op."""

    __slots__ = ("_name", "_source", "_normalize")

    def __init__(self, name: str, source: str, normalize: Any) -> None:
        """Bind the descriptor to its field name, op source, and normalizer."""

        self._name = name
        self._source = source
        self._normalize = normalize

    def __repr__(self) -> str:
        """Return a debugging repr naming the mirrored field."""
        return f"<Layer mirror descriptor {self._name!r}>"

    def __get__(self, layer: Any, objtype: Any = None) -> Any:
        """Return the per-layer shadow when present, else the op mirror."""

        if layer is None:
            return self
        value = layer.__dict__.get(self._name, _LAYER_UNSET)
        if value is not _LAYER_UNSET:
            if value is _LAYER_DELETED:
                raise AttributeError(self._name)
            return value
        rep = _layer_rep_op(layer)
        if rep is None:
            raise AttributeError(self._name)
        value = getattr(rep, self._source)
        normalize = self._normalize
        return normalize(value) if normalize is not None else value

    def __set__(self, layer: Any, value: Any) -> None:
        """Write a per-layer shadow (mirroring permanently stops).

        Relation-view fields normalize to their immutable view on finished
        layers, so a direct assignment can never re-expose a mutable
        relation container (the invariant the op-cell descriptors enforce).
        """

        layer.__dict__[self._name] = _layer_normalize_relation_write(layer, self._name, value)

    def __delete__(self, layer: Any) -> None:
        """Tombstone the field so it stays deleted instead of re-mirroring."""

        instance_dict = layer.__dict__
        if instance_dict.get(self._name, _LAYER_UNSET) is _LAYER_DELETED:
            raise AttributeError(self._name)
        instance_dict[self._name] = _LAYER_DELETED


#: Layer fields whose ``GradFn`` value is stored as a key.
_GRAD_FN_KEYED_FIELDS = ("grad_fn", "grad_fn_handle")


class LayerGradFnKey:
    """Key of one ``GradFn`` record in its trace's grad-fn table.

    The table (``Trace._grad_fn_logs``, keyed by ``grad_fn_object_id``) owns
    the record; the key names it by that id and keeps a weak reference so a
    read returns the identical record without a table lookup.
    """

    __slots__ = ("_record_ref", "object_id")

    def __init__(self, record: Any) -> None:
        """Key ``record`` by its autograd object id."""

        self.object_id = record.grad_fn_object_id
        self._record_ref = weakref.ref(record)

    def resolve(self, layer: Any) -> Any:
        """Return the keyed record (``None`` when the trace no longer has it)."""

        record = self._record_ref()
        if record is not None:
            return record
        logs = _grad_fn_table(layer)
        try:
            return logs[self.object_id] if logs is not None else None
        except KeyError:
            return None


def _grad_fn_table(layer: Any) -> Any:
    """Return the grad-fn table of the trace ``layer`` is bound to, or ``None``."""

    ref = layer.__dict__.get("_source_trace_ref")
    trace = ref() if ref is not None else None
    return trace.__dict__.get("_grad_fn_logs") if trace is not None else None


def _grad_fn_key(layer: Any, value: Any) -> Any:
    """Return the key for a ``GradFn`` record the layer's trace table owns, else ``value``."""

    if value.__class__.__name__ != "GradFn":
        return value
    logs = _grad_fn_table(layer)
    try:
        owned = logs is not None and logs[value.grad_fn_object_id] is value
    except (KeyError, AttributeError, TypeError):
        owned = False
    return LayerGradFnKey(value) if owned else value


def key_layer_grad_fns(layer: Any) -> None:
    """Replace raw ``GradFn`` records in a layer's state with keys (after a trace bind)."""

    state = layer.__dict__
    for name in _GRAD_FN_KEYED_FIELDS:
        value = state.get(name)
        if value is not None and value.__class__.__name__ == "GradFn":
            state[name] = _grad_fn_key(layer, value)


class _LayerGradFnField(_LayerMirrorField):
    """Mirror descriptor for a field whose ``GradFn`` value is stored as a key."""

    __slots__ = ()

    def __get__(self, layer: Any, objtype: Any = None) -> Any:
        """Return the keyed record, else the shadow or op mirror."""

        if layer is None:
            return self
        value = layer.__dict__.get(self._name, _LAYER_UNSET)
        if value.__class__ is LayerGradFnKey:
            return value.resolve(layer)
        return _LayerMirrorField.__get__(self, layer, objtype)

    def __set__(self, layer: Any, value: Any) -> None:
        """Write a per-layer shadow, keying a ``GradFn`` record the trace owns."""

        layer.__dict__[self._name] = _grad_fn_key(layer, value)
