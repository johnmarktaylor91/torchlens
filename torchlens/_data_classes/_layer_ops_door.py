"""The ``Layer.ops`` door: the op accessor a layer hands out keeps the layer's trace.

A ``Layer`` view holds its trace, and the layer's op accessor lives in the
layer state the trace holds, so the accessor itself cannot hold the trace (the
trace would hold itself). In ``tl.trace(model, x).layers["relu_1"].ops[0]``
the layer is a temporary: it is gone, and the trace with it, before the op is
looked up, and the op built then has no trace to read.

So a read of ``layer.ops`` on a layer that holds a trace hands out a second
accessor over the same state that holds the trace too, exactly as
``Trace.ops`` and ``Trace.layers`` do (``_trace_views.py``). The op views it
builds are built while the trace is alive, so each holds the trace in turn.
The second accessor is cached weakly on the stored one: identity holds while
anyone keeps it, and a trace nobody holds is still freed by reference
counting.

A layer with no trace to hold (while the trace is being built, or copied or
restored on its own) hands out the stored accessor unchanged.
"""

from __future__ import annotations

import weakref
from typing import Any

_OPS = "ops"

#: ``OpAccessor`` and an empty subclass of it, bound by ``_layer_accessors`` at
#: import so this module imports no record class.
_ACCESSOR: list[Any] = []


def bind_op_accessor(accessor_class: type) -> None:
    """Bind the ``OpAccessor`` class the door hands out."""

    # An instance of the empty subclass is created with no ``__init__`` run and
    # then made an ``OpAccessor``: the same layout, none of the setup.
    empty = type(
        "_HandedLayerOps",
        (accessor_class,),
        {"__slots__": (), "__init__": object.__init__, "__doc__": "An empty op accessor."},
    )
    _ACCESSOR[:] = [accessor_class, empty]


def _over(accessor: Any) -> Any:
    """Return a new ``OpAccessor`` object over the state of ``accessor`` (it holds no trace)."""

    accessor_class, empty = _ACCESSOR
    twin = empty()
    twin.__class__ = accessor_class
    twin.__dict__ = accessor.__dict__
    return twin


class LayerOpsDoor:
    """Data descriptor for ``Layer.ops``: the stored accessor, or one that keeps the trace."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a debugging repr naming the field."""

        return "<Layer ops door>"

    def __get__(self, layer: Any, objtype: Any = None) -> Any:
        """Return the layer's op accessor; a layer that holds a trace hands out one that does too."""

        if layer is None:
            return self
        try:
            stored = layer.__dict__[_OPS]
        except KeyError:
            raise AttributeError(_OPS) from None
        try:
            owner = layer._owner
        except AttributeError:
            return stored
        if owner is None or stored.__class__ is not _ACCESSOR[0]:
            return stored
        try:
            handed = stored._handed()
        except AttributeError:
            handed = None
        if handed is None or handed._owner is not owner:
            handed = _over(stored)
            handed._owner = owner
            stored._handed = weakref.ref(handed)
        return handed

    def __set__(self, layer: Any, value: Any) -> None:
        """Store ``value``; an accessor that keeps a trace is stored as one that does not."""

        if value.__class__ is _ACCESSOR[0]:
            try:
                owned = value._owner is not None
            except AttributeError:
                owned = False
            if owned:
                # The layer state is held by the trace: storing the handed-out
                # accessor there would make the trace hold itself.
                value = _over(value)
        layer.__dict__[_OPS] = value

    def __delete__(self, layer: Any) -> None:
        """Delete the stored accessor; a missing one raises ``AttributeError``."""

        try:
            del layer.__dict__[_OPS]
        except KeyError:
            raise AttributeError(_OPS) from None


def install(layer_class: type) -> None:
    """Install the door on the ``Layer`` class."""

    setattr(layer_class, _OPS, LayerOpsDoor())
