"""One call of a module, presented to facet recipes as a single-call Module.

Facet recipes are written against a Module record: they read
``module._single_call_or_error()``, ``module.calls[0]`` and the module's
children by address. A module called more than once (an ALBERT layer shared
across depth) refuses those single-call reads, so its recipes see nothing.
:class:`CallBoundModule` wraps one ``ModuleCall``: the reads a single-call
Module delegates to its call (outputs, input and output ops, forward
arguments, its facet view) answer for the bound call, and every identity or
configuration read (class name, parameters, captured attributes, trace)
answers for the owning Module, which records them; the call's own copies of
those can be empty. Children resolve to the child's call made inside the same
parent call, so a recipe reading ``query``'s output gets the output of that
pass.
"""

from __future__ import annotations

from typing import Any

from .._trace_core.record_refs import RecordSlot

# Names both records carry whose Module reading delegates to its single call;
# on a bound subject they answer for the bound call.
_PER_CALL_SHARED = frozenset(
    {
        "backward_duration",
        "call_children",
        "call_parent",
        "facets",
        "forward_args",
        "forward_args_summary",
        "forward_args_template",
        "forward_duration",
        "forward_kwargs",
        "forward_kwargs_summary",
        "forward_kwargs_template",
        "func_calls_duration",
        "grad",
        "grads",
        "input_layers",
        "input_ops",
        "num_edges",
        "num_input_edges",
        "num_internal_edges",
        "num_output_edges",
        "out",
        "out_shape",
        "out_shapes",
        "outs",
        "output_layers",
        "output_ops",
        "output_structure",
        "projective_field",
        "receptive_field",
    }
)


# The two record slots: a read of an unset slot must not recurse through ``__getattr__``.
_BOUND = frozenset({"_call", "_module"})


def _carries(record: Any, name: str) -> bool:
    """Return whether ``record`` defines ``name`` itself, without evaluating it."""

    return name in vars(record) or hasattr(type(record), name)


class CallBoundModule:
    """A Module record seen through exactly one of its calls.

    The Module and the call are kept as row keys (``RecordSlot``), so a facet
    view cached on the call does not hold either record.
    """

    _call = RecordSlot("_call")
    _module = RecordSlot("_module")

    def __init__(self, module: Any, call: Any) -> None:
        """Bind ``module`` to its ``call``.

        Parameters
        ----------
        module:
            The Module record that owns ``call``.
        call:
            One ModuleCall of ``module``.
        """

        self._module = module
        self._call = call

    def __getattr__(self, name: str) -> Any:
        """Answer per-call reads from the call and every other read from the Module."""

        if (name.startswith("__") and name.endswith("__")) or name in _BOUND:
            raise AttributeError(name)
        call = self._call
        if name in _PER_CALL_SHARED or (_carries(call, name) and not _carries(self._module, name)):
            return getattr(call, name)
        return getattr(self._module, name)

    @property
    def calls(self) -> Any:
        """The bound call, as the only call, in the Module's own call accessor type."""

        return type(self._module.calls)({self._call.call_index: self._call})

    @property
    def ops(self) -> Any:
        """The bound call, as the only per-call entry (``Module.ops`` holds calls)."""

        return self.calls

    @property
    def num_calls(self) -> int:
        """Always one: the subject is a single call."""

        return 1

    def _single_call_or_error(self, remedy: str | None = None) -> Any:
        """Return the bound call; a bound subject is never ambiguous."""

        return self._call

    def __repr__(self) -> str:
        """Name the bound call."""

        return f"CallBoundModule({getattr(self._call, 'label', '?')!r})"


def call_subject(call: Any) -> Any:
    """Return ``call`` bound to its owning Module for recipe evaluation.

    A call detached from its trace has no Module to bind; it is returned as is,
    so its view keeps the structural facets alone.
    """

    if getattr(call, "_source_trace", None) is None:
        return call
    return CallBoundModule(call.module, call)


def bound_child(subject: CallBoundModule, child: Any) -> Any | None:
    """Return ``child`` bound to its call made inside ``subject``'s call.

    ``None`` when the child module ran no call inside this parent call (a
    branch not taken on this pass).
    """

    children = set(getattr(subject._call, "call_children", ()) or ())
    inside = [call for call in child.calls.values() if getattr(call, "label", None) in children]
    if len(inside) == 1:
        return CallBoundModule(child, inside[0])
    if not inside and child.num_calls == 1 and subject._module.num_calls == 1:
        return CallBoundModule(child, child.calls[0])
    return None
