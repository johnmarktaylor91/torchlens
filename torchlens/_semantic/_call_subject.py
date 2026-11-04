"""One call of a module, presented to facet recipes as a single-call Module.

Facet recipes are written against a Module record: they read
``module._single_call_or_error()``, ``module.calls[0]`` and the module's
children by address. A module called more than once (an ALBERT layer shared
across depth) refuses those single-call reads, so its recipes see nothing.
:class:`CallBoundModule` wraps one ``ModuleCall`` so that every per-call read a
recipe makes answers for that call, and every other read (class, config,
parameters, trace) answers for the owning Module. Children resolve to the
child's call made inside the same parent call, so a recipe reading
``query``'s output gets the output of that pass.
"""

from __future__ import annotations

from typing import Any

# Fields a ModuleCall carries for its own invocation; a recipe reading one of
# these on the bound subject gets this call's value, never the Module's.
_PER_CALL_FIELDS = frozenset(
    {
        "call_children",
        "call_index",
        "call_parent",
        "code_context",
        "forward_args",
        "forward_kwargs",
        "input_layers",
        "input_ops",
        "label",
        "module_call_stack",
        "output_layers",
        "output_ops",
        "output_paths",
        "output_structure",
    }
)


class CallBoundModule:
    """A Module record seen through exactly one of its calls."""

    __slots__ = ("_call", "_module")

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
        """Answer per-call fields from the call and everything else from the Module."""

        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _PER_CALL_FIELDS:
            return getattr(self._call, name)
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

    @property
    def out(self) -> Any:
        """The bound call's single output."""

        return self._call.out

    @property
    def outs(self) -> Any:
        """The bound call's outputs."""

        return self._call.outs

    def _single_call_or_error(self) -> Any:
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
