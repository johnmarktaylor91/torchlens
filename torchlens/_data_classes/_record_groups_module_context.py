"""The module_context group of an ``Op`` or ``Layer``: ``op.module_context.call_depth``, ...

Where the record sits in the module hierarchy: the module calls active around
it, the module calls its output feeds and the module calls it is the output of.
Stored members read the record's private storage attribute
(``_module_context_<leaf>``, same persisted column as before); derived members
compute on read from those columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import OP_GROUP_VIEWS
from ._record_group_base import _LayerGroup, _RecordGroup

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class OpModuleContext(_RecordGroup):
    """Module-nesting facts of one Op: the module calls around it, fed by it and output by it.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.ops["linear_1_1"].module_context, tl.types.OpModuleContext)
    True
    """

    __slots__ = ()
    _record: Op

    @property
    def atomic_module(self) -> Any:
        """The `Module` record for the atomic module this Op is the output of, or `None`."""

        address = self.atomic_module_address
        if address is None:
            return None
        trace = self._record._source_trace
        if trace is None:
            return None
        try:
            return trace.modules[address]
        except (KeyError, TypeError):
            return None

    @property
    def atomic_module_address(self) -> Any:
        """Module address, without the pass suffix, of the atomic module this Op outputs from, or `None`."""

        label = self._record._module_context_atomic_module_call
        if label is None:
            return None
        return str(label).rsplit(":", 1)[0]

    @property
    def atomic_module_call(self) -> Any:
        """Stored `ModuleCall` label for an atomic-module output Op, or `None`."""

        return self._record._module_context_atomic_module_call

    @property
    def call_depth(self) -> Any:
        """Depth of this Op's active `ModuleCall` nesting (the length of `module_call_stack`)."""

        return len(self._record.modules)

    @property
    def call_stack(self) -> Any:
        """Root-first labels of the `ModuleCall`s active when this Op ran."""

        return self._record._module_context_call_stack

    @property
    def entry_arg_keys(self) -> Any:
        """Mapping from `ModuleCall` label to the argument keys this Op filled at module entry."""

        return self._record._module_context_entry_arg_keys

    @property
    def input_to_module_calls(self) -> Any:
        """Labels of the `ModuleCall` records whose input this Op's output fed."""

        return self._record._module_context_input_to_module_calls

    @property
    def input_to_modules(self) -> Any:
        """Module addresses, without the pass suffix, whose input this Op's output fed."""

        seen: dict[str, None] = {}
        for call_label in self._record._module_context_input_to_module_calls:
            seen.setdefault(str(call_label).rsplit(":", 1)[0], None)
        return list(seen)

    @property
    def is_atomic_module(self) -> Any:
        """Whether this Op is the sole operation output of an atomic module call."""

        return self._record._module_context_is_atomic_module

    @property
    def is_module_input(self) -> Any:
        """Whether this Op's output feeds at least one `ModuleCall` as an input."""

        return bool(self._record._module_context_input_to_module_calls)

    @property
    def is_module_output(self) -> Any:
        """Whether this Op is the output of a `ModuleCall`."""

        return self._record._module_context_is_module_output

    @property
    def output_of_module_calls(self) -> Any:
        """Labels of the `ModuleCall` records whose output this Op is."""

        return self._record._module_context_output_of_module_calls

    @property
    def output_of_modules(self) -> Any:
        """Module addresses, without the pass suffix, whose output this Op is."""

        return self._record._module_context_output_of_modules


class LayerModuleContext(_LayerGroup):
    """Module-nesting facts of one Layer: its nesting depth and the modules it is the output of.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.layers["linear_1"].module_context, tl.types.LayerModuleContext)
    True
    """

    __slots__ = ()
    _record: Layer
    _GROUP = "module_context"

    @property
    def call_depth(self) -> Any:
        """Depth of module nesting around this layer."""

        return len(self._record.modules)

    @property
    def is_atomic_module(self) -> Any:
        """Whether this layer's op is the only op of a module call, a single-op leaf module."""

        return self._record._module_context_is_atomic_module

    @property
    def is_module_input(self) -> Any:
        """Whether this layer's representative op feeds into at least one `ModuleCall`, for single-op layers."""

        return bool(self._record._single_pass_or_error("input_to_module_calls"))

    @property
    def output_of_module_calls(self) -> Any:
        """Labels of the module calls whose output this layer is."""

        return self._record._module_context_output_of_module_calls

    @property
    def output_of_modules(self) -> Any:
        """Addresses of the modules whose output this layer is."""

        return self._record._module_context_output_of_modules


OP_GROUP_VIEWS["module_context"] = OpModuleContext

__all__ = ["LayerModuleContext", "OpModuleContext"]
