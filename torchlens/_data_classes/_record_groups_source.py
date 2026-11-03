"""The source group of an ``Op`` or ``Layer``: ``op.source.frames``, ``layer.source.fx_path``, ...

Where the record's call sits in the user's source: call-stack frames,
argument expressions, assigned variable names and the ``torch.fx``-style
identity. Stored members read the record's private storage attribute
(``_source_<leaf>``, same persisted column as before); a Layer's members read
its single pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._record_group_base import _LayerGroup, _RecordGroup

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class OpSource(_RecordGroup):
    """Where one Op's call sits in the user's source.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.ops["linear_1_1"].source, tl.types.OpSource)
    True
    """

    __slots__ = ()
    _record: Op

    @property
    def arg_expressions(self) -> Any:
        """Source expressions of this Op's call arguments, parsed lazily from `code_context`; an empty list when unavailable."""

        return self._record._source_arg_expressions

    @property
    def frames(self) -> Any:
        """Call-stack frames showing where this Op ran."""

        return self._record._source_frames

    @property
    def func_qualname(self) -> Any:
        """Qualified name of the function this Op called."""

        return self._record._source_func_qualname

    @property
    def fx_call_index(self) -> Any:
        """Call index in the `torch.fx`-style identity of this Op, when applicable."""

        return self._record._source_fx_call_index

    @property
    def fx_path(self) -> Any:
        """Module path in the `torch.fx`-style identity of this Op, when applicable."""

        return self._record._source_fx_path

    @property
    def var_names(self) -> Any:
        """Variable names the Op's result is assigned to at its call site; empty when unresolved."""

        return self._record._source_var_names


class LayerSource(_LayerGroup):
    """Where one Layer's call sits in the user's source; per-pass members require a single-pass layer.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.layers["linear_1"].source, tl.types.LayerSource)
    True
    """

    __slots__ = ()
    _record: Layer
    _GROUP = "source"

    @property
    def frames(self) -> Any:
        """The captured call stack of the user code that ran this layer's function, for single-pass layers."""

        return self._record._single_pass_or_error("code_context")

    @property
    def fx_call_index(self) -> Any:
        """The FX-style call index of a single-op layer."""

        return self._record._single_pass_or_error("fx_call_index")

    @property
    def fx_path(self) -> Any:
        """The FX-style qualified path of a single-op layer, or `None`."""

        return self._record._single_pass_or_error("fx_qualpath")


__all__ = ["LayerSource", "OpSource"]
