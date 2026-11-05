"""Lazy routing doors shared by Op, Module and Trace, without stored fields."""

from __future__ import annotations

from typing import Any

from ._moe_structure import _identity
from ._moe_tables import _routing_table


class _OpRouting:
    """Expose derived routing identity without enlarging an Op shell."""

    __slots__ = ()

    @property
    def is_router(self: Any) -> bool:
        """Whether this op belongs to a structurally proven expert router.

        Examples
        --------
        >>> trace = tl.trace(model, x)
        >>> any(op.is_router for op in trace.ops)
        False
        """
        return _identity(self)[0]

    @property
    def expert(self: Any) -> int | None:
        """Expert number for an op inside a routed expert, otherwise None.

        Numbers follow the container's registration order, including for a
        ModuleDict. A fused expert operation has no single expert number.

        Examples
        --------
        >>> trace = tl.trace(model, x)
        >>> all(op.expert is None for op in trace.ops)
        True
        """
        return _identity(self)[1]


class _ModuleRouting:
    """Expose a module's routing table without adding stored metadata."""

    @property
    def routing_table(self: Any) -> list[dict[str, Any]] | None:
        """Return routing rows for this MoE block, or None for other modules.

        Each row describes one router invocation: flattened token selections,
        gate weights, token counts and load fractions per expert. Requires a
        live source model and saved routing activations; see Trace.routing_table.

        Examples
        --------
        >>> trace = tl.trace(model, x)
        >>> trace.modules["self"].routing_table is None
        True
        """
        trace = self._source_trace
        return None if trace is None else _routing_table(trace, self.address) or None


class _TraceRouting:
    """Expose the union of routing rows for a trace."""

    @property
    def routing_table(self: Any) -> list[dict[str, Any]]:
        """Return lazy routing rows for every structurally recognized MoE block.

        Rows contain module, router, call, token_shape, expert_indices,
        gate_weights, tokens_per_expert and load. Tokens flatten the leading
        routing dimensions in row-major order; load is assignments divided by
        the total assignments, including top-k multiplicity. No model execution
        or capture-time work is added. Recognition requires the live source
        model's expert containers or batched matrices and captured index dependencies. Unrecognized
        structures yield no rows. Missing saved routing outputs refuse with
        PayloadUnavailableError and code moe_routing_activation_missing;
        capture with save='all' to keep the required activations.

        Examples
        --------
        >>> trace = tl.trace(model, x)
        >>> trace.routing_table
        []
        """
        return _routing_table(self)
