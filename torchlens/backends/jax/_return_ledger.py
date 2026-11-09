"""Trace-time ledger for JAX module returns (the leaf half of ``_module_return_probe``).

``modules.py`` imports this module for the module trees' ``module_returns`` field and for
:func:`note_return`, which the scoped ``__call__`` wrappers call at every module exit. It
imports nothing from the JAX backend, so ``jaxpr.py`` (which imports ``modules.py``) stays
free of an import cycle; the jaxpr-dependent steps live in ``_module_return_probe.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._module_returns import ModuleReturnLedger


@dataclass
class JaxReturnLedger(ModuleReturnLedger):
    """Module-return ledger plus the trace-time state JAX needs to fill it.

    Parameters
    ----------
    armed:
        True only while the extended callable traces the model.
    pending:
        ``(address, call_index, returned_value)`` per module exit, in exit order.
    top_trace:
        Trace object of the top-level ``make_jaxpr`` trace, or ``None`` when unknown.
    leaves:
        Top-level tracer leaves, in the order they were appended as extra outputs.
    extra_values:
        Concrete values of those extra outputs after interpretation.
    producers:
        Capture index whose output is each extra value, or ``None`` (an input or constant).
    raw_labels:
        Raw producing-op label keyed by ``id`` of each recorded leaf tracer.
    extended_jaxpr:
        The extended closed jaxpr until it is interpreted, else ``None``.
    """

    armed: bool = False
    pending: list[tuple[str, int, Any]] = field(default_factory=list)
    top_trace: Any = None
    leaves: list[Any] = field(default_factory=list)
    extra_values: tuple[Any, ...] = ()
    producers: tuple[int | None, ...] = ()
    raw_labels: dict[int, str] = field(default_factory=dict)
    extended_jaxpr: Any = None

    def release(self) -> None:
        """Drop every tracer and value held for one capture."""

        self.armed = False
        self.pending = []
        self.top_trace = None
        self.leaves = []
        self.extra_values = ()
        self.producers = ()
        self.raw_labels = {}
        self.extended_jaxpr = None


def note_return(tree: Any, address: str, call_index: int, value: Any) -> Any:
    """Remember one module call's return while the extended callable traces; return it."""

    ledger = getattr(tree, "module_returns", None)
    if isinstance(ledger, JaxReturnLedger) and ledger.armed:
        ledger.pending.append((address, call_index, value))
    return value
