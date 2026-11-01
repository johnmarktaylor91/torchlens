"""Doors from the records to the packages above them (Rule F).

The records (``Trace``, ``Op``, ``Layer``, ``Bundle``) live in layer L1 and
may not import the packages above them at module top. This module is the
one FACADE-role door they use instead: one function per destination
package, each holding a single deferred import of that package's record
destination modules (the modules the record lanes fill) and returning the
package module, so a thin record method reads
``_doors.report()._trace_summary.summary(trace)``. The imports are plain
``from`` statements (never ``importlib`` strings) so ``tools/layering/graph.py``
keeps seeing every edge. No other module in ``_data_classes`` imports this
one; a lane that needs a new door appends a function here in its own commit.
"""

from __future__ import annotations

import sys
from types import ModuleType

__tl_layer__ = "FACADE"

__all__: list[str] = []


def _package_of(module: ModuleType) -> ModuleType:
    """Return the already-imported parent package of a destination module."""

    return sys.modules[module.__name__.rpartition(".")[0]]


def report() -> ModuleType:
    """Door to ``torchlens.report`` (facts, summary and narrate destinations)."""

    from ..report import _agent_json, _narrate, _trace_facts, _trace_summary  # noqa: F401

    return _package_of(_trace_summary)


def visualization() -> ModuleType:
    """Door to ``torchlens._visualization`` (draw and collapse destinations)."""

    from .._visualization import _trace_collapse, _trace_draw  # noqa: F401

    return _package_of(_trace_draw)


def intervention() -> ModuleType:
    """Door to ``torchlens.intervention`` (edits, selection edits, hooks, run and op-edit destinations)."""

    from ..intervention import (  # noqa: F401
        _op_edits,
        _trace_edits,
        _trace_hooks,
        _trace_run,
        _trace_selection_edits,
    )

    return _package_of(_trace_edits)


def validation() -> ModuleType:
    """Door to ``torchlens.validation`` (the trace checks destination)."""

    from ..validation import _trace_checks

    return _package_of(_trace_checks)


def export() -> ModuleType:
    """Door to ``torchlens.export`` (the trace tables destination)."""

    from ..export import _trace_tables

    return _package_of(_trace_tables)


def selection() -> ModuleType:
    """Door to ``torchlens._selection`` (resolver and do-plan, the record-facing modules)."""

    from .._selection import do_plan, resolver  # noqa: F401

    return _package_of(resolver)


def backends_torch() -> ModuleType:
    """Door to ``torchlens.backends.torch`` (the trace backward destination)."""

    from ..backends.torch import _trace_backward

    return _package_of(_trace_backward)


def recording() -> ModuleType:
    """Door to ``torchlens.recording`` (the trace preview destination)."""

    from ..recording import _trace_preview

    return _package_of(_trace_preview)


def debug() -> ModuleType:
    """Door to ``torchlens.debug`` (audit and NaN search, the record-facing modules)."""

    from ..debug import _audit, _nan  # noqa: F401

    return _package_of(_audit)


def receptive_field() -> ModuleType:
    """Door to ``torchlens.receptive_field`` (trace, op and layer field destinations)."""

    from ..receptive_field import _layer_fields, _op_fields, _trace_tables  # noqa: F401

    return _package_of(_trace_tables)


def trace_slice() -> ModuleType:
    """Door to ``torchlens._trace_slice`` (the ``between`` and ``subgraph`` slice builders)."""

    from .. import _trace_slice

    return _trace_slice
