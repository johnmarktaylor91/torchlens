"""Ownership audit: every strong path from a trace to its record objects.

Run ``python -m tools.record_ownership`` for the holder table on live traces
(``--json FILE`` writes it), and ``python -m tools.record_ownership.uses`` for
the classification of every use of the trace's op list and label
dictionaries. ``tests/test_record_ownership.py`` pins the holder set.
"""

from __future__ import annotations

from typing import Any

from .graph import Audit, HolderEdge, audit_roots

__all__ = ["Audit", "HolderEdge", "audit_roots", "record_types", "run_case", "signature"]


def record_types() -> dict[type, str]:
    """Return the record classes the audit stops at, keyed to their kind name."""

    from torchlens._data_classes.backward_pass import BackwardPass
    from torchlens._data_classes.buffer import Buffer
    from torchlens._data_classes.grad_fn import GradFn
    from torchlens._data_classes.grad_fn_call import GradFnCall
    from torchlens._data_classes.layer import Layer
    from torchlens._data_classes.module import Module, ModuleCall
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.param import Param

    return {
        Op: "Op",
        Layer: "Layer",
        Module: "Module",
        ModuleCall: "ModuleCall",
        Param: "Param",
        Buffer: "Buffer",
        GradFn: "GradFn",
        GradFnCall: "GradFnCall",
        BackwardPass: "BackwardPass",
    }


def signature(edge: HolderEdge) -> str:
    """Render a holder edge as one line: the path, then the record kind it reaches."""

    return "".join(edge.path) + " -> " + edge.record_kind


def run_case(name: str) -> dict[str, Any]:
    """Run one case and return its holder counts as plain data."""

    from .cases import CASES

    audit = audit_roots(CASES[name](), record_types())
    trace_holders: dict[str, int] = {}
    for edge in audit.trace_edges:
        trace_holders[signature(edge)] = trace_holders.get(signature(edge), 0) + 1
    record_holders: dict[str, int] = {}
    for edge in audit.record_edges:
        record_holders[signature(edge)] = record_holders.get(signature(edge), 0) + 1
    return {
        "alive": dict(sorted(audit.alive.items())),
        "visited": audit.visited,
        "trace_holders": dict(sorted(trace_holders.items())),
        "record_holders": dict(sorted(record_holders.items())),
        "weak_back": dict(sorted(audit.weak_back.items())),
        "unexplained": audit.unexplained,
    }
