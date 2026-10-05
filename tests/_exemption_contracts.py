"""Validation contract clauses an exemption audit record may cite.

Shared by the exemption ledger (``test_validation_exemption_ledger.py``) and
the per-entry registry ledger (``test_validation_registry_ledger.py``) so both
check citations against one list.
"""

from __future__ import annotations

VALIDATION_CONTRACTS = frozenset(
    {
        "C1 replay determinism",
        "C2 perturbation sensitivity",
        "C3 replay-surface completeness",
        "C4 numeric resolution",
    }
)
