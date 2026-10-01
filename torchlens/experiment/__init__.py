"""torchlens.experiment — multi-run experiments as first-class objects (F03).

The F03 ledger-memo surface: the private candidate engine and its
``site_sweep`` face (items 6-7), the head-ablation sugar (7b), the effect
-table reads (item 8), and the persistent event-sourced experiment ledger
with its read-only serving tools (items 9-11). The Bundle is the MATERIAL
record (always on, travels with the artifact); the ledger is the SEMANTIC
record (hypothesis, metric choice, verdict — opt-in, references the material
record, never duplicates it).

Deliberately NOT here (ledger memo): no default metric, no default edit, no
default retention (each is a required argument — what to measure and what
evidence to destroy are the user's scientific choices); no iterated search
(TorchLens enumerates, executes, and records; ACDC-style circuit discovery
is user/agent strategy); no verdict computation ever (TorchLens may lint,
never decide).

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification; no
root ``tl.*`` names are registered here (the F35 registration sweep owns the
root facade).
"""

from __future__ import annotations

from ..bundle._bytes import bundle_retained_bytes as retained_bytes
from ._engine import EffectsView, TopK, site_sweep, top_k
from ._head_sugar import head_ablation_candidates
from ._ledger import (
    ExperimentLedger,
    LedgerEntry,
    active_ledger,
    ledger,
)
from ._mcp import ledger_entry, ledger_evidence, ledger_overview

__all__ = [
    "EffectsView",
    "ExperimentLedger",
    "LedgerEntry",
    "TopK",
    "active_ledger",
    "head_ablation_candidates",
    "ledger",
    "ledger_entry",
    "ledger_evidence",
    "ledger_overview",
    "retained_bytes",
    "site_sweep",
    "top_k",
]
