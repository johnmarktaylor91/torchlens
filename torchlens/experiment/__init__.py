"""torchlens.experiment: multi-run experiments as first-class objects.

The candidate engine and its two faces, ``sweep_sites`` (a flat candidate
plan) and ``sweep_grid`` (named axes, a labeled N-dimensional result), the
head-ablation sugar, and the effect-table reads. The Bundle is the material
record of an experiment: it is always on, travels with the artifact, and
carries the complete per-candidate effect table.

Deliberately NOT here: no default metric, no default edit, no default
retention (each is a required argument; what to measure and what evidence to
destroy are the user's scientific choices); no iterated search (TorchLens
enumerates, executes, and records; ACDC-style circuit discovery is user or
agent strategy); no verdict computation ever (TorchLens may lint, never
decide). The experiment ledger (the opt-in semantic record) was removed on
1 Nov 2026; the Bundle is the record that stays (``docs/reference/deprecations.md``).

No root ``tl.*`` names are registered here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils.facade import resolve_facade_attr
from ..bundle._bytes import bundle_retained_bytes as retained_bytes
from ._effects import EffectsView
from ._engine import sweep_sites
from ._grid import GridCell, SweepGrid, sweep_grid
from ._head_sugar import head_ablation_candidates

__all__ = [
    "EffectsView",
    "GridCell",
    "SweepGrid",
    "head_ablation_candidates",
    "retained_bytes",
    "sweep_grid",
    "sweep_sites",
]

_LEDGER_REDIRECT = (
    "the experiment ledger was removed (1 Nov 2026); the Bundle is the record: "
    "torchlens.experiment.sweep_sites returns one whose effect table and operation "
    "chronology persist with the artifact (bundle.effects(), bundle.history)"
)

#: Removed spellings: each raises the typed ``facade_redirect`` error naming
#: the record that stays (remove-and-rename, no alias).
_REDIRECTS: dict[str, str] = {
    "ExperimentLedger": _LEDGER_REDIRECT,
    "LedgerEntry": _LEDGER_REDIRECT,
    "active_ledger": _LEDGER_REDIRECT,
    "ledger": _LEDGER_REDIRECT,
    "read_ledger": _LEDGER_REDIRECT,
}


if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Raise the typed teaching refusal for a removed spelling.

        Parameters
        ----------
        name:
            Attribute name being looked up on ``torchlens.experiment``.

        Returns
        -------
        Any
            Never returns: every name this package defines is bound eagerly.

        Raises
        ------
        AttributeError
            Typed ``facade_redirect`` for a removed ledger spelling; plain for
            any other unknown name.
        """

        return resolve_facade_attr(
            owner=__name__, name=name, module_globals=globals(), redirects=_REDIRECTS
        )
