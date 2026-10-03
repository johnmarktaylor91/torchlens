"""S2 marker-combination table TOTALITY test (P1 duty, S2-RATIFICATION sec 5).

Transcribes the ratified marker-combination table (S2-RATIFICATION.md sec 5)
plus the S2-AMENDMENT-01 deltas verbatim and proves TOTALITY: every cell of
the full axis product {capture_kind} x {structure_only} x {truncation} x
{per-step provenance tier} resolves to either a ratified LEGAL row or a typed
refusal -- no cell is silently unspecified, and any combination not listed is
REFUSED until an amendment adds it. Reachable refusing cells are provoked
behaviorally; cells whose spelling has not shipped are tripwired so the change
that makes them reachable must amend this table in the same change.

Amendment deltas transcribed here:

- S2-AMENDMENT-01 item A.4: the checkpoint-invocation ambiguity term is an
  IDENTITY delta on this table (no new axis, no new row, no cell changes).
- S2-AMENDMENT-01 item B.5: ``declared_late_bind`` joins the provenance-tier
  AXIS only; legality rows are unchanged, so every cell carrying it refuses
  under the sec 5 default until a further amendment adds a row.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import derive_episode_status
from torchlens.errors import EpisodeDeclarationError
from torchlens.options import EpisodeSpec
from torchlens.runnable import StateSource

# ---------------------------------------------------------------------------
# Axes (S2 sec 5 + amendment B.5). ``mixed_exact_ledger`` is the wildcard
# row's own tier value ("mixed exact+ledger within one episode").
# ---------------------------------------------------------------------------

CAPTURE_KINDS = ("plain", "episode")
STRUCTURE_ONLY = (False, True)
TRUNCATION = ("absent", "present")
PROVENANCE_TIERS = ("n/a", "exact", "ledger_only", "declared_late_bind", "mixed_exact_ledger")

LEGAL = "legal"
REFUSE = "typed_refuse"

#: The ratified table, transcribed in DOCUMENT ORDER (first match wins; the
#: ``episode x structure_only=True`` refusal row precedes the mixed-tier
#: wildcard row exactly as in S2 sec 5). ``"*"`` is a wildcard. The mixed-tier
#: row is scoped to ``capture_kind="episode"`` by its own text ("within one
#: episode"); a plain capture has no per-step tier to mix.
_TABLE: tuple[tuple[tuple[object, object, object, object], str], ...] = (
    (("plain", False, "absent", "n/a"), LEGAL),  # today's baseline
    (("plain", False, "present", "n/a"), LEGAL),  # run-result term only
    (("plain", True, "absent", "n/a"), LEGAL),  # L7a core
    (("plain", True, "present", "n/a"), REFUSE),  # nothing executes to truncate
    (("episode", False, "absent", "exact"), LEGAL),  # episode core
    (("episode", False, "absent", "ledger_only"), LEGAL),  # degrades fail-closed on load
    (("episode", False, "present", "exact"), LEGAL),  # per-step truncation disclosed
    (("episode", False, "present", "ledger_only"), LEGAL),  # same fail-closed degradation
    (("episode", True, "*", "*"), REFUSE),  # structure-only episodes out of scope
    (("episode", "*", "*", "mixed_exact_ledger"), LEGAL),  # tier = MINIMUM of members
)


def legality(cell: tuple[str, bool, str, str]) -> str:
    """Resolve one axis-product cell against the ratified table.

    Parameters
    ----------
    cell:
        ``(capture_kind, structure_only, truncation, provenance_tier)``.

    Returns
    -------
    str
        ``"legal"`` for a ratified row; ``"typed_refuse"`` otherwise (sec 5:
        any combination not listed is REFUSED typed until an amendment).
    """

    for pattern, verdict in _TABLE:
        if all(want == "*" or want == have for want, have in zip(pattern, cell, strict=True)):
            return verdict
    return REFUSE


_ALL_CELLS = tuple(itertools.product(CAPTURE_KINDS, STRUCTURE_ONLY, TRUNCATION, PROVENANCE_TIERS))

#: The exact ratified LEGAL cell set, pinned as a literal so ANY drift in the
#: table transcription or the resolver is a loud diff reviewed against
#: S2-RATIFICATION.md sec 5, never a silent legality change.
_EXPECTED_LEGAL_CELLS = frozenset(
    {
        ("plain", False, "absent", "n/a"),
        ("plain", False, "present", "n/a"),
        ("plain", True, "absent", "n/a"),
        ("episode", False, "absent", "exact"),
        ("episode", False, "absent", "ledger_only"),
        ("episode", False, "present", "exact"),
        ("episode", False, "present", "ledger_only"),
        ("episode", False, "absent", "mixed_exact_ledger"),
        ("episode", False, "present", "mixed_exact_ledger"),
    }
)


def test_table_is_total_and_legal_set_is_pinned() -> None:
    """Every product cell resolves; the LEGAL set matches S2 sec 5 exactly."""

    assert len(_ALL_CELLS) == 40  # 2 x 2 x 2 x 5
    verdicts = {cell: legality(cell) for cell in _ALL_CELLS}
    assert set(verdicts.values()) <= {LEGAL, REFUSE}
    assert {cell for cell, verdict in verdicts.items() if verdict == LEGAL} == (
        _EXPECTED_LEGAL_CELLS
    )


def test_amendment_01_checkpoint_delta_is_identity() -> None:
    """S2-AMENDMENT-01 A.4: no new axis, no new row, no cell changes."""

    assert len(_TABLE) == 10  # the ten ratified sec 5 rows, nothing appended
    assert len(PROVENANCE_TIERS) == 5  # B.5's axis value + the wildcard tier only


def test_declared_late_bind_tier_has_no_legal_cell_yet() -> None:
    """S2-AMENDMENT-01 B.5: axis value added, legality rows unchanged.

    Every cell carrying the tier refuses under the sec 5 default. The
    ``StateSource`` tripwire below fires when the R-L7B-1 spelling ships:
    adding the enum member without amending the table (and this test) fails
    here, forcing the legality decision to be made explicitly.
    """

    for cell in _ALL_CELLS:
        if cell[3] == "declared_late_bind":
            assert legality(cell) == REFUSE
    spellings = {member.value for member in StateSource}
    assert not any("late" in spelling or "declared" in spelling for spelling in spellings), (
        "a declared-late-bind StateSource member shipped: amend the S2 "
        "marker-combination table (and this test) in the same change"
    )


# ---------------------------------------------------------------------------
# Behavioral provocations for the reachable refusing cells.
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Minimal stepped model: embedding -> mean -> head logits."""

    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class _GreedyRunner(nn.Module):
    """Episode root: steps the inner model, greedy decode."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            logits = self.model(current)
            next_token = logits.argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


def test_episode_structure_only_cell_refuses_typed() -> None:
    """episode x structure_only=True refuses at declaration time (row 9)."""

    model = _TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            _GreedyRunner(model, 2),
            torch.tensor([[1, 2, 3]]),
            capture=tl.options.CaptureOptions(structure_only=True),
            episode=EpisodeSpec(stepped_module=model),
        )
    assert excinfo.value.fields["code"] == "structure_only_episode_unsupported"


def test_plain_structure_only_truncation_cell_refuses_typed() -> None:
    """plain x structure_only x truncation-present refuses typed (row 4).

    Truncation is a RUN-RESULT term (``until=``); a structure-only capture
    executes no values to truncate, so the value-requiring run consumer
    refuses through the ONE structure-only capability chokepoint.
    """

    from torchlens.capture.structure_only import StructureOnlyCapabilityError

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        trace.run(inputs=torch.randn(2, 4), until="relu_1_2")
    assert excinfo.value.fields["code"] == "structure_only_replay_unsupported"


@pytest.mark.smoke
def test_mixed_tier_episode_folds_to_minimum() -> None:
    """The wildcard row's tier rule: episode tier = MINIMUM of member tiers."""

    mixed = derive_episode_status(
        [("COMPLETE", None), ("COMPLETE", None)],
        n_declared=2,
        ledger=None,
        member_tiers=["exact", "ledger_only"],
    )
    assert mixed.provenance_tier == "ledger_only"
    uniform = derive_episode_status(
        [("COMPLETE", None), ("COMPLETE", None)],
        n_declared=2,
        ledger=None,
        member_tiers=["exact", "exact"],
    )
    assert uniform.provenance_tier == "exact"
