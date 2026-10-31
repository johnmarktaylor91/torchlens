"""W051 FIX2 -- the declared-crossing chain arm of the alignment license.

The audit-3.1 license (``tests/test_w051_episode_join_grading.py``) refuses
tail alignment unless the source width is the emitted-only or the measured
entry-plus-one-per-step shape. That refused the DECLARED tool-call shape: a
``generate()``-style root returning the full fed ids carries prompt +
emissions + the injected tool tokens, and tail alignment there would
misattribute a tool token to the step before the injection.

The fix derives the column at the MEASURED entry-chain positions -- each
step's emission is the position right after its own measured entry, the
root must reproduce every entry as a prefix, and a surplus is admitted only
at a step declared in ``EpisodeSpec(crossings=...)`` -- and DISCLOSES the
positions read in the header (``step_output_positions``, bound by the
capture digest) so a loaded product re-derives without the live session.
An undeclared surplus stays refused: measured alone it is indistinguishable
from multi-token decoding.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_derivation import (
    rederived_evidence_matches,
    validate_step_output_positions,
)
from torchlens.capture._episode_ledger import (
    EpisodeLedger,
    EpisodeLedgerHeader,
    EpisodeLedgerRow,
    episode_ledger_for,
)
from torchlens.errors import TorchLensWarning
from torchlens.errors.episode import EpisodeDeclarationError
from torchlens.options import EpisodeSpec

V = 16
PROMPT = torch.tensor([[1, 2, 3, 4]])


class _Step(nn.Module):
    """next token = (last token + 1) % V; a deterministic permutation LM."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _ToolInject(nn.Module):
    """generate()-shaped loop: two tool tokens enter before step ``inject_at``."""

    def __init__(self, n: int = 3, inject_at: int = 1, corrupt_prompt: bool = False) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n
        self.inject_at = inject_at
        self.corrupt_prompt = corrupt_prompt

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for i in range(self.n):
            if i == self.inject_at:
                ids = torch.cat([ids, torch.full((ids.shape[0], 2), 9, dtype=ids.dtype)], dim=1)
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        if self.corrupt_prompt:
            ids = ids.clone()
            ids[:, 0] = 15
        return ids


class _PromptPlusCompletion(nn.Module):
    """A generate()-shaped root: prompt + one emission per step."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids


def _header(trace: tl.Trace) -> dict:
    return trace.annotations["episode"]["header"]


def _trace(model: nn.Module, **spec) -> tl.Trace:
    return tl.trace(
        model, PROMPT, episode=EpisodeSpec(stepped_module=model.step, n_steps=3, **spec)
    )


# ---------------------------------------------------------------------------
# The declared tool-call shape derives at the measured chain positions.
# ---------------------------------------------------------------------------


def test_declared_crossing_derives_at_measured_chain_positions() -> None:
    log = _trace(_ToolInject(), crossings=(1,))
    ledger = episode_ledger_for(log)
    assert ledger is not None
    # Entries: [1,2,3,4] (w=4) -> [1,2,3,4,5,9,9] (w=7) -> [...,10] (w=8); root W=9.
    assert ledger.header.step_output_positions == (4, 7, 8)
    assert _header(log)["step_output_positions"] == [4, 7, 8]
    assert tuple(row.step_output for row in ledger.rows) == ((5,), (10,), (11,))
    envelope = _header(log)["step_join"]
    assert envelope["grades"] == [None, "declared", "continuous"]
    assert envelope["break_step"] == 1


@pytest.mark.smoke
def test_undeclared_injection_refuses_and_teaches_crossings() -> None:
    """Measured alone, the surplus is ambiguous with multi-token decoding."""

    with pytest.raises(EpisodeDeclarationError) as info:
        _trace(_ToolInject())
    assert info.value.fields["code"] == "episode_declaration_invalid"
    message = str(info.value)
    assert "9 positions" in message
    assert "exactly 3 (emitted-only root) or 7" in message
    assert "crossings=(k, ...)" in message


def test_declared_crossing_requires_the_root_to_reproduce_the_entry_chain() -> None:
    """A root that does not return the fed ids cannot be chain-aligned."""

    with pytest.raises(EpisodeDeclarationError) as info:
        _trace(_ToolInject(corrupt_prompt=True), crossings=(1,))
    assert info.value.fields["code"] == "episode_declaration_invalid"


def test_declaring_the_wrong_crossing_step_still_refuses() -> None:
    """The surplus sits at step 1's entry; declaring step 2 licenses nothing."""

    with pytest.raises(EpisodeDeclarationError) as info:
        _trace(_ToolInject(), crossings=(2,))
    assert info.value.fields["code"] == "episode_declaration_invalid"


def test_pure_chain_discloses_the_tail_positions() -> None:
    log = _trace(_PromptPlusCompletion())
    ledger = episode_ledger_for(log)
    assert ledger is not None
    assert ledger.header.step_output_positions == (4, 5, 6)
    assert ledger.step_output_series() == ((5,), (6,), (7,))


# ---------------------------------------------------------------------------
# The disclosure travels, re-derives, and validates fail-closed.
# ---------------------------------------------------------------------------


def test_chain_positions_round_trip_and_rederive(tmp_path) -> None:
    log = _trace(_ToolInject(), crossings=(1,))
    path = tmp_path / "tool_call.tlspec"
    tl.save(log, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        loaded = tl.load(path)
    ledger = episode_ledger_for(loaded)
    assert ledger is not None
    assert ledger.header.step_output_positions == (4, 7, 8)
    assert tuple(row.step_output for row in ledger.rows) == ((5,), (10,), (11,))
    assert rederived_evidence_matches(loaded, loaded.annotations["episode"]) is True
    assert loaded.episode_coupling.bound is True


def test_tail_rederivation_without_disclosure_reads_the_tail() -> None:
    """A pre-disclosure v2 payload (no key) loads as None and re-derives at the tail."""

    log = _trace(_PromptPlusCompletion())
    payload = _header(log)
    legacy = {key: value for key, value in payload.items() if key != "step_output_positions"}
    header = EpisodeLedgerHeader.from_payload(legacy)
    assert header.step_output_positions is None
    assert "step_output_positions" in header.to_payload()
    rows = log.annotations["episode"]["rows"]
    assert rederived_evidence_matches(log, {"header": legacy, "rows": rows}) is True


@pytest.mark.smoke
def test_forged_positions_do_not_rederive() -> None:
    """Rewritten positions read a different column: the anchor reports a mismatch."""

    log = _trace(_ToolInject(), crossings=(1,))
    payload = dict(_header(log))
    payload["step_output_positions"] = [6, 7, 8]
    rows = log.annotations["episode"]["rows"]
    assert rederived_evidence_matches(log, {"header": payload, "rows": rows}) is False
    payload["step_output_positions"] = [4, 7, 30]
    assert rederived_evidence_matches(log, {"header": payload, "rows": rows}) is False


@pytest.mark.parametrize(
    ("positions", "kind"),
    [
        ([4, 7, 8], "none"),
        ([4, True, 8], "tokens"),
        ([4, 7, 7], "tokens"),
        ([4, -1, 8], "tokens"),
        ("478", "tokens"),
    ],
)
def test_positions_validator_refuses_malformed_slots(positions, kind) -> None:
    with pytest.raises(ValueError):
        validate_step_output_positions(positions, step_output_kind=kind)


def test_positions_validator_geometry_against_rows() -> None:
    assert validate_step_output_positions([4, 7, 8], step_output_kind="tokens", n_rows=3) == (
        4,
        7,
        8,
    )
    with pytest.raises(ValueError):
        validate_step_output_positions([4, 7], step_output_kind="tokens", n_rows=3)
    with pytest.raises(ValueError):
        validate_step_output_positions(
            [4, 7, 8], step_output_kind="tokens", n_rows=3, rows_with_output=2
        )


def test_ledger_assembly_refuses_positions_geometry_mismatch() -> None:
    log = _trace(_ToolInject(), crossings=(1,))
    ledger = episode_ledger_for(log)
    assert ledger is not None
    header = EpisodeLedgerHeader.from_payload(
        {**ledger.header.to_payload(), "step_output_positions": [4, 7]}
    )
    with pytest.raises(ValueError):
        EpisodeLedger(header, ledger.rows)
    header = EpisodeLedgerHeader.from_payload({**ledger.header.to_payload(), "step_join": None})
    truncated = [
        EpisodeLedgerRow.from_payload({**row.to_payload(), "step_output": None})
        for row in ledger.rows
    ]
    with pytest.raises(ValueError):
        EpisodeLedger(header, truncated)
