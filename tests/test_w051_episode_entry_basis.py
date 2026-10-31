"""W051 regressions for the step-join ENTRY BASIS (audit 3.3).

The join basis was "first tensor argument" with no disclosure of WHICH
argument was read: a kwargs-first masked LM (``forward(attention_mask=None,
input_ids=None)``) graded its attention MASK and reported false exogenous
positions on a true continuation; a timestep-first denoiser graded its
timestep. The entry is now chosen by a disclosed preference rule (preferred
keyword names, then non-auxiliary arguments with scalar timesteps passed
over), an explicit ``step_input_from`` declaration is honored exactly, and
the chosen argument is persisted per step in the envelope's ``entry_basis``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_join import (
    EpisodeJoinSession,
    _select_entry,
    validate_step_join_envelope,
)
from torchlens.capture._episode_ledger import ResolvedEpisode
from torchlens.options import EpisodeSpec

V = 16


class _Step(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _MaskLM(nn.Module):
    """kwargs-first signature: the mask precedes the ids."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _Step()

    def forward(self, attention_mask=None, input_ids=None):
        return self.inner(input_ids)


class _KwRunner(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.step = _MaskLM()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            mask = torch.ones_like(ids)
            nxt = self.step(attention_mask=mask, input_ids=ids).argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids[:, -self.n :]


def _envelope(trace: tl.Trace) -> dict:
    return trace.annotations["episode"]["header"]["step_join"]


def test_kwargs_first_masked_lm_grades_the_ids_not_the_mask() -> None:
    model = _KwRunner()
    log = tl.trace(
        model,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=3),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "continuous", "continuous"]
    assert envelope["exogenous_positions"] == {}
    assert envelope["entry_basis"] == {"1": "kwarg[input_ids]", "2": "kwarg[input_ids]"}


class _Denoiser(nn.Module):
    """Timestep-first signature (torchdiffeq / flow-matching convention)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x - 0.1 * self.lin(x) * t


class _Flow(nn.Module):
    def __init__(self, replace: bool) -> None:
        super().__init__()
        self.step = _Denoiser()
        self.replace = replace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(3):
            x = self.step(torch.tensor(float(i)), x)
            if self.replace:
                # The carried sample is REPLACED (shape-derived from x, so the
                # loop stays graph-connected and every step is recorded).
                x = torch.randn_like(x) + 100.0
        return x


def test_timestep_first_denoiser_grades_the_sample() -> None:
    torch.manual_seed(0)
    model = _Flow(replace=False)
    log = tl.trace(
        model,
        torch.randn(1, 4),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=3, step_output_kind="none"),
    )
    envelope = _envelope(log)
    assert envelope["entry_basis"] == {"1": "positional[1]", "2": "positional[1]"}
    # x flows step to step byte-identically (the exit IS the next entry).
    assert envelope["grades"] == [None, "continuous", "continuous"]
    assert envelope["basis"] == {"1": "digest_direct", "2": "digest_direct"}

    replaced = _Flow(replace=True)
    log2 = tl.trace(
        replaced,
        torch.randn(1, 4),
        episode=EpisodeSpec(stepped_module=replaced.step, n_steps=3, step_output_kind="none"),
    )
    envelope2 = _envelope(log2)
    assert envelope2["entry_basis"] == {"1": "positional[1]", "2": "positional[1]"}
    # The replaced sample is NOT the prior exit (digest_direct fails on the
    # sample, where the timestep basis could never see the replacement); the
    # open-feed graph witness then grades the captured randn_like transform.
    assert envelope2["basis"] == {"1": "graph", "2": "graph"}
    assert envelope2["grades"] == [None, "transformed", "transformed"]
    # Under the strict arm the replacement is a closed-feed violation.
    strict = _Flow(replace=True)
    with pytest.raises(Exception) as info:
        tl.trace(
            strict,
            torch.randn(1, 4),
            episode=EpisodeSpec(
                stepped_module=strict.step, n_steps=3, step_output_kind="none", feed="closed"
            ),
        )
    assert info.value.fields["code"] == "episode_feed_closed_violation"


@pytest.mark.smoke
def test_declared_step_input_from_is_honored_and_misses_disclose() -> None:
    torch.manual_seed(0)
    resolved_kwargs = {
        "episode_id": "ep-test",
        "address": "step",
        "n_steps": 3,
        "step_axis": -1,
        "step_output_kind": "none",
        "step_output_from": None,
        "forced_tokens": None,
        "escalated_from": None,
        "reason": None,
    }
    assert ResolvedEpisode(**resolved_kwargs).step_input_from is None

    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    # Explicit keyword declaration wins over every preference.
    assert _select_entry((), {"attention_mask": mask, "input_ids": ids}, "attention_mask") == (
        "kwarg[attention_mask]",
        mask,
    )
    # Explicit positional declaration.
    t = torch.tensor(0.5)
    x = torch.randn(1, 4)
    assert _select_entry((t, x), {}, 0) == ("positional[0]", t)
    # A declaration naming no tensor argument selects nothing (never a guess).
    assert _select_entry((t, x), {}, "sample") == (None, None)
    # The session records the miss as an unchecked reason, not a grade.
    session = EpisodeJoinSession(
        _Denoiser(), feed="open", on_feed_break="disclose", crossings=(), step_input_from="sample"
    )
    session.arm()
    try:
        session._stepped_module(t, x)
    finally:
        session.disarm()
    assert session.boundaries[0].unavailable_reason == "step_input_not_found:'sample'"


def test_preference_rule_table() -> None:
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    t = torch.tensor(0.5)
    x = torch.randn(1, 4)
    # Preferred keyword beats positional order.
    assert _select_entry((mask,), {"input_ids": ids}, None)[0] == "kwarg[input_ids]"
    # Auxiliary keywords are passed over for any non-auxiliary tensor.
    assert _select_entry((), {"attention_mask": mask, "ids": ids}, None)[0] == "kwarg[ids]"
    # Only auxiliary tensors present: the first one is still read (disclosed).
    assert _select_entry((), {"attention_mask": mask}, None)[0] == "kwarg[attention_mask]"
    # A scalar float timestep beside a wider sample is passed over.
    assert _select_entry((t, x), {}, None)[0] == "positional[1]"
    # Two wide positionals: positional order stands.
    assert _select_entry((x, x.clone()), {}, None)[0] == "positional[0]"
    # No tensors at all.
    assert _select_entry((1, "a"), {"k": 2}, None) == (None, None)


def test_envelope_entry_basis_is_optional_and_validated() -> None:
    envelope = {
        "schema": "episode_step_join_v1",
        "claim": "measured",
        "basis": {"1": "tokens"},
        "grades": [None, "continuous"],
        "break_step": None,
        "live_break_step": None,
        "exogenous_positions": {},
        "reasons": {},
    }
    validate_step_join_envelope(envelope, 2)  # pre-W051 envelope: still valid
    validate_step_join_envelope({**envelope, "entry_basis": {"1": "kwarg[input_ids]"}}, 2)
    with pytest.raises(ValueError, match="entry_basis"):
        validate_step_join_envelope({**envelope, "entry_basis": {"1": 7}}, 2)
    with pytest.raises(ValueError, match="in-range"):
        validate_step_join_envelope({**envelope, "entry_basis": {"0": "positional[0]"}}, 2)
    with pytest.raises(ValueError, match="exactly the keys"):
        validate_step_join_envelope({**envelope, "extra": {}}, 2)
