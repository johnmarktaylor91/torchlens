"""Episode join honesty surfaces (lane F40c): per-artifact claims only.

The F40a claim-off switch flips WITH the measurement behind it, and every
surface keys on THIS artifact's envelope: an absent envelope reads
unmeasured, a measured break renders its break fact, and the build-level
token never upgrades an artifact's own evidence.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import _capture_honesty
from torchlens.capture import _episode_ledger as episode_ledger_module
from torchlens.capture._episode_ledger import episode_step_join_claim
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class GreedyRunner(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class ToolInjectRunner(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int, inject_at: int = 2):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.inject_at = inject_at

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for step in range(self.n_steps):
            if step == self.inject_at:
                tool = torch.full((ids.shape[0], 2), 9, dtype=ids.dtype)
                current = torch.cat([current, tool], dim=1)
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


def _capture(runner_cls, **spec_kwargs):
    model = TinyLM()
    return tl.trace(
        runner_cls(model, 4),
        torch.tensor([[1, 2, 3]]),
        episode=EpisodeSpec(stepped_module=model, n_steps=4, **spec_kwargs),
    )


def test_build_switch_is_flipped_with_the_measurement_behind_it() -> None:
    assert episode_ledger_module.EPISODE_STEP_JOIN_MEASURED is True
    assert episode_step_join_claim() == "measured"


def test_facts_read_the_artifact_envelope_never_the_build_switch() -> None:
    continuous = _capture(GreedyRunner)
    assert _capture_honesty.episode_facts(continuous)["step_join"] == "continuous"
    broken = _capture(ToolInjectRunner)
    assert _capture_honesty.episode_facts(broken)["step_join"] == "broken_at_step_2"
    declared = _capture(ToolInjectRunner, crossings=(2,))
    assert _capture_honesty.episode_facts(declared)["step_join"] == "declared_crossing_at_step_2"
    # Per-artifact honesty: stripping the envelope makes the SAME build read
    # this artifact as unmeasured -- the switch never upgrades evidence.
    stripped = _capture(GreedyRunner)
    stripped.annotations["episode"]["header"]["step_join"] = None
    assert _capture_honesty.episode_facts(stripped)["step_join"] == "unmeasured"


def test_explain_renders_break_and_unmeasured_lines() -> None:
    broken = _capture(ToolInjectRunner)
    report = tl.report.explain(broken)
    assert "Cross-step join BREAK measured (broken_at_step_2)" in report
    assert "chain-shaped at that join" in report
    stripped = _capture(GreedyRunner)
    stripped.annotations["episode"]["header"]["step_join"] = None
    unmeasured_report = tl.report.explain(stripped)
    assert "Cross-step continuity is DECLARED, not measured" in unmeasured_report


def test_banner_and_preamble_carry_the_join_fact() -> None:
    broken = _capture(ToolInjectRunner)
    preamble = "\n".join(_capture_honesty.honesty_preamble_lines(broken))
    assert "step_join=broken_at_step_2" in preamble
    banner = "\n".join(_capture_honesty.honesty_banner_lines(broken))
    assert "step_join=broken_at_step_2" in banner
    continuous = _capture(GreedyRunner)
    assert "step_join=continuous" in "\n".join(_capture_honesty.honesty_preamble_lines(continuous))
