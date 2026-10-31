"""The travel-policy arm on a REAL model (lane F40a; foldA s6 F40 bullet).

Capture a real generation episode on distilgpt2 (the R0 roster's REAL
upstream class built from its vendored config, zero network), re-run on
different inputs, and assert no foreign ledger and no
VERIFIED-with-foreign-evidence product.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import episode_ledger_for
from torchlens.options import EpisodeSpec
from torchlens.runnable import PathFaithfulness

pytest.importorskip("transformers")

from tests.real_model.r0.families import SEED, VOCAB, build_distilgpt2

# heavy, not smoke: the real 2-layer distilgpt2 episode + full rerun measures
# ~7 s on the dev box (over the 5 s smoke budget; tiered honestly).
pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

N_STEPS = 3


class _GreedyGenerate(nn.Module):
    """Greedy generation loop over the REAL distilgpt2 LM head."""

    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            logits = self.model(input_ids=ids).logits
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids[:, -self.n_steps :]


def _prompt(offset: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + offset)
    return torch.randint(0, VOCAB, (1, 8), generator=generator)


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_distilgpt2_fresh_run_product_never_carries_foreign_ledger():
    """Real-model travel arm: capture on x1, re-run on x2, no foreign evidence.

    The append-all greedy feed grows per-pass shapes (seq 8 -> 10 across
    steps), so the full-rerun refresh emits its DISCLOSED shape-change
    warning on this arm; the warning is scoped-allowed here because the arm
    asserts the travel policy, not the refresh matcher's shape bookkeeping
    (which stays another lane's territory).
    """

    runner = _GreedyGenerate(build_distilgpt2("eager"), N_STEPS)
    x1 = _prompt()
    x2 = _prompt(offset=1)
    assert not torch.equal(x1, x2)

    trace = tl.trace(
        runner,
        x1,
        episode=EpisodeSpec(stepped_module=runner.model, n_steps=N_STEPS),
    )
    ledger = episode_ledger_for(trace)
    assert ledger is not None
    assert trace.capture_kind == "episode"
    assert len(ledger.rows) == N_STEPS
    assert all(row.status == "complete" for row in ledger.rows)
    assert all(row.step_output is not None for row in ledger.rows)

    result = trace.run(inputs=x2)

    # No foreign ledger on the product, whatever the faithfulness verdict --
    # and a VERIFIED product with foreign step evidence is the exact
    # silent-wrongness class this lane exists to kill.
    assert episode_ledger_for(result.trace) is None
    assert result.trace.capture_kind == "plain"
    note = result.trace.annotations.get("episode")
    assert note is not None and note.get("quarantined") is True
    assert note.get("code") == "episode_evidence_dropped_fresh_execution"
    if result.report.path_faithfulness is PathFaithfulness.VERIFIED:
        assert result.trace.episode is None

    # The source episode capture keeps its own honest evidence.
    assert episode_ledger_for(trace) is not None
