"""Two ``do``-era doors the teaching pages use.

``fork.do(spec)`` accepts a spec loaded with ``tl.io.load_intervention_spec``
(every stored rule re-attaches through its portable helper and the replay
engine propagates), and ``with trace.fork() as scoped:`` detaches the hooks the
last ``do()`` inside the block staged when the block ends.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    return model, torch.randn(3, 4)


def test_do_applies_a_loaded_intervention_spec(tmp_path) -> None:
    model, x = _model_and_input()
    options = tl.options.CaptureOptions(replayable=True)
    fork = tl.trace(model, x, capture=options).fork()
    fork.do(tl.module("1"), tl.scale(0.5))
    path = tmp_path / "scale.tlspec"
    tl.io.save_intervention(fork, path, level="portable")

    spec = tl.io.load_intervention_spec(path)
    fresh = tl.trace(model, x, capture=options).fork()
    reloaded = fresh.do(spec, intervention=tl.options.InterventionOptions(confirm_mutation=True))

    assert torch.equal(reloaded.output_ops[-1].out, fork.output_ops[-1].out)
    row = reloaded.intervention_audit[-1]
    assert row["kind"] == "ACT"
    assert "scale" in row["edit"]


def test_scoped_fork_detaches_the_hooks_do_staged_inside_the_block() -> None:
    model, x = _model_and_input()
    clean = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    with clean.fork("scoped") as scoped:
        scoped.do(tl.module("1"), tl.zero_ablate())
        assert len(scoped._intervention_spec.hook_specs) == 1
        edited_out = scoped.output_ops[-1].out.clone()
    assert len(scoped._intervention_spec.hook_specs) == 0
    assert torch.equal(scoped.output_ops[-1].out, edited_out)
    assert len(clean._intervention_spec.hook_specs if clean._intervention_spec else ()) == 0
