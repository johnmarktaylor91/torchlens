"""EAP and EAP-IG on real GPT-2 small against an independent implementation.

Pre-registered bar CIRC-2a (definition exactness, gating): on GPT-2 small,
the first 8 rows of the pinned ``mib-bench/ioi`` train split kept by MIB's
tokenisation filter (the vendored rows beside this file; clean prompt and its
``s2_io_flip_counterfactual``), float64, ``tl.mechinterp.edge_scores`` with
EAP and with EAP-IG at ``m`` in {1, 5} equals a plain-PyTorch implementation
on the Hugging Face model (``tests/support/eap_reference.py``, computed with
torch unwrapped, no TorchLens code path) within 1e-8 relative plus 1e-10
absolute on every edge. Metric: the logit difference, logit(indirect object)
minus logit(subject), at each prompt's last token. Prompts are right-padded
with EOS (causal attention: the padding never reaches a scored position).
Runs only in the preflighted offline venue (GATE-ID R1_OFFLINE_VENUE).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.real_model, pytest.mark.real_checkpoint, pytest.mark.slow]

_ROWS = json.loads((Path(__file__).parent / "data" / "circ1_mib_ioi_rows.json").read_text())


def _ioi_batch(tokenizer, n: int):
    """Clean and corrupted ids ``[n, width]``, last-token positions and answer ids."""

    rows = _ROWS["rows"]["gpt2"][:n]

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    clean = [encode(row["prompt"]) for row in rows]
    corrupt = [encode(row["s2_io_flip_counterfactual"]) for row in rows]
    assert [len(ids) for ids in clean] == [len(ids) for ids in corrupt]
    width = max(len(ids) for ids in clean)
    pad = tokenizer.eos_token_id

    def padded(batch):
        return torch.tensor([ids + [pad] * (width - len(ids)) for ids in batch])

    last = torch.tensor([len(ids) - 1 for ids in clean])
    answer = torch.tensor([encode(" " + row["indirect_object"])[0] for row in rows])
    distractor = torch.tensor([encode(" " + row["subject"])[0] for row in rows])
    return padded(clean), padded(corrupt), last, answer, distractor


def test_edge_scores_match_the_independent_reference_on_gpt2_small(r1_loader) -> None:
    from support.eap_reference import TreeGPT2, edge_names

    import torchlens as tl
    from torchlens.backends.torch.wrappers import unwrap_torch

    loaded = r1_loader("r1-gpt2")
    model = copy.deepcopy(loaded["model"]).to(torch.float64).eval()
    model.set_attn_implementation("eager")
    clean, corrupt, last, answer, distractor = _ioi_batch(loaded["tokenizer"], 8)
    rows = torch.arange(clean.shape[0])

    def logit_difference(logits: torch.Tensor) -> torch.Tensor:
        return logits[rows, last, answer] - logits[rows, last, distractor]

    unwrap_torch()
    tree = TreeGPT2(model)
    cases = (("eap", None), ("eap_ig", 1), ("eap_ig", 5))
    want = {
        case: tree.edge_scores(clean, corrupt, logit_difference, method=case[0], steps=case[1] or 1)
        for case in cases
    }
    assert len(want[cases[0]]) == len(edge_names(12, 12)) == 32_491
    for method, steps in cases:
        extra = {} if steps is None else {"steps": steps}
        table = tl.mechinterp.edge_scores(
            model, clean, corrupted=corrupt, metric=logit_difference, method=method, **extra
        )
        got = {address[0]: row.score for (_t, _k, address), row in table.items()}
        bias = [name for name in got if name.startswith("b")]
        worst_rel = 0.0
        for edge, value in want[(method, steps)].items():
            error = abs(got[edge] - value)
            worst_rel = max(worst_rel, error / max(abs(value), 1e-30))
            assert error <= 1e-8 * abs(value) + 1e-10, (method, steps, edge, got[edge], value)
        print(
            f"CIRC2A-BAR method={method} steps={steps} edges={len(want[(method, steps)])} "
            f"bias_edges={len(bias)} worst_rel={worst_rel:.3e} "
            f"wall_s={table.provenance.timing_s:.1f}"
        )
        assert all(got[name] == 0.0 for name in bias)
