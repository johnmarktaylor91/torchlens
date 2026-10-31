"""Resource row: a ``site_sweep`` over all 144 heads of a 12-layer, 12-head GPT-2.

The sweep ablates every head (one scoped v-facet candidate each) under
explicit ``top_k`` retention and a byte ceiling. It must return the complete
144-row effect table, keep exactly the baseline plus the ``k`` most-changed
members, release the rest with their numbers intact, keep the retained bytes
within the ceiling, refuse a ceiling one byte too small before the first
candidate runs, and agree with a direct ``trace.fork().do(...)`` of the same
head ablation. The model is built from the real GPT-2 config through the
HuggingFace ``from_dict`` path with 12 layers and 12 heads at a reduced width
(no download), and runs on the replay engine, whose cost per candidate is the
edit's cone.
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.episode import BundleExperimentError
from torchlens.experiment import site_sweep, top_k
from torchlens.experiment._engine import _member_bytes
from torchlens.options import CaptureOptions, InterventionOptions

pytestmark = pytest.mark.slow

_LAYERS = 12
_HEADS = 12
_K = 3


def _gpt2_12x12() -> nn.Module:
    pytest.importorskip("transformers")
    sys.path.insert(0, ".")
    from transformers import GPT2Config, GPT2LMHeadModel

    from tests.real_model.r0.families import VOCAB, _hf_config

    shrink = {
        "n_layer": _LAYERS,
        "n_head": _HEADS,
        "n_embd": 96,
        "vocab_size": VOCAB,
        "n_positions": 32,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "use_cache": False,
    }
    torch.manual_seed(0)
    return GPT2LMHeadModel(_hf_config("gpt2", GPT2Config, shrink, "eager")).eval()


def _candidates() -> dict[str, object]:
    return {
        f"L{layer}H{head}": tl.head(head, "v").in_module(f"transformer.h.{layer}.attn")
        for layer in range(_LAYERS)
        for head in range(_HEADS)
    }


def _metric(member: tl.Trace) -> float:
    return float(member[member.output_layers[0]].out[0, -1].sum().item())


def test_all_144_heads_sweep_within_the_retention_ceiling() -> None:
    model = _gpt2_12x12()
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (1, 8))
    baseline = tl.trace(model, ids, capture=CaptureOptions(replayable=True))
    candidates = _candidates()
    per_member = _member_bytes(baseline)
    assert per_member > 0
    ceiling = per_member * (_K + 1)

    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates=candidates,
            edit=tl.zero_ablate(),
            metric=_metric,
            retain=top_k(_K),
            retain_ceiling_bytes=ceiling - 1,
            engine="replay",
        )
    assert excinfo.value.fields["code"] == "retention_projection_over_budget"
    assert excinfo.value.fields["projected_bytes"] == ceiling

    bundle = site_sweep(
        baseline,
        candidates=candidates,
        edit=tl.zero_ablate(),
        metric=_metric,
        retain=top_k(_K),
        retain_ceiling_bytes=ceiling,
        engine="replay",
    )
    view = bundle.effects()
    rows = {row.candidate_id: row for row in view.rows if row.candidate_id != "__baseline__"}
    assert set(rows) == set(candidates)
    assert all(row.value is not None and row.resolved_site_count == 1 for row in rows.values())
    retained = sorted(name for name, row in rows.items() if row.member_name is not None)
    released = [name for name, row in rows.items() if row.status == "released"]
    assert len(retained) == _K and len(released) == len(candidates) - _K
    top = sorted(name for name, _value in view.most_changed()[:_K])
    assert retained == top
    assert len(bundle.names) == _K + 1 and bundle.names[0] == "baseline"
    retained_bytes = sum(_member_bytes(bundle[name]) for name in bundle.names)
    assert retained_bytes <= ceiling

    for name in ("L0H0", "L6H5", "L11H11"):
        direct = baseline.fork().do(
            candidates[name], tl.zero_ablate(), intervention=InterventionOptions(engine="replay")
        )
        assert rows[name].value == _metric(direct)
