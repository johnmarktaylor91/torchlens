"""Explorer sign-off 6: the op-tier gate row beside the module-tier rows (D25).

The memo splits the overhead gates into two rows, module tier and op tier;
``tests/test_explorer_watch_gates.py`` carries the module-tier CUDA rows. This
module adds the op-tier row through the SHIPPED door (``watch(grain="op")``,
U32.1), measured by the same noise-disciplined A/B harness:

- a CPU structural row (unmarked): the row runs end to end on a small model and
  produces a complete A/B record; no number is claimed from a shared CPU box;
- the CUDA row (``rare``, real gpt2 on a dedicated GPU, the funded GPU hour):
  publishes the op-tier full-train-step ratio only when it sits outside the
  measured noise band. The memo sets no op-tier threshold, so the row is a
  receipt, never a pass/fail claim on the number.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.trackers as trk
from torchlens.observability import measure_ab, measure_noise_band


def _op_grain_row(model_a, model_b, step_fn, select, example, *, repeats, note):  # type: ignore[no-untyped-def]
    """Interleaved A/B: plain train step vs the same step under op-grain watch."""

    sink = trk.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = trk.watch(
            model_b,
            to=sink,
            signals=("activations",),
            grain="op",
            select=select,
            example_input=example,
            allow_forward_wrap=True,
            every=1,
        )
    counter = {"i": 0}

    def watched() -> None:
        i = counter["i"]
        counter["i"] += 1
        with session.step(i):
            step_fn(model_b)

    band = measure_noise_band(lambda: step_fn(model_a), repeats=repeats, warmup=1)
    row = measure_ab(
        "op-tier watch (grain='op')",
        lambda: step_fn(model_a),
        watched,
        repeats=repeats,
        warmup=1,
        noise_band=band,
        note=note,
    )
    session.close()
    return row, session, sink


def test_op_tier_gate_row_runs_end_to_end_on_cpu() -> None:
    """Structural CPU row: complete record, the watch really observed every step."""

    torch.manual_seed(0)

    def make() -> torch.nn.Module:
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))

    model_a, model_b = make(), make()
    x = torch.randn(8, 16)

    def step_fn(model: torch.nn.Module) -> None:
        model(x).square().mean().backward()
        model.zero_grad(set_to_none=True)

    row, _session, sink = _op_grain_row(
        model_a, model_b, step_fn, tl.func("relu"), x, repeats=3, note="cpu structural row"
    )
    assert row.repeats == 3 and row.median_a_s > 0 and row.median_b_s > 0
    assert row.noise_band >= 1.0
    observed = {p.step for p in sink.scalars if p.tag.startswith("activations/norm/")}
    assert len(observed) >= 3, "every timed watched step must have been observed"


@pytest.mark.rare
@pytest.mark.real_model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the D25 op-tier row runs on CUDA")
def test_op_tier_overhead_row_cuda_gpt2() -> None:
    """The funded GPU-hour row: gpt2-124M, b2 x s128, op grain on one block's ops."""

    transformers = pytest.importorskip("transformers")
    device = "cuda"
    torch.manual_seed(0)
    model_a = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device).train()
    model_b = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device).train()
    ids = torch.randint(0, 50257, (2, 128), device=device)
    optimizers = {
        id(model_a): torch.optim.AdamW(model_a.parameters(), lr=1e-5),
        id(model_b): torch.optim.AdamW(model_b.parameters(), lr=1e-5),
    }

    def step_fn(model: torch.nn.Module) -> None:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizers[id(model)].step()
        optimizers[id(model)].zero_grad()
        torch.cuda.synchronize()

    row, session, _sink = _op_grain_row(
        model_a,
        model_b,
        step_fn,
        "transformer.h.0",
        ids,
        repeats=20,
        note="D25 op-tier row; gpt2-124M b2xs128; ops of transformer.h.0",
    )
    receipt = {
        "row": row.verdict(),
        "ratio": row.ratio,
        "noise_band": row.noise_band,
        "publishable": row.publishable,
        "repeats": row.repeats,
        "calibration": session.collector.calibration.to_artifact()
        if session.collector is not None and session.collector.calibration is not None
        else None,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    out_dir = Path(os.environ.get("TORCHLENS_TEST_OUTPUTS_DIR", ".")) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "explorer_op_tier_gate_row.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    assert row.repeats >= 20
