"""Multi-receiver edge patching on GPT-2 small against plain forward hooks.

In float64 on the first two vendored MIB IOI rows (clean prompt and its
``s2_io_flip_counterfactual``), one ``tl.mechinterp.patch_edges`` call that
patches every edge outside a random kept set (every receiver of the graph
is patched) equals the independent forward-hook implementation of the same
convention (``tests/support/multi_patch_reference.py``) within 1e-10
absolute on every layer's ``c_attn`` and ``ln_2`` outputs, ``ln_f`` and the
logits. Runs only in the preflighted offline venue (GATE-ID R1_OFFLINE_VENUE).
"""

from __future__ import annotations

import copy
import random
import time

import pytest
import torch

pytestmark = [pytest.mark.real_model, pytest.mark.real_checkpoint, pytest.mark.slow]


def test_a_gpt2_small_circuit_runs_as_one_call_and_matches_plain_hooks(r1_loader) -> None:
    from support.multi_patch_reference import HookedPatch, gpt2_spec

    import torchlens as tl
    from real_model.r1.test_circ2_edge_scores_real import _ioi_batch

    loaded = r1_loader("r1-gpt2")
    model = copy.deepcopy(loaded["model"]).to(torch.float64).eval()
    model.set_attn_implementation("eager")
    clean_ids, corrupt_ids, *_ = _ioi_batch(loaded["tokenizer"], 2)
    capture = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)
    clean = tl.trace(model, clean_ids, capture=capture)
    corrupt_trace = tl.trace(model, corrupt_ids, capture=capture)
    try:
        graph = tl.mechinterp.component_graph(clean)
        corrupt = tl.mechinterp.component_graph(corrupt_trace)
        rng = random.Random(0)
        edges = [edge for edge in graph.edges.values() if rng.random() >= 0.05]
        assert len({edge.receiver.name for edge in edges}) == len(graph.receivers) == 445
        started = time.perf_counter()
        patched = tl.mechinterp.patch_edges(clean, edges, corrupt=corrupt)
        print(
            f"patch_edges: {len(edges)} edges into 445 receivers in {time.perf_counter() - started:.1f} s"
        )
        hooked = HookedPatch(model, gpt2_spec(model))
        corrupt_nodes, _ = hooked.run(corrupt_ids)
        plan: dict[str, list[str]] = {}
        for edge in edges:
            plan.setdefault(edge.receiver.name, []).append(edge.sender.name)
        _, want = hooked.run(clean_ids, plan, corrupt_nodes)
        got = {
            "ln_f": patched.modules["transformer.ln_f"].out,
            "lm_head": patched.modules["lm_head"].out,
        }
        for layer in range(12):
            got[f"c_attn{layer}"] = patched.modules[f"transformer.h.{layer}.attn.c_attn"].out
            got[f"ln_2.{layer}"] = patched.modules[f"transformer.h.{layer}.ln_2"].out
        worst = max(float((got[name] - value).abs().max()) for name, value in want.items())
        print(f"max |TorchLens - plain hooks| = {worst:.3e}")
        for name, value in want.items():
            torch.testing.assert_close(got[name], value, rtol=0, atol=1e-10, msg=name)
    finally:
        clean.cleanup()
        corrupt_trace.cleanup()
