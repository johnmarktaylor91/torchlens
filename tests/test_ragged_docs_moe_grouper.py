"""A-RAGGED-DOCS: routed-MoE neutral-grouper regression (COORDINATE test).

Pins foldB D11 on a real routed mixture-of-experts architecture family: the
landed episode grouper keys on successive top-level calls of the declared
stepped module, never on pass-interior graph equality — a ragged interior
(experts absent from some steps) cannot change the step count, and pass
indices stay per-site occurrence counters.

REALISM CAVEAT (foldB s5 "ragged MoE reads" row): this is a CONFIG-BUILT
Qwen3Moe — a coordinate test pinning the architecture family's shape, NOT
the realism gate. The realism gate needs one roster-pinned public routed-MoE
CHECKPOINT, whose identity is an open TP9/P04 acquisition (the requirement
is settled 3-0); the episode-capture-caveats lint stays armed until
F-EPISODE's real-model matrix closes it. Nothing here claims that gate
closed.

Ground truth is self-consistent: an eager pre-run with a forward hook on the
fused experts module records which experts each step actually hit, and the
capture's facts are checked against that record. Raggedness is found by a
deterministic bounded seed scan (decode steps carry few tokens, so with more
experts than ``top_k`` most seeds route unevenly).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import EpisodeSpec

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_moe")

from transformers.models.qwen3_moe import (  # noqa: E402
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

pytestmark = pytest.mark.heavy

N_STEPS = 4
PROMPT = [[1, 2]]
SEED_SCAN = range(8)


def _build_model(seed: int) -> Qwen3MoeForCausalLM:
    """Config-built tiny routed Qwen3Moe (4 experts, top-1 routing), eval mode."""

    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model


class GreedyRunner(nn.Module):
    """Episode root: steps the causal LM N times, greedy decode, no KV cache."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            logits = self.model(current).logits
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


def _experts_module(model: Qwen3MoeForCausalLM) -> nn.Module:
    """The fused experts module of the single decoder layer's sparse MLP."""

    return model.model.layers[0].mlp.experts


def _eager_ground_truth(model: Qwen3MoeForCausalLM) -> tuple[list[list[int]], list[int]]:
    """Run the loop eagerly; record per-step hit-expert sets and emitted tokens."""

    hits: list[list[int]] = []

    def record(module: nn.Module, args: tuple, kwargs: dict, output: object) -> None:
        top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
        hits.append(sorted(set(top_k_index.reshape(-1).tolist())))

    handle = _experts_module(model).register_forward_hook(record, with_kwargs=True)
    try:
        with torch.no_grad():
            tokens = GreedyRunner(model, N_STEPS)(torch.tensor(PROMPT))
    finally:
        handle.remove()
    return hits, tokens.reshape(-1).tolist()


@pytest.fixture(scope="module")
def moe_episode():
    """Seed-scanned ragged fixture: eager ground truth + one episode capture.

    Both model variants (the scan models and the capture model) are
    constructed BEFORE the first ``tl.trace`` in the process, and the
    ground-truth model never enters the capture.
    """

    experts_probe = _experts_module(_build_model(0))
    if type(experts_probe).forward is nn.Module.forward:
        pytest.skip(
            "this coordinate test assumes the fused-experts Qwen3Moe "
            "implementation (transformers >= 5); per-expert ModuleList "
            "implementations need a different ground-truth hook"
        )
    for seed in SEED_SCAN:
        model = _build_model(seed)
        hits, eager_tokens = _eager_ground_truth(model)
        if len({len(step_hits) for step_hits in hits}) > 1:
            break
    else:
        pytest.fail(
            f"no seed in {SEED_SCAN!r} produced ragged routing (hit-expert "
            "counts constant across steps); the fixture assumption broke"
        )
    capture_model = _build_model(seed)
    runner = GreedyRunner(capture_model, N_STEPS)
    log = tl.trace(
        runner,
        torch.tensor(PROMPT),
        episode=EpisodeSpec(stepped_module=capture_model, num_steps=N_STEPS),
    )
    try:
        yield {"log": log, "hits": hits, "eager_tokens": eager_tokens, "seed": seed}
    finally:
        log.cleanup()


def test_ragged_interior_cannot_bend_the_ledger(moe_episode: dict) -> None:
    """D11: grouping keys on top-level calls; raggedness never changes the count."""

    hits = moe_episode["hits"]
    assert len({len(step_hits) for step_hits in hits}) > 1, (
        "fixture must be ragged (hit-expert counts vary across steps)"
    )
    ledger = moe_episode["log"].annotations["episode"]
    rows = ledger["rows"]
    assert len(rows) == N_STEPS
    assert [row["status"] for row in rows] == ["complete"] * N_STEPS
    assert [row["episode_step"] for row in rows] == list(range(N_STEPS))
    assert ledger["header"]["stepped_module"] == "model"


def test_capture_settles_complete(moe_episode: dict) -> None:
    """The ragged interior settles COMPLETE — no degradation, no refusal."""

    outcome = moe_episode["log"].outcome
    assert outcome.status.name == "COMPLETE"


def test_wrapped_tokens_match_eager(moe_episode: dict) -> None:
    """The wrapped episode emits the same greedy tokens as the eager loop."""

    rows = moe_episode["log"].annotations["episode"]["rows"]
    ledger_tokens = [token for row in rows for token in row["step_output"]]
    assert ledger_tokens == moe_episode["eager_tokens"]


def test_always_on_interior_site_counts_per_call(moe_episode: dict) -> None:
    """The router linear (runs every call) groups to exactly one pass per step."""

    log = moe_episode["log"]
    router_records = [
        op
        for op in log
        if op.func_name == "linear" and any("gate" in str(module) for module in (op.modules or ()))
    ]
    assert len(router_records) == N_STEPS
    assert len({op.layer_label for op in router_records}) == 1
    assert {op.num_passes for op in router_records} == {N_STEPS}


def test_expert_interior_sites_are_per_occurrence(moe_episode: dict) -> None:
    """Expert-interior sites count their OWN occurrences; nothing is padded.

    grouping rule rows 17, 26 and 51: a layer is one place in the experts
    module's code, so an op there has one pass per time that place ran. An op
    that runs once per call of the experts module has one pass per step (row
    26); an op inside a Python loop over the hit experts would have one pass
    per expert execution, summed over the steps (rows 17 and 51: a different
    expert weight per pass is not evidence). Measured on this fused
    implementation, every interior op runs once per call. A grouper that
    padded sites to a rectangular count (``N_STEPS * max hits``, or one per
    expert per step) or split them per expert weight lands outside both
    honest counts and fails here.
    """

    log = moe_episode["log"]
    hits = moe_episode["hits"]
    total_hits = sum(len(step_hits) for step_hits in hits)
    assert total_hits not in {N_STEPS, N_STEPS * max(len(step_hits) for step_hits in hits)}
    expert_ops = [
        op for op in log if any("experts" in str(module) for module in (op.modules or ()))
    ]
    assert expert_ops, "expected op records inside the fused experts module"
    counts = {op.num_passes for op in expert_ops}
    assert counts <= {N_STEPS, total_hits}, (
        f"expert-interior pass counts {counts} are neither per-call ({N_STEPS}) nor the "
        f"recorded per-occurrence execution count ({total_hits}) — the grouper is "
        "inventing rectangularity or splitting a site by its weights"
    )


# Per-step expert routing for the in-test Python-loop MoE: expert 0 is hit on
# every step, expert 1 on two of three, expert 2 never.
LOOP_ROUTES = ([0, 1, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0])


class _LoopExpert(nn.Module):
    """One expert: a linear map and a relu."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _LoopMoE(nn.Module):
    """MoE with a Python loop over experts that skips experts with no tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(_LoopExpert() for _ in range(3))
        self.scales = nn.ParameterList(
            nn.Parameter(torch.full((4,), 0.5 + e)) for e in range(len(self.experts))
        )

    def forward(self, x: torch.Tensor, route: list[int]) -> torch.Tensor:
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            idx = [i for i, r in enumerate(route) if r == e]
            if not idx:
                continue
            y = expert(x[idx]) * self.scales[e]
            out = out.index_add(0, torch.tensor(idx), y)
        return out


class _LoopMoESteps(nn.Module):
    """Calls the MoE once per route from one loop line."""

    def __init__(self) -> None:
        super().__init__()
        self.moe = _LoopMoE()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for route in LOOP_ROUTES:
            x = torch.tanh(x + self.moe(x, route))
        return x


def _address(op: object) -> str:
    """Innermost module address of an op, without its call index."""

    module = getattr(op, "module", None)
    return str(module).rsplit(":", 1)[0] if module else ""


def test_python_loop_moe_pass_counts_show_ragged_routing() -> None:
    """A per-expert Python loop exposes ragged routing in the pass counts.

    The fused Qwen3Moe experts above run every interior op once per call, so
    their pass counts cannot show routing. This in-test MoE restores that
    coverage. Grouping rule rows 26 and 29: each expert is a distinct module
    instance, so its ops are its own layers, and one pass per step it was hit
    (an expert with no tokens has no passes at all). Rows 17 and 51: the scale
    multiply is one place in the MoE loop, so it is one layer with one pass
    per expert execution summed over the steps, although each pass uses a
    different expert's parameter. Padding (every expert every step) or
    splitting the scale multiply by parameter fails these counts.
    """

    hits = [sum(e in route for route in LOOP_ROUTES) for e in range(3)]
    assert hits == [3, 2, 0]
    model = _LoopMoESteps()
    trace = tl.trace(model, torch.randn(4, 4))
    try:
        linear_passes = {
            _address(op): op.num_passes for op in trace.layer_list if op.func_name == "linear"
        }
        assert linear_passes == {"moe.experts.0.lin": 3, "moe.experts.1.lin": 2}
        assert not any(_address(op).startswith("moe.experts.2") for op in trace.layer_list)

        moe_ops = [op for op in trace.layer_list if _address(op) == "moe"]
        scale_layers = {
            op.layer_label: op.num_passes for op in moe_ops if op.func_name in {"mul", "__mul__"}
        }
        assert list(scale_layers.values()) == [sum(hits)], scale_layers
        add_layers = {
            op.layer_label: op.num_passes for op in moe_ops if op.func_name == "index_add"
        }
        assert list(add_layers.values()) == [sum(hits)], add_layers
        zeros_layers = {
            op.layer_label: op.num_passes for op in moe_ops if op.func_name == "zeros_like"
        }
        assert list(zeros_layers.values()) == [len(LOOP_ROUTES)], zeros_layers
    finally:
        tl.release_model(model)
        trace.cleanup()


def test_pass_window_on_moe_episode_discloses(moe_episode: dict) -> None:
    """The pass-axis disclosure fires on the MoE episode with trace-true counts.

    Most sites in a no-cache greedy loop shape-drift across steps (the token
    dimension grows) and the cross-pass statistic refuses on those, so the
    test scans the capture's full-count sites for one whose resolve succeeds
    (on this fixture: the fused experts module's constant-geometry weight
    transposes) and asserts the disclosure on it.
    """

    log = moe_episode["log"]
    full_count_labels = sorted({op.layer_label for op in log if op.num_passes == N_STEPS})
    for label in full_count_labels:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                tl.pass_variance(above=-1.0, within=label).resolve(log)
            except tl.selection.SelectionError:
                continue
        coded = [
            w.message
            for w in caught
            if getattr(w.message, "fields", {}).get("code") == "episode_pass_window_occurrence_axis"
        ]
        assert len(coded) == 1
        assert coded[0].fields["episode_steps"] == N_STEPS
        assert coded[0].fields["site_passes"] == {label: N_STEPS}
        return
    pytest.fail(
        "no full-count site on the MoE episode resolves a pass window "
        f"(candidates: {full_count_labels!r}); the disclosure cannot be "
        "pinned on this fixture geometry"
    )
