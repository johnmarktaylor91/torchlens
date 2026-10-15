"""One-call logit-lens trajectory statistics (``tl.stats.measure_trajectory``).

The call must equal a hand loop of per-batch captures, logit-lens
projections and hand-written statistics folded into ``Spine``s, exactly in
float64; ``mask_input=`` must ban the input's own tokens before every
reduction (also on ``logit_lens_predictions``); and the shared per-layer lens
helper must leave ``logit_lens_predictions`` bit-identical to reducing the
``logit_lens`` projections.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import FacetSpec
from torchlens.semantic.logit_lens import logit_lens, logit_lens_predictions

VOCAB, D_MODEL, SEQ = 13, 6, 5


class TrajBlock(nn.Module):
    """Residual toy block exposing resid_post."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.lin(x))


class TrajLM(nn.Module):
    """Toy causal LM over token ids: embed, two blocks, final norm, head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.b0 = TrajBlock()
        self.b1 = TrajBlock()
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_f(self.b1(self.b0(self.embed(ids)))))


class EmbedFedLM(TrajLM):
    """Same model fed embeddings (no token ids to mask)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.lm_head(self.ln_f(self.b1(self.b0(x))))


def _block_recipe(module: Any) -> dict[str, Any]:
    op = module.trace.ops[module.calls[0].output_ops[0]]
    return {"resid_post": FacetSpec.from_home(op, recipe_id="traj_toy_block")}


def _head_recipe(module: Any) -> dict[str, Any]:
    from torchlens.semantic.recipes._helpers import config_value, parameter_spec

    trace = module.trace
    head = trace.modules["lm_head"]
    norm = trace.modules["ln_f"]
    head_out = trace.ops[head.calls[0].output_ops[0]]
    result: dict[str, Any] = {
        "logits": FacetSpec.from_home(head_out, recipe_id="traj_toy_head"),
        "unembed_weight": parameter_spec(head, "weight", "traj_toy_head"),
        "final_norm_kind": "layer_norm",
        "final_norm_gamma": parameter_spec(norm, "weight", "traj_toy_head"),
        "final_norm_beta": parameter_spec(norm, "bias", "traj_toy_head"),
    }
    eps = config_value(norm, "eps", "variance_epsilon")
    if isinstance(eps, (int, float)):
        result["final_norm_eps"] = eps
    return result


_HEAD_FACETS = (
    "logits",
    "unembed_weight",
    "final_norm_kind",
    "final_norm_eps",
    "final_norm_gamma",
    "final_norm_beta",
)


@pytest.fixture(autouse=True, scope="module")
def _register_recipes() -> Iterator[None]:
    """Register the toy recipes at run time, restoring the registry after."""

    from torchlens.semantic import facets as _facets

    saved = list(_facets._REGISTRY)
    tl.facets.register(class_name="TrajBlock", target_scope="module", facets=("resid_post",))(
        _block_recipe
    )
    for class_name in ("TrajLM", "EmbedFedLM"):
        tl.facets.register(class_name=class_name, target_scope="module", facets=_HEAD_FACETS)(
            _head_recipe
        )
    try:
        yield
    finally:
        _facets._REGISTRY[:] = saved
        _facets._REGISTRY_VERSION += 1


def _model() -> TrajLM:
    torch.manual_seed(3)
    return TrajLM().double().eval()


def _batches(n: int = 3, rows: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    out = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (rows, SEQ), generator=generator)
        targets = torch.randint(0, VOCAB, (rows, SEQ), generator=generator)
        out.append((ids, targets))
    return out


def _capture(model: nn.Module, ids: torch.Tensor) -> Any:
    return tl.trace(
        model, ids, capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True)
    )


def _mask(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Hand mask: each row's own input ids get -inf at every position."""

    out = logits.clone()
    for row in range(out.shape[0]):
        out[row, :, ids[row].unique()] = float("-inf")
    return out


def _hand_values(
    logits: torch.Tensor, final: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Hand-written statistics for one layer, ``[batch, positions]`` each."""

    lp = torch.log_softmax(logits, dim=-1)
    p = lp.exp()
    q_log = torch.log_softmax(final, dim=-1)
    q = q_log.exp()
    tgt = targets.long().unsqueeze(-1)
    return {
        "entropy": -torch.where(p > 0, p * lp, 0.0).sum(dim=-1),
        "kl_to_final": torch.where(q > 0, q * (q_log - lp), 0.0).sum(dim=-1),
        "max_prob": p.amax(dim=-1),
        "cross_entropy": -lp.gather(-1, tgt).squeeze(-1),
        "target_rank": ((logits > logits.gather(-1, tgt)).sum(dim=-1) + 1).to(logits.dtype),
    }


def _hand_loop(model: nn.Module, batches: list[Any], *, mask: bool) -> dict[str, Any]:
    """Per-batch capture + logit lens + hand statistics, folded by hand."""

    sums: dict[str, torch.Tensor] = {}
    spines: dict[str, dict[str, tl.stats.Spine]] = {}
    examples = 0
    for ids, targets in batches:
        result = logit_lens(_capture(model, ids))
        final = result.final_logits
        assert final is not None
        if mask:
            final = _mask(final, ids)
        for row, entry in enumerate(result.entries):
            logits = _mask(entry.logits, ids) if mask else entry.logits
            for name, value in _hand_values(logits, final, targets).items():
                sums.setdefault(name, torch.zeros(len(result.entries), SEQ, dtype=torch.float64))
                sums[name][row] += value.sum(dim=0)
                spine = spines.setdefault(name, {}).setdefault(entry.address, tl.stats.Spine())
                spine.update(value)
        examples += int(ids.shape[0])
    return {"means": {k: v / examples for k, v in sums.items()}, "spines": spines, "n": examples}


@pytest.mark.parametrize("mask", [False, True])
def test_trajectory_equals_hand_loop_float64(mask: bool) -> None:
    """Every statistic equals the hand loop exactly (means and Spine fields)."""

    model, batches = _model(), _batches()
    got = tl.stats.measure_trajectory(model, batches, mask_input=mask)
    want = _hand_loop(model, batches, mask=mask)
    assert got.statistics == tl.stats._trajectory.TRAJECTORY_STATISTICS
    assert got.examples == want["n"] == 6 and got.positions == SEQ
    assert got.layers == ("b0", "b1") and got.validated and got.mask_input is mask
    for name in got.statistics:
        assert got.position_means[name].dtype == torch.float64
        assert torch.equal(got.position_means[name], want["means"][name]), name
        for layer in got.layers:
            assert got.spines[name][layer].result() == want["spines"][name][layer].result()


def test_trajectory_default_statistics_without_targets() -> None:
    """Batches without targets measure the untargeted statistics only."""

    model = _model()
    ids = [batch[0] for batch in _batches(2)]
    got = tl.stats.measure_trajectory(model, ids)
    assert got.statistics == ("entropy", "kl_to_final", "max_prob")
    assert set(got.summary("entropy")) == {"b0", "b1"}
    assert "2 layers x 5 positions" in repr(got)
    single = tl.stats.measure_trajectory(model, ids[0], statistics=["entropy"])
    assert single.examples == 2 and single.statistics == ("entropy",)


def test_masked_tokens_carry_zero_probability() -> None:
    """With mask_input the input's own tokens cannot be predicted at any layer."""

    model = _model()
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    target = torch.tensor([[1, 1, 1, 1, 1]])
    got = tl.stats.measure_trajectory(model, [(ids, target)], mask_input=True)
    assert torch.isinf(got.position_means["cross_entropy"]).all()
    plain = tl.stats.measure_trajectory(model, [(ids, target)])
    assert torch.isfinite(plain.position_means["cross_entropy"]).all()


def test_logit_lens_predictions_mask_input_bans_input_tokens() -> None:
    """``mask_input=True`` keeps banned ids out of top-k and renormalises."""

    model = _model()
    ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    trace = _capture(model, ids)
    masked = logit_lens_predictions(trace, k=3, tokens=[0, 12], mask_input=True)
    projections = logit_lens(trace)
    for row, entry in zip(masked.rows, projections.entries, strict=False):
        for batch in range(2):
            banned = set(ids[batch].tolist())
            assert not banned & set(row.top_ids[batch].reshape(-1).tolist())
        expected = torch.logsumexp(_mask(entry.logits.float(), ids), dim=-1)
        assert torch.equal(row.logsumexp, expected)
        assert (row.token_probs[0, :, 0] == 0).all()
    native = masked.rows[-1]
    assert native.provenance == "native output"
    assert not {0, 1, 2, 3, 4} & set(native.top_ids[0].reshape(-1).tolist())


def test_logit_lens_predictions_match_logit_lens_projection() -> None:
    """The shared per-layer helper leaves predictions bit-identical to the projections."""

    from torchlens.semantic.logit_lens import PROVENANCE_PROJECTED, _reduce_projection_row

    model = _model()
    trace = _capture(model, _batches(1)[0][0])
    rows = logit_lens_predictions(trace, k=4, tokens=[2, 5], include_native=False).rows
    entries = logit_lens(trace).entries
    assert len(rows) == len(entries) == 2
    for row, entry in zip(rows, entries, strict=True):
        expected = _reduce_projection_row(
            entry.logits,
            address=entry.address,
            layer_index=entry.layer_index,
            facet=entry.facet,
            provenance=PROVENANCE_PROJECTED,
            position_base=0,
            full_length=SEQ,
            k=4,
            positions=None,
            token_ids=(2, 5),
        )
        for field in ("top_ids", "top_logits", "top_probs", "logsumexp", "token_ranks"):
            assert torch.equal(getattr(row, field), getattr(expected, field)), field
        assert row.positions == expected.positions and row.address == expected.address


def test_logit_lens_predictions_mask_input_refuses_embeddings() -> None:
    """A trace fed embeddings has no token ids to ban: typed refusal."""

    torch.manual_seed(3)
    model = EmbedFedLM().double().eval()
    trace = _capture(model, torch.randn(1, SEQ, D_MODEL, dtype=torch.float64))
    with pytest.raises(tl.errors.TorchLensError) as info:
        logit_lens_predictions(trace, mask_input=True)
    assert info.value.fields["code"] == "logit_lens_mask_input_unavailable"
    rows = logit_lens_predictions(trace).rows
    assert rows


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("targets_missing", "stats_trajectory_targets_missing"),
        ("mask_floats", "stats_trajectory_mask_input_unavailable"),
        ("unknown_statistic", "stats_trajectory_statistic_unknown"),
        ("positions_changed", "stats_trajectory_positions_changed"),
        ("flag_type", "stats_trajectory_flag_type"),
        ("empty", "stats_trajectory_empty"),
        ("data_type", "stats_trajectory_data_type"),
        ("targets_float", "stats_trajectory_targets_type"),
        ("targets_out_of_vocab", "stats_trajectory_targets_type"),
    ],
)
def test_trajectory_refusals(case: str, code: str) -> None:
    """Every refusal carries its stable code."""

    model = _model()
    ids = _batches(1)[0][0]
    calls = {
        "targets_missing": lambda: tl.stats.measure_trajectory(
            model, [ids], statistics=["cross_entropy"]
        ),
        "mask_floats": lambda: tl.stats.measure_trajectory(
            EmbedFedLM().double().eval(),
            [torch.randn(1, SEQ, D_MODEL, dtype=torch.float64)],
            mask_input=True,
        ),
        "unknown_statistic": lambda: tl.stats.measure_trajectory(
            model, [ids], statistics=["perplexity"]
        ),
        "positions_changed": lambda: tl.stats.measure_trajectory(model, [ids, ids[:, :3]]),
        "flag_type": lambda: tl.stats.measure_trajectory(model, [ids], mask_input="yes"),
        "empty": lambda: tl.stats.measure_trajectory(model, []),
        "data_type": lambda: tl.stats.measure_trajectory(model, 5),
        "targets_float": lambda: tl.stats.measure_trajectory(model, [(ids, ids.double())]),
        "targets_out_of_vocab": lambda: tl.stats.measure_trajectory(
            model, [(ids, ids + VOCAB)], statistics=["cross_entropy"]
        ),
    }
    with pytest.raises(tl.errors.TorchLensError) as info:
        calls[case]()
    assert info.value.fields["code"] == code


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("layers_changed", "stats_trajectory_layers_changed"),
        ("projection_rank", "stats_trajectory_projection_rank"),
        ("final_missing", "stats_trajectory_final_unavailable"),
        ("final_sliced", "stats_trajectory_final_unavailable"),
    ],
)
def test_trajectory_internal_refusals(case: str, code: str) -> None:
    """Refusals a toy model cannot reach end to end, provoked at their helper."""

    from torchlens.stats import _trajectory

    def layers_changed() -> None:
        folder = _trajectory._TrajectoryFolder(None)
        folder._start(("blocks.0",), "identity", True)
        folder._start(("blocks.1",), "identity", True)

    calls = {
        "layers_changed": layers_changed,
        "projection_rank": lambda: _trajectory._as_bpv(torch.zeros(VOCAB)),
        "final_missing": lambda: _trajectory._final_log_probs(None, None, ("kl_to_final",), None),
        "final_sliced": lambda: _trajectory._align_final(
            torch.zeros(1, 1, VOCAB), torch.zeros(1, SEQ, VOCAB)
        ),
    }
    with pytest.raises(tl.errors.TorchLensError) as info:
        calls[case]()
    assert info.value.fields["code"] == code


def test_trajectory_spines_save_and_reload(tmp_path: Any) -> None:
    """The result's Spines are ordinary accumulators: they save and reload."""

    got = tl.stats.measure_trajectory(_model(), _batches(1))
    spine = got.spines["entropy"]["b1"]
    path = tl.stats.save_fitted(spine, tmp_path / "entropy-b1.safetensors")
    assert tl.stats.load_fitted(path).result() == spine.result()
