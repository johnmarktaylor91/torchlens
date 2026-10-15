"""Per-unit ``tl.stats.Quantile`` and ``tl.stats.TopK`` (neurons and SAE latents).

Every top-k case is checked against a brute-force oracle that holds every
activation in memory. Weights and inputs are small integers, so each
activation is exact in float32 whatever the batch size, and integer values
make ties common; the oracle and the accumulator must agree exactly,
example ids and positions included.
"""

from __future__ import annotations

import math
import re
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.stats import _per_unit

N_EXAMPLES = 7
TOKENS = 5


class _MLP(nn.Module):
    """Linear -> ReLU -> Linear with integer weights (exact arithmetic)."""

    def __init__(self, d_in: int = 4, hidden: int = 6, d_out: int = 3) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, d_out)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                layer.weight.copy_(torch.randint(-2, 3, layer.weight.shape, generator=generator))
                layer.bias.copy_(torch.randint(-1, 2, layer.bias.shape, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the ReLU site directly (the oracle's view)."""

        return self.act(self.fc1(x))


class _SAE(nn.Module):
    """A tiny SAE encoder over the ReLU site, with one dead latent."""

    def __init__(self, hidden: int = 6, latents: int = 8) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        self.w_enc = nn.Parameter(
            torch.randint(-1, 2, (hidden, latents), generator=generator).float()
        )
        self.b_enc = nn.Parameter(torch.randint(-1, 1, (latents,), generator=generator).float())
        with torch.no_grad():
            self.w_enc[:, 0] = 0.0
            self.b_enc[0] = -1.0

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h @ self.w_enc + self.b_enc)


def _inputs(*shape: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=generator).float()


def _batches(data: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [data[start : start + batch_size] for start in range(0, len(data), batch_size)]


def _relu_label(model: nn.Module, x: torch.Tensor) -> str:
    log = tl.trace(model, x)
    try:
        return next(str(op.layer_label) for op in log.layer_list if op.layer_type == "relu")
    finally:
        log.cleanup()


def _oracle_topk(acts: torch.Tensor, k: int) -> tuple[list, list, list]:
    """Brute force: per unit (last axis), sort every non-NaN value with its ids."""

    acts = acts.double()
    positions_shape = acts.shape[1:-1]
    values, ids, positions = [], [], []
    for unit in range(acts.shape[-1]):
        column = acts[..., unit]
        entries = []
        for flat, value in enumerate(column.reshape(-1).tolist()):
            if math.isnan(value):
                continue
            example, rest = divmod(flat, max(math.prod(positions_shape), 1))
            position = []
            for size in reversed(positions_shape):
                rest, index = divmod(rest, size)
                position.append(index)
            position.reverse()
            entries.append((-value, example, tuple(position)))
        entries.sort()
        top = entries[:k]
        values.append([-entry[0] for entry in top])
        ids.append([entry[1] for entry in top])
        positions.append([list(entry[2]) for entry in top])
    return values, ids, positions


def _assert_matches_oracle(result: tl.stats.UnitTopK, acts: torch.Tensor, k: int) -> None:
    values, ids, positions = _oracle_topk(acts, k)
    assert result.values.shape == (acts.shape[-1], k)
    for unit in range(acts.shape[-1]):
        kept = len(values[unit])
        assert result.values[unit, :kept].tolist() == values[unit]
        assert result.example_ids[unit, :kept].tolist() == ids[unit]
        assert result.positions[unit, :kept].tolist() == positions[unit]
        assert torch.isnan(result.values[unit, kept:]).all()
        assert (result.example_ids[unit, kept:] == -1).all()


@pytest.mark.parametrize("batch_size", [2, 3])
def test_topk_neurons_match_oracle(batch_size: int) -> None:
    model = _MLP().eval()
    data = _inputs(N_EXAMPLES, 4)
    label = _relu_label(model, data[:2])
    results = tl.aggregate(
        model,
        _batches(data, batch_size),
        metrics={label: tl.stats.TopK(k=4, unit_axis=-1)},
    )
    with torch.no_grad():
        acts = model.hidden(data)
    result = results[label]
    assert isinstance(result, tl.stats.UnitTopK)
    assert result.positions.shape == (6, 4, 0)
    _assert_matches_oracle(result, acts, k=4)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_topk_latents_match_oracle(batch_size: int) -> None:
    model = _MLP().eval()
    sae = _SAE()
    data = _inputs(N_EXAMPLES, TOKENS, 4)
    label = _relu_label(model, data[:2])
    results = tl.aggregate(
        model,
        _batches(data, batch_size),
        metrics={label: tl.stats.TopK(k=6, unit_axis=-1, encoder=sae.encode)},
    )
    with torch.no_grad():
        latents = sae.encode(model.hidden(data))
    result = results[label]
    assert result.positions.shape == (8, 6, 1)
    # The dead latent ties at zero everywhere: the earliest examples win.
    assert result.example_ids[0].tolist() == [0, 0, 0, 0, 0, 1]
    assert result.positions[0, :, 0].tolist() == [0, 1, 2, 3, 4, 0]
    _assert_matches_oracle(result, latents, k=6)


@pytest.mark.parametrize("site", ["neurons", "latents"])
def test_quantile_per_unit_exact_under_reservoir(site: str) -> None:
    model = _MLP().eval()
    sae = _SAE()
    shape = (N_EXAMPLES, 4) if site == "neurons" else (N_EXAMPLES, TOKENS, 4)
    data = _inputs(*shape)
    label = _relu_label(model, data[:2])
    encoder = sae.encode if site == "latents" else None
    levels = (0.1, 0.5, 0.9)
    results = tl.aggregate(
        model,
        _batches(data, 3),
        metrics={label: tl.stats.Quantile(levels, unit_axis=-1, encoder=encoder)},
    )
    with torch.no_grad():
        acts = model.hidden(data)
        if encoder is not None:
            acts = encoder(acts)
    flat = acts.double().reshape(-1, acts.shape[-1])
    for q in levels:
        expected = torch.stack([torch.quantile(flat[:, unit], q) for unit in range(flat.shape[1])])
        assert torch.equal(results[label][q], expected)


def test_quantile_per_unit_rank_error_beyond_reservoir() -> None:
    generator = torch.Generator().manual_seed(1)
    stream = torch.stack(
        [
            torch.randn(6000, generator=generator),
            torch.rand(6000, generator=generator) * 10,
            torch.randn(6000, generator=generator).exp(),
        ],
        dim=-1,
    )
    reservoir_size = 256
    stat = tl.stats.Quantile((0.1, 0.5, 0.9), reservoir_size=reservoir_size, unit_axis=-1)
    for batch in _batches(stream, 700):
        stat.update(batch)
    estimates = stat.result()
    for q, values in estimates.items():
        bound = 4 * math.sqrt(q * (1 - q) / reservoir_size)
        for unit in range(3):
            rank = float((stream[:, unit].double() <= values[unit]).double().mean())
            assert abs(rank - q) <= bound, (q, unit, rank)
    exact = torch.quantile(stream.double(), 0.5, dim=0)
    assert not torch.equal(estimates[0.5], exact)


def test_per_unit_same_seed_identical_twice() -> None:
    stream = _inputs(900, 3, seed=9) + torch.linspace(0, 1, 900).unsqueeze(-1)

    def run(seed: int) -> tuple[dict, tl.stats.UnitTopK]:
        quantile = tl.stats.Quantile(reservoir_size=64, seed=seed, unit_axis=1)
        top = tl.stats.TopK(k=5, unit_axis=1)
        for batch in _batches(stream, 128):
            quantile.update(batch)
            top.update(batch)
        return quantile.result(), top.result()

    first, second, other = run(4), run(4), run(5)
    for q in first[0]:
        assert torch.equal(first[0][q], second[0][q])
    assert any(not torch.equal(first[0][q], other[0][q]) for q in first[0])
    for field in ("values", "example_ids", "positions"):
        assert torch.equal(getattr(first[1], field), getattr(second[1], field))


def test_topk_per_unit_nan_matches_pooled() -> None:
    data = _inputs(9, 4, seed=2)
    data[1, 0] = math.nan
    data[4:, 0] = math.nan
    data[:, 3] = math.nan
    per_unit = tl.stats.TopK(k=3, unit_axis=-1)
    pooled = [tl.stats.TopK(k=3) for _ in range(4)]
    for batch in _batches(data, 4):
        per_unit.update(batch)
        for unit, stat in enumerate(pooled):
            stat.update(batch[:, unit])
    result = per_unit.result()
    for unit, stat in enumerate(pooled):
        expected = stat.result()
        kept = result.values[unit][~result.values[unit].isnan()].tolist()
        assert kept == expected
    assert result.example_ids[3].tolist() == [-1, -1, -1]
    _assert_matches_oracle(result, data, k=3)


def test_topk_explicit_example_ids_tie_break() -> None:
    first = torch.tensor([[1.0, 5.0], [2.0, 5.0], [2.0, 4.0]])
    second = torch.tensor([[2.0, 5.0], [0.0, 6.0]])
    stat = tl.stats.TopK(k=3, unit_axis=-1)
    stat.update(first, example_ids=[10, 3, 7])
    stat.update(second, example_ids=torch.tensor([1, 20]))
    result = stat.result()
    # Unit 0: three 2.0 values from ids 3, 7 and 1; the lowest ids win.
    assert result.values[0].tolist() == [2.0, 2.0, 2.0]
    assert result.example_ids[0].tolist() == [1, 3, 7]
    assert result.values[1].tolist() == [6.0, 5.0, 5.0]
    assert result.example_ids[1].tolist() == [20, 1, 3]
    # Batch boundaries do not matter: one batch gives the same answer.
    whole = tl.stats.TopK(k=3, unit_axis=-1)
    whole.update(torch.cat([first, second]), example_ids=[10, 3, 7, 1, 20])
    assert torch.equal(whole.result().example_ids, result.example_ids)


def test_pooled_behaviour_unchanged_without_unit_axis() -> None:
    data = torch.tensor([[3.0, 1.0], [2.0, math.nan], [5.0, 4.0]])
    top = tl.stats.TopK(3)
    quantile = tl.stats.Quantile([0.5])
    top.update(data)
    quantile.update(torch.tensor([1.0, 2.0, 3.0]))
    assert top.result() == [5.0, 4.0, 3.0]
    assert quantile.result() == {0.5: 2.0}
    with pytest.raises(tl.errors.TorchLensError) as info:
        top.update(data, example_ids=[0, 1, 2])
    assert info.value.fields["code"] == "stats_example_ids_without_unit_axis"
    shifted = tl.stats.TopK(2, encoder=lambda v: v * 2)
    shifted.update(torch.tensor([1.0, 3.0, 2.0]))
    assert shifted.result() == [6.0, 4.0]


def test_state_size_warning_once_names_cheaper_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_per_unit, "STATE_WARN_BYTES", 1000)
    monkeypatch.setattr(_per_unit, "_WARNED_CODES", set())
    batch = _inputs(3, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.stats.Quantile(unit_axis=-1, reservoir_size=64).update(batch)
        tl.stats.Quantile(unit_axis=-1, reservoir_size=64).update(batch)
    coded = [w for w in caught if getattr(w.message, "fields", {}).get("code")]
    assert len(coded) == 1
    message = str(coded[0].message)
    assert coded[0].message.fields["code"] == "stats_per_unit_state_large"
    assert "4 units x 512 bytes" in message and "nothing is dropped" in message
    spelling = re.search(r"Remedy: (tl\.stats\.Quantile\([^)]*\))", message)
    narrow = re.search(r"encoder=(lambda v: v\.narrow\([^)]*\))", message)
    assert spelling is not None and narrow is not None
    monkeypatch.setattr(_per_unit, "_WARNED_CODES", set())
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        cheaper = eval(spelling.group(1), {"tl": tl})  # noqa: S307 -- our own message
        cheaper.update(batch)
        fewer = tl.stats.Quantile(unit_axis=-1, reservoir_size=64, encoder=eval(narrow.group(1)))  # noqa: S307
        fewer.update(batch)
    assert not [w for w in again if getattr(w.message, "fields", {}).get("code")]
    assert cheaper.result()[0.5].shape == (4,)
    assert fewer.result()[0.5].shape == (1,)


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("batch_axis", "stats_unit_axis_is_batch_axis"),
        ("out_of_range", "stats_unit_axis_out_of_range"),
        ("unit_count_changed", "stats_unit_count_changed"),
        ("example_ids_shape", "stats_example_ids_invalid"),
    ],
)
def test_per_unit_refusals(case: str, code: str) -> None:
    batch = torch.zeros(2, 3)
    with pytest.raises(tl.errors.TorchLensError) as info:
        if case == "batch_axis":
            tl.stats.TopK(unit_axis=0).update(batch)
        elif case == "out_of_range":
            tl.stats.Quantile(unit_axis=5).update(batch)
        elif case == "unit_count_changed":
            stat = tl.stats.TopK(unit_axis=-1)
            stat.update(batch)
            stat.update(torch.zeros(2, 4))
        else:
            tl.stats.TopK(unit_axis=-1).update(batch, example_ids=[1, 2, 3])
    assert info.value.fields["code"] == code
    assert "Remedy" in str(info.value) or "remedy" in str(info.value).lower()
