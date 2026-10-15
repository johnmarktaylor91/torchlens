"""``tl.aggregate(..., cache=True)``: a rerun of the same run skips the loop.

Forward passes are counted through a module-level counter (an attribute on
the model would itself change the model's key). Every cached result is
compared with an uncached run of the same setup.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.stats import _state_io

N_EXAMPLES = 7
FORWARDS = {"count": 0}


class _MLP(nn.Module):
    """Linear -> ReLU -> Linear with integer weights; counts its forward calls."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.fc1 = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                layer.weight.copy_(torch.randint(-2, 3, layer.weight.shape, generator=generator))
                layer.bias.copy_(torch.randint(-1, 2, layer.bias.shape, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        FORWARDS["count"] += 1
        return self.fc2(self.act(self.fc1(x)))


class _SAE(nn.Module):
    """A tiny SAE encoder over the ReLU site."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        self.w_enc = nn.Parameter(torch.randint(-1, 2, (6, 8), generator=generator).float())

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h @ self.w_enc)


def _data(seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, (N_EXAMPLES, 4), generator=generator).float()


def _batches(data: torch.Tensor, batch_size: int = 3) -> list[torch.Tensor]:
    return [data[start : start + batch_size] for start in range(0, len(data), batch_size)]


def _relu_label(model: nn.Module) -> str:
    log = tl.trace(model, _data()[:2])
    try:
        return next(str(op.layer_label) for op in log.layer_list if op.layer_type == "relu")
    finally:
        log.cleanup()


def _metrics(label: str, k: int = 3) -> dict[str, Any]:
    return {
        label: tl.stats.Aggregator(
            tl.stats.TopK(k, unit_axis=-1, name="top"),
            tl.stats.Quantile(reservoir_size=4, unit_axis=-1, name="quantiles"),
            name="relu",
        ),
        "output": tl.stats.Mean(),
    }


def _state(metrics: dict[str, Any]) -> str:
    state_json, tensors, _paths = _state_io.encode_state(metrics)
    return _state_io.state_digest(state_json, tensors)


def _run(
    model: nn.Module, data: Any, metrics: dict[str, Any], cache_dir: Path | None, **kwargs: Any
) -> tuple[dict[str, Any], int]:
    """Run ``tl.aggregate`` and return its results and the forward passes it made."""

    before = FORWARDS["count"]
    results = tl.aggregate(
        model, data, metrics, cache=cache_dir is not None, cache_dir=cache_dir, **kwargs
    )
    return results, FORWARDS["count"] - before


def _assert_equal_results(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert list(left) == list(right)
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict):
            _assert_equal_results(a, b)
        elif isinstance(a, tl.stats.UnitTopK):
            for field in ("values", "example_ids", "positions"):
                torch.testing.assert_close(
                    getattr(a, field), getattr(b, field), rtol=0, atol=0, equal_nan=True
                )
        elif isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        else:
            assert a == b


@pytest.fixture
def model() -> nn.Module:
    return _MLP().eval()


def test_cache_hit_skips_forward_and_equals(model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    batches = _batches(_data())
    uncached, _ = _run(model, batches, _metrics(label), None)
    first_metrics = _metrics(label)
    first, first_forwards = _run(model, batches, first_metrics, tmp_path)
    assert first_forwards > 0
    _assert_equal_results(first, uncached)
    assert len(list((tmp_path / "aggregate").glob("*.safetensors"))) == 1
    second_metrics = _metrics(label)
    second, second_forwards = _run(model, batches, second_metrics, tmp_path)
    assert second_forwards == 0
    _assert_equal_results(second, first)
    assert _state(second_metrics) == _state(first_metrics)


@pytest.mark.parametrize("change", ["parameter", "dataset", "accumulator_config"])
def test_cache_misses_on_change(change: str, model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    data = _data()
    _run(model, _batches(data), _metrics(label), tmp_path)
    k = 3
    if change == "parameter":
        with torch.no_grad():
            model.fc1.bias += 1.0
    elif change == "dataset":
        data = data.clone()
        data[5, 2] += 1.0
    else:
        k = 2
    expected, _ = _run(model, _batches(data), _metrics(label, k=k), None)
    changed, forwards = _run(model, _batches(data), _metrics(label, k=k), tmp_path)
    assert forwards > 0
    _assert_equal_results(changed, expected)
    assert len(list((tmp_path / "aggregate").glob("*.safetensors"))) == 2


class _Opaque:
    """A batch leaf whose contents the key cannot read."""


class _CallableObject:
    """An encoder object whose state the key cannot read."""

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


@pytest.mark.parametrize(
    "case",
    [
        "one_shot_iterator",
        "opaque_batch_leaf",
        "unseeded_quantile",
        "training_mode",
        "opaque_encoder",
    ],
)
def test_cache_unkeyable_refuses(case: str, model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    data: Any = _batches(_data())
    metrics = _metrics(label)
    if case == "one_shot_iterator":
        data = iter(data)
    elif case == "opaque_batch_leaf":
        data = [(batch, _Opaque()) for batch in data]
    elif case == "unseeded_quantile":
        metrics = {label: tl.stats.Quantile(seed=None)}
    elif case == "training_mode":
        model.train()
    else:
        metrics = {label: tl.stats.TopK(3, unit_axis=-1, encoder=_CallableObject())}
    before = FORWARDS["count"]
    with pytest.raises(tl.errors.TorchLensError) as info:
        tl.aggregate(model, data, metrics, cache=True, cache_dir=tmp_path)
    assert info.value.fields["code"] == "stats_aggregate_cache_unkeyable"
    assert "drop cache=" in info.value.fields["remedy"]
    assert FORWARDS["count"] == before


def test_cache_grad_target_hits(model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    data = _data()
    targets = torch.arange(N_EXAMPLES) % 3
    batches = [(x, y) for x, y in zip(_batches(data), _batches(targets), strict=True)]

    def loss_fn(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(output, labels)

    def metrics() -> dict[str, Any]:
        return {label: tl.stats.Mean()}

    expected, _ = _run(model, batches, metrics(), None, target="grad", loss_fn=loss_fn)
    first, first_forwards = _run(
        model, batches, metrics(), tmp_path, target="grad", loss_fn=loss_fn
    )
    second, second_forwards = _run(
        model, batches, metrics(), tmp_path, target="grad", loss_fn=loss_fn
    )
    assert first_forwards > 0
    assert second_forwards == 0
    _assert_equal_results(first, expected)
    _assert_equal_results(second, expected)


def test_cache_corrupt_entry_recomputes(model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    batches = _batches(_data())
    expected, _ = _run(model, batches, _metrics(label), tmp_path)
    (entry,) = (tmp_path / "aggregate").glob("*.safetensors")
    entry.write_bytes(b"not a safetensors file")
    recomputed, forwards = _run(model, batches, _metrics(label), tmp_path)
    assert forwards > 0
    _assert_equal_results(recomputed, expected)
    again, again_forwards = _run(model, batches, _metrics(label), tmp_path)
    assert again_forwards == 0
    _assert_equal_results(again, expected)


def test_cache_encoder_weights_change_misses(model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    batches = _batches(_data())
    sae = _SAE()

    def metrics() -> dict[str, Any]:
        return {label: tl.stats.TopK(3, unit_axis=-1, encoder=sae.encode)}

    _run(model, batches, metrics(), tmp_path)
    _hit, hit_forwards = _run(model, batches, metrics(), tmp_path)
    assert hit_forwards == 0
    with torch.no_grad():
        sae.w_enc += 1.0
    expected, _ = _run(model, batches, metrics(), None)
    changed, forwards = _run(model, batches, metrics(), tmp_path)
    assert forwards > 0
    _assert_equal_results(changed, expected)


class _ShiftingLoader:
    """A re-iterable loader that yields different batches on every pass."""

    def __init__(self, batches: list[torch.Tensor]) -> None:
        self.batches = batches
        self.passes = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        self.passes += 1
        offset = float(self.passes)
        return iter([batch + offset for batch in self.batches])


def test_cache_unstable_loader_warns_and_stores_nothing(model: nn.Module, tmp_path: Path) -> None:
    label = _relu_label(model)
    loader = _ShiftingLoader(_batches(_data()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _results, forwards = _run(model, loader, _metrics(label), tmp_path)
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "stats_aggregate_cache_data_unstable" in codes
    assert forwards > 0
    assert not list((tmp_path / "aggregate").glob("*.safetensors"))


@pytest.mark.parametrize(
    "call",
    [
        lambda m, d, s, p: tl.aggregate(m, d, s, cache="yes"),
        lambda m, d, s, p: tl.aggregate(m, d, s, cache=True, cache_dir=3),
    ],
    ids=["cache_not_bool", "cache_dir_not_path"],
)
def test_cache_argument_type_refuses(
    call: Callable[..., Any], model: nn.Module, tmp_path: Path
) -> None:
    with pytest.raises(tl.errors.TorchLensError) as info:
        call(model, _batches(_data()), {"output": tl.stats.Mean()}, tmp_path)
    assert info.value.fields["code"] == "stats_aggregate_cache_type"
    assert "cache=True" in info.value.fields["remedy"]
