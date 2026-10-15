"""Save and reload of ``tl.stats`` accumulators (``save_fitted`` / ``load_fitted``).

Each accumulator folds the first half of a dataset, is saved, reloaded into
a fresh object and folds the rest; its state (random streams included) and
its results must equal one uninterrupted run exactly. Seven examples give a
ragged last batch at both batch sizes.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import torchlens as tl
from torchlens.stats import _state_io

N_EXAMPLES = 7
TOKENS = 5
HIDDEN = 6


class _SAE(nn.Module):
    """A tiny SAE encoder: integer weights, so latents are exact."""

    def __init__(self, hidden: int = HIDDEN, latents: int = 8) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        self.w_enc = nn.Parameter(
            torch.randint(-1, 2, (hidden, latents), generator=generator).float()
        )
        self.b_enc = nn.Parameter(torch.randint(-1, 1, (latents,), generator=generator).float())

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h @ self.w_enc + self.b_enc)


SAE = _SAE()


def _acts(seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, (N_EXAMPLES, TOKENS, HIDDEN), generator=generator).float()


def _batches(data: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [data[start : start + batch_size] for start in range(0, len(data), batch_size)]


def _accumulators() -> dict[str, Callable[[], Any]]:
    """Every accumulator ``tl.aggregate`` folds, pooled and per unit."""

    return {
        "mean": tl.stats.Mean,
        "norm": lambda: tl.stats.Norm(p=1.0),
        "covariance": tl.stats.Covariance,
        "pca": lambda: tl.stats.PCA(3),
        "quantile_pooled": lambda: tl.stats.Quantile(reservoir_size=20, seed=4),
        "topk_pooled": lambda: tl.stats.TopK(5),
        "quantile_units": lambda: tl.stats.Quantile(reservoir_size=4, seed=2, unit_axis=-1),
        "topk_units": lambda: tl.stats.TopK(3, unit_axis=-1),
        "quantile_latents": lambda: tl.stats.Quantile(
            reservoir_size=4, unit_axis=-1, encoder=SAE.encode
        ),
        "topk_latents": lambda: tl.stats.TopK(3, unit_axis=-1, encoder=SAE.encode),
        "histogram": tl.stats.Histogram,
        "spine": tl.stats.Spine,
    }


def _state(stat: Any) -> str:
    """Return the exact state digest of an accumulator (random streams included)."""

    state_json, tensors, _paths = _state_io.encode_state(stat)
    return _state_io.state_digest(state_json, tensors)


def _assert_same(left: Any, right: Any) -> None:
    """Assert two results are identical, NaN equal to NaN."""

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif dataclasses.is_dataclass(left) and not isinstance(left, type):
        assert type(left) is type(right)
        for field in dataclasses.fields(left):
            _assert_same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float) and math.isnan(right)
    else:
        assert left == right


def _encoder_of(stat: Any) -> Any:
    return getattr(stat, "encoder", None)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_resume_matches_uninterrupted(batch_size: int, tmp_path: Path) -> None:
    batches = _batches(_acts(), batch_size)
    half = len(batches) // 2
    for name, make in _accumulators().items():
        whole = make()
        for batch in batches:
            whole.update(batch)
        first = make()
        for batch in batches[:half]:
            first.update(batch)
        path = tl.stats.save_fitted(first, tmp_path / f"{name}.safetensors")
        resumed = tl.stats.load_fitted(path, encoder=_encoder_of(first))
        assert type(resumed) is type(first), name
        assert _state(resumed) == _state(first), name
        for batch in batches[half:]:
            resumed.update(batch)
        assert _state(resumed) == _state(whole), name
        _assert_same(resumed.result(), whole.result())


def test_resume_two_input_accumulators(tmp_path: Path) -> None:
    a_batches = _batches(_acts(seed=3).reshape(N_EXAMPLES, -1), 3)
    b_batches = _batches(_acts(seed=8).reshape(N_EXAMPLES, -1)[:, :7], 3)
    for make in (tl.stats.CrossCovariance, tl.stats.CKA):
        whole = make()
        for a, b in zip(a_batches, b_batches, strict=True):
            whole.update(a, b)
        first = make()
        first.update(a_batches[0], b_batches[0])
        resumed = tl.stats.load_fitted(tl.stats.save_fitted(first, tmp_path / "two.safetensors"))
        for a, b in zip(a_batches[1:], b_batches[1:], strict=True):
            resumed.update(a, b)
        assert _state(resumed) == _state(whole)
        _assert_same(resumed.result(), whole.result())


def test_resume_aggregator(tmp_path: Path) -> None:
    batches = _batches(_acts(), 2)

    def make() -> tl.stats.Aggregator:
        return tl.stats.Aggregator(
            tl.stats.Mean(name="mean"),
            tl.stats.TopK(2, unit_axis=-1, name="top"),
            name="site",
        )

    whole = make()
    for batch in batches:
        whole.update(batch)
    first = make()
    for batch in batches[:2]:
        first.update(batch)
    resumed = tl.stats.load_fitted(tl.stats.save_fitted(first, tmp_path / "agg.safetensors"))
    assert isinstance(resumed, tl.stats.Aggregator)
    for batch in batches[2:]:
        resumed.update(batch)
    assert _state(resumed) == _state(whole)
    _assert_same(resumed.result(), whole.result())


@pytest.mark.parametrize(
    ("case", "code"),
    [("unit_count", "stats_unit_count_changed"), ("ndim", "stats_unit_layout_changed")],
)
def test_reload_keeps_unit_layout(case: str, code: str, tmp_path: Path) -> None:
    stat = tl.stats.TopK(3, unit_axis=-1)
    stat.update(_acts()[:2])
    resumed = tl.stats.load_fitted(tl.stats.save_fitted(stat, tmp_path / "top.safetensors"))
    batch = torch.zeros(2, TOKENS, HIDDEN + 1) if case == "unit_count" else torch.zeros(2, HIDDEN)
    with pytest.raises(tl.errors.TorchLensError) as info:
        resumed.update(batch)
    assert info.value.fields["code"] == code


def _rewrite(path: Path, *, metadata: dict[str, str] | None = None, bump: bool = False) -> None:
    """Rewrite a state file's metadata, or nudge one tensor, keeping the rest."""

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        old_metadata = dict(handle.metadata() or {})
        names = list(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in names}
    if bump:
        key = sorted(tensors)[0]
        tensors[key] = tensors[key] + 1
    save_file(tensors, str(path), metadata={**old_metadata, **(metadata or {})})


def test_unknown_format_version_refuses(tmp_path: Path) -> None:
    path = tl.stats.save_fitted(tl.stats.Mean(), tmp_path / "mean.safetensors")
    _rewrite(path, metadata={"schema": "tl_stats_state_v99"})
    with pytest.raises(tl.stats.FittedArtifactError) as info:
        tl.stats.load_fitted(path)
    assert info.value.fields["code"] == "stats_state_version_unknown"
    assert "tl.stats.save_fitted(stat, path)" in str(info.value)


def test_digest_mismatch_refuses(tmp_path: Path) -> None:
    stat = tl.stats.TopK(3, unit_axis=-1)
    stat.update(_acts()[:3])
    path = tl.stats.save_fitted(stat, tmp_path / "top.safetensors")
    _rewrite(path, bump=True)
    with pytest.raises(tl.stats.FittedArtifactError) as info:
        tl.stats.load_fitted(path)
    assert info.value.fields["code"] == "stats_state_digest_mismatch"


def test_malformed_state_refuses(tmp_path: Path) -> None:
    path = tl.stats.save_fitted(tl.stats.Mean(), tmp_path / "mean.safetensors")
    forged = '{"attrs":[],"cls":"NotAnAccumulator","encoders":[],"t":"obj"}'
    _rewrite(path, metadata={"state": forged, "digest": _state_io.state_digest(forged, {})})
    with pytest.raises(tl.stats.FittedArtifactError) as info:
        tl.stats.load_fitted(path)
    assert info.value.fields["code"] == "stats_state_invalid"


@pytest.mark.parametrize("case", ["missing", "unexpected"])
def test_encoder_must_match_saved(case: str, tmp_path: Path) -> None:
    encoder = SAE.encode if case == "missing" else None
    stat = tl.stats.TopK(3, unit_axis=-1, encoder=encoder)
    stat.update(_acts()[:2])
    path = tl.stats.save_fitted(stat, tmp_path / "top.safetensors")
    with pytest.raises(tl.stats.FittedArtifactError) as info:
        tl.stats.load_fitted(path, encoder=None if case == "missing" else SAE.encode)
    assert info.value.fields["code"] == "stats_state_encoder_mismatch"
    remedy = info.value.fields["remedy"]
    spelling = "encoder=sae.encode" if case == "missing" else "tl.stats.load_fitted(path)"
    assert spelling in remedy


def test_save_refuses_unsupported_state(tmp_path: Path) -> None:
    stat = tl.stats.Mean()
    stat.extra = object()  # type: ignore[attr-defined]
    with pytest.raises(tl.stats.FittedArtifactError) as info:
        tl.stats.save_fitted(stat, tmp_path / "mean.safetensors")
    assert info.value.fields["code"] == "stats_state_unsupported"
    assert not list(tmp_path.iterdir())
    with pytest.raises(tl.stats.FittedArtifactError) as other:
        tl.stats.save_fitted(object(), tmp_path / "x.safetensors")
    assert other.value.fields["code"] == "pca_fitted_record_invalid"
