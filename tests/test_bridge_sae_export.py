"""SAE latent export for SAEDashboard and delphi: what is written equals a hand top-k.

A tiny next-token model and a hand-built SAE give latents two ways: one batch
through ``splice`` (``SpliceResult.latents``) and a whole dataset through
``tl.aggregate`` with ``tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)``.
Each writer's files are read back with ``json`` and ``safetensors`` (no
consumer package) and compared with the hand computation: every latent's top
examples ranked by value, then lowest example, then lowest position, and
every nonzero activation for the delphi cache. The consumers' own loaders run
in ``tests/test_bridge_real_sae_export.py``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
from safetensors import numpy as safetensors_numpy
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.bridge import _sae_export, sae as sae_bridge
from torchlens.errors import TorchLensWarning


class _TinyLM(nn.Module):
    """Embedding, one ReLU MLP site, unembedding."""

    def __init__(self, vocab: int = 13, width: int = 6) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.hidden = nn.Linear(width, width)
        self.unembed = nn.Linear(width, vocab)

    def site(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.hidden(self.embed(tokens)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.site(tokens))


class _TinySAE(nn.Module):
    """Minimal encode/decode pair standing in for a real SAE."""

    def __init__(self, d_in: int = 6, d_hidden: int = 10) -> None:
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.dec(latents)


@pytest.fixture(autouse=True)
def _fresh_warning_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees the once-per-session warnings as not yet fired."""

    monkeypatch.setattr(sae_bridge, "_WARNED_CODES", set())


@pytest.fixture()
def setup() -> tuple[_TinyLM, _TinySAE, torch.Tensor]:
    torch.manual_seed(5)
    model, sae = _TinyLM().eval(), _TinySAE().eval()
    tokens = torch.randint(0, 13, (7, 5), generator=torch.Generator().manual_seed(9))
    return model, sae, tokens


def _hand_latents(model: _TinyLM, sae: _TinySAE, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return sae.encode(model.site(tokens)).double()


#: The dataset the top-k tests stream through ``tl.aggregate``, as token-row slices.
_DATASET_BATCHES = (slice(0, 3), slice(3, 5), slice(5, 7))


def _hand_dataset_latents(model: _TinyLM, sae: _TinySAE, tokens: torch.Tensor) -> torch.Tensor:
    """Latents computed batch by batch, as the streamed top-k sees them.

    The same token gives the same latent within one forward, but not always across
    forwards of different batch sizes: on some BLAS instruction paths the last bits
    depend on the matrix shape, which reorders near-ties. Ranking the values the
    statistic actually saw keeps the exact comparison and its tie-break check.
    """

    return torch.cat([_hand_latents(model, sae, tokens[rows]) for rows in _DATASET_BATCHES])


def _hand_topk(latents: torch.Tensor, unit: int, k: int) -> list[tuple[int, int, float]]:
    """Rank every (example, position) of one latent: value down, example up, position up."""

    entries = [
        (example, position, float(latents[example, position, unit]))
        for example in range(latents.shape[0])
        for position in range(latents.shape[1])
    ]
    return sorted(entries, key=lambda item: (-item[2], item[0], item[1]))[:k]


def _relu_label(model: _TinyLM, tokens: torch.Tensor) -> str:
    trace = tl.trace(model, tokens[:1])
    return next(op.layer_label for op in trace.layer_list if op.layer_type == "relu")


def _dataset_topk(model, sae, tokens, k):
    label = _relu_label(model, tokens)
    batches = [tokens[rows] for rows in _DATASET_BATCHES]
    stat = tl.stats.TopK(k=k, unit_axis=-1, encoder=sae.encode, name="latents")
    return tl.aggregate(model, batches, metrics={label: tl.stats.Aggregator(stat)})[label][
        "latents"
    ]


def test_dashboard_top_examples_over_a_dataset_equal_a_hand_topk(setup, tmp_path: Path) -> None:
    """A dataset top-k: per latent, the ranked examples and the ranked token's value."""

    model, sae, tokens = setup
    top = _dataset_topk(model, sae, tokens, k=3)
    path = sae_bridge.write_sae_dashboard(top, tokens, path=tmp_path / "dash.json")
    data = json.loads(path.read_text())
    hand = _hand_dataset_latents(model, sae, tokens)
    fired = [u for u in range(hand.shape[-1]) if bool((hand[..., u].abs() > 1e-6).any())]
    assert sorted(int(key) for key in data["feature_data_dict"]) == fired
    for unit in fired:
        group = data["feature_data_dict"][str(unit)]["sequence_data"]["seq_group_data"][0]
        expected = _hand_topk(hand, unit, 3)
        got = [(s["original_index"], s["qualifying_token_index"]) for s in group["seq_data"]]
        assert got == [(example, position) for example, position, _ in expected]
        for record, (example, position, value) in zip(group["seq_data"], expected, strict=True):
            assert record["token_ids"] == tokens[example].tolist()
            acts = [0.0] * tokens.shape[1]
            acts[position] = round(value, 4)
            assert record["feat_acts"] == pytest.approx(acts, abs=1e-12)
    assert data["feature_stats"]["max"] == []


def test_dashboard_from_one_spliced_batch_carries_every_token(setup, tmp_path: Path) -> None:
    """``SpliceResult.latents``: each sequence holds the latent at every token."""

    model, sae, tokens = setup
    batch = tokens[:4]
    result = sae_bridge.splice(model, batch, site=tl.func("relu"), sae=sae)
    hand = _hand_latents(model, sae, batch)
    assert torch.allclose(result.latents.double(), hand, atol=1e-6)
    path = sae_bridge.write_sae_dashboard(
        result.latents, batch, latent_ids=[0, 3], k=2, path=tmp_path / "one.json"
    )
    data = json.loads(path.read_text())
    assert sorted(data["feature_data_dict"]) == ["0", "3"]
    latents = result.latents.double()
    for index, unit in enumerate((0, 3)):
        group = data["feature_data_dict"][str(unit)]["sequence_data"]["seq_group_data"][0]
        expected = _hand_topk(latents, unit, 2)
        assert [(s["original_index"], s["qualifying_token_index"]) for s in group["seq_data"]] == [
            (example, position) for example, position, _ in expected
        ]
        for record, (example, _, _) in zip(group["seq_data"], expected, strict=True):
            assert record["feat_acts"] == [round(v, 4) for v in latents[example, :, unit].tolist()]
        column = latents[..., unit].reshape(-1)
        assert data["feature_stats"]["max"][index] == pytest.approx(float(column.max()))
        assert data["feature_stats"]["frac_nonzero"][index] == pytest.approx(
            float((column.abs() > 1e-6).double().mean())
        )


def _read_delphi(folder: Path) -> tuple[set[tuple[int, int, int]], dict, list[Path]]:
    entries: dict[tuple[int, int, int], float] = {}
    files = sorted(folder.glob("*.safetensors"), key=lambda p: int(p.stem.split("_")[0]))
    for file in files:
        start = int(file.stem.split("_")[0])
        split = safetensors_numpy.load_file(str(file))
        for (example, position, latent), value in zip(
            split["locations"].astype("int64").tolist(), split["activations"].tolist(), strict=True
        ):
            entries[(example, position, latent + start)] = value
    return set(entries), entries, files


def test_delphi_cache_holds_every_nonzero_activation(setup, tmp_path: Path) -> None:
    """Dense latents: every activation above delphi's 1e-5, in delphi's split layout."""

    model, sae, tokens = setup
    hand = _hand_latents(model, sae, tokens)
    folder = sae_bridge.write_delphi(
        hand, tokens, site="hidden", model_name="tiny", n_splits=3, directory=tmp_path
    )
    assert folder == tmp_path / "hidden"
    keys, values, files = _read_delphi(folder)
    bounds = torch.linspace(0, 10, steps=4).long().tolist()
    assert [f.name for f in files] == [
        f"{start}_{stop - 1}.safetensors"
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    expected = {tuple(i) for i in torch.nonzero(hand.abs() > 1e-5).tolist()}
    assert keys == expected
    for example, position, latent in expected:
        want = float(hand[example, position, latent].half())
        assert values[(example, position, latent)] == want
    split = safetensors_numpy.load_file(str(files[0]))
    assert split["locations"].dtype.name == "uint16"
    assert split["activations"].dtype.name == "float16"
    assert split["tokens"].tolist() == tokens.tolist()
    config = json.loads((folder / "config.json").read_text())
    assert config["model_name"] == "tiny" and config["n_splits"] == 3
    assert config["cache_ctx_len"] == 5 and config["n_tokens"] == 35


def test_delphi_cache_from_a_dataset_topk_holds_the_retained_entries(setup, tmp_path: Path) -> None:
    """A dataset top-k: exactly the retained nonzero top entries, at their example rows."""

    model, sae, tokens = setup
    top = _dataset_topk(model, sae, tokens, k=2)
    folder = sae_bridge.write_delphi(
        top, tokens, site="hidden", model_name="tiny", directory=tmp_path
    )
    keys, _, _ = _read_delphi(folder)
    hand = _hand_dataset_latents(model, sae, tokens)
    expected = {
        (example, position, unit)
        for unit in range(hand.shape[-1])
        for example, position, value in _hand_topk(hand, unit, 2)
        if abs(value) > 1e-5
    }
    assert keys == expected


def test_export_refusals_name_their_code(setup, tmp_path: Path) -> None:
    """Wrong tokens, wrong latents, an out-of-range latent id: typed refusals."""

    model, sae, tokens = setup
    hand = _hand_latents(model, sae, tokens)
    cases = [
        (
            lambda: sae_bridge.write_delphi(
                hand, tokens.float(), site="s", model_name="m", directory=tmp_path
            ),
            "sae_export_tokens_invalid",
        ),
        (
            lambda: sae_bridge.write_delphi(
                hand, tokens[:3], site="s", model_name="m", directory=tmp_path
            ),
            "sae_export_tokens_invalid",
        ),
        (
            lambda: sae_bridge.write_sae_dashboard(hand[None], tokens, path=tmp_path / "a.json"),
            "sae_export_latents_invalid",
        ),
        (
            lambda: sae_bridge.write_sae_dashboard(
                hand, tokens, latent_ids=[99], path=tmp_path / "b.json"
            ),
            "sae_export_latents_invalid",
        ),
        (
            lambda: sae_bridge.write_sae_dashboard(
                _dataset_topk(model, sae, tokens, 2), tokens[:2], path=tmp_path / "c.json"
            ),
            "sae_export_tokens_invalid",
        ),
    ]
    for call, code in cases:
        with pytest.raises(InvalidArgumentError) as caught:
            call()
        assert caught.value.fields["code"] == code


def test_large_dashboard_warns_once_and_names_a_spelling_that_runs(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the size guard: one coded warning; its remedy (``latent_ids=``) runs quietly."""

    model, sae, tokens = setup
    hand = _hand_latents(model, sae, tokens)
    monkeypatch.setattr(_sae_export, "_DASHBOARD_WARN_BYTES", 2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sae_bridge.write_sae_dashboard(hand, tokens, path=tmp_path / "big.json")
        sae_bridge.write_sae_dashboard(hand, tokens, path=tmp_path / "big2.json")
    coded = [w for w in caught if isinstance(w.message, TorchLensWarning)]
    assert [w.message.fields["code"] for w in coded] == ["sae_export_dashboard_large"]
    assert "latent_ids=[0, 1, 2]" in str(coded[0].message)
    monkeypatch.setattr(sae_bridge, "_WARNED_CODES", set())
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        sae_bridge.write_sae_dashboard(
            hand, tokens, latent_ids=[0, 1, 2], k=2, path=tmp_path / "small.json"
        )
    assert not [w for w in quiet if isinstance(w.message, TorchLensWarning)]
