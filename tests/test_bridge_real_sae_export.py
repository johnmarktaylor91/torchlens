"""SAE latent export read back by the consumers' own loaders (sae-dashboard, eai-delphi).

The files ``tl.bridge.sae.write_sae_dashboard`` and ``write_delphi`` write are
loaded with SAEDashboard's ``SaeVisData.load_json`` and rendered with its
``save_feature_centric_vis`` (sae-dashboard 0.8.0), and with delphi's
``TensorBuffer`` and ``LatentDataset`` (eai-delphi 0.1.3), and what the
consumers read equals a hand top-k over the same latents. The tokenizer is the
vendored GPT-2 one, so nothing downloads. Both packages are test-only pins
installed by the peer-recipe leg; elsewhere the module skips.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import sae as sae_bridge

sae_vis_data = pytest.importorskip("sae_dashboard.sae_vis_data")
delphi_loader = pytest.importorskip("delphi.latents.loader")

_VENDORED = Path(__file__).parent / "real_model" / "vendored" / "gpt2"


def _tokenizer():
    from transformers import GPT2TokenizerFast

    vocab = json.loads(gzip.decompress((_VENDORED / "vocab.json.gz").read_bytes()))
    merges_text = (_VENDORED / "merges.txt").read_text(encoding="utf-8")
    merges = [
        tuple(line.split(" ", 1))
        for line in merges_text.splitlines()
        if line and not line.startswith("#")
    ]
    return GPT2TokenizerFast(vocab=vocab, merges=merges)


class _TinyLM(nn.Module):
    """Embedding, one ReLU MLP site, unembedding."""

    def __init__(self, vocab: int = 50, width: int = 6) -> None:
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


@pytest.fixture()
def setup():
    torch.manual_seed(5)
    model, sae = _TinyLM().eval(), _TinySAE().eval()
    tokens = torch.randint(0, 50, (8, 8), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        hand = sae.encode(model.site(tokens)).double()
    return model, sae, tokens, hand


def _hand_topk(latents: torch.Tensor, unit: int, k: int) -> list[tuple[int, int, float]]:
    entries = [
        (example, position, float(latents[example, position, unit]))
        for example in range(latents.shape[0])
        for position in range(latents.shape[1])
    ]
    return sorted(entries, key=lambda item: (-item[2], item[0], item[1]))[:k]


def _dataset_topk(model, sae, tokens, k):
    trace = tl.trace(model, tokens[:1])
    label = next(op.layer_label for op in trace.layer_list if op.layer_type == "relu")
    stat = tl.stats.TopK(k=k, unit_axis=-1, encoder=sae.encode, name="latents")
    batches = [tokens[0:3], tokens[3:6], tokens[6:8]]
    return tl.stats.aggregate(model, batches, metrics={label: tl.stats.Aggregator(stat)})[label][
        "latents"
    ]


class _Model:
    """What ``save_feature_centric_vis`` reads from the model: its tokenizer."""

    def __init__(self) -> None:
        self.tokenizer = _tokenizer()


def test_sae_dashboard_loads_and_renders_the_dense_export(setup, tmp_path: Path) -> None:
    """SAEDashboard's loader reads the ranked sequences; its renderer writes the page."""

    from sae_dashboard.components_config import SequencesConfig
    from sae_dashboard.data_writing_fns import save_feature_centric_vis
    from sae_dashboard.layout import Column, SaeVisLayoutConfig

    _, _, tokens, hand = setup
    path = sae_bridge.write_sae_dashboard(hand, tokens, k=4, path=tmp_path / "dash.json")
    layout = SaeVisLayoutConfig(
        columns=[Column(SequencesConfig(buffer=None, stack_mode="stack-none"))]
    )
    fired = [u for u in range(10) if bool((hand[..., u].abs() > 1e-6).any())]
    cfg = sae_vis_data.SaeVisConfig(
        hook_point="hidden", features=fired, feature_centric_layout=layout
    )
    loaded = sae_vis_data.SaeVisData.load_json(path, cfg=cfg, model=_Model(), encoder=None)
    assert sorted(loaded.feature_data_dict) == fired
    for unit in fired:
        group = loaded.feature_data_dict[unit].sequence_data.seq_group_data[0]
        expected = _hand_topk(hand, unit, 4)
        assert [(s.original_index, s.qualifying_token_index) for s in group.seq_data] == [
            (example, position) for example, position, _ in expected
        ]
        for record, (example, position, value) in zip(group.seq_data, expected, strict=True):
            assert record.token_ids == tokens[example].tolist()
            assert record.feat_acts[position] == pytest.approx(value, abs=5e-5)
    html = tmp_path / "dash.html"
    save_feature_centric_vis(loaded, html)
    assert html.stat().st_size > 0 and "TOP ACTIVATIONS" in html.read_text()


def test_sae_dashboard_loads_the_dataset_topk_export(setup, tmp_path: Path) -> None:
    """The dataset top-k path through SAEDashboard's loader and renderer."""

    from sae_dashboard.components_config import SequencesConfig
    from sae_dashboard.data_writing_fns import save_feature_centric_vis
    from sae_dashboard.layout import Column, SaeVisLayoutConfig

    model, sae, tokens, hand = setup
    top = _dataset_topk(model, sae, tokens, k=3)
    path = sae_bridge.write_sae_dashboard(top, tokens, path=tmp_path / "top.json")
    layout = SaeVisLayoutConfig(
        columns=[Column(SequencesConfig(buffer=None, stack_mode="stack-none"))]
    )
    cfg = sae_vis_data.SaeVisConfig(hook_point="hidden", features=[], feature_centric_layout=layout)
    loaded = sae_vis_data.SaeVisData.load_json(path, cfg=cfg, model=_Model(), encoder=None)
    for unit, feature in loaded.feature_data_dict.items():
        seqs = feature.sequence_data.seq_group_data[0].seq_data
        assert [(s.original_index, s.qualifying_token_index) for s in seqs] == [
            (example, position) for example, position, _ in _hand_topk(hand, unit, 3)
        ]
    save_feature_centric_vis(loaded, tmp_path / "top.html")
    assert (tmp_path / "top.html").stat().st_size > 0


@pytest.mark.parametrize("source", ["dense", "dataset_topk"])
def test_delphi_reads_the_cache_and_builds_latent_records(setup, tmp_path: Path, source) -> None:
    """delphi's buffers hold the exported activations; its dataset builds records from them."""

    from delphi.config import ConstructorConfig, SamplerConfig

    model, sae, tokens, hand = setup
    latents = hand if source == "dense" else _dataset_topk(model, sae, tokens, k=6)
    sae_bridge.write_delphi(
        latents, tokens, site="hidden", model_name="gpt2", n_splits=2, directory=tmp_path
    )
    dataset = delphi_loader.LatentDataset(
        tmp_path,
        SamplerConfig(n_examples_train=2, n_examples_test=1, n_quantiles=1, train_type="top"),
        # n_non_activating=0: on a dataset this small a latent can have exactly one
        # non-activating window, which delphi 0.1.3 indexes as a 0-d tensor and fails on
        # (delphi/latents/constructors.py, random_non_activating_windows).
        ConstructorConfig(min_examples=1, example_ctx_len=4, n_non_activating=0),
        tokenizer=_tokenizer(),
        modules=["hidden"],
    )
    assert torch.equal(dataset.tokens, tokens)
    read: dict[tuple[int, int, int], float] = {}
    for buffer in dataset.buffers:
        locations, activations, _ = buffer.load()
        for (example, position, latent), value in zip(
            locations.tolist(), activations.tolist(), strict=True
        ):
            read[(example, position, latent)] = value
    if source == "dense":
        expected = {tuple(i) for i in torch.nonzero(hand.abs() > 1e-5).tolist()}
    else:
        expected = {
            (example, position, unit)
            for unit in range(10)
            for example, position, value in _hand_topk(hand, unit, 6)
            if abs(value) > 1e-5
        }
    assert set(read) == expected
    for example, position, latent in expected:
        assert read[(example, position, latent)] == float(hand[example, position, latent].half())
    records = list(dataset)
    assert records, "delphi built no latent record"
    for record in records:
        best = max(example.max_activation for example in record.train)
        unit = record.latent.latent_index
        assert best == pytest.approx(float(hand[..., unit].max().half()), abs=1e-3)
