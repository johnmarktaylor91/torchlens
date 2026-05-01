"""Phase 12a launch-tier bridge invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.callbacks as tl_callbacks
import torchlens.compat as tl_compat


class _TinyCnn(nn.Module):
    """Small deterministic CNN fixture for bridge tests."""

    def __init__(self) -> None:
        """Initialize the CNN layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(72, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN forward pass.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.fc(self.flatten(self.relu(self.conv(x))))


class _TinyTransformer(nn.Module):
    """Tiny transformer-like fixture with an exposed hidden projection."""

    def __init__(self) -> None:
        """Initialize the model layers."""

        super().__init__()
        self.embed = nn.Embedding(13, 4)
        self.proj = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the model forward pass.

        Parameters
        ----------
        tokens:
            Integer token IDs.

        Returns
        -------
        torch.Tensor
            Per-token logits.
        """

        hidden = torch.relu(self.proj(self.embed(tokens)))
        return self.head(hidden)


class _TinySae:
    """Small SAE-like encoder fixture."""

    def __init__(self, width: int) -> None:
        """Initialize deterministic encoder weights.

        Parameters
        ----------
        width:
            Activation width.
        """

        self.weight = torch.eye(width)

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        """Encode an activation.

        Parameters
        ----------
        activation:
            Tensor activation.

        Returns
        -------
        torch.Tensor
            Encoded activation.
        """

        return activation @ self.weight.to(activation.device)


class _OfflineBenchmark:
    """Callable Brain-Score-style offline benchmark fixture."""

    def __call__(self, activation: torch.Tensor, *, layer: str) -> float:
        """Score an activation.

        Parameters
        ----------
        activation:
            Layer activation.
        layer:
            Layer label.

        Returns
        -------
        float
            Deterministic fixture score.
        """

        return float(activation.detach().float().mean().item() + len(layer) * 0.001)


class _TraceObject:
    """nnsight-like cached trace fixture."""

    def to_dict(self) -> dict[str, Any]:
        """Return a trace dictionary.

        Returns
        -------
        dict[str, Any]
            Trace payload.
        """

        return {"nodes": [{"name": "embed"}, {"name": "proj"}], "source": "offline-fixture"}


def _cnn_log() -> tuple[_TinyCnn, torch.Tensor, Any]:
    """Build a deterministic CNN log.

    Returns
    -------
    tuple[_TinyCnn, torch.Tensor, Any]
        Model, input tensor, and TorchLens log.
    """

    torch.manual_seed(12)
    model = _TinyCnn().eval()
    x = torch.randn(3, 1, 8, 8)
    log = tl.log_forward_pass(model, x, layers_to_save="all")
    return model, x, log


def test_captum_bridge_matches_direct_layer_integrated_gradients() -> None:
    """Captum bridge attribution should match direct Captum on a tiny CNN."""

    captum_attr = pytest.importorskip("captum.attr")
    model, x, log = _cnn_log()
    layer = tl.bridge.captum.layer(log, "conv")
    method = captum_attr.LayerIntegratedGradients(model, layer)

    bridge_result = tl.bridge.captum.attribute(log, method, 0, inputs=x, n_steps=4)
    direct_result = method.attribute(x, target=0, n_steps=4)

    assert torch.allclose(bridge_result, direct_result, atol=1e-5)


def test_sae_lens_bridge_encode_matches_direct_sae() -> None:
    """SAE Lens bridge encoding should match a direct SAE encode call."""

    pytest.importorskip("sae_lens")
    torch.manual_seed(13)
    model = _TinyTransformer().eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    log = tl.log_forward_pass(model, tokens, layers_to_save="all")
    activation = log.resolve_sites("linear", max_fanout=1).first().activation
    sae = _TinySae(width=activation.shape[-1])

    bridge_result = tl.bridge.sae_lens.encode(log, "linear", sae)
    direct_result = sae.encode(activation)

    assert torch.allclose(bridge_result, direct_result)


def test_rsatoolbox_bridge_rdm_matches_direct_dataset() -> None:
    """rsatoolbox bridge Dataset should produce the same RDM as a direct Dataset."""

    rsa = pytest.importorskip("rsatoolbox")
    _model, _x, log = _cnn_log()
    bridge_dataset = tl.bridge.rsatoolbox.dataset(log)
    output = log[log.output_layers[0]].activation.detach().cpu().reshape(3, -1).numpy()
    direct_dataset = rsa.data.Dataset(
        measurements=output,
        obs_descriptors={"presentation": np.arange(output.shape[0])},
        channel_descriptors={"neuroid": np.arange(output.shape[1])},
        descriptors={"source": "torchlens"},
    )

    bridge_rdm = rsa.rdm.calc_rdm(bridge_dataset)
    direct_rdm = rsa.rdm.calc_rdm(direct_dataset)

    assert np.allclose(bridge_rdm.dissimilarities, direct_rdm.dissimilarities)


def test_brain_score_bridge_mocked_offline_fixture_matches_direct_scores() -> None:
    """Brain-Score bridge should match a direct offline benchmark fixture."""

    _model, _x, log = _cnn_log()
    benchmark = _OfflineBenchmark()
    sites = ["conv2d", "linear"]

    bridge_scores = tl.bridge.brain_score.per_layer(log, benchmark, sites=sites)
    direct_scores = {
        layer.layer_label: benchmark(layer.activation, layer=layer.layer_label)
        for layer in [log.resolve_sites(site, max_fanout=1).first() for site in sites]
    }

    assert bridge_scores == direct_scores


def test_lightning_layer_profiler_callback_fires_and_saves_results(tmp_path: Path) -> None:
    """Lightning callback should profile one batch and persist a JSONL record."""

    try:
        import lightning

        lightning_base = lightning.LightningModule
    except Exception:
        lightning_base = nn.Module

    class TinyLightningModule(lightning_base):  # type: ignore[valid-type, misc]
        """Tiny LightningModule fixture."""

        def __init__(self) -> None:
            """Initialize the wrapped network."""

            super().__init__()
            self.net = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the model forward pass.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Output logits.
            """

            return self.net(x)

    class TrainerStub:
        """Small trainer fixture exposing ``global_step``."""

        global_step = 7

    output_path = tmp_path / "lightning_layers.jsonl"
    callback = tl_callbacks.lightning.LayerProfilerCallback(output_path)
    batch = (torch.randn(2, 4), torch.tensor([0, 1]))

    callback.on_validation_batch_end(TrainerStub(), TinyLightningModule(), None, batch, 0)

    assert callback.records
    saved = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["stage"] == "validation"
    assert saved[0]["num_layers"] > 0


def test_hf_compat_loaders_match_direct_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF and timm compat loaders should match their direct library calls."""

    transformers = pytest.importorskip("transformers")
    timm = pytest.importorskip("timm")

    try:
        direct_hf = transformers.AutoModel.from_pretrained(
            "distilbert-base-uncased", local_files_only=True
        )
    except Exception:

        class FakeAutoModel:
            """Offline Transformers AutoModel fixture."""

            @staticmethod
            def from_pretrained(model_id: str, *, local_files_only: bool = True) -> nn.Module:
                """Return a fake model for offline cache misses.

                Parameters
                ----------
                model_id:
                    Model identifier.
                local_files_only:
                    Offline loading flag.

                Returns
                -------
                nn.Module
                    Deterministic fake model.
                """

                model = nn.Linear(2, 2)
                model.model_id = model_id  # type: ignore[attr-defined]
                model.local_files_only = local_files_only  # type: ignore[attr-defined]
                return model

        monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
        direct_hf = transformers.AutoModel.from_pretrained(
            "distilbert-base-uncased", local_files_only=True
        )

    bridge_hf = tl_compat.from_huggingface("distilbert-base-uncased")
    assert type(bridge_hf) is type(direct_hf)

    direct_timm = timm.create_model("resnet18", pretrained=False, num_classes=2)
    bridge_timm = tl_compat.from_timm("resnet18", pretrained=False, num_classes=2)
    assert type(bridge_timm) is type(direct_timm)
    assert list(bridge_timm.state_dict()) == list(direct_timm.state_dict())


def test_profiler_join_merges_per_op_timing(tmp_path: Path) -> None:
    """Profiler join should attach Kineto event durations to TorchLens ops."""

    _model, _x, log = _cnn_log()
    conv_layer = log.resolve_sites("conv2d", max_fanout=1).first()
    trace_path = tmp_path / "kineto.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": f"{conv_layer.layer_label} cpu", "ph": "X", "dur": 12.5},
                    {"name": "unrelated", "ph": "X", "dur": 99.0},
                ],
                "metadata": {"fixture": True},
            }
        ),
        encoding="utf-8",
    )

    joined = tl.bridge.profiler.join(log, trace_path)
    conv_row = next(row for row in joined["ops"] if row["layer_label"] == conv_layer.layer_label)

    assert conv_row["kineto_event_count"] == 1
    assert conv_row["kineto_duration_us"] == 12.5


def test_nnsight_bridge_normalizes_offline_trace_fixture() -> None:
    """nnsight bridge should normalize an offline cached trace-like object."""

    payload = tl.bridge.nnsight.from_trace(_TraceObject())

    assert payload["schema"] == "torchlens.nnsight_trace.v1"
    assert payload["nodes"] == [{"name": "embed"}, {"name": "proj"}]
    assert payload["metadata"]["source"] == "offline-fixture"
