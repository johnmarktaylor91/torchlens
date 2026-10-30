"""Real-model panel definitions for the capture benchmarks.

Each entry names the model, how it is built, its device class, its hook sites, and
the generation cells it carries. A generation cell names the entry point that runs it
(``record`` for the per-step recorder, ``bound_edit`` for the bound live edit) or is
marked refused with the reason, so the table shows it either way. Only entries with
``smoke=True`` build without downloads or an accelerator; the rest are defined here
and run in the full matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

GENERATION_STEPS = (1, 32, 128, 1000)
FULL_TRACE_STEP_LIMIT = 100


@dataclass(frozen=True)
class PanelEntry:
    """One real model in the panel."""

    name: str
    family: str
    source: str
    device: str
    params: str
    site: str
    sites: tuple[str, ...]
    final_site: str
    smoke: bool = False
    fx_traceable: bool = True
    generation: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


PANEL = (
    PanelEntry(
        "resnet18",
        "CNN",
        "torchvision.models.resnet18(weights=None)",
        "cpu",
        "11.7M",
        "layer2",
        ("layer1", "layer2", "layer3", "layer4"),
        "fc",
        smoke=True,
    ),
    PanelEntry(
        "lstm-tagger",
        "recurrent net",
        "in-file: embedding, 2-layer LSTM, linear head",
        "cpu",
        "0.6M",
        "lstm",
        ("embed", "lstm", "head"),
        "head",
        smoke=True,
        fx_traceable=False,
        notes="nn.LSTM is one fused kernel; its gates are not operations",
    ),
    PanelEntry(
        "encoder-small",
        "encoder",
        "in-file: torch.nn.TransformerEncoder, 4 layers, d=128",
        "cpu",
        "0.8M",
        "encoder.layers.1",
        ("encoder.layers.0", "encoder.layers.1", "encoder.layers.2", "encoder.layers.3"),
        "head",
        smoke=True,
        fx_traceable=False,
    ),
    PanelEntry(
        "bert-base",
        "encoder",
        "transformers bert-base-uncased (public, ungated)",
        "cpu",
        "110M",
        "encoder.layer.6",
        ("encoder.layer.0", "encoder.layer.6", "encoder.layer.11"),
        "pooler",
    ),
    PanelEntry(
        "gpt2",
        "decoder",
        "transformers gpt2 (public, ungated); mech-interp workloads",
        "cpu",
        "124M",
        "transformer.h.6",
        ("transformer.h.0", "transformer.h.6", "transformer.h.11"),
        "lm_head",
        generation=True,
    ),
    PanelEntry(
        "pythia-1.4b",
        "decoder",
        "transformers EleutherAI/pythia-1.4b (public, ungated)",
        "cpu",
        "1.4B",
        "gpt_neox.layers.12",
        ("gpt_neox.layers.0", "gpt_neox.layers.12", "gpt_neox.layers.23"),
        "embed_out",
        generation=True,
    ),
    PanelEntry(
        "decoder-8b",
        "decoder",
        "a public, ungated 7-9B decoder, fp16 or bf16",
        "cuda",
        "7-9B",
        "model.layers.16",
        ("model.layers.0", "model.layers.16", "model.layers.31"),
        "lm_head",
        generation=True,
        notes="GPU only; full trace at about 3 ms host time per op takes tens of seconds",
    ),
    PanelEntry(
        "vit-b-16",
        "vision transformer",
        "torchvision.models.vit_b_16(weights=None)",
        "cpu",
        "86M",
        "encoder.layers.encoder_layer_6",
        (
            "encoder.layers.encoder_layer_0",
            "encoder.layers.encoder_layer_6",
            "encoder.layers.encoder_layer_11",
        ),
        "heads",
    ),
    PanelEntry(
        "unet-diffusion",
        "diffusion U-Net",
        "diffusers UNet2DModel, a small published configuration, random init",
        "cuda",
        "~100M",
        "mid_block",
        ("down_blocks.1", "mid_block", "up_blocks.1"),
        "conv_out",
    ),
    PanelEntry(
        "dynamic-router",
        "dynamic custom model",
        "in-file: data-dependent routing between two experts per block",
        "cpu",
        "0.1M",
        "blocks.2",
        ("blocks.0", "blocks.2", "blocks.3"),
        "head",
        smoke=True,
        fx_traceable=False,
        notes="Python control flow on tensor values; symbolic tracing refuses it",
    ),
)
PANEL_BY_NAME = {entry.name: entry for entry in PANEL}


def generation_cells(entry: PanelEntry) -> list[dict[str, Any]]:
    """Return the generation cells of a decoder entry, each with its entry point.

    Full trace refuses past :data:`FULL_TRACE_STEP_LIMIT` steps today, so a cell past
    the limit names the recorder and the bound edit and marks full trace refused.

    Parameters
    ----------
    entry:
        A panel entry.

    Returns
    -------
    list[dict[str, Any]]
        One row per (steps, entry point); empty for an entry without generation.
    """

    if not entry.generation:
        return []
    rows = []
    for steps in GENERATION_STEPS:
        rows.append({"steps": steps, "entry_point": "bound_edit", "status": "planned"})
        rows.append({"steps": steps, "entry_point": "record", "status": "planned"})
        if steps > FULL_TRACE_STEP_LIMIT:
            rows.append(
                {
                    "steps": steps,
                    "entry_point": "full_trace",
                    "status": "refused",
                    "reason": f"full trace refuses past {FULL_TRACE_STEP_LIMIT} generation steps",
                }
            )
        else:
            rows.append({"steps": steps, "entry_point": "full_trace", "status": "planned"})
    return rows


class _LSTMTagger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True)
        self.head = nn.Linear(128, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out, _state = self.lstm(self.embed(ids))
        return self.head(out)


class _EncoderSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.head = nn.Linear(128, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _Expert(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            return torch.relu(self.a(x)) + x
        return torch.tanh(self.b(x)) + x


class _DynamicRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[_Expert(64) for _ in range(4)])
        self.head = nn.Linear(64, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


def build_panel_entry(name: str) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Build a smoke-runnable panel model, its seeded input and its site metadata.

    Parameters
    ----------
    name:
        Panel entry name with ``smoke=True``.

    Returns
    -------
    tuple[nn.Module, Any, dict[str, Any]]
        Model in eval mode, input, and the entry's sites.

    Raises
    ------
    KeyError
        For an entry that needs downloads or an accelerator (not built here).
    """

    entry = PANEL_BY_NAME[name]
    torch.manual_seed(0)
    if name == "resnet18":
        import torchvision

        model: nn.Module = torchvision.models.resnet18(weights=None)
        x: Any = torch.randn(1, 3, 64, 64)
    elif name == "lstm-tagger":
        model = _LSTMTagger()
        x = torch.randint(0, 1000, (1, 32))
    elif name == "encoder-small":
        model = _EncoderSmall()
        x = torch.randn(1, 32, 128)
    elif name == "dynamic-router":
        model = _DynamicRouter()
        x = torch.randn(1, 16, 64)
    else:
        raise KeyError(f"{name} is defined but not built in the CPU smoke")
    meta = {
        "site": entry.site,
        "sites": list(entry.sites),
        "final_site": entry.final_site,
        "family": entry.family,
        "fx_traceable": entry.fx_traceable,
    }
    return model.eval(), x, meta


def smoke_panel() -> list[str]:
    """Return the panel entries the CPU smoke runs."""

    return [entry.name for entry in PANEL if entry.smoke]
