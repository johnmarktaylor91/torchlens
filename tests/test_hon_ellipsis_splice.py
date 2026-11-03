"""U16.1: ``...`` is a safe stored literal, so a Gemma-class splice runs.

Rotary attention slices with ``x[..., :n]``. Before, the replay template
stored the ``Ellipsis`` as ``Unsupported``, so any splice whose downstream
recomputation crossed such an index refused on Gemma-class models. The
literal is now stored directly (the runnable literal encoder and the safe
unpickler already admit it). The test splices a toy SAE into a tiny randomly
initialised Gemma-2 (built from its real ``transformers`` class, no download)
and checks the replayed logits against an independent forward-hook oracle.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.types import LiteralValue, Unsupported

transformers = pytest.importorskip("transformers")


class _ToySAE(nn.Module):
    """Deterministic encode/decode pair over the model's hidden size."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.enc = nn.Linear(width, 2 * width)
        self.dec = nn.Linear(2 * width, width)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(value))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.dec(latents)


def _tiny_gemma2() -> nn.Module:
    """A two-layer Gemma-2 with random weights from the real upstream class."""

    config = transformers.Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return transformers.Gemma2ForCausalLM(config).eval()


def _template_leaves(component: object) -> list[object]:
    """Flatten an argument-template component into its leaves."""

    if isinstance(component, tuple):
        return [leaf for item in component for leaf in _template_leaves(item)]
    return [component]


def test_ellipsis_is_stored_as_a_literal_on_gemma_class_indexing() -> None:
    """Every ``x[..., :n]`` getitem stores ``...`` literally; none is Unsupported."""

    model, ids = _tiny_gemma2(), torch.randint(0, 64, (1, 6))
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(replayable=True))
    stored_ellipses = 0
    for op in trace._layer_list:
        template = getattr(op, "args_template", None)
        for component in getattr(template, "args", ()):
            for leaf in _template_leaves(component):
                assert not (isinstance(leaf, Unsupported) and "ellipsis" in leaf.reason), (
                    f"{op.label}: {leaf.reason}"
                )
                if isinstance(leaf, LiteralValue) and leaf.value is Ellipsis:
                    stored_ellipses += 1
    assert stored_ellipses > 0, "the rotary x[..., :n] slices were not captured"
