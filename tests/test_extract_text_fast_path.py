"""``tl.extract`` given raw text takes the hooked fast path that token ids take.

A hands-on GPT-2 run measured raw text at 34.7x a plain forward against 1.1x
for token ids: text skipped the fast path and ran a full capture. Text now
resolves through the shared input door first, so both inputs run one hooked
forward and return the same values as the full capture.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import torchlens as tl
from torchlens._extract_fast import try_module_output_extract
from torchlens._extraction._front_door import _extract_layers_with_trace


class _CharTokenizer:
    """A callable tokenizer stub: one id per character, ``input_ids`` like Hugging Face."""

    def __call__(self, text: str, return_tensors: str | None = None) -> SimpleNamespace:
        """Return ``input_ids`` of shape ``(1, len(text))``."""

        ids = [[ord(char) % 32 for char in text]]
        return SimpleNamespace(input_ids=torch.tensor(ids))


class _TinyLM(nn.Module):
    """Embedding, block and head: a language model shape with an attached tokenizer."""

    def __init__(self) -> None:
        """Build the layers and attach the tokenizer stub."""

        super().__init__()
        self.emb = nn.Embedding(32, 8)
        self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.head = nn.Linear(8, 32)
        self.tokenizer = _CharTokenizer()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return per-position logits."""

        return self.head(self.block(self.emb(ids)))


def _tiny_lm() -> _TinyLM:
    """Build the tiny language model in eval mode with a fixed seed."""

    torch.manual_seed(0)
    return _TinyLM().eval()


def test_extract_raw_text_takes_the_hooked_fast_path() -> None:
    """Raw text resolves through the input door and runs one hooked forward."""

    model = _tiny_lm()
    fast = try_module_output_extract(model, "hello world", ["block", "head"], detach=True)
    assert fast is not None, "raw text fell back to a full capture"
    _trace, full, _views = _extract_layers_with_trace(model, "hello world", ["block", "head"])
    assert set(fast) == {"block", "head"}
    for name in ("block", "head"):
        assert torch.equal(fast[name], full[name]), name


def test_extract_raw_text_matches_token_ids() -> None:
    """``tl.extract`` on text equals ``tl.extract`` on the text's token ids."""

    model = _tiny_lm()
    ids = model.tokenizer("hello world").input_ids
    from_text = tl.extract(model, "hello world", {"mid": "block"})
    from_ids = tl.extract(model, ids, {"mid": "block"})
    assert torch.equal(from_text["mid"], from_ids["mid"])


def test_extract_text_without_tokenizer_falls_back_to_the_full_capture() -> None:
    """With no tokenizer the fast path declines, so the full capture owns the refusal."""

    model = _tiny_lm()
    del model.tokenizer
    assert try_module_output_extract(model, "hello", ["block"], detach=True) is None
