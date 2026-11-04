"""Tracing a generation loop through a bound ``generate`` method.

``tl.trace(model.generate, ...)`` is the README's generation-loop example. The
toy tests pin, without Hugging Face: module addresses mirror a plain trace of
the model under the documented ``owner.`` prefix, each generated token is one
call of every module, a loop that keeps its values as tensors traces without
warnings, an escape warning names the caller's line, the caller's
``input_kwargs`` survive an input transform, text reaches a bound method of a
model with a resolvable tokenizer, and a plain module address on such a trace
names the ``owner.`` spelling. The GPT-2 test runs the README snippet itself.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import AmbiguousOpLookupError, ScalarEscapeWarning

N_NEW = 3


class _ToyGenerator(nn.Module):
    """Two-module decoder stand-in with an HF-shaped ``generate`` loop."""

    def __init__(self, *, stop_token: int | None = None) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.stop_token = stop_token

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(ids[:, -1:]))

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int = N_NEW) -> torch.Tensor:
        for _ in range(n_new):
            next_token = self(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            # A stopping check reads the token as a Python int, as HF's does.
            if self.stop_token is not None and int(next_token) == self.stop_token:
                break
        return ids


def _ids() -> torch.Tensor:
    return torch.tensor([[1, 4, 2]])


def _module_addresses(trace: tl.Trace) -> set[str]:
    return {str(module.address) for module in trace.modules}


def test_generate_trace_addresses_owner_modules_like_a_plain_trace() -> None:
    model = _ToyGenerator().eval()
    plain = tl.trace(model, _ids())
    gen = tl.trace(model.generate, _ids(), input_kwargs={"n_new": N_NEW})

    expected = {"self", "owner"} | {f"owner.{a}" for a in _module_addresses(plain) - {"self"}}
    assert _module_addresses(gen) == expected
    assert gen["owner.head"].num_calls == N_NEW
    for call in range(1, N_NEW + 1):
        assert gen[f"owner.head:{call}"].out.shape == (1, 1, 16)
    # The first call reads the same input as a plain forward, so it computes the same values.
    assert torch.equal(gen["owner.head:1"].out, plain["head"].out)


def test_plain_generate_trace_emits_no_warning() -> None:
    model = _ToyGenerator().eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model.generate, _ids(), input_kwargs={"n_new": N_NEW})
    assert [f"{w.category.__name__}: {w.message}" for w in caught] == []


def test_scalar_escape_warning_names_the_callers_line() -> None:
    model = _ToyGenerator(stop_token=99).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model.generate, _ids(), input_kwargs={"n_new": N_NEW})
    escapes = [w for w in caught if issubclass(w.category, ScalarEscapeWarning)]
    assert len(escapes) == 1
    assert escapes[0].filename == __file__


@pytest.mark.parametrize(
    "transform",
    [lambda ids: {"ids": ids}, lambda ids: ids.clone()],
    ids=["mapping", "positional"],
)
def test_input_kwargs_survive_an_input_transform(transform: Any) -> None:
    model = _ToyGenerator().eval()
    gen = tl.trace(
        model.generate,
        _ids(),
        input_kwargs={"n_new": 2},
        capture=tl.options.CaptureOptions(input_transform=transform),
    )
    assert gen["owner.head"].num_calls == 2


class _FakeTokenizer:
    """Tokenizer stand-in: whitespace words to ids, HF call signature."""

    name_or_path = "toy-generator"

    def __call__(self, text: str, return_tensors: str = "pt", padding: bool = True) -> dict:
        return {"ids": torch.tensor([[len(word) for word in text.split()]])}


def test_text_input_reaches_a_bound_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens.bridge import transformers as hf_bridge

    model = _ToyGenerator().eval()
    model.config = SimpleNamespace(name_or_path="toy-generator")
    monkeypatch.setattr(hf_bridge, "_resolve_tokenizer", lambda _model: _FakeTokenizer())

    gen = tl.trace(model.generate, "a bb ccc", input_kwargs={"n_new": 2})
    by_ids = tl.trace(model.generate, torch.tensor([[1, 2, 3]]), input_kwargs={"n_new": 2})

    assert gen["owner.head"].num_calls == 2
    assert torch.equal(gen.output, by_ids.output)
    # generate returns token ids, not logits: no tokenizer is attached for output decoding.
    assert gen._output_tokenizer is None
    assert not hasattr(model, "_torchlens_output_tokenizer")


def test_plain_module_address_names_the_owner_spelling() -> None:
    model = _ToyGenerator().eval()
    gen = tl.trace(model.generate, _ids(), input_kwargs={"n_new": N_NEW})
    with pytest.raises(AmbiguousOpLookupError, match=r"use 'owner\.head'"):
        gen["head"]


@pytest.mark.real_model
@pytest.mark.slow
def test_gpt2_generate_readme_snippet() -> None:
    """The README's generation-loop snippet, run as written on GPT-2 small."""

    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(
            model.generate, "The capital of France is", input_kwargs={"max_new_tokens": 4}
        )

    mlp = "owner.transformer.h.5.mlp"
    assert trace[mlp].num_calls == 4
    assert trace[f"{mlp}:1"].out.shape == (1, 5, 768)  # the prefill reads all 5 prompt tokens
    for call in (2, 3, 4):
        assert trace[f"{mlp}:{call}"].out.shape == (1, 1, 768)  # one new token per call
    assert trace.output.shape == (1, 9)
    # generate's stopping check reads tensors as Python values; that one advisory is
    # documented (limitations.md) and must name the caller's line, not contextlib.
    torchlens_warnings = [w for w in caught if issubclass(w.category, tl.errors.TorchLensWarning)]
    assert [w.category for w in torchlens_warnings] == [ScalarEscapeWarning]
    assert torchlens_warnings[0].filename == __file__
