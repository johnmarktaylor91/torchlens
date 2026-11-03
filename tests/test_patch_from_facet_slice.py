"""``patch_from`` on a facet or head target copies only the matching donor slice.

A ``tl.head(h, "v")`` / ``tl.facet(...)`` target edits one slice of its home op
(GPT-2's fused ``c_attn`` output). ``patch_from(source)`` used to hand back the
source's WHOLE home output, so single-head patching failed on shape
(``expected (1, 5, 8), got (1, 5, 24)``). The donor is now the same facet view
resolved on the source trace. The model is a hand-written attention module
named like GPT-2's (the built-in ``GPT2Attention`` facet recipe: fused
``c_attn``, ``c_proj``), so no pretrained weights or transformers are needed;
the oracle is a plain-torch forward hook on ``c_attn``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions, InterventionOptions

N_EMBD, N_HEAD, SEQ = 8, 2, 5
D_HEAD = N_EMBD // N_HEAD


class GPT2Attention(nn.Module):
    """GPT-2-shaped causal self-attention: fused ``c_attn``, per-head split, ``c_proj``."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = N_HEAD
        self.head_dim = D_HEAD
        self.embed_dim = N_EMBD
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=-1)
        q, k, v = (t.view(batch, seq, N_HEAD, D_HEAD).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-1, -2) / D_HEAD**0.5
        causal = torch.ones(seq, seq, dtype=torch.bool).tril()
        pattern = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        z = (pattern @ v).transpose(1, 2).reshape(batch, seq, N_EMBD)
        return self.c_proj(z)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = GPT2Attention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class _TinyGpt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.h = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(N_EMBD, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.h:
            x = block(x)
        return self.head(x)


def _oracle(
    model: _TinyGpt, clean_x: torch.Tensor, corrupt_x: torch.Tensor, facet: str, head: int
) -> torch.Tensor:
    """Run ``corrupt_x`` with block 1's ``c_attn`` facet head columns taken from ``clean_x``."""

    c_attn = model.h[1].attn.c_attn
    captured: dict[str, torch.Tensor] = {}
    grab = c_attn.register_forward_hook(lambda _m, _a, out: captured.setdefault("clean", out))
    with torch.no_grad():
        model(clean_x)
    grab.remove()
    offset = {"q": 0, "k": N_EMBD, "v": 2 * N_EMBD}[facet] + head * D_HEAD

    def _swap(_module: nn.Module, _args: Any, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out[..., offset : offset + D_HEAD] = captured["clean"][..., offset : offset + D_HEAD]
        return out

    swap = c_attn.register_forward_hook(_swap)
    try:
        with torch.no_grad():
            return model(corrupt_x).detach()
    finally:
        swap.remove()


def _final(trace: tl.Trace) -> torch.Tensor:
    return trace[trace.output_layers[0]].out.detach()


@pytest.fixture(scope="module")
def case() -> Iterator[tuple[_TinyGpt, torch.Tensor, torch.Tensor, tl.Trace, tl.Trace]]:
    model = _TinyGpt().eval()
    generator = torch.Generator().manual_seed(1)
    clean_x = torch.randn(1, SEQ, N_EMBD, generator=generator)
    corrupt_x = torch.randn(1, SEQ, N_EMBD, generator=generator)
    options = CaptureOptions(replayable=True)
    clean = tl.trace(model, clean_x, capture=options)
    corrupt = tl.trace(model, corrupt_x, capture=options)
    try:
        yield model, clean_x, corrupt_x, clean, corrupt
    finally:
        clean.cleanup()
        corrupt.cleanup()


@pytest.mark.parametrize("engine", ["replay", "rerun"])
@pytest.mark.parametrize(("facet", "head"), [("v", 0), ("v", 1), ("q", 1), ("k", 0)])
def test_single_head_patch_from_copies_only_the_head_slice(
    case: tuple[_TinyGpt, torch.Tensor, torch.Tensor, tl.Trace, tl.Trace],
    engine: str,
    facet: str,
    head: int,
) -> None:
    model, clean_x, corrupt_x, clean, corrupt = case
    site = tl.head(head, facet).in_module("h.1.attn")
    kwargs: dict[str, Any] = {"model": model, "inputs": corrupt_x} if engine == "rerun" else {}
    patched = corrupt.fork().do(
        site, tl.patch_from(clean), intervention=InterventionOptions(engine=engine), **kwargs
    )
    expected = _oracle(model, clean_x, corrupt_x, facet, head)
    torch.testing.assert_close(_final(patched), expected, atol=1e-5, rtol=0)
    # The control: the patch changed the corrupt run, and it is not the clean run.
    assert not torch.allclose(_final(patched), _final(corrupt), atol=1e-5)
    assert not torch.allclose(_final(patched), _final(clean), atol=1e-5)
    # Only the head's columns of the home op moved; the rest stay corrupt.
    home = patched.modules["h.1.attn.c_attn"].out.detach()
    corrupt_home = corrupt.modules["h.1.attn.c_attn"].out.detach()
    clean_home = clean.modules["h.1.attn.c_attn"].out.detach()
    offset = {"q": 0, "k": N_EMBD, "v": 2 * N_EMBD}[facet] + head * D_HEAD
    written = torch.zeros(3 * N_EMBD, dtype=torch.bool)
    written[offset : offset + D_HEAD] = True
    torch.testing.assert_close(home[..., written], clean_home[..., written])
    torch.testing.assert_close(home[..., ~written], corrupt_home[..., ~written])


def test_whole_facet_patch_from_copies_the_facet_group(
    case: tuple[_TinyGpt, torch.Tensor, torch.Tensor, tl.Trace, tl.Trace],
) -> None:
    """``tl.facet("v")`` (every head) takes the donor's whole v block, not all of c_attn."""

    _model, _clean_x, _corrupt_x, clean, corrupt = case
    patched = corrupt.fork().do(
        tl.facet("v").in_module("h.1.attn"),
        tl.patch_from(clean),
        intervention=InterventionOptions(engine="replay"),
    )
    home = patched.modules["h.1.attn.c_attn"].out.detach()
    clean_home = clean.modules["h.1.attn.c_attn"].out.detach()
    corrupt_home = corrupt.modules["h.1.attn.c_attn"].out.detach()
    torch.testing.assert_close(home[..., 2 * N_EMBD :], clean_home[..., 2 * N_EMBD :])
    torch.testing.assert_close(home[..., : 2 * N_EMBD], corrupt_home[..., : 2 * N_EMBD])


def test_whole_site_patch_from_is_unchanged(
    case: tuple[_TinyGpt, torch.Tensor, torch.Tensor, tl.Trace, tl.Trace],
) -> None:
    """Outside a facet write the donor is still the source's whole site value."""

    _model, _clean_x, _corrupt_x, clean, corrupt = case
    label = corrupt.modules["h.1.attn.c_attn"].output_layers[0]
    patched = corrupt.fork().do(
        label, tl.patch_from(clean), intervention=InterventionOptions(engine="replay")
    )
    torch.testing.assert_close(
        patched.modules["h.1.attn.c_attn"].out.detach(),
        clean.modules["h.1.attn.c_attn"].out.detach(),
    )
    assert not torch.allclose(_final(patched), _final(corrupt), atol=1e-5)


def test_single_head_patch_from_on_the_real_gpt2_attention_class() -> None:
    """The same patch on a randomly initialised ``transformers`` GPT-2 (eager attention)."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=64,
        n_positions=16,
        bos_token_id=0,
        eos_token_id=0,
        use_cache=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    generator = torch.Generator().manual_seed(0)
    clean_ids = torch.randint(0, 64, (1, SEQ), generator=generator)
    corrupt_ids = torch.randint(0, 64, (1, SEQ), generator=generator)
    options = CaptureOptions(replayable=True)
    clean = tl.trace(model, clean_ids, capture=options)
    corrupt = tl.trace(model, corrupt_ids, capture=options)
    patched = corrupt.fork().do(
        tl.head(2, "v").in_module("transformer.h.1.attn"),
        tl.patch_from(clean),
        intervention=InterventionOptions(engine="replay"),
    )

    c_attn = model.get_submodule("transformer.h.1.attn.c_attn")
    d_head = 32 // 4
    offset = 2 * 32 + 2 * d_head
    captured: dict[str, torch.Tensor] = {}
    grab = c_attn.register_forward_hook(lambda _m, _a, out: captured.setdefault("clean", out))
    with torch.no_grad():
        model(clean_ids)
    grab.remove()

    def _swap(_module: nn.Module, _args: Any, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out[..., offset : offset + d_head] = captured["clean"][..., offset : offset + d_head]
        return out

    swap = c_attn.register_forward_hook(_swap)
    try:
        with torch.no_grad():
            expected = model(corrupt_ids).logits.detach()
    finally:
        swap.remove()
    torch.testing.assert_close(_final(patched), expected, atol=1e-5, rtol=0)
    assert not torch.allclose(_final(patched), _final(corrupt), atol=1e-5)
