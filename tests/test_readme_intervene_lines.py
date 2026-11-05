"""The README Intervene lines run as written on a plain ``tl.trace(model, x)`` trace.

The README promises that edits "can propagate through the saved graph without
rerunning the model, or apply to a fresh run", and its snippets build traces with
plain ``tl.trace``. These tests pin the two README spellings on default captures:

- ``trace.do(tl.func("gelu") & tl.in_module("stages.2"), tl.zero_ablate())``
- ``patched = italy.fork().do(tl.module("transformer.h.5"), tl.patch_from(france))``

Each replay is checked against the rerun engine and a hand-built PyTorch ground
truth. The refusal for a trace that genuinely cannot replay (a ``tl.record``
recording keeps no argument templates) must name the fix.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
import torch

import torchlens as tl

pytestmark = pytest.mark.filterwarnings("ignore::torchlens.errors.InterventionWarning")


def _output(trace: Any) -> torch.Tensor:
    return cast(torch.Tensor, trace[trace.output_layers[0]].out)


class _ConvNeXtShaped(torch.nn.Module):
    """Three ``stages`` of blocks with a GELU each, so ``stages.2`` holds several GELUs."""

    def __init__(self) -> None:
        super().__init__()

        def stage() -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1), torch.nn.GELU()),
                torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1), torch.nn.GELU()),
                torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1), torch.nn.GELU()),
            )

        self.stem = torch.nn.Conv2d(3, 2, 1)
        self.stages = torch.nn.ModuleList([stage(), stage(), stage()])
        self.head = torch.nn.Linear(2, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x.mean(dim=(2, 3)))


class _Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class _Transformer(torch.nn.Module):
    def __init__(self, vocab: int, width: int, n_layer: int) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(vocab, width)
        self.h = torch.nn.ModuleList([_Block(width) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(width)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(ids)
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class _GPT2Shaped(torch.nn.Module):
    """``transformer.h.0`` to ``transformer.h.5`` then a tied-free ``lm_head``, like GPT-2."""

    def __init__(self, vocab: int = 16, width: int = 8) -> None:
        super().__init__()
        self.transformer = _Transformer(vocab, width, n_layer=6)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.transformer(ids))


def test_readme_gelu_zero_ablate_line_on_default_trace() -> None:
    """``trace.do(tl.func("gelu") & tl.in_module("stages.2"), tl.zero_ablate())`` replays."""

    torch.manual_seed(0)
    model = _ConvNeXtShaped().eval()
    x = torch.rand(1, 3, 4, 4)

    trace = tl.trace(model, x)
    assert trace.is_replayable is False
    before = _output(trace).detach().clone()

    returned = trace.do(tl.func("gelu") & tl.in_module("stages.2"), tl.zero_ablate())
    after = _output(trace)
    assert returned is trace
    assert trace.last_run["engine"] == "replay"
    assert not torch.equal(before, after)

    # Fresh-run engine on a second default trace.
    rerun = tl.trace(model, x).fork()
    rerun.do(tl.func("gelu") & tl.in_module("stages.2"), tl.zero_ablate(), model=model, inputs=x)
    assert torch.equal(after, _output(rerun))

    # Ground truth: plain PyTorch forward hooks zeroing every GELU in stages.2.
    handles = [
        module.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        for name, module in model.named_modules()
        if name.startswith("stages.2.") and isinstance(module, torch.nn.GELU)
    ]
    assert len(handles) == 3
    try:
        with torch.no_grad():
            hooked = model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.allclose(after, hooked, atol=1e-6)


def test_readme_patch_from_fork_line_on_default_traces() -> None:
    """``patched = italy.fork().do(tl.module("transformer.h.5"), tl.patch_from(france))``."""

    torch.manual_seed(0)
    model = _GPT2Shaped().eval()
    france_ids = torch.tensor([[1, 2, 3, 4, 5]])
    italy_ids = torch.tensor([[1, 2, 3, 6, 5]])

    france = tl.trace(model, france_ids)
    italy = tl.trace(model, italy_ids)
    assert france.is_replayable is False and italy.is_replayable is False
    italy_before = _output(italy).detach().clone()

    patched = italy.fork().do(tl.module("transformer.h.5"), tl.patch_from(france))
    assert patched is not italy
    assert patched.last_run["engine"] == "replay"

    # Block 5 carries France's residual stream at every position, so from there on
    # the patched Italy run equals the France run; Italy itself is untouched.
    assert torch.equal(_output(patched), _output(france))
    assert not torch.equal(_output(patched), italy_before)
    assert torch.equal(_output(italy), italy_before)

    # Fresh-run engine agrees exactly.
    rerun = tl.trace(model, italy_ids).fork()
    rerun.do(tl.module("transformer.h.5"), tl.patch_from(france), model=model, inputs=italy_ids)
    assert torch.equal(_output(patched), _output(rerun))


def test_recording_trace_refusal_names_the_fix() -> None:
    """A trace that keeps no argument templates still refuses, naming both remedies."""

    torch.manual_seed(0)
    model = _ConvNeXtShaped().eval()
    x = torch.rand(1, 3, 4, 4)
    recorded = tl.record(model, x, save=tl.func("gelu")).to_trace()
    assert recorded.is_replayable is False

    with pytest.raises(tl.errors.ReplayPreconditionError) as info:
        recorded.do(tl.func("gelu") & tl.in_module("stages.2"), tl.zero_ablate())
    message = str(info.value)
    assert "tl.trace(model, inputs)" in message
    assert "model=model, inputs=inputs" in message
    assert "replayable=True" not in message


@pytest.mark.heavy
def test_readme_gelu_line_on_real_convnext_tiny() -> None:
    """The ConvNeXt README line on timm ``convnext_tiny`` (random weights, no download)."""

    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    model = timm.create_model("convnext_tiny", pretrained=False).eval()
    x = torch.rand(1, 3, 224, 224)

    trace = tl.trace(model, x)
    before = _output(trace).detach().clone()
    trace.do(tl.func("gelu") & tl.in_module("stages.2"), tl.zero_ablate())
    after = _output(trace)
    assert trace.last_run["engine"] == "replay"
    assert not torch.equal(before, after)

    handles = [
        module.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        for name, module in model.named_modules()
        if name.startswith("stages.2.") and type(module).__name__ == "GELU"  # timm's own GELU class
    ]
    assert len(handles) == 9
    try:
        with torch.no_grad():
            hooked = model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.allclose(after, hooked, atol=1e-5)


@pytest.mark.heavy
def test_readme_patch_from_line_on_real_gpt2_small() -> None:
    """The GPT-2 README line on a GPT-2 small architecture (random weights, no download)."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config()  # GPT-2 small: 12 layers, 768 wide
    model = transformers.GPT2LMHeadModel(config).eval()
    france_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    italy_ids = torch.tensor([[464, 3139, 286, 8031, 318]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        france = tl.trace(model, france_ids)
        italy = tl.trace(model, italy_ids)
        italy_before = _output(italy).detach().clone()
        patched = italy.fork().do(tl.module("transformer.h.5"), tl.patch_from(france))
    assert patched is not italy
    assert patched.last_run["engine"] == "replay"
    assert torch.allclose(_output(patched), _output(france), atol=1e-5)
    assert not torch.allclose(_output(patched), italy_before, atol=1e-3)
    assert torch.equal(_output(italy), italy_before)
