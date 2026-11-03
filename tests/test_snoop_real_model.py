"""Echo narration on real architectures (snoop memo section 5; R0 realism).

Toy-only validation is disqualifying: these rows exercise a config-built
GPT-2 (real architecture, zero network), the real torchvision resnet18 with
its pretrained weights, and the real distilgpt2 checkpoint. Crash rows use
REAL failure shapes (position overflow inside the embedding; incompatible
add over real logits).
"""

from __future__ import annotations

import io

import pytest
import torch

import torchlens as tl
from torchlens.options import NarrationOptions

transformers = pytest.importorskip("transformers")


def _tiny_gpt2() -> tuple[torch.nn.Module, torch.Tensor]:
    """Config-built GPT-2 (real architecture, zero network) plus input ids."""

    config = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, n_positions=64, vocab_size=128)
    model = transformers.GPT2LMHeadModel(config).eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (1, 9))
    return model, input_ids


@pytest.mark.real_model
def test_gpt2_scoped_narration_stays_inside_the_block() -> None:
    """Memo test 2: tl.in_module scoping on a transformer block."""

    model, input_ids = _tiny_gpt2()
    sink = io.StringIO()
    tl.record(
        model,
        input_ids,
        narrate=NarrationOptions(sites=tl.in_module("transformer.h.1"), sink=sink),
    )
    lines = [line for line in sink.getvalue().split("\n") if line]
    body = [line for line in lines if not line.startswith("-- echo:")]
    assert body, "block scope narrated nothing"
    op_lines = [line for line in body if line.lstrip().startswith("#")]
    assert op_lines
    assert all("@transformer.h.1" in line for line in op_lines)
    structure = [line for line in body if line.lstrip().startswith((">", "<"))]
    # Held-ancestor lines for the match's OWN chain print; unselected sibling
    # modules never do.
    for line in structure:
        address = line.lstrip().lstrip("><").split()[0]
        assert "transformer.h.1".startswith(address) or address.startswith("transformer.h.1"), line
    assert not any("h.0" in line for line in body)


@pytest.mark.heavy
@pytest.mark.real_model
def test_resnet18_scoped_echo_leaves_logits_unchanged() -> None:
    """Memo test 3: scoped record-tier echo on real resnet18 weights.

    Narration must be observationally invisible: logits equal the raw
    forward, nothing is retained solely because echo printed it.
    """

    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        raw_logits = model(x)
    sink = io.StringIO()
    recording = tl.record(
        model,
        x,
        narrate=NarrationOptions(sites=tl.in_module("layer3"), sink=sink),
    )
    output = recording.output
    assert torch.equal(raw_logits, output)
    assert recording.records == []
    lines = [line for line in sink.getvalue().split("\n") if line]
    op_lines = [line for line in lines if line.lstrip().startswith("#")]
    assert op_lines
    assert all("@layer3" in line for line in op_lines)


def _distilgpt2_checkpoint_cached() -> bool:
    """Return whether the local HF cache holds every distilgpt2 file the demo loads.

    Tests that load only the distilgpt2 model (config and weights) leave a
    partial cache entry behind. ``AutoTokenizer.from_pretrained(...,
    local_files_only=True)`` then builds an empty-vocabulary tokenizer instead
    of raising ``OSError``, and it encodes any prompt to zero tokens. Checking
    the files up front also keeps the skip out of an ``except`` handler, so a
    load that fails with the files present fails loudly.

    Returns
    -------
    bool
        True when ``config.json``, the weights, and the tokenizer files
        (``tokenizer.json``, or both ``vocab.json`` and ``merges.txt``) are cached.
    """

    from huggingface_hub import try_to_load_from_cache

    def cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache("distilgpt2", filename), str)

    weights = cached("model.safetensors") or cached("pytorch_model.bin")
    tokenizer = cached("tokenizer.json") or (cached("vocab.json") and cached("merges.txt"))
    return cached("config.json") and weights and tokenizer


@pytest.mark.rare
@pytest.mark.real_model
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata zero-sync gate")
def test_cuda_metadata_echo_is_zero_sync() -> None:
    """Memo test 4: metadata echo under set_sync_debug_mode('error')."""

    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU()).cuda()
    x = torch.randn(8, 64, device="cuda")
    sink = io.StringIO()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tl.record(model, x, narrate=NarrationOptions(sites=True, sink=sink, stats="off"))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert "relu" in sink.getvalue()
