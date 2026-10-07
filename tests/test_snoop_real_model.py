"""Echo narration on real architectures (snoop memo section 5; R0 realism).

Toy-only validation is disqualifying: these rows exercise a config-built
GPT-2 (real architecture, zero network), the real torchvision resnet18 with
its pretrained weights, and the real distilgpt2 checkpoint. Crash rows use
REAL failure shapes (position overflow inside the embedding; incompatible
add over real logits).
"""

from __future__ import annotations

import io
import warnings

import pytest
import torch

import torchlens as tl
from torchlens.options import EchoOptions

transformers = pytest.importorskip("transformers")


def _tiny_gpt2() -> tuple[torch.nn.Module, torch.Tensor]:
    """Config-built GPT-2 (real architecture, zero network) plus input ids."""

    config = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, n_positions=64, vocab_size=128)
    model = transformers.GPT2LMHeadModel(config).eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 128, (1, 9))
    return model, input_ids


@pytest.mark.real_model
def test_gpt2_full_narration_line_count_and_label_resolution() -> None:
    """Memo test 1: line count == narrated events; labels resolve on record."""

    model, input_ids = _tiny_gpt2()
    sink = io.StringIO()
    recording = tl.record(
        model,
        input_ids,
        save=lambda ctx: ctx.kind == "op",
        echo=EchoOptions(select=True, sink=sink),
    )
    lines = [line for line in sink.getvalue().split("\n") if line]
    op_lines = [line for line in lines if line.lstrip().startswith("#")]
    footer = lines[-1]
    assert footer.startswith("-- echo:")
    assert f"{len(lines) - 1} lines narrated" in footer
    # Every printed op label is the spelling the returned object accepts.
    printed_labels = {line.split()[1] for line in op_lines}
    # Recording indexes build lazily on first *records* access (pre-existing
    # fastlog behavior; by_label alone reads empty before materialization).
    assert len(recording.records) > 0
    resolvable = set(recording.by_label)
    op_only = {label for label in printed_labels if not label.startswith(("input_", "buffer_"))}
    assert op_only and op_only <= resolvable


@pytest.mark.real_model
def test_gpt2_scoped_narration_stays_inside_the_block() -> None:
    """Memo test 2: tl.in_module scoping on a transformer block."""

    model, input_ids = _tiny_gpt2()
    sink = io.StringIO()
    tl.record(
        model,
        input_ids,
        echo=EchoOptions(select=tl.in_module("transformer.h.1"), sink=sink),
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


@pytest.mark.real_model
def test_gpt2_position_overflow_crash_narrates_post_hoc() -> None:
    """Memo crash B: tokens beyond n_positions raise inside the embedding."""

    model, _ = _tiny_gpt2()
    too_long = torch.randint(0, 128, (1, 128))  # > n_positions=64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as excinfo:
            tl.trace(model, too_long)
    partial = getattr(excinfo.value, "partial_log", None)
    assert partial is not None
    rendered = partial.narrate(20)
    assert "!! forward failed" in rendered
    assert "the raising call is not in the record" in rendered


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
    output, recording = tl.record(
        model,
        x,
        echo=EchoOptions(select=tl.in_module("layer3"), sink=sink),
        return_output=True,
    )
    assert torch.equal(raw_logits, output)
    assert recording.records == []
    lines = [line for line in sink.getvalue().split("\n") if line]
    op_lines = [line for line in lines if line.lstrip().startswith("#")]
    assert op_lines
    assert all("@layer3" in line for line in op_lines)


@pytest.mark.heavy
@pytest.mark.real_model
def test_resnet18_incompatible_add_crash_trio() -> None:
    """Memo crash C: (1000,) + (3,) over real logits; tail joins the partial."""

    torchvision = pytest.importorskip("torchvision")
    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ).eval()

    class BadHead(torch.nn.Module):
        """Adds an incompatible bias to real logits."""

        def __init__(self) -> None:
            """Wrap the backbone with a mis-shaped bias."""

            super().__init__()
            self.backbone = backbone
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Crash on the final add."""

            return self.backbone(x) + self.bias

    sink = io.StringIO()
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError) as excinfo:
            tl.record(
                BadHead(),
                torch.randn(1, 3, 224, 224),
                save=tl.func("linear"),
                echo=EchoOptions(select=True, sink=sink),
                on_forward_error="attach_partial",
            )
    transcript = sink.getvalue()
    assert "!! forward failed: RuntimeError" in transcript
    assert "attempted call=" in transcript
    partial = getattr(excinfo.value, "partial_recording", None)
    assert partial is not None and partial.failed
    tail_labels = [
        line.split()[1]
        for line in transcript.split("\n")
        if line.lstrip().startswith("#") and "linear" in line
    ]
    assert tail_labels
    assert any(record.ctx.label == tail_labels[-1] for record in partial.records)


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


@pytest.mark.heavy
@pytest.mark.real_model
def test_distilgpt2_real_checkpoint_scoped_narration() -> None:
    """Memo headline demo: scoped narration on the real distilgpt2 checkpoint."""

    if not _distilgpt2_checkpoint_cached():
        pytest.skip("distilgpt2 checkpoint or tokenizer files not in the local HF cache")
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "distilgpt2", local_files_only=True
    ).eval()
    inputs = tokenizer("The keys to the cabinet", return_tensors="pt")
    assert inputs["input_ids"].numel() > 0, "the cached distilgpt2 tokenizer encoded nothing"
    sink = io.StringIO()
    tl.record(
        model,
        inputs["input_ids"],
        echo=EchoOptions(select=tl.in_module("transformer.h.3"), sink=sink),
    )
    lines = [line for line in sink.getvalue().split("\n") if line]
    op_lines = [line for line in lines if line.lstrip().startswith("#")]
    assert op_lines, "h.3 scope narrated nothing on the real checkpoint"
    assert all("@transformer.h.3" in line for line in op_lines)


@pytest.mark.rare
@pytest.mark.real_model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata zero-sync gate")
def test_cuda_metadata_echo_is_zero_sync() -> None:
    """Memo test 4: metadata echo under set_sync_debug_mode('error')."""

    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU()).cuda()
    x = torch.randn(8, 64, device="cuda")
    sink = io.StringIO()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tl.record(model, x, echo=EchoOptions(select=True, sink=sink, stats="off"))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert "relu" in sink.getvalue()
