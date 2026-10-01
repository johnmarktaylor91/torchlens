"""R0 family builders (testing MEMO 4.1 roster; build row A3).

Fourteen architecture families, each instantiated from its REAL upstream
class with its REAL vendored ``config.json`` (registry-pinned revision)
shrunk through explicit overrides, weights built from a fixed seed, ZERO
network. Plus the six structural fixtures (tied-weight, multi-input,
in-place, lazy-parameter, container-output, train-mode-BatchNorm).

R0 proves structure -- class/config registries, kwargs transport,
attention-implementation invariance, aliasing, multi-pass identity,
absence-claim honesty -- and NEVER counts as pretrained evidence
(registry never-count enforcement).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from tests.real_model.registry import VENDORED_DIR

SEED = 20260826
# Shared tiny-input shapes: one fixed token batch for every text family.
BATCH, SEQ, VOCAB = 1, 8, 512


def _token_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)


def _pixels(size: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randn(BATCH, 3, size, size, generator=generator)


def vendored_config(family: str) -> dict[str, Any]:
    """Parse the vendored REAL ``config.json`` for ``family``."""

    path = VENDORED_DIR / family / "config.json"
    return json.loads(path.read_text())


def load_vendored_gpt2_tokenizer(tmp_dir: str) -> Any:
    """Build the REAL GPT-2 tokenizer from the vendored files, zero network.

    ``vocab.json`` is stored gzip-compressed (deterministic bytes, mtime=0)
    to fit the repo's 500 KB large-file gate; the decompressed digest is
    ledgered in ``vendored/PROVENANCE.jsonl``. transformers 5.x builds fast
    tokenizers from in-memory ``vocab``/``merges`` objects, so the loader
    parses the vendored files itself (``tmp_dir`` kept for interface
    stability with file-based tokenizer families).
    """

    import gzip

    from transformers import GPT2TokenizerFast

    del tmp_dir
    vocab = json.loads(gzip.decompress((VENDORED_DIR / "gpt2" / "vocab.json.gz").read_bytes()))
    merges_text = (VENDORED_DIR / "gpt2" / "merges.txt").read_text(encoding="utf-8")
    merges = [
        tuple(line.split(" ", 1))
        for line in merges_text.splitlines()
        if line and not line.startswith("#")
    ]
    return GPT2TokenizerFast(vocab=vocab, merges=merges)


@dataclass(frozen=True)
class FamilySpec:
    """One R0 roster family."""

    name: str
    artifact_id: str
    build: Callable[[str], nn.Module]
    input_kwargs: Callable[[], dict[str, Any]]
    impls: tuple[str, ...]
    dims: dict[str, int]
    # Families whose upstream class takes no attn_implementation axis at all
    # (attention-free or plain torch) carry a single "eager" impl entry.
    # input_args covers plain-torch families: the kwargs-only spelling on a
    # plain module raises "got multiple values" today (enumerated-red row in
    # expectations.KNOWN_RED, pinned by the axes suite).
    input_args: Callable[[], tuple[Any, ...]] = lambda: ()


def _hf_config(family: str, config_cls: Any, shrink: dict[str, Any], impl: str) -> Any:
    """Build a shrunk config THROUGH the real ``from_dict`` parsing path."""

    payload = dict(vendored_config(family))
    payload.update(shrink)
    config = config_cls.from_dict(payload)
    config._attn_implementation = impl
    return config


def _seeded(model: nn.Module) -> nn.Module:
    model.eval()
    return model


def build_gpt2(impl: str) -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    shrink = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "vocab_size": VOCAB,
        "n_positions": 64,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "use_cache": False,
    }
    torch.manual_seed(SEED)
    return _seeded(GPT2LMHeadModel(_hf_config("gpt2", GPT2Config, shrink, impl)))


def build_gpt2_default_cache(impl: str) -> nn.Module:
    """The shrunk GPT-2 with the HuggingFace DEFAULT ``use_cache=True`` (AUD-HONESTY H1).

    ``build_gpt2`` pins ``use_cache=False`` so the runnable artifact can round-trip; this
    sibling keeps the KV cache on -- the configuration every real user has -- so the
    live ``run()`` surface is exercised against a ``DynamicCache``-bearing ModelOutput.
    """

    from transformers import GPT2Config, GPT2LMHeadModel

    shrink = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "vocab_size": VOCAB,
        "n_positions": 64,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    torch.manual_seed(SEED)
    return _seeded(GPT2LMHeadModel(_hf_config("gpt2", GPT2Config, shrink, impl)))


def build_distilgpt2(impl: str) -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    shrink = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "vocab_size": VOCAB,
        "n_positions": 64,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "use_cache": False,
    }
    torch.manual_seed(SEED)
    return _seeded(GPT2LMHeadModel(_hf_config("distilgpt2", GPT2Config, shrink, impl)))


def build_llama(impl: str) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    shrink = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
        "use_cache": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": None,
    }
    torch.manual_seed(SEED)
    return _seeded(LlamaForCausalLM(_hf_config("llama", LlamaConfig, shrink, impl)))


def build_qwen2(impl: str) -> nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    shrink = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "use_cache": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    torch.manual_seed(SEED)
    return _seeded(Qwen2ForCausalLM(_hf_config("qwen2", Qwen2Config, shrink, impl)))


def build_albert(impl: str) -> nn.Module:
    from transformers import AlbertConfig, AlbertModel

    shrink = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 64,
        "embedding_size": 32,
        "intermediate_size": 128,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    torch.manual_seed(SEED)
    return _seeded(AlbertModel(_hf_config("albert", AlbertConfig, shrink, impl)))


def build_distilbert(impl: str) -> nn.Module:
    from transformers import DistilBertConfig, DistilBertModel

    shrink = {
        "n_layers": 2,
        "n_heads": 2,
        "dim": 64,
        "hidden_dim": 128,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
    }
    torch.manual_seed(SEED)
    return _seeded(DistilBertModel(_hf_config("distilbert", DistilBertConfig, shrink, impl)))


def build_bert(impl: str) -> nn.Module:
    # bert_uncased_L-2_H-128_A-2 is ALREADY tiny (2 layers, H=128): the one
    # roster family whose real config builds unshrunk; only the vocab is cut
    # so the fixed-seed embedding stays small.
    from transformers import BertConfig, BertModel

    shrink = {"vocab_size": VOCAB, "max_position_embeddings": 64, "pad_token_id": 0}
    torch.manual_seed(SEED)
    return _seeded(BertModel(_hf_config("bert", BertConfig, shrink, impl)))


def build_t5(impl: str) -> nn.Module:
    from transformers import T5Config, T5ForConditionalGeneration

    shrink = {
        "num_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "d_model": 64,
        "d_ff": 128,
        "d_kv": 16,
        "vocab_size": VOCAB,
        "use_cache": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
    }
    torch.manual_seed(SEED)
    return _seeded(T5ForConditionalGeneration(_hf_config("t5", T5Config, shrink, impl)))


def build_vit(impl: str) -> nn.Module:
    from transformers import ViTConfig, ViTModel

    shrink = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "image_size": 32,
        "patch_size": 16,
    }
    torch.manual_seed(SEED)
    return _seeded(ViTModel(_hf_config("vit", ViTConfig, shrink, impl)))


def build_clip(impl: str) -> nn.Module:
    from transformers import CLIPConfig, CLIPModel

    payload = dict(vendored_config("clip"))
    payload["text_config"] = dict(
        payload.get("text_config", {}),
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    payload["vision_config"] = dict(
        payload.get("vision_config", {}),
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        image_size=32,
        patch_size=16,
    )
    payload["projection_dim"] = 32
    config = CLIPConfig.from_dict(payload)
    config._attn_implementation = impl
    torch.manual_seed(SEED)
    return _seeded(CLIPModel(config))


def build_whisper(impl: str) -> nn.Module:
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    shrink = {
        "encoder_layers": 2,
        "decoder_layers": 2,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "d_model": 64,
        "encoder_ffn_dim": 128,
        "decoder_ffn_dim": 128,
        "vocab_size": VOCAB,
        "num_mel_bins": 16,
        "max_source_positions": 32,
        "max_target_positions": 32,
        "use_cache": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 1,
        "suppress_tokens": [],
        "begin_suppress_tokens": [],
        "forced_decoder_ids": None,
    }
    torch.manual_seed(SEED)
    return _seeded(
        WhisperForConditionalGeneration(_hf_config("whisper", WhisperConfig, shrink, impl))
    )


def build_mamba(impl: str) -> nn.Module:
    from transformers import MambaConfig, MambaForCausalLM

    shrink = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "state_size": 8,
        "vocab_size": VOCAB,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": None,
        "use_cache": False,
    }
    torch.manual_seed(SEED)
    return _seeded(MambaForCausalLM(_hf_config("mamba", MambaConfig, shrink, impl)))


def build_rwkv(impl: str) -> nn.Module:
    from transformers import RwkvConfig, RwkvForCausalLM

    shrink = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "attention_hidden_size": 64,
        "intermediate_size": 128,
        "vocab_size": VOCAB,
        "context_length": 64,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "use_cache": False,
    }
    torch.manual_seed(SEED)
    return _seeded(RwkvForCausalLM(_hf_config("rwkv", RwkvConfig, shrink, impl)))


def build_detection(impl: str) -> nn.Module:
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

    torch.manual_seed(SEED)
    return _seeded(
        fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=3, min_size=64, max_size=64
        )
    )


class RNNStack(nn.Module):
    """LSTM+GRU stack (roster family: recurrent torch natives)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32)
        self.lstm = nn.LSTM(32, 32, num_layers=2, batch_first=True)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(self.emb(input_ids))
        hidden, _ = self.gru(hidden)
        return self.head(hidden)


def build_lstm_gru(impl: str) -> nn.Module:
    torch.manual_seed(SEED)
    return _seeded(RNNStack())


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "gpt2",
        "r0-gpt2",
        build_gpt2,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "distilgpt2",
        "r0-distilgpt2",
        build_distilgpt2,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "llama",
        "r0-llama",
        build_llama,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 4, "kv_heads": 2, "d_head": 16, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "qwen2",
        "r0-qwen2",
        build_qwen2,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 4, "kv_heads": 2, "d_head": 16, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "albert",
        "r0-albert",
        build_albert,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "distilbert",
        "r0-distilbert",
        build_distilbert,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "bert",
        "r0-bert",
        build_bert,
        lambda: {"input_ids": _token_ids()},
        ("eager", "sdpa"),
        {"hidden": 128, "heads": 2, "d_head": 64, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "t5",
        "r0-t5",
        build_t5,
        lambda: {"input_ids": _token_ids(), "decoder_input_ids": _token_ids()},
        ("eager",),
        {"hidden": 64, "heads": 2, "d_head": 16, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "vit",
        "r0-vit",
        build_vit,
        lambda: {"pixel_values": _pixels()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": 5, "vocab": 0},
    ),
    FamilySpec(
        "clip",
        "r0-clip",
        build_clip,
        lambda: {"input_ids": _token_ids(), "pixel_values": _pixels()},
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "whisper",
        "r0-whisper",
        build_whisper,
        lambda: {
            "input_features": torch.randn(
                BATCH, 16, 64, generator=torch.Generator().manual_seed(SEED)
            ),
            "decoder_input_ids": _token_ids(),
        },
        ("eager", "sdpa"),
        {"hidden": 64, "heads": 2, "d_head": 32, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "mamba",
        "r0-mamba",
        build_mamba,
        lambda: {"input_ids": _token_ids()},
        ("eager",),
        {"hidden": 64, "heads": 0, "d_head": 0, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "rwkv",
        "r0-rwkv",
        build_rwkv,
        lambda: {"input_ids": _token_ids()},
        ("eager",),
        {"hidden": 64, "heads": 0, "d_head": 0, "seq": SEQ, "vocab": VOCAB},
    ),
    FamilySpec(
        "detection",
        "r0-detection",
        build_detection,
        lambda: {"images": [torch.randn(3, 64, 64, generator=torch.Generator().manual_seed(SEED))]},
        ("eager",),
        {"hidden": 0, "heads": 0, "d_head": 0, "seq": 0, "vocab": 0},
    ),
    FamilySpec(
        "lstm_gru",
        "r0-lstm-gru",
        build_lstm_gru,
        lambda: {},
        ("eager",),
        {"hidden": 32, "heads": 0, "d_head": 0, "seq": SEQ, "vocab": VOCAB},
        input_args=lambda: (_token_ids(),),
    ),
)

FAMILY_BY_NAME = {spec.name: spec for spec in FAMILIES}


# --- structural fixtures (memo 4.1: "Plus tied-weight, multi-input, in-place,
# lazy-parameter, container-output, and train-mode-BatchNorm fixtures.") ---


class TiedWeightLM(nn.Module):
    """Embedding/unembedding weight tying (one Parameter, two consumers)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32)
        self.body = nn.Linear(32, 32)
        self.unembed = nn.Linear(32, VOCAB, bias=False)
        self.unembed.weight = self.emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.unembed(torch.tanh(self.body(self.emb(input_ids))))


class MultiInput(nn.Module):
    """Two positional tensors plus one keyword tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_a = nn.Linear(16, 16)
        self.proj_b = nn.Linear(16, 16)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.proj_a(a) + self.proj_b(b)
        if scale is not None:
            out = out * scale
        return out


class InPlaceNet(nn.Module):
    """In-place op realism: relu_ and add_ on the live activation."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        out.relu_()
        out.add_(1.0)
        return out


class ContainerOutput(nn.Module):
    """Mixed container output: dict holding a tensor, a list, and a tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        out = self.lin(x)
        return {"main": out, "parts": [out * 2, out * 3], "pair": (out.sum(), out.mean())}


class TrainBN(nn.Module):
    """Train-mode BatchNorm: running stats mutate during capture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


def build_structural(name: str) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any]]:
    """Build one structural fixture: (model, input_args, input_kwargs)."""

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    if name == "tied-weight":
        return TiedWeightLM().eval(), (_token_ids(),), {}
    if name == "multi-input":
        a = torch.randn(2, 16, generator=generator)
        b = torch.randn(2, 16, generator=generator)
        scale = torch.randn(2, 16, generator=generator)
        return MultiInput().eval(), (a, b), {"scale": scale}
    if name == "in-place":
        return InPlaceNet().eval(), (torch.randn(2, 16, generator=generator),), {}
    if name == "lazy-param":
        model = nn.Sequential(nn.LazyLinear(16), nn.ReLU(), nn.Linear(16, 4))
        x = torch.randn(2, 8, generator=generator)
        with torch.no_grad():
            model(x)  # materialize the lazy parameter BEFORE capture
        return model.eval(), (x,), {}
    if name == "container-output":
        return ContainerOutput().eval(), (torch.randn(2, 16, generator=generator),), {}
    if name == "train-bn":
        model = TrainBN().train()
        return model, (torch.randn(2, 3, 8, 8, generator=generator),), {}
    raise ValueError(f"unknown structural fixture {name!r}")


STRUCTURAL_FIXTURES = (
    "tied-weight",
    "multi-input",
    "in-place",
    "lazy-param",
    "container-output",
    "train-bn",
)
