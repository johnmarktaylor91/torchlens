"""U36.2 deployment legs for the TorchLens HF attention backend, on CPU.

device_map dispatch, disk offload and bitsandbytes quantization: each loads
the same saved grouped-query checkpoint twice (sdpa and torchlens), checks
every recorded call against SDPA at the semantic reconstruction tolerance,
the probabilities against the independent reference of
``tests/test_hf_attention_backend.py``, and the logits against the sdpa load
at the existing SDPA ``z`` tolerance (atol 1e-5, rtol 1e-4).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from transformers import AttentionInterface, AutoModelForCausalLM  # noqa: E402
from transformers.masking_utils import AttentionMaskInterface  # noqa: E402

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402
from tests.test_hf_attention_backend import (  # noqa: E402
    _assert_matches_reference,
    _inputs,
    _keep_mask,
    _Spy,
)
from torchlens.bridge import hf as tl_hf  # noqa: E402
from torchlens.bridge.hf_attention import BACKEND_KEY  # noqa: E402

DISK_MAP = {
    "model.embed_tokens": "cpu",
    "model.rotary_emb": "cpu",
    "model.layers.0": "cpu",
    "model.layers.1": "disk",
    "model.norm": "cpu",
    "lm_head": "cpu",
}


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    """Register the backend and restore both registries after each test."""

    attention_before = dict(AttentionInterface._global_mapping)
    mask_before = dict(AttentionMaskInterface._global_mapping)
    tl_hf.register_attention()
    yield
    AttentionInterface._global_mapping.clear()
    AttentionInterface._global_mapping.update(attention_before)
    AttentionMaskInterface._global_mapping.clear()
    AttentionMaskInterface._global_mapping.update(mask_before)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved grouped-query llama checkpoint (4 query heads, 2 key/value heads)."""

    path = tmp_path_factory.mktemp("tl_hf_attention_ckpt")
    FAMILY_BY_NAME["llama"].build("sdpa").save_pretrained(path)
    return path


def _load(path: Path, impl: str, **kwargs: Any) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation=impl, **kwargs)
    return model.eval()


def _check_leg(reference: torch.nn.Module, model: torch.nn.Module, atol: float = 1e-5) -> None:
    """Reference, SDPA and logits checks for one loaded leg."""

    spy = _Spy()
    AttentionInterface._global_mapping[BACKEND_KEY] = spy
    batch = _inputs("left_pad")
    with torch.no_grad():
        expected = reference(**batch).logits
        with tl_hf.record_attention(model, check_sdpa=True) as recording:
            logits = model(**batch).logits
    assert len(recording.calls) == model.config.num_hidden_layers
    assert all(call.sdpa_checked for call in recording.calls)
    for recorded_call, call in zip(recording.calls, spy.calls, strict=True):
        q_len, kv_len = call["q"].shape[-2], call["k"].shape[-2]
        keep = _keep_mask(batch["attention_mask"], q_len, kv_len, causal=True)
        _assert_matches_reference(recorded_call.probs, call, keep)
    valid = batch["attention_mask"].bool()
    torch.testing.assert_close(logits[valid], expected[valid], atol=atol, rtol=1e-4)


def test_device_map_cpu_leg(checkpoint: Path) -> None:
    """from_pretrained(device_map="cpu", attn_implementation="torchlens")."""

    reference = _load(checkpoint, "sdpa")
    model = _load(checkpoint, BACKEND_KEY, device_map="cpu")
    assert {p.device.type for p in model.parameters()} == {"cpu"}
    _check_leg(reference, model)


def test_accelerate_cpu_offload_leg(checkpoint: Path) -> None:
    """accelerate.cpu_offload: weights stream in through hooks on every forward."""

    import accelerate

    reference = _load(checkpoint, "sdpa")
    model = accelerate.cpu_offload(
        _load(checkpoint, BACKEND_KEY), execution_device=torch.device("cpu")
    )
    assert hasattr(model.model.layers[0].self_attn.q_proj, "_hf_hook")
    _check_leg(reference, model)


def test_disk_offload_leg(checkpoint: Path, tmp_path: Path) -> None:
    """A device_map with one decoder layer offloaded to disk."""

    reference = _load(checkpoint, "sdpa")
    model = _load(checkpoint, BACKEND_KEY, device_map=DISK_MAP, offload_folder=str(tmp_path))
    assert "disk" in set(model.hf_device_map.values())
    _check_leg(reference, model)


@pytest.mark.parametrize("bits", [8, 4])
def test_bitsandbytes_quantized_leg(checkpoint: Path, bits: int) -> None:
    """bitsandbytes 8/4-bit linears on CPU: sdpa and torchlens loads agree."""

    pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig

    if bits == 8:
        quant = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float32)
    reference = _load(checkpoint, "sdpa", quantization_config=quant, device_map="cpu")
    model = _load(checkpoint, BACKEND_KEY, quantization_config=quant, device_map="cpu")
    _check_leg(reference, model)
