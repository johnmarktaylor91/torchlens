"""TorchLens as a registered Hugging Face attention backend (U36.2).

Per-head probabilities recorded by the backend are checked against an
INDEPENDENT eager reference: float64 QK-softmax built in this file from the
spied query/key tensors and a mask built here from the user's 2D padding mask
and the causal flag (never from the 4D mask transformers hands the backend),
with grouped-query heads expanded by ``repeat_interleave``. The SDPA legs
check every call against ``scaled_dot_product_attention`` at the semantic
reconstruction tolerance (``torchlens.semantic.tolerances``) and the model
output against the sdpa backend at the existing SDPA ``z`` tolerance
(``tests/test_sem_attn_gpt2_sdpa.py``: atol 1e-5, rtol 1e-4).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import AttentionInterface  # noqa: E402
from transformers.masking_utils import AttentionMaskInterface  # noqa: E402

from tests.real_model.r0.families import FAMILY_BY_NAME, VOCAB  # noqa: E402
from torchlens._errors import InvalidArgumentError, MissingDependencyError  # noqa: E402
from torchlens.bridge import hf as tl_hf  # noqa: E402
from torchlens.bridge.hf_attention import (  # noqa: E402
    BACKEND_KEY,
    _check_against_sdpa,
    full_matrix_bytes,
    torchlens_attention_forward,
)
from torchlens.errors import ValidationError  # noqa: E402
from torchlens.semantic.tolerances import within_reconstruction_tolerance  # noqa: E402

SEQ = 7
#: family -> (builder name, causal decoder?, attention modules suffix)
FAMILIES = {
    "gpt2": ("gpt2", True),
    "llama_gqa": ("llama", True),
    "qwen2_gqa": ("qwen2", True),
    "bert": ("bert", False),
}
MASKS = ("none", "right_pad", "left_pad")


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    """Register the backend for each test and restore both registries after."""

    attention_before = dict(AttentionInterface._global_mapping)
    mask_before = dict(AttentionMaskInterface._global_mapping)
    tl_hf.register_attention()
    yield
    AttentionInterface._global_mapping.clear()
    AttentionInterface._global_mapping.update(attention_before)
    AttentionMaskInterface._global_mapping.clear()
    AttentionMaskInterface._global_mapping.update(mask_before)


class _Spy:
    """Wrap the registered backend to keep each call's query/key/kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, module, query, key, value, attention_mask, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(
            {"module": module, "q": query.detach(), "k": key.detach(), "kwargs": kwargs}
        )
        return torchlens_attention_forward(module, query, key, value, attention_mask, **kwargs)


@pytest.fixture
def spy() -> Iterator[_Spy]:
    """Install the spy under the backend key for one test."""

    recorder = _Spy()
    AttentionInterface._global_mapping[BACKEND_KEY] = recorder
    yield recorder


def _inputs(mask_case: str) -> dict[str, torch.Tensor]:
    """Batch of two sequences; the second is padded per ``mask_case``."""

    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(2, VOCAB, (2, SEQ), generator=generator)
    mask = torch.ones(2, SEQ, dtype=torch.long)
    if mask_case == "right_pad":
        mask[1, -3:] = 0
    elif mask_case == "left_pad":
        mask[1, :3] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _build(family: str, impl: str) -> torch.nn.Module:
    return FAMILY_BY_NAME[FAMILIES[family][0]].build(impl)


def _reference_probs(
    q: torch.Tensor, k: torch.Tensor, scale: float, keep: torch.Tensor, bias: Any
) -> torch.Tensor:
    """Independent float64 masked QK-softmax (masked entries exactly zero)."""

    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q.double(), k.double().transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias.double()
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1)


def _keep_mask(pad: torch.Tensor, q_len: int, kv_len: int, causal: bool) -> torch.Tensor:
    """(batch, 1, q, kv) boolean keep-mask from the 2D padding mask alone."""

    keep = pad.bool()[:, None, None, :kv_len].expand(-1, 1, q_len, kv_len)
    if causal:
        offset = kv_len - q_len
        rows = torch.arange(q_len)[:, None] + offset
        keep = keep & (torch.arange(kv_len)[None, :] <= rows)
    return keep


def _assert_matches_reference(recorded: torch.Tensor, call: dict[str, Any], keep) -> None:
    """Recorded probabilities equal the independent reference on defined rows."""

    scale = call["kwargs"].get("scaling") or call["q"].shape[-1] ** -0.5
    reference = _reference_probs(
        call["q"], call["k"], scale, keep, call["kwargs"].get("position_bias")
    )
    defined = keep.any(dim=-1)  # (batch, 1, q)
    rows = defined.expand(-1, recorded.shape[1], -1)
    assert within_reconstruction_tolerance(recorded[rows], reference[rows].to(recorded.dtype))
    # Masked entries carry exactly zero probability, never a min-dtype residue.
    masked = (~keep).expand_as(recorded) & rows[..., None]
    assert torch.count_nonzero(recorded[masked]) == 0


@pytest.mark.parametrize("mask_case", MASKS)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_probs_match_independent_reference(family: str, mask_case: str, spy: _Spy) -> None:
    """Per-head probabilities equal the independent masked QK-softmax."""

    model = _build(family, "sdpa")
    tl_hf.register_attention(model)
    batch = _inputs(mask_case)
    with torch.no_grad(), tl_hf.record_attention(model) as recording:
        model(**batch)
    causal = FAMILIES[family][1]
    assert len(recording.calls) == len(spy.calls) == model.config.num_hidden_layers
    for recorded_call, call in zip(recording.calls, spy.calls, strict=True):
        q_len, kv_len = call["q"].shape[-2], call["k"].shape[-2]
        keep = _keep_mask(batch["attention_mask"], q_len, kv_len, causal)
        _assert_matches_reference(recorded_call.probs, call, keep)
        assert recorded_call.num_key_value_heads == call["k"].shape[1]
    if family.endswith("_gqa"):
        assert spy.calls[0]["k"].shape[1] < spy.calls[0]["q"].shape[1]


@pytest.mark.parametrize("mask_case", MASKS)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_output_matches_sdpa(family: str, mask_case: str) -> None:
    """Every call passes the SDPA check; model output equals the sdpa backend."""

    model = _build(family, "sdpa")
    batch = _inputs(mask_case)
    with torch.no_grad():
        sdpa_out = model(**batch)[0]
        tl_hf.register_attention(model)
        with tl_hf.record_attention(model, check_sdpa=True) as recording:
            tl_out = model(**batch)[0]
    assert recording.calls and all(call.sdpa_checked for call in recording.calls)
    valid = batch["attention_mask"].bool()
    torch.testing.assert_close(tl_out[valid], sdpa_out[valid], atol=1e-5, rtol=1e-4)


def test_attention_only_registration_is_the_mask_trap() -> None:
    """Tripwire for the pairing: without the mask function padding leaks in.

    Registering the attention function WITHOUT a mask function makes
    transformers pass no mask at all. The backend still applies causality
    itself, but padded keys receive probability; with the paired
    registration they receive exactly zero.
    """

    AttentionInterface.register("tl_trap_demo", torchlens_attention_forward)
    model = _build("gpt2", "sdpa")
    batch = _inputs("left_pad")
    pad_keys = ~batch["attention_mask"].bool()[1]
    with torch.no_grad():
        model.set_attn_implementation("tl_trap_demo")
        trapped = model(**batch, output_attentions=True).attentions[0]
        tl_hf.register_attention(model)
        paired = model(**batch, output_attentions=True).attentions[0]
    valid_rows = batch["attention_mask"].bool()[1]
    assert torch.count_nonzero(trapped[1][:, valid_rows][..., pad_keys]) > 0
    assert torch.count_nonzero(paired[1][:, valid_rows][..., pad_keys]) == 0
    future = torch.ones(SEQ, SEQ, dtype=torch.bool).triu(1)
    assert torch.count_nonzero(trapped[0][:, future]) == 0  # causality held by the backend


def test_register_is_idempotent_and_pairs_the_mask() -> None:
    """One call registers both functions under the same key, repeatably."""

    assert tl_hf.register_attention() == BACKEND_KEY == tl_hf.register_attention()
    assert BACKEND_KEY in AttentionInterface._global_mapping
    assert BACKEND_KEY in AttentionMaskInterface._global_mapping
    assert "sdpa" in AttentionInterface._global_mapping  # built-ins untouched
    assert AttentionInterface._global_mapping["sdpa"] is not torchlens_attention_forward


def test_register_refuses_a_foreign_key() -> None:
    """A foreign function under the key refuses typed instead of overwriting."""

    AttentionInterface._global_mapping[BACKEND_KEY] = lambda *a, **k: None
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl_hf.register_attention()
    assert excinfo.value.fields["code"] == "hf_attention_key_taken"


def test_register_refuses_without_the_mask_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """No mask registry means no paired registration: refuse, never half-register."""

    monkeypatch.setitem(sys.modules, "transformers.masking_utils", None)
    with pytest.raises(MissingDependencyError) as excinfo:
        tl_hf.register_attention()
    assert excinfo.value.fields["code"] == "hf_attention_registry_unavailable"


def test_record_refuses_a_model_on_another_backend() -> None:
    """Recording a model left on sdpa refuses instead of staying silently empty."""

    model = _build("gpt2", "sdpa")
    with pytest.raises(InvalidArgumentError) as excinfo, tl_hf.record_attention(model):
        pass
    assert excinfo.value.fields["code"] == "hf_attention_backend_not_selected"


def test_repeated_calls_refuse_single_read_and_serve_calls_for() -> None:
    """Two forwards in one recording: [] refuses typed, calls_for serves both."""

    model = _build("llama_gqa", "sdpa")
    tl_hf.register_attention(model)
    ids = torch.randint(0, VOCAB, (1, SEQ))
    with torch.no_grad(), tl_hf.record_attention(model) as recording:
        model(input_ids=ids)
        model(input_ids=ids)
    name = next(iter(recording))
    assert name == "model.layers.0.self_attn"
    with pytest.raises(InvalidArgumentError) as excinfo:
        recording[name]
    assert excinfo.value.fields["code"] == "hf_attention_multiple_calls"
    first, second = recording.calls_for(name)
    assert torch.equal(first.probs, second.probs)


def test_output_attentions_returns_the_recorded_probs() -> None:
    """output_attentions=True under the backend returns the recorded tensors."""

    model = _build("qwen2_gqa", "sdpa")
    tl_hf.register_attention(model)
    batch = _inputs("right_pad")
    with torch.no_grad(), tl_hf.record_attention(model) as recording:
        out = model(**batch, output_attentions=True)
    assert len(out.attentions) == len(recording.calls)
    for returned, call in zip(out.attentions, recording.calls, strict=True):
        assert torch.equal(returned, call.probs)


def test_nbytes_matches_the_disclosed_formula() -> None:
    """The retained bytes equal batch*heads*q*kv*itemsize*layers."""

    model = _build("llama_gqa", "sdpa")
    tl_hf.register_attention(model)
    batch = _inputs("none")
    with torch.no_grad(), tl_hf.record_attention(model) as recording:
        model(**batch)
    cfg = model.config
    expected = full_matrix_bytes(
        batch=2, heads=cfg.num_attention_heads, q_len=SEQ, kv_len=SEQ,
        layers=cfg.num_hidden_layers, dtype=torch.float32,
    )  # fmt: skip
    assert recording.nbytes == expected
    # The disclosed 7B example: 32 layers x 32 heads at 4096 tokens in bf16.
    assert (
        full_matrix_bytes(
            batch=1, heads=32, q_len=4096, kv_len=4096, layers=32, dtype=torch.bfloat16
        )
        == 32 * 2**30
    )


def test_generate_with_cache_records_every_decode_step(spy: _Spy) -> None:
    """KV-cache decoding: one call per layer per step, each matching the reference."""

    model = _build("llama_gqa", "sdpa")
    model.config.use_cache = True
    model.generation_config.pad_token_id = 0
    tl_hf.register_attention(model)
    batch = _inputs("left_pad")
    new_tokens = 3
    with torch.no_grad(), tl_hf.record_attention(model) as recording:
        model.generate(**batch, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                       do_sample=False)  # fmt: skip
    layers = model.config.num_hidden_layers
    assert len(recording.calls) == layers * (1 + new_tokens)
    full_pad = torch.cat([batch["attention_mask"], torch.ones(2, new_tokens, dtype=torch.long)], 1)
    for recorded_call, call in zip(recording.calls, spy.calls, strict=True):
        q_len, kv_len = call["q"].shape[-2], call["k"].shape[-2]
        keep = _keep_mask(full_pad, q_len, kv_len, causal=True)
        _assert_matches_reference(recorded_call.probs, call, keep)


def test_unsupported_sparse_kwargs_refuse_typed() -> None:
    """Sparse-index attention kwargs refuse instead of computing dense math."""

    module = torch.nn.Module()
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(InvalidArgumentError) as excinfo:
        torchlens_attention_forward(module, q, q, q, None, indices=torch.zeros(1))
    assert excinfo.value.fields["code"] == "hf_attention_kwarg_unsupported"


def test_sdpa_check_is_live_on_a_corrupted_output() -> None:
    """Tripwire: the SDPA check refuses a sign-flipped attention output."""

    q = torch.randn(1, 2, 4, 8)
    probs = torch.softmax(q @ q.transpose(-1, -2), dim=-1)
    out = probs @ q
    facts = {"attention_mask": None, "mask_kind": "none", "scale": 1.0, "position_bias": None}
    _check_against_sdpa("m", q, q, q, out, probs, facts)
    with pytest.raises(ValidationError) as excinfo:
        _check_against_sdpa("m", q, q, q, -out, probs, facts)
    assert excinfo.value.fields["code"] == "hf_attention_sdpa_mismatch"


def test_t5_position_bias_and_cross_attention(spy: _Spy) -> None:
    """T5: relative position bias, causal decoder and cross attention."""

    model = FAMILY_BY_NAME["t5"].build("sdpa")
    tl_hf.register_attention(model)
    batch = _inputs("right_pad")
    decoder_ids = torch.randint(0, VOCAB, (2, 5))
    with torch.no_grad(), tl_hf.record_attention(model, check_sdpa=True) as recording:
        model(**batch, decoder_input_ids=decoder_ids)
    assert all(call.sdpa_checked for call in recording.calls)
    for recorded_call, call in zip(recording.calls, spy.calls, strict=True):
        q_len, kv_len = call["q"].shape[-2], call["k"].shape[-2]
        is_decoder_self = (
            recorded_call.module_name.startswith("decoder")
            and q_len == kv_len
            and ("EncDecAttention" not in recorded_call.module_name)
        )
        pad = batch["attention_mask"] if kv_len == SEQ else torch.ones(2, kv_len)
        keep = _keep_mask(pad, q_len, kv_len, causal=is_decoder_self)
        _assert_matches_reference(recorded_call.probs, call, keep)
