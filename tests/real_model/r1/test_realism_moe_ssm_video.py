"""Real MoE, SSM and video checkpoints: real weights, capture, validation.

forks.3 adds one small mixture-of-experts (Switch Transformer, 8 experts,
top-1 routing) and one small state-space model (Mamba 130M); U55.4's CPU
fallback adds one video transformer (VideoMAE small, Kinetics-400). Each test
first proves the registry checkpoint's REAL weights loaded: an exact SHA-256
of one named weight tensor, then the checkpoint's known output on a fixed
input. A config-built random-weight twin fails both. Only then does it check
that the traced output equals the direct forward and that forward replay
validation passes.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

# Measured in the pinned CPU venv (torch 2.13.0, transformers 5.18.0): capture plus
# forward validation takes 24-55 s per checkpoint, so the module is slow-tier.
pytestmark = [pytest.mark.slow, pytest.mark.real_model, pytest.mark.real_checkpoint]

MAMBA_EMBEDDINGS_SHA256 = (
    "69a427766e7b1661d7d84f3c5519cafa7ea709a876d207308f0b1bc0c254ed20"  # pragma: allowlist secret
)
MAMBA_PROMPT = "The capital of France is"
MAMBA_NEXT_TOKEN = " in"

SWITCH_SHARED_SHA256 = (
    "228d5f09066d373d9d55f23e7223ad63c4d0e85a43d794f9bf02c922e1ad5686"  # pragma: allowlist secret
)
SWITCH_SPAN_INPUT = "The <extra_id_0> walks in <extra_id_1> park."
SWITCH_GREEDY_IDS = [0, 32099, 200, 32098, 8, 32097, 5]

VIDEOMAE_PATCH_SHA256 = (
    "4155f19e0ecddb1852a8e9e663de1e158caea9d21423d425a76515d1c0a5026f"  # pragma: allowlist secret
)
VIDEOMAE_TOP1 = 346  # "tai chi" on the seeded clip


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _capture_matches_direct_and_validates(model, inputs: dict, direct: torch.Tensor, extra=None):
    """Trace, check traced output == direct forward, validate; return the trace facts."""

    import torchlens as tl

    trace = tl.trace(model, (), dict(inputs))
    try:
        if extra is not None:
            extra(trace)
        assert trace.outcome.status.name == "COMPLETE"
        assert any(
            isinstance(op.out, torch.Tensor)
            and op.out.shape == direct.shape
            and torch.equal(op.out, direct)
            for op in trace.output_ops
        ), "traced output != direct forward on the real checkpoint"
        called = {module.address for module in trace.modules if module.num_calls > 0}
        uncalled = {str(module) for module in trace.uncalled_modules}
    finally:
        trace.cleanup()
    assert tl.validate(model, (), dict(inputs), scope="forward") is True
    return called, uncalled


def test_mamba_130m_real_weights_capture_and_validate(r1_loader):
    loaded = r1_loader("r1-mamba-130m-hf")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    embeddings = model.get_parameter("backbone.embeddings.weight")
    assert _tensor_sha256(embeddings) == MAMBA_EMBEDDINGS_SHA256, "not the pinned real weights"
    encoded = tokenizer(MAMBA_PROMPT, return_tensors="pt")
    with torch.no_grad():
        direct = model(**encoded).logits
    assert tokenizer.decode([int(direct[0, -1].argmax())]) == MAMBA_NEXT_TOKEN
    called, _ = _capture_matches_direct_and_validates(model, dict(encoded), direct)
    mixers = {address for address in called if address.endswith(".mixer")}
    assert len(mixers) == model.config.num_hidden_layers, (
        f"expected one selective-scan mixer call per layer, saw {sorted(mixers)}"
    )


# Peak RSS 16.7 GB measured: over the 16 GB hosted runner, so a large-memory venue runs it.
@pytest.mark.big_memory
def test_switch_base_8_real_weights_sparse_routing_capture_and_validate(r1_loader):
    loaded = r1_loader("r1-switch-base-8")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    assert _tensor_sha256(model.get_parameter("shared.weight")) == SWITCH_SHARED_SHA256
    encoded = tokenizer(SWITCH_SPAN_INPUT, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=6, do_sample=False)
    assert generated[0].tolist() == SWITCH_GREEDY_IDS
    inputs = {**dict(encoded), "decoder_input_ids": generated[:, :4]}
    routed = set()
    num_experts = model.config.num_experts

    def _record_routed(router_name: str):
        def _hook(_module, _args, output) -> None:
            # The router's first output is the one-hot (capacity-masked) expert choice per token.
            selected = output[0]
            assert selected.shape[-1] == num_experts, "router output contract changed"
            chosen = selected.reshape(-1, num_experts).any(dim=0).nonzero().flatten()
            expert_prefix = router_name.removesuffix(".router") + ".experts.expert_"
            routed.update(f"{expert_prefix}{int(index)}" for index in chosen)

        return _hook

    handles = [
        module.register_forward_hook(_record_routed(name))
        for name, module in model.named_modules()
        if name.endswith(".mlp.router")
    ]
    try:
        with torch.no_grad():
            direct = model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()
    called, _ = _capture_matches_direct_and_validates(model, inputs, direct)
    # Compare expert modules only: a called expert's children (wi, wo, act, dropout) are not
    # experts, so counting them would make the subset check fail even when routing is sparse.
    every_expert = {
        name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1].startswith("expert_")
    }
    experts = called & every_expert
    assert experts, "no expert MLP ran: routing never reached an expert"
    assert experts < every_expert, "every expert ran on a short input: routing is not sparse"
    assert experts == routed, "captured expert calls differ from the router's top-1 choices"


def test_videomae_small_real_weights_tubelet_capture_and_validate(r1_loader):
    """CPU CI video row (U55.4 fallback): 16 frames -> 8 x 14 x 14 = 1568 tubelet tokens."""

    model = r1_loader("r1-videomae-small-kinetics")["model"]
    patch = model.get_parameter("videomae.embeddings.patch_embeddings.projection.weight")
    assert _tensor_sha256(patch) == VIDEOMAE_PATCH_SHA256, "not the pinned real weights"
    generator = torch.Generator().manual_seed(0)
    inputs = {"pixel_values": torch.rand(1, 16, 3, 224, 224, generator=generator)}
    with torch.no_grad():
        direct = model(**inputs).logits
    assert int(direct[0].argmax()) == VIDEOMAE_TOP1
    tokens = (16 // model.config.tubelet_size) * (224 // model.config.patch_size) ** 2

    def _has_tubelet_sequence(trace) -> None:
        assert any(
            isinstance(op.out, torch.Tensor)
            and tuple(op.out.shape) == (1, tokens, model.config.hidden_size)
            for op in trace.ops
        ), f"no op carried the {tokens}-token tubelet sequence"

    _capture_matches_direct_and_validates(model, inputs, direct, extra=_has_tubelet_sequence)
