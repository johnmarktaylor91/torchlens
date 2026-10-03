"""A05 escrow-spill fix: the four-panel red tests (leverage B2, brainpipe 2,
transforms P7, tvscope B13).

Root cause (review T3, conceded 3-0): deferred final-numbering selectors (op
labels, positive ordinals, ``output``) escrow every candidate payload and
spill past the 64 MiB RAM budget via ``torch.save``; a captured payload's
``TensorMeta.label_storage`` pins its own ``UntypedStorage``, rides the
tensor's ``__dict__`` into the pickle, and the writer refuses two views of
one storage under different types ("Cannot save multiple tensors or storages
that view the same data as different types"). The fix strips the TorchLens
sidecar from the exclusively-owned payload at the spill line -- NO clone
(review A12: cloning is a diagnostic, not the fix).

Realism axis 4 (BELOW THE MODEL): every arm here escrows tensors that came
out of a real capture, never a bare constructor -- the bare-constructor
spelling is exactly what made the historical guarding test un-fail-able.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens.capture.plan import CapturePlan, RetentionKind, RetentionProfile
from torchlens.capture.session import CaptureSession

TORCH_HUB_CHECKPOINTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
HF_HUB_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def _require_torchvision_checkpoint(filename: str) -> None:
    """Skip out of venue: real-weights arms need the cached checkpoint file."""

    if not (TORCH_HUB_CHECKPOINTS / filename).exists():
        pytest.skip(
            f"GATE A05_REAL_WEIGHTS: torchvision checkpoint {filename} not cached;"
            " fetch it once online, then rerun offline."
        )


def _require_hf_snapshot(repo_dirname: str) -> None:
    """Skip out of venue: real-checkpoint HF arms need the cached snapshot."""

    if not (HF_HUB_CACHE / repo_dirname).exists():
        pytest.skip(
            f"GATE A05_REAL_WEIGHTS: HF snapshot {repo_dirname} not cached;"
            " fetch it once online, then rerun offline."
        )


def _eager_module_output(model: torch.nn.Module, module_path: str, *args, **kwargs):
    """Independent oracle: the named module's output from a bare eager forward."""

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["value"] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return None

    handle = dict(model.named_modules())[module_path].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        handle.remove()
    return captured["value"]


@pytest.fixture
def force_spill(monkeypatch):
    """Compile every capture plan with a 1-byte escrow RAM budget.

    Tiny-model arms cross the spill line with REAL capture payloads without
    needing >64 MiB of activations; the real-weights arms cross the default
    budget at real scale.
    """

    real_compile = CapturePlan.compile.__func__

    def tiny_budget_compile(cls, **kwargs):
        plan = real_compile(cls, **kwargs)
        return dataclasses.replace(
            plan,
            retention_profile=dataclasses.replace(
                plan.retention_profile, activation_ram_budget_bytes=1
            ),
        )

    monkeypatch.setattr(CapturePlan, "compile", classmethod(tiny_budget_compile))


class _TwoConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.relu(self.conv1(x)))


@pytest.mark.smoke
def test_spilled_real_capture_payload_materializes_bit_exact(force_spill):
    """Every escrowed real-capture payload survives the forced spill bit-exactly."""

    torch.manual_seed(0)
    model = _TwoConv().eval()
    x = torch.randn(2, 3, 8, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=["conv2d_2_3"]))
    oracle = _eager_module_output(model, "conv2", x)
    assert torch.equal(log["conv2d_2_3"].out, oracle), (
        "spilled escrow payload is not bit-exact against the eager module output"
    )


@pytest.mark.smoke
def test_spill_artifact_carries_no_torchlens_sidecar():
    """The spill file holds ONLY the payload: weights_only load, no ``_tl``.

    B2's negative control at the session level with a sidecar-bearing tensor
    (the shape every real capture produces): without the sidecar-stripping
    fix, ``torch.save`` at the spill line raises "Cannot save multiple
    tensors or storages that view the same data as different types".
    """

    from torchlens.backends.torch._tl import set_tensor_label

    session = CaptureSession(
        CapturePlan.compile(
            projection_target="trace",
            retention_profile=RetentionProfile(
                activation_kind=RetentionKind.ACTIVATION,
                activation_window=None,
                spillable=True,
                activation_ram_budget_bytes=1,
            ),
        )
    )
    tensor = torch.arange(16, dtype=torch.float32)
    set_tensor_label(tensor, "arange_1_1")
    session.escrow_candidate(1, tensor)
    payload = session.activation_escrow[1]
    assert payload.tensor is None and payload.spill_path is not None
    reloaded = torch.load(payload.spill_path, weights_only=True)
    assert torch.equal(reloaded, tensor)
    assert not hasattr(reloaded, "_tl"), "TorchLens sidecar leaked into the spill artifact"
    assert torch.equal(payload.materialize(), tensor)
    session.release()


@pytest.mark.smoke
@pytest.mark.real_model
def test_config_built_clip_visual_projection_bit_equal_through_forced_spill(force_spill):
    """tvscope B13 shape at R0 scale: the projected image embedding survives escrow.

    Config-built CLIP (real class, real config, zero network) with the
    visual-projection op saved through the deferred door and every escrow
    byte forced through the spill line.
    """

    pytest.importorskip("transformers")
    from tests.real_model.r0 import families

    torch.manual_seed(families.SEED)
    model = families.build_clip("eager").eval()
    generator = torch.Generator().manual_seed(families.SEED)
    kwargs = {
        "input_ids": torch.randint(0, 512, (2, 8), generator=generator),
        "pixel_values": torch.randn(2, 3, 32, 32, generator=generator),
    }
    kwargs["attention_mask"] = torch.ones_like(kwargs["input_ids"])

    structure = tl.trace(
        model, (), kwargs, capture=tl.options.CaptureOptions(layers_to_save="none")
    )
    projection_ops = [
        op
        for op in structure.layer_list
        if "visual_projection" in (op.output_of_modules or ())
        and not getattr(op, "is_output", False)
    ]
    assert projection_ops, "no visual_projection op captured on the R0 CLIP fixture"
    label = projection_ops[0].layer_label

    log = tl.trace(model, (), kwargs, capture=tl.options.CaptureOptions(layers_to_save=[label]))
    oracle = _eager_module_output(model, "visual_projection", **kwargs)
    assert torch.equal(log[label].out, oracle), (
        f"R0 CLIP {label} escrow payload not bit-equal to the eager visual projection"
    )


@pytest.mark.heavy
@pytest.mark.real_model
def test_resnet18_batch8_deferred_spellings_complete_and_materialize_bit_exact():
    """Leverage B2 gate + brainpipe 2 red: ResNet-18 IMAGENET1K_V1 batch 8.

    The three deferred final-numbering spellings (op label, positive ordinal,
    ``output``) all crashed at the spill line pre-fix; each must now complete
    with bit-exact payloads.
    """

    _require_torchvision_checkpoint("resnet18-f37072fd.pth")
    import torchvision

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    x = torch.randn(8, 3, 224, 224, generator=torch.Generator().manual_seed(1))

    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=["conv2d_5_15"]))
    op = log["conv2d_5_15"]
    assert op.output_of_modules == ("layer1.1.conv2",)
    assert torch.equal(op.out, _eager_module_output(model, "layer1.1.conv2", x))

    ordinal_log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=[5]))
    saved = [
        op
        for op in ordinal_log.layer_list
        if getattr(op, "has_saved_activation", False) and op.output_of_modules
    ]
    assert saved, "positive-ordinal save retained no module-attributed payloads"
    for op in saved:
        assert torch.equal(op.out, _eager_module_output(model, op.output_of_modules[-1], x))

    output_log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=["output"]))
    with torch.no_grad():
        direct = model(x)
    output_ops = [
        op for op in output_log.layer_list if getattr(op, "is_output", False) and op.out is not None
    ]
    assert output_ops and all(torch.equal(op.out, direct) for op in output_ops), (
        "output-label deferred save did not materialize the returned tensor bit-exactly"
    )


@pytest.mark.heavy
@pytest.mark.real_model
def test_resnet50_batch1_op_label_selective_save_completes():
    """tvscope B13 red test 1: op-label selector on resnet50 at batch 1."""

    _require_torchvision_checkpoint("resnet50-0676ba61.pth")
    import torchvision

    model = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=["conv2d_10_30"]))
    op = log["conv2d_10_30"]
    assert torch.equal(op.out, _eager_module_output(model, op.output_of_modules[-1], x))


@pytest.mark.heavy
@pytest.mark.real_model
def test_vit_b_16_op_label_selective_save_completes():
    """Transforms P7 red: torchvision vit_b_16 selective-save crash at the spill line."""

    _require_torchvision_checkpoint("vit_b_16-c867db91.pth")
    import torchvision

    model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").eval()
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    structure = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="none"))
    module_linears = [
        op
        for op in structure.layer_list
        if op.layer_label.startswith("linear_")
        and op.output_of_modules
        and not getattr(op, "is_output", False)
    ]
    assert len(module_linears) > 4, "expected module-attributed linear ops on vit_b_16"
    target = module_linears[4]
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=[target.layer_label]))
    op = log[target.layer_label]
    # Door parity is the exact claim: the deferred escrow door must return the
    # SAME BYTES the live predicate door returns for the same op. The bare
    # eager forward keeps torch's MHA fast path (disabled under any
    # torch-function interception), so eager comparison gets kernel-path
    # tolerance, not bit-exactness.
    live = tl.trace(model, x, save=tl.in_module(op.output_of_modules[-1]))
    assert torch.equal(op.out, live[target.layer_label].out), (
        "deferred escrow door and live predicate door disagree bit-wise"
    )
    torch.testing.assert_close(op.out, _eager_module_output(model, op.output_of_modules[-1], x))


@pytest.mark.heavy
@pytest.mark.real_model
def test_real_clip_linear_73_288_bit_equal_to_visual_projection():
    """tvscope B13 red test 2: CLIP ``linear_73_288`` bit-equal to the projected
    image embedding (real openai/clip-vit-base-patch32 weights, batch 8 images).

    ``CLIPOutput.image_embeds`` is L2-normalized in transformers 5.x, so the
    independent oracle is the eager ``visual_projection`` module output (the
    raw projected embedding the memo's claim is about).
    """

    _require_hf_snapshot("models--openai--clip-vit-base-patch32")
    transformers = pytest.importorskip("transformers")

    model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    generator = torch.Generator().manual_seed(0)
    kwargs = {
        "input_ids": torch.randint(0, 1000, (2, 7), generator=generator),
        "pixel_values": torch.randn(8, 3, 224, 224, generator=generator),
    }
    kwargs["attention_mask"] = torch.ones_like(kwargs["input_ids"])
    oracle = _eager_module_output(model, "visual_projection", **kwargs)

    log = tl.trace(
        model, (), kwargs, capture=tl.options.CaptureOptions(layers_to_save=["linear_73_288"])
    )
    op = log["linear_73_288"]
    assert op.output_of_modules == ("visual_projection",)
    assert torch.equal(op.out, oracle), (
        "CLIP linear_73_288 escrow payload is not bit-equal to the projected image embedding"
    )
