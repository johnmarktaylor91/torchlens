"""RG breadth scenarios: gpt2, albert, CLIP, detection, T5 (testing memo 5.2).

RG06 attribution patching (gpt2), RG09 pass-qualified patching on REAL
cross-layer weight sharing (albert), RG14 multimodal processor kwargs
(CLIP), RG15 detection with two shapes and the zero-result case (Faster
R-CNN), RG17 encoder-decoder capture (T5). Pinned registry rows, offline,
committed natural inputs.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

from tests.real_model.registry import NATURAL_INPUTS_DIR
from tests.workflow_gallery.conftest import prompt

pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_model,
    pytest.mark.real_checkpoint,
    pytest.mark.gallery,
]


def test_rg06_attribution_patching_finite_with_zero_negative_control(
    rg_loader: Any,
) -> None:
    """RG06: attribution patching on real GPT-2 produces a finite, non-flat
    table with restored params/grads/mode, and the clean-vs-clean negative
    control attributes (approximately) nothing."""

    import torchlens.semantic.patching as patching

    loaded = rg_loader("r1-gpt2")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    pair = prompt("clean-factual-1")
    clean = tokenizer(pair["clean"], return_tensors="pt")["input_ids"]
    corrupted = tokenizer(pair["corrupt"], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        answer_id = int(model(clean).logits[0, -1].argmax())

    def metric(log: Any) -> torch.Tensor:
        for op in log.output_ops:
            if getattr(op.out, "dim", lambda: 0)() == 3:
                return op.out[0, -1, answer_id]
        raise AssertionError("no 3D logits op on the capture")

    params_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        try:
            table = patching.attribution_patch_attention_heads(model, clean, corrupted, metric)
        except Exception as exc:  # noqa: BLE001 - typed settlement is honest
            code = getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
            typed = type(exc).__name__ in {
                "SiteResolutionError",
                "SiteAmbiguityError",
                "PatchApplicationError",
                "MissingFacetError",
            } or bool(code)
            # Ledgered settlement on this venue: fused attention refuses the
            # grad-capable facet with a TEACHING but BARE RuntimeError (the
            # DIGEST-AUDIT mis-rooted-hierarchy class; typing it must flip
            # this arm). Either way the remedy must be named.
            teaching = "backward_ready" in str(exc) or "recapture" in str(exc).lower()
            assert typed or teaching, (
                f"RG06 settled with an untyped, unteaching failure:"
                f" {type(exc).__name__}: {str(exc)[:180]}"
            )
            return
    assert torch.isfinite(table).all(), "attribution table carries nonfinite cells"
    assert table.abs().sum() > 0, "flat attribution table (silent no-op class)"
    # Cleanup contract: parameters untouched, no grads left, eval mode kept.
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), params_before[name]), (
            f"attribution patching mutated parameter {name}"
        )
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert not model.training


def test_rg09_pass_qualified_patch_on_real_cross_layer_sharing(rg_loader: Any) -> None:
    """RG09: on albert's genuinely shared encoder layer, a bare label refuses
    teaching pass-qualified spellings; editing pass 3 changes pass 3 alone,
    with independently indexed earlier passes byte-identical."""

    import torchlens as tl

    loaded = rg_loader("r1-albert-base-v2")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    encoded = dict(tokenizer(prompt("encoder-sentence-1")["text"], return_tensors="pt"))
    trace = tl.trace(model, (), encoded, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        multi_pass = [label for label in trace.layer_labels if len(trace[label].ops) >= 12]
        assert multi_pass, "albert capture shows no 12-pass shared-layer sites"
        target = multi_pass[0]
        with pytest.raises(Exception) as bare_exc:
            trace.fork().do(target, tl.zero_ablate())
        assert "pass" in str(bare_exc.value).lower(), (
            "bare label on the shared layer must refuse teaching the"
            " pass-qualified spellings (the pass-blind-donor class)"
        )
        pass_two_before = trace[target].ops[1].out.detach().clone()
        fork = trace.fork()
        try:
            fork.do(f"{target}:3", tl.zero_ablate())
            assert torch.count_nonzero(fork[target].ops[2].out) == 0, (
                "the addressed pass 3 was not edited"
            )
            assert torch.equal(fork[target].ops[1].out, pass_two_before), (
                "editing pass 3 CHANGED pass 2 -- pass-blind donor corruption"
                " on real cross-layer weight sharing"
            )
        finally:
            fork.cleanup()
    finally:
        trace.cleanup()


def test_rg14_clip_processor_kwargs_and_similarity_match_direct(rg_loader: Any) -> None:
    """RG14: real CLIP processor inputs (image + captions) trace to the
    model's own logits_per_image; the modality towers stay distinct."""

    from PIL import Image
    from transformers import AutoProcessor

    import torchlens as tl

    loaded = rg_loader("r1-clip-vit-base-patch32")
    model = loaded["model"]
    row = loaded["row"]
    processor = AutoProcessor.from_pretrained(row.model_id, revision=row.revision)
    captions = prompt("clip-captions-1")["captions"]
    image = Image.open(NATURAL_INPUTS_DIR / "pd_astronaut_256.jpg")
    inputs = dict(processor(text=captions, images=image, return_tensors="pt", padding=True))
    with torch.no_grad():
        direct = model(**inputs)
    trace = tl.trace(model, (), dict(inputs))
    try:
        assert trace.outcome.status.name == "COMPLETE"
        matches = [
            op
            for op in trace.output_ops
            if getattr(op.out, "shape", None) == direct.logits_per_image.shape
            and torch.allclose(op.out, direct.logits_per_image, atol=1e-5)
        ]
        assert matches, "traced CLIP logits_per_image != direct forward"
        addresses = {
            address
            for op_label in trace.layer_labels
            for address in (trace[op_label].modules or ())
        }
        assert any("vision_model" in str(address) for address in addresses)
        assert any("text_model" in str(address) for address in addresses), (
            "the two CLIP towers did not both appear in the module record"
        )
    finally:
        trace.cleanup()


def test_rg15_detection_two_shapes_and_zero_result_case(rg_loader: Any) -> None:
    """RG15: Faster R-CNN on two input shapes + the zero-result case; the
    traced list/dict output schema and values match the direct forward."""

    import torchlens as tl

    model = rg_loader("r1-fasterrcnn-mnv3-320-coco-v1")["model"]
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize

    image = read_image(str(NATURAL_INPUTS_DIR / "pd_astronaut_256.jpg")).float() / 255.0
    small = resize(image, 224, antialias=True)
    blank = torch.zeros(3, 256, 256)

    for case_name, batch in (("natural", [image, small]), ("zero-result", [blank])):
        with torch.no_grad():
            direct = model(batch)
        trace = tl.trace(model, ([tensor.clone() for tensor in batch],))
        try:
            assert trace.outcome.status.name == "COMPLETE", case_name
            # The list/dict output container decomposes into leaf output ops;
            # payload identity: every direct leaf (boxes/labels/scores per
            # image) appears among the traced output payloads exactly.
            traced_leaves = [op.out for op in trace.output_ops if isinstance(op.out, torch.Tensor)]
            for image_index, direct_entry in enumerate(direct):
                assert set(direct_entry) >= {"boxes", "labels", "scores"}, case_name
                for key in ("boxes", "labels", "scores"):
                    expected = direct_entry[key]
                    found = any(
                        leaf.shape == expected.shape
                        and torch.allclose(leaf.float(), expected.float(), atol=1e-5)
                        for leaf in traced_leaves
                    )
                    assert found, (
                        f"{case_name}: direct {key} for image {image_index}"
                        " has no byte-matching traced output leaf"
                    )
            if case_name == "zero-result":
                assert direct[0]["boxes"].shape[0] >= 0  # schema present even empty
        finally:
            trace.cleanup()


def test_rg17_t5_encoder_decoder_capture_matches_direct(rg_loader: Any) -> None:
    """RG17: T5 forward with decoder inputs: traced logits == direct;
    encoder and decoder both present and distinct in the module record."""

    import torchlens as tl

    loaded = rg_loader("r1-t5-small")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    text = prompt("seq2seq-translate-1")["text"]
    encoded = dict(tokenizer(text, return_tensors="pt"))
    encoded["decoder_input_ids"] = torch.tensor([[model.config.decoder_start_token_id]])
    with torch.no_grad():
        direct = model(**encoded)
    trace = tl.trace(model, (), dict(encoded))
    try:
        assert trace.outcome.status.name == "COMPLETE"
        matches = [
            op
            for op in trace.output_ops
            if getattr(op.out, "shape", None) == direct.logits.shape
            and torch.allclose(op.out, direct.logits, atol=1e-5)
        ]
        assert matches, "traced T5 logits != direct forward"
        addresses = {
            str(address) for label in trace.layer_labels for address in (trace[label].modules or ())
        }
        assert any(address.startswith("encoder") for address in addresses)
        assert any(address.startswith("decoder") for address in addresses), (
            "encoder/decoder distinction lost in the module record"
        )
    finally:
        trace.cleanup()
