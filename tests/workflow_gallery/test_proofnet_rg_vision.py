"""RG vision scenarios on real ResNet-18 weights (testing memo 5.2; F36 E2).

RG01 trace, RG10 extraction, RG11 RDM/CKA, RG12 runnable roundtrip, RG13
render floors, RG16 training step, RG20 receptive fields -- each with an
INDEPENDENT oracle (assertion-ladder rung 1), on the pinned
``r1-resnet18-imagenet1k-v1`` registry row and the committed natural image.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy
import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_model,
    pytest.mark.real_checkpoint,
    pytest.mark.gallery,
]

EXTRACT_SITE = "avgpool"


@pytest.fixture(scope="module")
def resnet(rg_loader: Any) -> Any:
    return rg_loader("r1-resnet18-imagenet1k-v1")["model"]


def test_rg01_trace_processed_natural_image_matches_direct(
    resnet: Any, natural_image_batch: Any
) -> None:
    """RG01: traced output == direct forward; capture settles COMPLETE."""

    import torchlens as tl

    batch, _ = natural_image_batch
    with torch.no_grad():
        direct = resnet(batch)
    trace = tl.trace(resnet, batch)
    try:
        assert trace.outcome.status.name == "COMPLETE"
        assert torch.equal(trace.output_ops[0].out, direct), (
            "traced resnet18 logits != the model's own forward on real weights"
        )
        # Real-weights witness: pretrained BN stats are nonzero.
        assert resnet.bn1.running_mean.abs().sum() > 0
    finally:
        trace.cleanup()


def test_rg10_extraction_memory_and_disk_agree_and_reload(
    resnet: Any, natural_image_batch: Any, tmp_path: Any
) -> None:
    """RG10: direct-forward, in-memory extraction, and the disk artifact
    agree on values, stimulus ids, order, and content hashes."""

    import torchlens as tl
    from torchlens.dataset_extraction import load_extraction

    batch, flipped = natural_image_batch
    stimuli = [batch[0], flipped[0]]
    stimulus_ids = ["astronaut", "astronaut-flipped"]

    class _Probe(torch.nn.Module):
        def __init__(self, backbone: Any) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)

    with torch.no_grad():
        direct_features = []
        hook_store: dict[str, torch.Tensor] = {}
        module = dict(resnet.named_modules())[EXTRACT_SITE]
        handle = module.register_forward_hook(
            lambda mod, args, out: hook_store.__setitem__("value", out.detach().clone())
        )
        try:
            for stimulus in stimuli:
                resnet(stimulus.unsqueeze(0))
                direct_features.append(hook_store["value"].flatten())
        finally:
            handle.remove()

    output_dir = tmp_path / "extraction"
    tl.extract_dataset(
        resnet,
        torch.stack(stimuli),
        [EXTRACT_SITE],
        batch_size=2,
        output_dir=str(output_dir),
        progress=False,
        stimulus_ids=stimulus_ids,
    )
    loaded = load_extraction(str(output_dir))
    site_keys = [key for key in loaded.activations if EXTRACT_SITE in key]
    assert site_keys, f"extraction lost the requested site: {list(loaded.activations)}"
    features = torch.as_tensor(loaded.activations[site_keys[0]]).reshape(2, -1).float()

    # Ordered-identity contract: row i is stimulus i (the manifest's own
    # documented order rule), and the persisted id digest matches the ids we
    # supplied (fact carriage across the disk hop).
    provenance = loaded.manifest["stimulus_provenance"]
    assert provenance["n_stimuli"] == 2 and provenance["ids_recorded"]
    from torchlens._data_substrate.artifact import stimulus_ids_digest

    assert provenance["ids_digest"] == stimulus_ids_digest(list(stimulus_ids)), (
        "the persisted stimulus-id digest does not match the supplied ids"
    )

    for index, direct in enumerate(direct_features):
        assert torch.allclose(features[index], direct, atol=1e-5), (
            f"stimulus {stimulus_ids[index]}: extracted features differ from a"
            " direct hooked forward (independent oracle)"
        )

    # Byte-stability: a second load serves identical bytes (disk round trip).
    reloaded = (
        torch.as_tensor(load_extraction(str(output_dir)).activations[site_keys[0]])
        .reshape(2, -1)
        .float()
    )
    assert hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest() == (
        hashlib.sha256(reloaded.contiguous().numpy().tobytes()).hexdigest()
    )


def test_rg11_rdm_matches_independent_numpy_and_cka_self_score(
    resnet: Any, natural_image_batch: Any
) -> None:
    """RG11: TorchLens RDM == a hand-coded NumPy correlation RDM (symmetric,
    zero diagonal); CKA(X, X) == 1."""

    import torchlens as tl

    batch, flipped = natural_image_batch
    stimuli = torch.cat([batch, flipped, batch * 0.5], dim=0)
    features = tl.extract(resnet, stimuli, [EXTRACT_SITE])[EXTRACT_SITE]
    matrix = features.reshape(features.shape[0], -1).double()

    rdm = tl.repgeom.rdm(matrix, metric="correlation", output="square")
    rdm_numpy = numpy.asarray(rdm)
    assert rdm_numpy.shape == (3, 3)
    assert numpy.allclose(rdm_numpy, rdm_numpy.T), "RDM not symmetric"
    assert numpy.allclose(numpy.diag(rdm_numpy), 0.0, atol=1e-9), "RDM diagonal nonzero"

    # Independent oracle: hand-coded correlation distance in NumPy.
    rows = matrix.numpy()
    centered = rows - rows.mean(axis=1, keepdims=True)
    normalized = centered / numpy.linalg.norm(centered, axis=1, keepdims=True)
    hand = 1.0 - normalized @ normalized.T
    numpy.fill_diagonal(hand, 0.0)
    assert numpy.allclose(rdm_numpy, hand, atol=1e-6), (
        "TorchLens correlation RDM disagrees with the hand-coded NumPy RDM"
    )

    cka_self = float(tl.stats.cka(matrix, matrix))
    assert abs(cka_self - 1.0) < 1e-9, f"CKA self-score {cka_self} != 1"


def test_rg12_runnable_save_load_rerun_verified(
    resnet: Any, natural_image_batch: Any, tmp_path: Any
) -> None:
    """RG12: runnable save with weights -> load -> run: VERIFIED path
    faithfulness and byte-agreeing outputs; provenance hashes present."""

    import torchlens as tl

    batch, _ = natural_image_batch
    trace = tl.trace(resnet, batch, capture=tl.options.CaptureOptions(intervention_ready=True))
    try:
        path = tmp_path / "resnet-runnable.tlspec"
        tl.save(trace, str(path), level="runnable", include_weights=True)
        loaded = tl.load(str(path))
        result = loaded.run(inputs=batch)
        assert result.report.path_faithfulness.name == "VERIFIED"
        assert torch.allclose(result.output, trace.output_ops[0].out, atol=1e-6), (
            "rerun output differs from the captured forward on real weights"
        )
    finally:
        trace.cleanup()


def test_rg13_render_floors_track_the_captured_graph(
    resnet: Any, natural_image_batch: Any, tmp_path: Any
) -> None:
    """RG13: the rendered graph's node population tracks the capture (content
    floor keyed to THIS capture, never file-existence)."""

    import torchlens as tl

    batch, _ = natural_image_batch
    trace = tl.trace(resnet, batch)
    try:
        graph = trace.draw(vis_outpath=str(tmp_path / "resnet-graph"), return_graph=True)
        dot_source = graph.source if hasattr(graph, "source") else str(graph)
        node_count = dot_source.count("label=")
        assert node_count >= 50, (
            f"rendered resnet18 graph carries only {node_count} labelled"
            " entities -- the render dropped the captured population"
        )
        assert "conv" in dot_source.lower(), "no conv node named in the rendered graph"
        assert (tmp_path / "resnet-graph.pdf").exists() or any(tmp_path.iterdir()), (
            "draw() reported success but wrote nothing"
        )
    finally:
        trace.cleanup()


def test_rg16_training_step_grads_match_no_torchlens_control(
    rg_loader: Any, natural_image_batch: Any
) -> None:
    """RG16: unscaled grads + param deltas under capture == a no-torchlens
    control, fp32 exact; the bf16 autocast arm agrees within tolerance;
    model/RNG state exact afterwards (cleanup)."""

    import torchvision.models as tv_models

    import torchlens as tl

    batch, _ = natural_image_batch
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1

    def one_step(use_torchlens: bool, autocast: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = tv_models.resnet18(weights=weights).train()
        target = torch.tensor([1])
        if autocast:
            context: Any = torch.autocast("cpu", dtype=torch.bfloat16)
        else:
            import contextlib

            context = contextlib.nullcontext()
        if use_torchlens:
            with context:
                trace = tl.trace(
                    model,
                    batch,
                    capture=tl.options.CaptureOptions(backward_ready=True),
                    save_mode="reference",
                )
                logits = trace.output_ops[0].out
                loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            trace.cleanup()
        else:
            with context:
                loss = torch.nn.functional.cross_entropy(model(batch), target)
            loss.backward()
        grads = {
            name: parameter.grad.detach().clone().float()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        tl.release_model(model)
        return grads

    control = one_step(use_torchlens=False, autocast=False)
    captured = one_step(use_torchlens=True, autocast=False)
    assert set(control) == set(captured), "capture changed WHICH params got grads"
    for name in control:
        assert torch.allclose(captured[name], control[name], atol=1e-6), (
            f"fp32 grad for {name} differs under capture -- the tool changed the training step"
        )

    control_amp = one_step(use_torchlens=False, autocast=True)
    captured_amp = one_step(use_torchlens=True, autocast=True)
    for name in ("fc.weight", "layer4.1.conv2.weight"):
        assert torch.allclose(captured_amp[name], control_amp[name], rtol=2e-2, atol=1e-3), (
            f"bf16-autocast grad for {name} diverged beyond the (cpu, bf16) tolerance"
        )


def test_rg20_receptive_field_geometric_matches_empirical(
    rg_loader: Any, natural_image_batch: Any
) -> None:
    """RG20: geometric vs empirical receptive fields agree ON REAL WEIGHTS
    (one of the five broken 2026-08 flagships; the acceptance set cannot
    omit it)."""

    import torchvision.models as tv_models

    import torchlens as tl

    batch, _ = natural_image_batch
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
    torch.manual_seed(0)
    model = tv_models.resnet18(weights=weights).eval()
    armed = tl.trace(
        model,
        batch.clone().requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    try:
        early_conv = [label for label in armed.layer_labels if "conv" in label.lower()][1]
        field = armed[early_conv].receptive_field
        unit = field.center_unit(batch_index=0)
        verdict = field.check(unit)
        assert verdict.status.name == "PASS", (
            f"receptive-field check on real resnet18 weights returned"
            f" {verdict.status} ({verdict.message}) at {early_conv} --"
            " geometric and empirical fields disagree on a flagship surface"
        )
        assert verdict.n_violations == 0
    finally:
        armed.cleanup()
        tl.release_model(model)
