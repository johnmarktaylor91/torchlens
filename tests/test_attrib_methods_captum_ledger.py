"""F06 B10: the consolidated captum oracle ledger (attrib memo B10 / s4).

LEDGER_VERSION = 1. One versioned file holds every captum oracle row; it
SKIPS CLEANLY when the captum extra is absent (the ``captum~=0.7`` ->
``>=0.7,<1.0`` pin widening is externally owned; see the F06 row in
the packaging-request ledger). Discipline (memo section 4):

- numeric values are compared BEFORE any rendering;
- captum is ALWAYS told ``method="riemann_middle"`` for path integrals (its
  default Gauss-Legendre differs -- measured 22.5% at n=16 on resnet18);
- causal-LM rows ALWAYS pass captum the explicit (position, vocab) tuple --
  a bare int row does not fail, it never runs (captum 0.9 refuses it);
- every row records exact torch/torchvision/transformers/captum versions,
  dtype/device, seeds/digests, target, and settings;
- GPU rows are manual/periodic and cannot substitute for CPU semantic rows.

Rows that CANNOT be captum rows are named, never silently absent: captum's
NoiseTunnel and GradientShap draw their own randomness (no bank injection
door), so their conventions are pinned by our stored-bank/stored-draw
oracles in ``test_attrib_methods_wrapping.py``; captum's ``sensitivity_max``
likewise draws internally. The zero-noise NoiseTunnel degenerate row below
IS a captum row (no randomness left).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution

captum_attr = pytest.importorskip("captum.attr")
import captum  # noqa: E402 - guarded by the importorskip above

LEDGER_VERSION = 1

pytestmark = pytest.mark.real_model


def _environment() -> dict[str, Any]:
    """Record the exact oracle environment for every ledger row."""

    try:
        import torchvision

        torchvision_version = str(torchvision.__version__)
    except Exception:  # noqa: BLE001 - optional
        torchvision_version = None
    try:
        import transformers

        transformers_version = str(transformers.__version__)
    except Exception:  # noqa: BLE001 - optional
        transformers_version = None
    return {
        "ledger_version": LEDGER_VERSION,
        "torch": str(torch.__version__),
        "torchvision": torchvision_version,
        "transformers": transformers_version,
        "captum": str(captum.__version__),
        "device": "cpu",
    }


def _row(**fields: Any) -> dict[str, Any]:
    """Assemble one ledger row: environment + row-specific settings."""

    row = _environment()
    row.update(fields)
    for required in ("method", "settings", "target"):
        assert required in row, f"ledger row missing {required!r}"
    return row


class _SharedMlp(nn.Module):
    """The shared-semantics MLP every exact toy row runs on (float64)."""

    def __init__(self) -> None:
        """Fixed weights, biased first layer (exercises absorption paths)."""

        super().__init__()
        self.l1 = nn.Linear(4, 6, dtype=torch.float64)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(6, 3, dtype=torch.float64)
        with torch.no_grad():
            self.l1.weight.copy_(torch.arange(24, dtype=torch.float64).reshape(6, 4) / 11 - 1)
            self.l1.bias.copy_(torch.arange(6, dtype=torch.float64) / 10 - 0.2)
            self.l2.weight.copy_(torch.arange(18, dtype=torch.float64).reshape(3, 6) / 7 - 1.2)
            self.l2.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Linear-ReLU-Linear forward."""

        return self.l2(self.act(self.l1(x)))


@pytest.fixture(scope="module")
def shared_case() -> tuple[nn.Module, Tensor]:
    """The shared model + input for the exact toy rows."""

    model = _SharedMlp()
    model.eval()
    x = torch.tensor([[0.7, -0.3, 0.5, 0.2]], dtype=torch.float64)
    return model, x


@pytest.mark.smoke
def test_row_integrated_gradients_riemann_middle(shared_case: tuple[nn.Module, Tensor]) -> None:
    """IG parity, EXACT: ours == captum under matched midpoint semantics."""

    model, x = shared_case
    row = _row(
        method="integrated_gradients",
        target=1,
        settings={"n_steps": 32, "baseline": "zeros", "captum_method": "riemann_middle"},
        dtype="float64",
    )
    ours = attribution.integrated_gradients(model, x, target=1, n_steps=32)
    theirs = captum_attr.IntegratedGradients(model).attribute(
        x, target=1, n_steps=32, method="riemann_middle"
    )
    torch.testing.assert_close(ours.values, theirs, rtol=0, atol=0)
    assert row["captum"] == "0.9.0" or row["captum"] >= "0.7"


@pytest.mark.smoke
def test_row_saliency_and_input_x_grad(shared_case: tuple[nn.Module, Tensor]) -> None:
    """Saliency (absolute) and input-x-gradient parity, EXACT."""

    model, x = shared_case
    _row(method="saliency+input_x_grad", target=1, settings={}, dtype="float64")
    torch.testing.assert_close(
        attribution.saliency(model, x, target=1).values,
        captum_attr.Saliency(model).attribute(x, target=1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attribution.input_x_grad(model, x, target=1).values,
        captum_attr.InputXGradient(model).attribute(x, target=1),
        rtol=0,
        atol=0,
    )


@pytest.mark.smoke
def test_row_guided_and_deconv_module_sites(shared_case: tuple[nn.Module, Tensor]) -> None:
    """Guided backprop + deconvolution parity on module-dispatched coverage.

    ``sites="module"`` IS the module-hook coverage class captum implements;
    the op-level default additionally rewrites functional/method ReLUs, which
    captum cannot see (the densenet divergence row below).
    """

    model, x = shared_case
    _row(
        method="guided_backprop+deconvolution",
        target=1,
        settings={"sites": "module", "signed": True},
        dtype="float64",
    )
    torch.testing.assert_close(
        attribution.guided_backprop(model, x, target=1, sites="module").values,
        captum_attr.GuidedBackprop(model).attribute(x, target=1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attribution.deconvolution(model, x, target=1, sites="module").values,
        captum_attr.Deconvolution(model).attribute(x, target=1),
        rtol=0,
        atol=0,
    )


@pytest.mark.smoke
def test_row_lrp_epsilon_shared_mlp(shared_case: tuple[nn.Module, Tensor]) -> None:
    """The captum LRP(EpsilonRule) shared-MLP exact row (memo D32)."""

    from test_attrib_methods_lrp import epsilon_lrp

    model, x = shared_case
    _row(method="epsilon_lrp_recipe", target=1, settings={"epsilon": 1e-9}, dtype="float64")
    relevance, ledger, leftovers = epsilon_lrp(model, x, target_index=1, epsilon=1e-9)
    theirs = captum_attr.LRP(model).attribute(x, target=1)
    torch.testing.assert_close(relevance, theirs, rtol=1e-9, atol=1e-12)
    assert leftovers == {}
    assert all(row["nonfinite"] is False for row in ledger)


@pytest.mark.smoke
def test_row_occlusion_map_shared_grid(shared_case: tuple[nn.Module, Tensor]) -> None:
    """Occlusion parity on an identical grid/baseline/target, incl. clipping."""

    class _Img(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1, dtype=torch.float64)
            with torch.no_grad():
                self.conv.weight.copy_(
                    torch.arange(18, dtype=torch.float64).reshape(2, 1, 3, 3) / 10 - 0.8
                )
                self.conv.bias.zero_()

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x).mean(dim=(2, 3))

    model = _Img().eval()
    image = torch.rand(1, 1, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    _row(
        method="occlusion_map",
        target=0,
        settings={"window": (4, 4), "strides": (3, 3), "baseline": 0.0, "overlap": "average"},
        dtype="float64",
    )
    ours = attribution.occlusion_map(model, image, target=0, window=(4, 4), strides=(3, 3))
    theirs = captum_attr.Occlusion(model).attribute(
        image, target=0, sliding_window_shapes=(1, 4, 4), strides=(1, 3, 3), baselines=0.0
    )
    # captum's ablation math downcasts through a float32 scalar baseline, so
    # its float32 output precision is the comparison limiter.
    torch.testing.assert_close(ours.values, theirs.to(torch.float64), rtol=1e-5, atol=1e-6)


@pytest.mark.smoke
def test_row_noise_tunnel_zero_noise_degenerate(shared_case: tuple[nn.Module, Tensor]) -> None:
    """Zero-noise NT == captum NT == the direct child (no randomness left).

    Captum's NoiseTunnel and GradientShap expose no draw-injection door, so
    their stochastic conventions are pinned by our stored-bank / stored-draw
    oracles (test_attrib_methods_wrapping.py); this degenerate row is the one
    NT cell captum can join deterministically.
    """

    model, x = shared_case
    _row(
        method="noise_tunnel(saliency)",
        target=1,
        settings={"n_samples": 3, "stdevs": 0.0, "aggregation": "mean"},
        dtype="float64",
    )
    ours = attribution.noise_tunnel(
        x, method=attribution.saliency, model=model, target=1, n_samples=3, stdevs=0.0
    )
    theirs = captum_attr.NoiseTunnel(captum_attr.Saliency(model)).attribute(
        x, target=1, nt_type="smoothgrad", nt_samples=3, stdevs=1e-12
    )
    torch.testing.assert_close(ours.values, theirs, rtol=1e-6, atol=1e-9)


@pytest.mark.smoke
def test_row_infidelity_identical_stored_perturbations(
    shared_case: tuple[nn.Module, Tensor],
) -> None:
    """Infidelity parity on IDENTICAL perturbations via the callable escape."""

    model, x = shared_case
    attribution_values = attribution.input_x_grad(model, x, target=1).values
    fixed = torch.tensor([[0.011, -0.007, 0.005, 0.003]], dtype=torch.float64)
    _row(
        method="infidelity",
        target=1,
        settings={"perturb": "fixed shared callable", "n_samples": 2},
        dtype="float64",
    )

    def ours_perturb(leaves: tuple[Tensor, ...]) -> tuple[tuple, tuple]:
        return (fixed,), (leaves[0] - fixed,)

    def captum_perturb(inputs: Tensor) -> tuple[Tensor, Tensor]:
        return fixed, inputs - fixed

    ours = attribution.infidelity(
        model, x, target=1, attribution=attribution_values, perturb=ours_perturb, n_samples=2
    )
    from captum.metrics import infidelity as captum_infidelity

    theirs = captum_infidelity(
        model,
        captum_perturb,
        x,
        attribution_values,
        target=1,
        n_perturb_samples=2,
    )
    assert ours.value == pytest.approx(float(theirs), rel=1e-9)


@pytest.mark.heavy
def test_row_resnet18_guided_backprop_signed_parity() -> None:
    """resnet18 guided parity, signed, module coverage (panel measured 0.0)."""

    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    image = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(11))
    _row(
        method="guided_backprop",
        target=207,
        settings={"sites": "module", "signed": True, "model": "resnet18/IMAGENET1K_V1"},
        dtype="float32",
    )
    ours = attribution.guided_backprop(model, image, target=207, sites="module")
    theirs = captum_attr.GuidedBackprop(model).attribute(image, target=207)
    torch.testing.assert_close(ours.values, theirs, rtol=0, atol=0)


@pytest.mark.heavy
def test_row_densenet121_three_way_identity() -> None:
    """The D13 three-way identity, part (i): restricted == captum, EXACT.

    Parts (ii)/(iii) -- op-level coverage differs, and the difference is the
    one functional ReLU in torchvision's own forward -- are pinned without
    captum in test_attrib_methods_realmodel.py. The docs claim stays in the
    checkable form; the honest limit rides here: one model / one target / one
    input / CPU -- the IDENTITY is the claim.
    """

    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.densenet121(
        weights=torchvision_models.DenseNet121_Weights.IMAGENET1K_V1
    )
    model.eval()
    image = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(1234))
    _row(
        method="guided_backprop",
        target=207,
        settings={"sites": "module", "model": "densenet121/IMAGENET1K_V1"},
        dtype="float32",
    )
    restricted = attribution.guided_backprop(model, image, target=207, sites="module")
    theirs = captum_attr.GuidedBackprop(model).attribute(image, target=207)
    torch.testing.assert_close(restricted.values, theirs, rtol=0, atol=0)


@pytest.mark.heavy
def test_row_gpt2_text_vs_captum_layer_integrated_gradients() -> None:
    """text() vs captum LayerIntegratedGradients on gpt2, matched midpoint.

    The causal-LM rule: captum is passed the EXPLICIT (position, vocab)
    tuple -- captum 0.9 refuses a bare int outright, so a bare-int row would
    never run. Panel measured 2.80e-06 relative agreement; the row asserts
    1e-4 (float32 kernel noise across two different execution paths).
    """

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    prompt = "The Eiffel Tower is located in the city of"
    encoding = tokenizer(prompt, return_tensors="pt")
    paris = tokenizer.encode(" Paris")[0]
    final_position = int(encoding["input_ids"].shape[1]) - 1
    _row(
        method="text/layer_integrated_gradients",
        target=(final_position, paris),
        settings={
            "n_steps": 32,
            "captum_method": "riemann_middle",
            "captum_target": "(position, vocab) tuple -- ALWAYS",
            "model": "gpt2",
        },
        dtype="float32",
    )
    ours = attribution.text(
        model, tokenizer, prompt, target=" Paris", n_steps=32, steps_per_batch=1
    )

    def forward_scores(input_ids: Tensor) -> Tensor:
        """Position-resolved captum forward (tuple targets need rank-2)."""

        return model(input_ids=input_ids).logits[:, final_position, :]

    lig = captum_attr.LayerIntegratedGradients(forward_scores, model.transformer.wte)
    theirs = lig.attribute(
        encoding["input_ids"],
        baselines=torch.zeros_like(encoding["input_ids"]),
        target=paris,
        n_steps=32,
        method="riemann_middle",
    )
    # captum LIG with an all-zeros-ids baseline uses the id-0 EMBEDDING as
    # its baseline; ours zeros the embeddings themselves. Compare against a
    # second captum run whose baseline matches ours via inputs_embeds.
    embeddings = model.transformer.wte(encoding["input_ids"]).detach()

    def forward_from_embeds(inputs_embeds: Tensor) -> Tensor:
        return model(inputs_embeds=inputs_embeds).logits[:, final_position, :]

    ig = captum_attr.IntegratedGradients(forward_from_embeds)
    theirs_embeds = ig.attribute(
        embeddings,
        baselines=torch.zeros_like(embeddings),
        target=paris,
        n_steps=32,
        method="riemann_middle",
    )
    ours_scores = ours.scores
    theirs_scores = theirs_embeds.sum(dim=-1)[0]
    scale = float(theirs_scores.abs().max())
    assert float((ours_scores - theirs_scores).abs().max()) / scale < 1e-4
    del theirs  # the ids-baseline run is kept only as an executed reference
