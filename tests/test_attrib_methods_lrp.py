"""F06 B9: the epsilon-LRP litmus recipe + the SiteStash mechanism (D30-D32).

The recipe below is the CLEANROOM docs recipe from
``docs/recipes/lrp_epsilon_litmus.md`` transcribed verbatim (keep the two in
lockstep) -- written from the published equations, never from LGPL code.
TorchLens ships only the pairing mechanism (``SiteStash``); rule catalogs
remain the LRP ecosystem's territory (no registry, twice reaffirmed).

Blocking rows (memo D32): the bias-free ReLU self-oracle (LRP-0 ==
input-x-gradient, exact conservation), the reused-module pairing proof, the
refusal-on-uncovered-op row, the concat toy (relevance splits by slice, no
free parameter), and the FULL squeezenet1_1 ledger (~5 MB checkpoint, fetched
once -- stated openly, even in blocking CI).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution import AttributionError, SiteStash

# Per-test tiers: everything here is smoke EXCEPT the squeezenet row, which is
# heavy + real_model (a smoke mark would combine tiers and fail the marker lint).
pytestmark = []

# --- BEGIN docs recipe (docs/recipes/lrp_epsilon_litmus.md, verbatim) -------


def epsilon_lrp(
    model: nn.Module,
    inputs: Tensor,
    target_index: int,
    epsilon: float = 1e-6,
) -> tuple[Tensor, list[dict[str, Any]], dict[str, int]]:
    """Cleanroom epsilon-LRP through paired forward/backward hooks.

    Convention: relevance rides the backward pass as ``c`` with
    ``R = value * c`` at every tensor. Under this convention the NATIVE
    backward of ReLU, max/avg pooling, reshape/flatten, dropout (eval), and
    concatenation ALREADY implements the standard LRP treatment -- a concat
    splits relevance by slice with no free parameter -- so hooks are needed
    ONLY on the parameterized linear/conv sites, where the epsilon rule
    replaces the gradient. Uncovered parameterized ops refuse BY EXACT LABEL;
    generic identity is forbidden. Epsilon (and bias) absorption is disclosed
    in the per-site conservation ledger rather than called exact
    conservation.
    """

    stash = SiteStash()
    ledger: list[dict[str, Any]] = []
    handles = []
    in_backward = {"flag": False}

    def make_forward_hook(site: nn.Module):
        def forward_hook(module: nn.Module, args: tuple, output: Tensor) -> None:
            if in_backward["flag"]:
                return  # the backward-rule recompute must not restash
            if len(args) != 1 or not isinstance(args[0], Tensor):
                raise AttributionError(
                    f"site {stash.label_of(module)!r} fired with an "
                    "unexpected argument tuple; the epsilon rule covers "
                    "single-tensor-input Linear/Conv2d sites"
                )
            stash.mark_firing(module)
            stash.stash(module, "input", args[0].detach())
            stash.stash(module, "output", output.detach())

        return forward_hook

    def make_backward_hook(site: nn.Module):
        def backward_hook(
            module: nn.Module,
            grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
        ) -> tuple[Tensor | None, ...] | None:
            if in_backward["flag"]:
                # The rule's own recompute backward passes through natively:
                # the autograd trick WANTS dz/dx with the substituted s.
                return None
            # Every gradient tuple slot is explicitly guarded: the recipe
            # never leans on a selector distinction.
            if len(grad_output) != 1 or grad_output[0] is None:
                raise AttributionError(
                    f"site {stash.label_of(module)!r} received a "
                    f"{len(grad_output)}-slot grad_output tuple; the recipe "
                    "expects exactly one used output"
                )
            if len(grad_input) != 1:
                raise AttributionError(
                    f"site {stash.label_of(module)!r} received a "
                    f"{len(grad_input)}-slot grad_input tuple; the recipe "
                    "expects exactly one tensor input"
                )
            x = stash.fetch(module, "input")
            z_saved = stash.fetch(module, "output")
            c_out = grad_output[0]
            in_backward["flag"] = True
            try:
                with torch.enable_grad():
                    x_live = x.clone().requires_grad_(True)
                    z = module(x_live)
                    stabilizer = epsilon * torch.where(
                        z.detach() >= 0,
                        torch.ones_like(z.detach()),
                        -torch.ones_like(z.detach()),
                    )
                    s = (z_saved * c_out) / (z_saved + stabilizer)
                    c_in = torch.autograd.grad(z, x_live, grad_outputs=s.detach())[0]
            finally:
                in_backward["flag"] = False
            output_relevance = float((z_saved * c_out).sum())
            input_relevance = float((x * c_in).sum())
            delta = input_relevance - output_relevance
            ledger.append(
                {
                    "site": stash.label_of(module),
                    "rule": "epsilon",
                    "grad_slots": len(grad_input),
                    "input_relevance_sum": input_relevance,
                    "output_relevance_sum": output_relevance,
                    "conservation_delta_abs": abs(delta),
                    "conservation_delta_rel": abs(delta) / max(abs(output_relevance), 1e-30),
                    "dtype": str(z_saved.dtype),
                    "device": str(z_saved.device),
                    "nonfinite": not bool(torch.isfinite(c_in).all()),
                }
            )
            return (c_in,)

        return backward_hook

    covered = (nn.Linear, nn.Conv2d)
    for label, module in model.named_modules():
        if list(module.children()):
            continue
        if any(True for _ in module.parameters(recurse=False)) and not isinstance(module, covered):
            raise AttributionError(
                f"epsilon-LRP litmus covers Linear and Conv2d sites only; "
                f"uncovered parameterized op at {label!r} "
                f"({type(module).__name__}). Generic identity is forbidden: "
                "write an explicit rule or stop here"
            )
        if isinstance(module, covered):
            stash.register_site(module, label)
            handles.append(module.register_forward_hook(make_forward_hook(module)))
            handles.append(module.register_full_backward_hook(make_backward_hook(module)))
    # In-place activations (nn.ReLU(inplace=True) and friends) would mutate
    # the hook-wrapped site outputs, which autograd forbids; flipping the
    # flag is semantics-preserving (same values, out of place) and restored.
    inplace_flags = [
        module for module in model.modules() if getattr(module, "inplace", False) is True
    ]
    try:
        for module in inplace_flags:
            module.inplace = False
        leaf = inputs.detach().clone().requires_grad_(True)
        model.eval()
        output = model(leaf)
        seed = torch.zeros_like(output)
        seed[..., target_index] = 1.0
        output.backward(gradient=seed)
        assert leaf.grad is not None
        relevance = (leaf * leaf.grad).detach()
    finally:
        for handle in handles:
            handle.remove()
        for module in inplace_flags:
            module.inplace = True
    return relevance, ledger, stash.leftovers()


# --- END docs recipe ---------------------------------------------------------


class _BiasFreeMlp(nn.Module):
    """Bias-free ReLU MLP: the LRP-0 == input-x-gradient self-oracle model."""

    def __init__(self) -> None:
        """Fixed bias-free weights."""

        super().__init__()
        self.lin1 = nn.Linear(4, 6, bias=False, dtype=torch.float64)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(6, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.lin1.weight.copy_(torch.arange(24, dtype=torch.float64).reshape(6, 4) / 11.0 - 1.0)
            self.lin2.weight.copy_(torch.arange(18, dtype=torch.float64).reshape(3, 6) / 7.0 - 1.2)

    def forward(self, x: Tensor) -> Tensor:
        """Linear-ReLU-Linear forward."""

        return self.lin2(self.act(self.lin1(x)))


class _ConcatFire(nn.Module):
    """Fire-like block: two convs whose outputs concatenate (one honest rule)."""

    def __init__(self) -> None:
        """Fixed bias-free convs."""

        super().__init__()
        self.left = nn.Conv2d(2, 2, 1, bias=False, dtype=torch.float64)
        self.right = nn.Conv2d(2, 2, 1, bias=False, dtype=torch.float64)
        self.head = nn.Conv2d(4, 3, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.left.weight.copy_(
                torch.arange(4, dtype=torch.float64).reshape(2, 2, 1, 1) / 3.0 + 0.1
            )
            self.right.weight.copy_(
                -torch.arange(4, dtype=torch.float64).reshape(2, 2, 1, 1) / 5.0 - 0.2
            )
            self.head.weight.copy_(
                torch.arange(12, dtype=torch.float64).reshape(3, 4, 1, 1) / 6.0 - 0.8
            )

    def forward(self, x: Tensor) -> Tensor:
        """Concat the two branches, project, and pool to logits."""

        merged = torch.cat([self.left(x), self.right(x)], dim=1)
        return self.head(torch.relu(merged)).mean(dim=(2, 3))


def test_lrp0_self_oracle_bias_free_relu() -> None:
    """LRP-0 (epsilon=0) on a bias-free ReLU net == input x gradient, exact."""

    model = _BiasFreeMlp()
    x = torch.tensor([[0.7, -0.3, 0.5, 0.2]], dtype=torch.float64)
    relevance, ledger, leftovers = epsilon_lrp(model, x, target_index=1, epsilon=0.0)
    reference = attribution.input_x_grad(model, x, target=1)
    torch.testing.assert_close(relevance, reference.values, rtol=1e-12, atol=1e-12)
    # Exact conservation at every site (no bias, no epsilon absorption).
    assert len(ledger) == 2
    for row in ledger:
        assert row["conservation_delta_rel"] < 1e-12, row
        assert row["nonfinite"] is False
        assert row["grad_slots"] == 1
    assert leftovers == {}


@pytest.mark.smoke
def test_epsilon_absorption_is_disclosed_not_hidden() -> None:
    """With epsilon > 0 the ledger discloses absorption; nothing pretends exactness."""

    model = _BiasFreeMlp()
    x = torch.tensor([[0.7, -0.3, 0.5, 0.2]], dtype=torch.float64)
    _relevance, ledger, _left = epsilon_lrp(model, x, target_index=1, epsilon=0.05)
    assert any(row["conservation_delta_abs"] > 0 for row in ledger)
    for row in ledger:
        assert {"input_relevance_sum", "output_relevance_sum", "dtype", "device"} <= set(row)


def test_uncovered_op_refuses_by_exact_label() -> None:
    """A parameterized non-Linear/Conv site refuses naming its exact label."""

    model = nn.Sequential(
        nn.Linear(4, 4, dtype=torch.float64),
        nn.BatchNorm1d(4, dtype=torch.float64),
        nn.Linear(4, 2, dtype=torch.float64),
    )
    with pytest.raises(AttributionError) as excinfo:
        epsilon_lrp(model, torch.randn(2, 4, dtype=torch.float64), target_index=0)
    message = str(excinfo.value)
    assert "'1'" in message and "BatchNorm1d" in message
    assert "Generic identity is forbidden" in message


def test_concat_splits_relevance_by_slice() -> None:
    """The concat rule is the honest slice split: branch sums add up."""

    model = _ConcatFire()
    x = torch.rand(1, 2, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    relevance, ledger, leftovers = epsilon_lrp(model, x, target_index=0, epsilon=0.0)
    assert leftovers == {}
    by_site = {row["site"]: row for row in ledger}
    assert set(by_site) == {"left", "right", "head"}
    # Bias-free, epsilon 0: the head's input relevance equals the two
    # branches' output relevance exactly -- the concat added no parameter.
    branch_output_total = (
        by_site["left"]["output_relevance_sum"] + by_site["right"]["output_relevance_sum"]
    )
    assert by_site["head"]["input_relevance_sum"] == pytest.approx(branch_output_total, rel=1e-9)
    assert relevance.shape == x.shape


def test_sitestash_pairing_proven_on_reused_module() -> None:
    """D30: stash/fetch pairing is correct when ONE module fires twice."""

    stash = SiteStash()
    module = nn.Linear(2, 2)
    stash.register_site(module, "shared")
    assert stash.mark_firing(module) == 0
    stash.stash(module, "input", "first-firing")
    assert stash.mark_firing(module) == 1
    stash.stash(module, "input", "second-firing")
    # Backward visits the SECOND firing first: LIFO pairing.
    assert stash.fetch(module, "input") == "second-firing"
    assert stash.fetch(module, "input") == "first-firing"
    with pytest.raises(AttributionError) as excinfo:
        stash.fetch(module, "input")
    assert excinfo.value.fields["code"] == "lrp_stash_unpaired"
    assert "shared" in str(excinfo.value)


def test_sitestash_leftovers_disclosed() -> None:
    """Un-fetched stashes are reported, never silently dropped."""

    stash = SiteStash()
    module = nn.Linear(2, 2)
    stash.register_site(module, "dead-branch")
    stash.mark_firing(module)
    stash.stash(module, "input", object())
    assert stash.leftovers() == {"dead-branch": 1}


@pytest.mark.heavy
@pytest.mark.real_model
def test_squeezenet_full_ledger_row() -> None:
    """The FULL squeezenet1_1 ledger (D32 blocking row; ~5 MB, fetched once).

    Zero BatchNorm, no residual add, exactly one concat per Fire module --
    the architecture the litmus recipe covers end to end. Every conv site
    lands one finite ledger row; the per-site conservation deltas are the
    DISCLOSURE (epsilon and bias absorption are absorption, never called
    exact conservation).
    """

    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.squeezenet1_1(
        weights=torchvision_models.SqueezeNet1_1_Weights.IMAGENET1K_V1
    )
    model.eval()
    generator = torch.Generator().manual_seed(7)
    image = torch.randn(1, 3, 224, 224, generator=generator)
    relevance, ledger, leftovers = epsilon_lrp(model, image, target_index=207, epsilon=1e-6)
    assert relevance.shape == image.shape
    assert bool(torch.isfinite(relevance).all())
    # squeezenet1_1 has 26 conv sites (first conv + 8 fires x 3 + classifier).
    assert len(ledger) == 26
    assert {row["site"] for row in ledger} >= {"features.0", "classifier.1"}
    for row in ledger:
        assert row["nonfinite"] is False, row["site"]
        assert row["grad_slots"] == 1
    assert leftovers == {}
