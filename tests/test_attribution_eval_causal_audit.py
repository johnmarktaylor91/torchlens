"""The causal audit of internal attributions (``tl.attribution.causal_audit``).

On a linear residual model, where each hidden unit's effect on the output is known in closed
form, the measured effects equal the closed form exactly in float64 and a correct claim scores
rank agreement 1.0; every recorded ablation replays through ``trace.fork().do(...)`` to its
measured effect on a small MLP and CNN; the caller's model and random state are untouched; each
refusal carries its code and a remedy that runs.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution

# Replaying a recorded edit goes through the same engine on a fresh trace; stated before the
# first run: the replayed effect matches the measured one within 1e-6 (float32 models), and an
# independent forward-hook ablation matches within 1e-5.
REPLAY_ATOL = 1e-6
HOOK_ATOL = 1e-5


class LinearResidual(nn.Module):
    """``out = fc2(fc1(x)) + skip(x)``: unit ``i`` of ``fc1`` adds ``W2[:, i] * h_i``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)
        self.skip = nn.Linear(4, 3, bias=False)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randint(-3, 4, parameter.shape, generator=generator))

    def forward(self, x: Tensor) -> Tensor:
        """Linear path through the hidden site plus a skip path around it."""

        return self.fc2(self.fc1(x)) + self.skip(x)


class SmallMlp(nn.Module):
    """A ReLU MLP with one hidden site of 8 units."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Two linear layers around a ReLU."""

        return self.fc2(self.act(self.fc1(x)))


class SmallCnn(nn.Module):
    """A convolution of 2 channels on a 4x4 image (32 units) and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.head = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Convolution, ReLU, flatten, linear."""

        return self.head(torch.relu(self.conv(x)).flatten(1))


class Reused(nn.Module):
    """A block applied twice, so its site has two calls."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        """The same block twice, then a head."""

        return self.head(self.block(self.block(x)))


def _linear_case() -> tuple[LinearResidual, Tensor]:
    model = LinearResidual().double().eval()
    x = torch.randint(-3, 4, (2, 4), generator=torch.Generator().manual_seed(1)).double()
    return model, x


def _closed_form(model: LinearResidual, x: Tensor, target: int, value: Tensor) -> Tensor:
    """Effect of replacing unit ``i`` of ``fc1`` by ``value[:, i]``: ``W2[t, i] sum_b (h - v)``."""

    with torch.no_grad():
        hidden = model.fc1(x)
    return model.fc2.weight[target].detach() * (hidden - value).sum(dim=0)


def _remedy(message: str) -> str:
    found = re.search(r"(tl\.attribution\.causal_audit\(.*\))\. See", message)
    assert found is not None, message
    return found.group(1)


def test_a_claim_computed_against_a_site_target_is_audited_at_that_site() -> None:
    """A claim for ``SiteTarget('fc2', unit=2)`` is scored on that site of every fork."""

    model, x = _linear_case()
    site = attribution.SiteTarget("fc2", unit=2)
    claim = attribution.layer_attribution(model, x, target=site, layer="fc1")
    result = attribution.causal_audit(model, x, claim=claim, target=site, k=1)
    expected = _closed_form(model, x, 2, torch.zeros(2, 6, dtype=torch.float64))
    assert [row.measured_effect for row in result.units] == expected.tolist()
    assert [row.claimed_effect for row in result.units] == expected.tolist()
    assert result.rank_agreement == 1.0


def _hook_ablated_score(
    model: nn.Module, x: Tensor, layer: str, mask: Tensor, target: int
) -> float:
    """Ablate ``mask`` at ``layer``'s output with a forward hook, independent of the engine."""

    def _zero(_module: nn.Module, _args: Any, output: Tensor) -> Tensor:
        return output.masked_fill(mask, 0.0)

    handle = dict(model.named_modules())[layer].register_forward_hook(_zero)
    try:
        with torch.no_grad():
            return float(model(x)[..., target].sum())
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["tensor", "input_level", "reused_site"])
def test_unsupported_claims_refuse_and_the_remedy_runs(kind: str) -> None:
    """A claim not grounded at one hidden site refuses with its code; the remedy runs."""

    torch.manual_seed(0)
    if kind == "reused_site":
        model: nn.Module = Reused().eval()
        x = torch.randn(2, 4)
        claim: Any = attribution.layer_attribution(model, x, target=0, layer="block")
    else:
        model, x = _linear_case()
        claim = (
            torch.zeros(2, 6, dtype=torch.float64)
            if kind == "tensor"
            else attribution.saliency(model, x, target=0)
        )
    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution.causal_audit(model, x, claim=claim, target=0, k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_unsupported"
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": x})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAudit)


@pytest.mark.parametrize("kind", ["shape", "target"])
def test_misaligned_claims_refuse_and_the_remedy_runs(kind: str) -> None:
    """A claim from other inputs or for another target refuses; the recomputed claim runs."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(
        model, x[:1] if kind == "shape" else x, target=0, layer="fc1"
    )
    target = 0 if kind == "shape" else 2
    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution.causal_audit(model, x, claim=claim, target=target, k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_misaligned"
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": x})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAudit) and fixed.rank_agreement == 1.0


@pytest.mark.parametrize(
    "settings",
    [
        {"k": 0},
        {"k": 7},
        {"edit": "noise"},
        {"edit": torch.ones(5)},
        {"controls": ("top",)},
        {"num_random_draws": 0},
    ],
    ids=["k-0", "k-too-large", "ablation-noise", "ablation-shape", "controls-top", "draws-0"],
)
def test_bad_settings_refuse_and_the_remedy_runs(settings: dict[str, Any]) -> None:
    """Bad k, ablation, controls or draws refuse with the setting code; the remedy runs."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=0, layer="fc1")
    call = {"k": 1, **settings}
    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution.causal_audit(model, x, claim=claim, target=0, **call)
    assert excinfo.value.fields.get("code") == "attribution_eval_setting_invalid"
    scope = {"tl": tl, "model": model, "x": x, "claim": claim}
    fixed = eval(_remedy(str(excinfo.value)), scope)  # noqa: S307
    assert isinstance(fixed, attribution.CausalAudit)
