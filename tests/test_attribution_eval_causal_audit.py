"""The causal audit of internal attributions (``tl.attribution.causal_audit``).

On a linear residual model, where each hidden unit's effect on the output is known in closed
form, the measured effects equal the closed form exactly in float64 and a correct claim scores
rank agreement 1.0; every recorded ablation replays through ``trace.fork().do(...)`` to its
measured effect on a small MLP and CNN; the caller's model and random state are untouched; each
refusal carries its code and a remedy that runs.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution
from torchlens.attribution._result import AttributionWarning

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


@pytest.mark.parametrize("ablation", ["zero", "mean", "baseline"])
def test_measured_effects_equal_the_closed_form_on_a_linear_residual_model(ablation: str) -> None:
    """Each unit's measured effect equals ``W2[t, i] * sum_b (h_i - v_i)`` exactly (float64)."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=1, layer="fc1")
    with torch.no_grad():
        hidden = model.fc1(x)
    values = {
        "zero": (torch.zeros_like(hidden), "zero"),
        "mean": (hidden.mean(dim=0, keepdim=True).expand_as(hidden), "mean"),
        "baseline": (torch.full_like(hidden, 2.0), 2.0),
    }
    reference, spelling = values[ablation]
    result = attribution.causal_audit(model, x, claim=claim, target=1, k=2, ablation=spelling)
    expected = _closed_form(model, x, 1, reference)
    assert [row.unit for row in result.units] == [(index,) for index in range(6)]
    assert [row.measured_effect for row in result.units] == expected.tolist()
    with torch.no_grad():
        assert result.original_score == float(model(x)[:, 1].sum())
    if ablation == "zero":
        assert [row.claimed_effect for row in result.units] == expected.tolist()
        assert result.rank_agreement == 1.0


def test_a_shuffled_claim_scores_below_the_correct_claim() -> None:
    """Shuffling the claim across units lowers its rank agreement below the correct 1.0."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=1, layer="fc1")
    shuffled = attribution.AttributionResult(
        method=claim.method,
        values=claim.values[:, torch.tensor([3, 0, 5, 1, 4, 2])],
        target_repr=claim.target_repr,
        extra=dict(claim.extra),
    )
    correct = attribution.causal_audit(model, x, claim=claim, target=1, k=2)
    wrong = attribution.causal_audit(model, x, claim=shuffled, target=1, k=2)
    assert correct.rank_agreement == 1.0
    assert wrong.rank_agreement is not None and wrong.rank_agreement < correct.rank_agreement
    assert [row.measured_effect for row in wrong.units] == [
        row.measured_effect for row in correct.units
    ]


def test_top_bottom_and_random_sets_add_up_on_the_linear_model() -> None:
    """Sets are chosen by claimed magnitude; on a linear model joint effects add exactly."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=0, layer="fc1")
    result = attribution.causal_audit(model, x, claim=claim, target=0, k=2, random_draws=3)
    per_unit = {row.unit: row.measured_effect for row in result.units}
    by_magnitude = sorted(per_unit, key=lambda unit: (-abs(per_unit[unit]), unit))
    assert [item.kind for item in result.sets] == ["top", "bottom", "random", "random", "random"]
    top, bottom = result.set_of("top")[0], result.set_of("bottom")[0]
    assert top.units == tuple(by_magnitude[:2]) and bottom.units == tuple(by_magnitude[-2:])
    for item in result.sets:
        assert item.measured_effect == math.fsum(per_unit[unit] for unit in item.units)
        assert item.claimed_effect == item.measured_effect
    again = attribution.causal_audit(model, x, claim=claim, target=0, k=2, random_draws=3)
    assert [item.units for item in again.set_of("random")] == [
        item.units for item in result.set_of("random")
    ]
    assert result.extra["random_measured_mean"] == pytest.approx(
        sum(item.measured_effect for item in result.set_of("random")) / 3, abs=1e-12
    )


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


@pytest.mark.parametrize("case", ["mlp", "cnn"])
def test_recorded_ablations_replay_to_their_measured_effects(case: str) -> None:
    """Every row's and set's recorded edit, replayed by ``trace.fork().do``, gives its effect."""

    torch.manual_seed(0)
    if case == "mlp":
        model, x, layer = SmallMlp().eval(), torch.randn(3, 5), "act"
    else:
        model, x, layer = SmallCnn().eval(), torch.randn(2, 1, 4, 4), "conv"
    claim = attribution.layer_attribution(model, x, target=1, layer=layer)
    result = attribution.causal_audit(model, x, claim=claim, target=1, k=3, random_draws=2)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    assert trace[result.site].out.shape == claim.values.shape
    for point in (*result.units, *result.sets):
        fork = trace.fork()
        try:
            for selection, edit in point.intervention:
                fork.do(selection, edit)
            replayed = float(fork[fork.output_layers[0]].out[..., 1].sum().detach())
        finally:
            fork.cleanup()
        assert result.original_score - replayed == pytest.approx(
            point.measured_effect, abs=REPLAY_ATOL
        )
        units = (point.unit,) if isinstance(point, attribution.CausalAuditUnit) else point.units
        mask = torch.zeros(claim.values.shape, dtype=torch.bool)
        for unit in units:
            mask[(slice(None), *unit)] = True
        hooked = _hook_ablated_score(model, x, layer, mask, 1)
        assert result.original_score - hooked == pytest.approx(point.measured_effect, abs=HOOK_ATOL)
    assert len(result.units) == claim.values[0].numel()


def test_the_caller_model_and_random_state_are_untouched() -> None:
    """Parameters and buffers stay bit-identical, the training flag and global RNG too."""

    torch.manual_seed(0)
    model = SmallCnn().train()
    x = torch.randn(2, 1, 4, 4)
    claim = attribution.layer_attribution(model, x, target=0, layer="conv")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    attribution.causal_audit(model, x, claim=claim, target=0, k=4, ablation="mean")
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), rng)


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
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, x, claim=claim, target=0, k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_unsupported"
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": x})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult)


@pytest.mark.parametrize("kind", ["shape", "target"])
def test_misaligned_claims_refuse_and_the_remedy_runs(kind: str) -> None:
    """A claim from other inputs or for another target refuses; the recomputed claim runs."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(
        model, x[:1] if kind == "shape" else x, target=0, layer="fc1"
    )
    target = 0 if kind == "shape" else 2
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, x, claim=claim, target=target, k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_misaligned"
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": x})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult) and fixed.rank_agreement == 1.0


@pytest.mark.parametrize(
    "settings",
    [
        {"k": 0},
        {"k": 7},
        {"ablation": "noise"},
        {"ablation": torch.ones(5)},
        {"controls": ("top",)},
        {"random_draws": 0},
    ],
    ids=["k-0", "k-too-large", "ablation-noise", "ablation-shape", "controls-top", "draws-0"],
)
def test_bad_settings_refuse_and_the_remedy_runs(settings: dict[str, Any]) -> None:
    """Bad k, ablation, controls or draws refuse with the setting code; the remedy runs."""

    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=0, layer="fc1")
    call = {"k": 1, **settings}
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, x, claim=claim, target=0, **call)
    assert excinfo.value.fields.get("code") == "attribution_eval_setting_invalid"
    scope = {"tl": tl, "model": model, "x": x, "claim": claim}
    fixed = eval(_remedy(str(excinfo.value)), scope)  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult)


def test_many_units_warn_once_and_name_per_unit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the per-unit guard the audit warns once; ``per_unit=False`` measures only the sets."""

    import torchlens.attribution._causal_audit as audit_module

    monkeypatch.setattr(audit_module, "_PER_UNIT_WARN_COUNT", 2)
    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=0, layer="fc1")
    with pytest.warns(AttributionWarning) as record:
        full = attribution.causal_audit(model, x, claim=claim, target=0, k=2)
    coded = [w for w in record if getattr(w.message, "fields", {}).get("code")]
    assert [w.message.fields["code"] for w in coded] == ["attribution_audit_units_many"]
    assert "per_unit=False" in str(coded[0].message)
    lean = attribution.causal_audit(model, x, claim=claim, target=0, k=2, per_unit=False)
    assert lean.units == () and lean.rank_agreement is None
    assert [item.measured_effect for item in lean.sets] == [
        item.measured_effect for item in full.sets
    ]
