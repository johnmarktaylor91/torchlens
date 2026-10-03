"""Patching-helper honesty fixes (A03: WT1 A-I items 1 and 4; mikit F7/F8).

Three defect classes pinned here:

* attribution patching crashed on ANY model WITH parameters -- the
  counterfactual guard's unconditional in-place restore bumped autograd
  version counters of params consumed by the captured forwards (the whole
  pre-fix test surface was zero-parameter toys, which is why it shipped).
* activation patching could return tables bitwise-equal to the corrupted
  baseline with no error when the facet hook never fired (publishable-looking
  null results); patched reruns now demand positive fire evidence.
* a user ``capture=`` silently dropped the helper's REQUIRED capture fields
  (facets came up absent / narrowed); required fields now compose, explicit
  conflicts refuse typed, and removed flat spellings refuse with the grouped
  remedy named.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.semantic import FacetSpec
from torchlens.semantic.patching import PatchApplicationError
from torchlens.utils._torch_compat import TorchCapabilityWarning

pytestmark = pytest.mark.smoke


class ParamHeads(nn.Module):
    """Two-head result source with REAL parameters (one important, one inert)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        return torch.stack((y, torch.zeros_like(y)), dim=2)


class ParamAttention(nn.Module):
    """Attention-like module exposing a writable per-head result facet."""

    def __init__(self) -> None:
        super().__init__()
        self.n_heads = 2
        self.result_source = ParamHeads()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.result_source(x).sum(dim=2)


class ParamBlock(nn.Module):
    """Residual block around the parameterized attention."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = ParamAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class ParamToy(nn.Module):
    """Toy model WITH parameters -- the class the pre-fix suite never had."""

    def __init__(self) -> None:
        super().__init__()
        self.block = ParamBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def param_attention_recipe(module: Any) -> dict[str, Any]:
    """Expose writable attention facets for the parameterized toy."""

    trace = module.trace
    result_module = trace.modules[f"{module.address}.result_source"]
    result_op = trace.ops[result_module.calls[0].output_ops[0]]
    output_op = trace.ops[module.calls[0].output_ops[0]]
    return {
        "result": FacetSpec.from_home(result_op, recipe_id="param_attention"),
        "attn_out": FacetSpec.from_home(output_op, recipe_id="param_attention"),
        "n_heads": 2,
        "head": module.facets.head,
    }


@pytest.fixture(autouse=True, scope="module")
def _register_recipes() -> Iterator[None]:
    """Register this module's recipes at RUN time, restoring the registry after."""

    from torchlens.semantic import facets as _facets

    saved = list(_facets._REGISTRY)
    tl.facets.register(
        class_name="ParamAttention",
        target_scope="module",
        facets=("result", "attn_out", "n_heads", "head"),
    )(param_attention_recipe)
    try:
        yield
    finally:
        _facets._REGISTRY[:] = saved
        _facets._REGISTRY_VERSION += 1


def _metric(log: Any) -> torch.Tensor:
    return log[log.output_layers[0]].out[0, 0, 0]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.zeros(1, 3, 4)
    corrupted = torch.zeros(1, 3, 4)
    clean[0, 0, 0] = 2.0
    return clean, corrupted


def test_attribution_patch_works_on_parameterized_model() -> None:
    """The autograd/param-restore crash: red on ANY parameterized model pre-fix.

    The guard's unconditional ``copy_`` bumped every parameter's autograd
    version counter between the two baseline captures, so ``log_backward``
    raised "modified by an inplace operation". Equality-gated restore leaves
    value-equal parameters untouched.
    """

    torch.manual_seed(0)
    model = ParamToy()
    clean, corrupted = _inputs()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    scores = tl.facets.patching.attribution_patch_attention_heads(
        model, clean.requires_grad_(), corrupted.requires_grad_(), _metric
    )

    assert scores.shape == (1, 2)
    assert bool(torch.isfinite(scores).all())
    # The inert head has exactly zero attributed effect.
    assert float(scores[0, 1]) == 0.0
    # Caller's model state is fully restored: values AND grad slots.
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name
        assert param.grad is None, name


def test_guard_restores_preexisting_param_grads() -> None:
    """A caller's pre-existing .grad survives the helper unchanged."""

    torch.manual_seed(0)
    model = ParamToy()
    seeded_grad = torch.full_like(model.block.attn.result_source.proj.weight, 0.25)
    model.block.attn.result_source.proj.weight.grad = seeded_grad.clone()
    clean, corrupted = _inputs()

    tl.facets.patching.attribution_patch_attention_heads(
        model, clean.requires_grad_(), corrupted.requires_grad_(), _metric
    )

    assert torch.equal(model.block.attn.result_source.proj.weight.grad, seeded_grad)


def test_activation_patch_on_parameterized_model_produces_real_effects() -> None:
    """Head patching on the parameterized toy: real values, real ordering."""

    torch.manual_seed(0)
    model = ParamToy()
    clean, corrupted = _inputs()
    table = tl.facets.patching.activation_patch_attention_heads(model, clean, corrupted, _metric)
    assert table.shape == (1, 2)
    # Patching the important head moves the metric; the inert head cannot.
    corrupted_metric = float(table[0, 1])
    assert float(table[0, 0]) != corrupted_metric


class _StubRecord:
    """Fire-record stand-in with a timestamp and a replaced flag."""

    def __init__(self, timestamp: float, replaced: bool) -> None:
        self.timestamp = timestamp
        self.replaced = replaced
        self.site_label = "stub_site_1"
        self.target_label = "stub_target_1"


class _StubLayer:
    def __init__(self, records: list[_StubRecord]) -> None:
        self.interventions = records


class _StubPatchedLog:
    """Minimal patched-trace stand-in for the fire-evidence seam."""

    def __init__(self, records: list[_StubRecord], hooks_fired: int) -> None:
        self.last_run = {"started_at": 100.0, "timestamp": 101.0, "hooks_fired": hooks_fired}
        self.layer_list = [_StubLayer(records)]


def test_fire_ledger_refuses_never_fired() -> None:
    """Zero fires + zero records = the silent-no-op shape; must refuse typed.

    Pre-fix the helpers published a table bitwise-equal to the corrupted
    baseline in this state (the measured real-HF silent no-op class).
    """

    from torchlens.semantic.patching import _require_effective_patch

    with pytest.raises(PatchApplicationError, match="never fired") as never_fired_exc:
        _require_effective_patch(
            _StubPatchedLog([], hooks_fired=0),
            where="facet 'resid_pre' on module 'block'",
            fire_ledger={"fires": 0, "identical": 0},
        )
    assert never_fired_exc.value.fields["code"] == "patch_ineffective"


def test_fire_ledger_refuses_all_refused_replacements() -> None:
    """Fires whose every replacement was refused (replaced=False) must refuse.

    This is the exact measured HF mechanism: the live-hook engine refused the
    replacement at an aliasing site with a warning only, and the appliances
    never read the fire record's replaced flag.
    """

    from torchlens.semantic.patching import _require_effective_patch

    with pytest.raises(PatchApplicationError, match="refused.*replaced=False"):
        _require_effective_patch(
            _StubPatchedLog([_StubRecord(100.5, replaced=False)], hooks_fired=1),
            where="facet 'resid_pre' on module 'block'",
            fire_ledger={"fires": 1, "identical": 0},
        )


def test_fire_ledger_accepts_replaced_fire() -> None:
    from torchlens.semantic.patching import _require_effective_patch

    _require_effective_patch(
        _StubPatchedLog([_StubRecord(100.5, replaced=True)], hooks_fired=1),
        where="facet 'resid_pre' on module 'block'",
        fire_ledger={"fires": 1, "identical": 0},
    )


def test_fire_ledger_ignores_stale_records_from_earlier_runs() -> None:
    """Records minted BEFORE this run cannot vouch for it."""

    from torchlens.semantic.patching import _require_effective_patch

    with pytest.raises(PatchApplicationError, match="never fired"):
        _require_effective_patch(
            _StubPatchedLog([_StubRecord(99.0, replaced=True)], hooks_fired=0),
            where="facet 'resid_pre' on module 'block'",
            fire_ledger={"fires": 0, "identical": 0},
        )


def test_all_identical_campaign_disclosed() -> None:
    """A campaign whose every fire replaced an identical value warns ONCE.

    Identical clean/corrupted inputs make every patched value identical; the
    table legitimately equals the baseline, and the disclosure names the
    mis-homed-facet possibility instead of staying silent.
    """

    torch.manual_seed(0)
    model = ParamToy()
    same = torch.ones(1, 3, 4)

    with pytest.warns(TorchLensWarning, match="IDENTICAL") as caught:
        table = tl.facets.patching.activation_patch_attention_output(
            model, same, same.clone(), _metric
        )
    assert table.shape == (1,)
    # Key on the disclosure code: unrelated TorchLensWarnings (e.g. a lingering
    # log's gc-time notice) may share the recording context in a full session.
    disclosure = next(
        w.message
        for w in caught
        if isinstance(w.message, TorchLensWarning)
        and w.message.fields.get("code") == "patch_campaign_all_identical"
    )
    assert disclosure.fields["remedy"].startswith("inspect")


def test_partially_identical_campaign_stays_silent() -> None:
    """Ordinary science (an inert head next to a live one) must NOT warn."""

    torch.manual_seed(0)
    model = ParamToy()
    clean, corrupted = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        # A floor-torch install may fire a one-time TorchCapabilityWarning
        # (itself a TorchLensWarning) from an unrelated capability probe
        # tripped by this capture; tolerate that category specifically
        # without loosening the "ordinary science stays silent" guarantee.
        warnings.simplefilter("ignore", TorchCapabilityWarning)
        tl.facets.patching.activation_patch_attention_heads(model, clean, corrupted, _metric)


def test_user_capture_composes_required_fields() -> None:
    """mikit F7: a user capture= no longer drops the helper's required fields.

    ``save_arg_values``/``layers_to_save`` used to vanish whenever ANY
    ``capture=`` was passed; an unrelated user option must compose with them.
    """

    torch.manual_seed(0)
    model = ParamToy()
    clean, corrupted = _inputs()
    table = tl.facets.patching.activation_patch_attention_output(
        model,
        clean,
        corrupted,
        _metric,
        trace_kwargs={"capture": tl.options.CaptureOptions(save_code_context=True)},
    )
    assert table.shape == (1,)


def test_explicitly_conflicting_capture_refuses_typed() -> None:
    with pytest.raises(ValueError, match="require capture options"):
        tl.facets.patching.activation_patch_attention_output(
            ParamToy(),
            *_inputs(),
            _metric,
            trace_kwargs={"capture": tl.options.CaptureOptions(layers_to_save=["block"])},
        )


def test_explicit_matching_capture_passes() -> None:
    torch.manual_seed(0)
    model = ParamToy()
    clean, corrupted = _inputs()
    table = tl.facets.patching.activation_patch_attention_output(
        model,
        clean,
        corrupted,
        _metric,
        trace_kwargs={
            "capture": tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True)
        },
    )
    assert table.shape == (1,)
