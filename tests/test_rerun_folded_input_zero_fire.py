"""The rerun's same-inputs check reads a folded input pass by pass.

Two inputs of identical shape and dtype consumed by one loop body fold into
one multi-pass input Layer, listed in ``input_layers`` once per pass under one
bare label. When a staged edit fires at zero sites, the rerun asks
``_same_rerun_inputs`` whether it replayed the source trace's graph on the
source inputs; that check read the Layer's ``out`` mirror, which raises on a
multi-pass Layer, so the rerun crashed with ``InvalidArgumentError`` instead of
the typed ``rerun_selector_unmatchable_live`` refusal or the zero-fire
warning. The check now pairs each listing with its own pass through the shared
per-listing rule. The oracles are the tensors the test passes in: equal inputs
compare equal, and a change to either input compares unequal, as for
single-pass inputs.

Under call-site grouping input nodes are singletons (grouping rule section 2),
so torch capture no longer folds the loop model's inputs; the module keeps the
loop model as the multi-pass case through its loop-body relu (rule row 2).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import LiveModeLabelError
from torchlens.intervention.rerun import _same_rerun_inputs
from torchlens.options import CaptureOptions, InterventionOptions


class _LoopOverInputs(nn.Module):
    """Two same-shape inputs consumed by one loop body (its relu is one 2-pass Layer)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(1, 4)
        for x in (a, b):
            h = h + torch.relu(self.lin(x))
        return h


class _TwoInputs(nn.Module):
    """Single-pass control: two inputs through separate code."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(a)) + torch.relu(self.lin(b)) * 2


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, generator=generator), torch.randn(1, 4, generator=generator)


def _traces(model_cls: type[nn.Module], inputs: tuple[Any, Any], other: tuple[Any, Any]) -> Any:
    torch.manual_seed(0)
    model = model_cls().eval()
    return tl.trace(model, inputs), tl.trace(model, other)


def test_folded_input_premise() -> None:
    """Premise under call-site grouping: input nodes never fold.

    Grouping rule section 2 (special nodes): input nodes are singletons, so the
    loop model's two inputs are two single-pass input Layers, like the
    control's; its loop-body relu is one 2-pass Layer (rule row 2). The
    per-listing pairing in ``_same_rerun_inputs`` is still exercised by the
    loop model's two listings, and the rest of this module pins the rerun on
    the loop model's multi-pass relu.
    """

    folded, _ = _traces(_LoopOverInputs, _inputs(0), _inputs(0))
    control, _ = _traces(_TwoInputs, _inputs(0), _inputs(0))
    assert folded.input_layers == ["input_1", "input_2"]
    assert all(len(folded[label].ops) == 1 for label in folded.input_layers)
    relu_layers = [layer for layer in folded._layer_logs.values() if layer.func_name == "relu"]
    assert [len(layer.ops) for layer in relu_layers] == [2]
    assert control.input_layers == ["input_1", "input_2"]


@pytest.mark.parametrize("model_cls", [_LoopOverInputs, _TwoInputs], ids=["folded", "single"])
def test_same_inputs_compare_equal(model_cls: type[nn.Module]) -> None:
    a, b = _inputs(0)
    source, rerun = _traces(model_cls, (a, b), (a.clone(), b.clone()))
    assert _same_rerun_inputs(source, rerun) is True


@pytest.mark.parametrize("model_cls", [_LoopOverInputs, _TwoInputs], ids=["folded", "single"])
@pytest.mark.parametrize("changed", [0, 1], ids=["first", "second"])
def test_a_changed_input_compares_unequal(model_cls: type[nn.Module], changed: int) -> None:
    inputs = _inputs(0)
    other = tuple(t + 1.0 if i == changed else t.clone() for i, t in enumerate(inputs))
    source, rerun = _traces(model_cls, inputs, other)
    assert _same_rerun_inputs(source, rerun) is False


def _relu_pass_labels(trace: Any) -> list[str]:
    # Pass-qualified op labels: the loop model's relu is one 2-pass Layer
    # (grouping rule row 2), so its bare layer label names no single op.
    return [str(site.label) for site in trace.find_sites(tl.func("relu"), max_fanout=8)]


def _intervention_case(
    model_cls: type[nn.Module],
) -> tuple[nn.Module, tuple[torch.Tensor, torch.Tensor], Any, str]:
    torch.manual_seed(0)
    model = model_cls().eval()
    inputs = _inputs(0)
    baseline = tl.trace(model, inputs, capture=CaptureOptions(replayable=True))
    # The relu that reads the second input: a change to that input changes it.
    label = _relu_pass_labels(baseline)[-1]
    return model, inputs, baseline, label


def _value_predicate(baseline: Any, label: str) -> Any:
    """Match the second-input relu by label and by its saved value.

    A live forward sees no ``layer_label``, so it fires at zero sites. On the
    source trace it matches one site. On a rerun whose second input changed,
    the value differs, so it matches nothing on the rerun's own trace either:
    there the miss is the inputs and the rerun warns instead of refusing.
    """

    reference = baseline[label].out.clone()

    layer_label, pass_index = label.rsplit(":", 1)

    def predicate(p: Any) -> bool:
        if getattr(p, "layer_label", None) != layer_label:
            return False
        if str(getattr(p, "pass_index", None)) != pass_index:
            return False
        out = getattr(p, "out", None)
        return isinstance(out, torch.Tensor) and torch.equal(out, reference)

    return tl.where(predicate)


def _rerun(baseline: Any, site: Any, model: nn.Module, inputs: tuple[Any, Any]) -> Any:
    return baseline.fork().do(
        site,
        tl.zero_ablate(),
        model=model,
        x=inputs,
        intervention=InterventionOptions(engine="rerun"),
    )


@pytest.mark.parametrize("model_cls", [_LoopOverInputs, _TwoInputs], ids=["folded", "single"])
def test_zero_fire_rerun_on_same_inputs_refuses_typed(model_cls: type[nn.Module]) -> None:
    model, inputs, baseline, label = _intervention_case(model_cls)
    member = baseline.fork()
    before = member[label].out.clone()
    with pytest.raises(LiveModeLabelError) as excinfo:
        member.do(
            _value_predicate(baseline, label),
            tl.zero_ablate(),
            model=model,
            x=tuple(t.clone() for t in inputs),
            intervention=InterventionOptions(engine="rerun"),
        )
    assert excinfo.value.fields["code"] == "rerun_selector_unmatchable_live"
    assert excinfo.value.fields["source_site_count"] == 1
    assert torch.equal(member[label].out, before)


@pytest.mark.parametrize("model_cls", [_LoopOverInputs, _TwoInputs], ids=["folded", "single"])
def test_zero_fire_rerun_on_a_changed_second_input_warns(model_cls: type[nn.Module]) -> None:
    model, (a, b), baseline, label = _intervention_case(model_cls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _rerun(baseline, _value_predicate(baseline, label), model, (a.clone(), b + 1.0))
    assert [w for w in caught if "fired at zero sites" in str(w.message)]
    assert result.last_run["hooks_fired"] == 0
    assert result.last_run["hooks_unfired"] > 0
