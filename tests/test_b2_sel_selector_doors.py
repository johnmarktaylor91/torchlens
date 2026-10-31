"""Module and stage names resolve to that module's output at every selector door (U26.1).

The rule (DOCUMENTED-UNSTABLE): a module address, a name relative to a wrapped
inner model, or a dotted name whose module was registered with dots flattened to
underscores resolves to that module's output at ``tl.extract``, ``tl.pluck``,
``trace[...]``, ``tl.trace(save=[...])``, and ``trace.run(until=...)``. The
replicas below reproduce the two timm wrapper shapes without importing timm:
``FeatureListNet`` (a ``ModuleDict`` whose forward returns every stage output, so
each stage is also a model output) and ``FeatureGetterNet`` (a wrapper whose stage
names are relative to its ``model`` child). The real timm models are exercised in
``tests/test_b2_sel_timm_stage_names.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import AmbiguousOpLookupError
from torchlens._trace_selector_helpers import canonical_module_site_key


def _stage(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())


class _FeatureList(nn.ModuleDict):
    """timm ``FeatureListNet`` shape: every named stage output is returned."""

    def __init__(self) -> None:
        super().__init__({"stem": nn.Conv2d(3, 4, 3, padding=1), "layer1": _stage(4)})
        self["layer2"] = _stage(4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for module in self.values():
            x = module(x)
            outs.append(x)
        return outs


class _Flattened(nn.ModuleDict):
    """timm ``FeatureListNet`` over nested stages: ``stages.0`` registered as ``stages_0``."""

    def __init__(self) -> None:
        super().__init__({"stem": nn.Conv2d(3, 4, 3, padding=1)})
        self["stages_0"] = _stage(4)
        self["stages_1"] = _stage(4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for name, module in self.items():
            x = module(x)
            if name.startswith("stages_"):
                outs.append(x)
        return outs


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(6, 6), nn.GELU()) for _ in range(3)])
        self.head = nn.Linear(6, 2)


class _Getter(nn.Module):
    """timm ``FeatureGetterNet`` shape: stage names are relative to ``self.model``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for block in self.model.blocks:
            x = block(x)
            outs.append(x)
        return outs


class _Classifier(nn.Module):
    """A resnet-like classifier whose final ``fc`` output is the model output."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Linear(4 * 5 * 5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


#: (model factory, input factory, stage names a user holds, address each names)
_CASES = {
    "feature_list": (
        _FeatureList,
        lambda: torch.randn(1, 3, 5, 5),
        {"stem": "stem", "layer1": "layer1", "layer2": "layer2"},
    ),
    "flattened": (
        _Flattened,
        lambda: torch.randn(1, 3, 5, 5),
        {"stages.0": "stages_0", "stages.1": "stages_1"},
    ),
    "getter": (
        _Getter,
        lambda: torch.randn(2, 6),
        {f"blocks.{i}": f"model.blocks.{i}" for i in range(3)},
    ),
    "classifier": (_Classifier, lambda: torch.randn(1, 3, 5, 5), {"fc": "fc"}),
}


def _hooked_outputs(
    model: nn.Module, x: torch.Tensor, addresses: list[str]
) -> dict[str, torch.Tensor]:
    """Return each module's forward output, captured by plain PyTorch hooks."""

    outputs: dict[str, torch.Tensor] = {}
    handles = [
        model.get_submodule(address).register_forward_hook(
            lambda _m, _i, out, address=address: outputs.setdefault(address, out.detach().clone())
        )
        for address in addresses
    ]
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()
    return outputs


@pytest.fixture(params=sorted(_CASES))
def case(request: pytest.FixtureRequest) -> tuple[nn.Module, torch.Tensor, dict[str, str], dict]:
    factory, make_input, names = _CASES[request.param]
    torch.manual_seed(0)
    model = factory().eval()
    x = make_input()
    return model, x, names, _hooked_outputs(model, x, list(names.values()))


def test_extract_and_pluck_resolve_stage_names(case) -> None:
    model, x, names, expected = case
    for name, address in names.items():
        listed = tl.extract(model, x, [name])
        assert list(listed) == [name]
        torch.testing.assert_close(listed[name], expected[address])
        torch.testing.assert_close(tl.extract(model, x, {"s": name})["s"], expected[address])
        torch.testing.assert_close(tl.pluck(model, x, name), expected[address])


def test_trace_getitem_and_save_list_resolve_stage_names(case) -> None:
    model, x, names, expected = case
    full = tl.trace(model, x)
    selective = tl.trace(model, x, save=list(names))
    for name, address in names.items():
        torch.testing.assert_close(full[name].out, expected[address])
        torch.testing.assert_close(selective[name].out, expected[address])
        assert full[name].label == full[address].label


def _until_outcome(trace: tl.Trace, x: torch.Tensor, site: str) -> tuple:
    """Return where an ``until=`` run stopped and how many ops it executed."""

    truncation = trace.run(inputs=x, until=site).report.truncation
    if truncation is None:
        return ("full run",)
    return (truncation.stopped_at, truncation.executed_count)


def test_run_until_resolves_stage_names(case) -> None:
    model, x, names, _expected = case
    trace = tl.trace(model, x)
    for name, address in names.items():
        label = trace[address].layer_label
        assert _until_outcome(trace, x, name) == _until_outcome(trace, x, label)


def test_relative_name_matching_several_modules_refuses_with_candidates() -> None:
    class _Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Sequential(nn.Linear(2, 2))
            self.b = nn.Sequential(nn.Linear(2, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x)) + 1

    trace = tl.trace(_Two(), torch.randn(1, 2))
    with pytest.raises(AmbiguousOpLookupError) as exc_info:
        trace["0"]
    assert "'a.0'" in str(exc_info.value) and "'b.0'" in str(exc_info.value)


def test_canonical_key_is_none_for_every_existing_name() -> None:
    """The name mapping never changes a lookup that resolves today."""

    for factory, make_input, _names in _CASES.values():
        trace = tl.trace(factory().eval(), make_input())
        names = set(trace._layer_dict_all_keys) | set(trace._layer_logs) | set(trace.modules.keys())
        for name in names:
            if isinstance(name, str):
                assert canonical_module_site_key(trace, name) is None, name


def test_pluck_resolves_the_module_it_stopped_at() -> None:
    """A halted trace's output alias does not steal the stop module's name."""

    class _TwoStage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.second = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(torch.relu(self.first(x)))

    torch.manual_seed(0)
    model, x = _TwoStage().eval(), torch.randn(2, 4)
    torch.testing.assert_close(tl.pluck(model, x, "first", stop_after="first"), model.first(x))
