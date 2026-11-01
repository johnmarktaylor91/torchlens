"""Backward validation keys module outputs the way the capture does.

Two classics-corpus models returned a bare False from ``scope="backward"``: `aenet-PyTorch`
shares one ``nn.Tanh`` between two ``nn.Sequential``s, and the capture addressed its calls by
the LAST registration path while stock-torch hooks (``named_modules()``) use the first; `IQL`'s
policy returns a ``torch.distributions`` object whose tensor leaves the stock side did not
enumerate, and four of them receive no gradient on either side. These tests pin the shared
fixes: first registration path is canonical, stock enumerates plain-object leaves with the
capture's walk, and "no gradient in stock or capture" is agreement, never evidence (Lead
ruling, 2026-10-31): the asymmetric cases still fail and a run that compares nothing still
returns False.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.classics_corpus._loader import load_model_module
from torchlens.validation import _layer_grad_report as layer_grad_report, last_validation_failure
from torchlens.validation.diagnostics import CHECK_BACKWARD


def _validate_backward(model: nn.Module, inputs: Any, **kwargs: Any) -> bool:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, inputs, scope="backward", seed=0, **kwargs))


def _reason() -> str:
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == CHECK_BACKWARD
    return failure.extra["reason"]


class _SharedActivation(nn.Module):
    """One ``nn.Tanh`` registered three times: ``act``, ``a.act`` and ``b.act``."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.a = nn.Sequential(OrderedDict(lin=nn.Linear(3, 4), act=self.act))
        self.b = nn.Sequential(OrderedDict(lin=nn.Linear(3, 4), act=self.act))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x).sum(dim=-1) + self.b(x).sum(dim=-1)


def test_shared_module_canonical_address_is_its_first_registration() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_SharedActivation(), torch.randn(2, 3))
    calls = {call.call_label for call in trace.module_calls}
    assert {"act:1", "act:2"} <= calls
    assert not {call for call in calls if call.startswith(("a.act", "b.act"))}


def test_shared_module_validates_backward() -> None:
    assert _validate_backward(_SharedActivation(), torch.randn(2, 3)) is True
    assert last_validation_failure() is None


class _Holder:
    """A plain object output (no dataclass, no ``to_tuple``) holding one tensor."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value


class _ReturnsObject(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> _Holder:
        return _Holder(torch.tanh(self.lin(x)))


class _UnwrapsObject(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _ReturnsObject()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).value.sum(dim=-1)


def test_plain_object_module_output_is_keyed_like_the_capture() -> None:
    assert _validate_backward(_UnwrapsObject(), torch.randn(2, 3)) is True
    assert last_validation_failure() is None


class _UnusedChild(nn.Module):
    """``b`` runs but the loss never depends on it: no gradient on either side."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 2)
        self.b = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.b(x)
        return torch.tanh(self.a(x)).sum(dim=-1)


class _OnlyUnusedChild(nn.Module):
    """The only child's output is never used: nothing to compare."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(3))
        self.b = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.b(x)
        return (x * self.w).sum(dim=-1)


def _capture_report(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    reports: list[Any] = []
    real = layer_grad_report._compare_module_output_grads

    def recording(*args: Any, **kwargs: Any) -> Any:
        report = real(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(layer_grad_report, "_compare_module_output_grads", recording)
    return reports


def test_no_gradient_on_either_side_is_agreement_not_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = _capture_report(monkeypatch)
    assert _validate_backward(_UnusedChild(), torch.randn(2, 3)) is True
    (report,) = reports
    assert report.coverage["b:1"] == "compared_no_grad"
    assert report.coverage["a:1"] == "covered"
    assert report.compared_no_grad_count == 1
    assert report.covered_count == 1


def test_run_that_compares_no_gradient_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = _capture_report(monkeypatch)
    assert _validate_backward(_OnlyUnusedChild(), torch.randn(2, 3)) is False
    (report,) = reports
    assert report.coverage["b:1"] == "compared_no_grad"
    assert report.covered_count == 0
    assert _reason() == "no_module_output_evidence"
    failure = last_validation_failure()
    assert failure is not None and failure.message.startswith("nothing was compared")


class _TraceWithoutGrad:
    """Proxy over a trace whose named output op reports no captured gradient."""

    def __init__(self, trace: Any, label: str) -> None:
        self._trace = trace
        self._label = label

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trace, name)

    def __getitem__(self, label: str) -> Any:
        layer = self._trace[label]
        if label == self._label:
            return _LayerWithoutGrad(layer)
        return layer


class _LayerWithoutGrad:
    def __init__(self, layer: Any) -> None:
        self._layer = layer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)

    @property
    def grad(self) -> None:
        return None


def test_capture_dropping_a_gradient_stock_has_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = layer_grad_report._compare_module_output_grads
    reports: list[Any] = []

    def dropping(trace: Any, stock: Any, *args: Any, **kwargs: Any) -> Any:
        label = next(iter(trace.module_calls["a:1"].output_ops))
        report = real(_TraceWithoutGrad(trace, label), stock, *args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(layer_grad_report, "_compare_module_output_grads", dropping)
    assert _validate_backward(_UnusedChild(), torch.randn(2, 3)) is False
    (report,) = reports
    assert report.coverage["a:1"] == "skipped_no_grad"
    assert _reason() == "module_output_grads"


def test_stock_dropping_a_gradient_capture_has_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = layer_grad_report._compare_module_output_grads
    reports: list[Any] = []

    def dropping(trace: Any, stock: dict[Any, Any], *args: Any, **kwargs: Any) -> Any:
        report = real(trace, {k: v for k, v in stock.items() if k[0] != "a"}, *args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(layer_grad_report, "_compare_module_output_grads", dropping)
    assert _validate_backward(_UnusedChild(), torch.randn(2, 3)) is False
    (report,) = reports
    assert report.coverage["a:1"] == "skipped_no_grad"
    assert _reason() == "module_output_grads"


def _sum_of_differentiable_leaves(output: Any) -> torch.Tensor:
    """The behaviour harness's loss: every floating leaf that requires grad, summed."""

    leaves: list[torch.Tensor] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point() and obj.requires_grad:
                leaves.append(obj.float().sum())
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                walk(item)

    walk(output)
    return torch.stack(leaves).sum()


@pytest.mark.parametrize(
    ("module_name", "build", "example_input"),
    [
        ("rs_L790_3_aenet_pytorch", "build_aenet_pytorch", "example_input_aenet_pytorch"),
        ("rs_L1573_iql", "build_iql", "example_input_iql"),
    ],
    ids=["aenet_shared_activation", "iql_distribution_output"],
)
def test_classics_rows_validate_backward(module_name: str, build: str, example_input: str) -> None:
    corpus_module = load_model_module(module_name)
    torch.manual_seed(0)
    model = getattr(corpus_module, build)()
    inputs = getattr(corpus_module, example_input)()
    verdict = _validate_backward(model, inputs, loss_fn=_sum_of_differentiable_leaves)
    assert verdict is True, last_validation_failure()
