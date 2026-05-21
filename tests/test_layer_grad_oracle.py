"""PATH E per-module-output gradient oracle tests."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from torchlens.validation import backward as backward_validation
from torchlens.validation._layer_grad_report import (
    LayerGradReport,
    _compare_module_output_grads,
)
from torchlens.validation._stock_layer_grads import (
    _StockModuleGradCollector,
    _candidate_module_call_for,
    _candidate_root_module,
    _first_leaf_tensor,
    _pass_index_from_layer_modules,
)


class TinyMLP(nn.Module):
    """Small MLP used by the module-output oracle tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.l1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.l2(self.relu(self.l1(x)))


class TinyRNN(nn.Module):
    """Small recurrent model used as the RNN acceptance fixture."""

    def __init__(self) -> None:
        """Initialize recurrent cell and output head."""

        super().__init__()
        self.cell = nn.RNNCell(3, 5)
        self.out = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 3, 3)``.

        Returns
        -------
        torch.Tensor
            Final output.
        """

        h = torch.zeros(x.shape[0], 5, device=x.device)
        for step in range(3):
            h = self.cell(x[:, step, :], h)
        return self.out(h)


class TinyResNet(nn.Module):
    """Small residual block used when torchvision is unavailable."""

    def __init__(self) -> None:
        """Initialize residual layers."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a residual forward pass."""

        return self.head(torch.relu(self.block(x) + x))


class IdentityWrapper(nn.Module):
    """Module with an identity-output submodule."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with identity output."""

        return self.identity(self.linear(x))


@dataclasses.dataclass
class TensorBox:
    """Dataclass container for first-leaf tests."""

    ignored: int
    tensor: torch.Tensor


class SyntheticTrace:
    """Minimal trace stub consumed by ``_compare_module_output_grads``."""

    def __init__(self, call_logs: list[Any], layers: dict[str, Any]) -> None:
        """Initialize the synthetic trace.

        Parameters
        ----------
        call_logs:
            Synthetic module-call logs.
        layers:
            Layer mapping by label.
        """

        self.modules = SimpleNamespace(_pass_dict={call.call_label: call for call in call_logs})
        self._layers = layers
        self.layer_list = list(layers.values())

    def __getitem__(self, label: str) -> Any:
        """Return one synthetic layer by label."""

        return self._layers[label]


def _loss(output: torch.Tensor) -> torch.Tensor:
    """Reduce a model output to a scalar loss."""

    return output.sum()


def _coverage_ratio(report: LayerGradReport) -> float:
    """Return PATH E module-output coverage ratio."""

    denom = (
        report.covered_count
        + report.mismatched_count
        + report.skipped_no_first_leaf_count
        + report.skipped_no_grad_count
    )
    return report.covered_count / denom if denom else 0.0


def _assert_acceptance(report: LayerGradReport) -> None:
    """Assert the P5 module-output acceptance criteria."""

    assert report.overall_passed
    assert _coverage_ratio(report) >= 0.80
    assert report.mismatched_count == 0
    assert report.skipped_no_grad_count == 0


def _synthetic_call(address: str, call_index: int, output_layers: list[str]) -> Any:
    """Build a synthetic module-call log."""

    return SimpleNamespace(
        address=address,
        call_index=call_index,
        call_label=f"{address}:{call_index}",
        output_layers=output_layers,
    )


def _synthetic_layer(
    label: str,
    grad: torch.Tensor | None,
    modules: list[str] | None = None,
) -> Any:
    """Build a synthetic layer log."""

    return SimpleNamespace(
        layer_label=label,
        grad=grad,
        has_saved_gradient=grad is not None,
        modules=modules if modules is not None else ["m:1"],
    )


def test_stock_module_grad_collector_captures_module_output_grad() -> None:
    """The stock collector captures gradients from module outputs."""

    model = TinyMLP()
    collector = _StockModuleGradCollector()
    collector.install(model)
    try:
        loss = model(torch.randn(2, 3)).sum()
        loss.backward()
        collector.collect_grads_after_backward()
    finally:
        collector.cleanup()
    assert ("l1", 1) in collector.stock_module_output_grads
    assert ("l2", 1) in collector.stock_module_output_grads


def test_first_leaf_tensor_traverses_supported_containers() -> None:
    """First-leaf discovery handles lists, dicts, and dataclasses."""

    first = torch.randn(1)
    second = torch.randn(1)
    assert _first_leaf_tensor({"a": None, "b": [first, second]}) is first
    assert _first_leaf_tensor(TensorBox(ignored=1, tensor=second)) is second


def test_pass_index_from_layer_modules_parses_string_and_tuple() -> None:
    """Module pass-index parsing supports current layer module formats."""

    assert _pass_index_from_layer_modules(SimpleNamespace(modules=["a.b:3"])) == 3
    assert _pass_index_from_layer_modules(SimpleNamespace(modules=[("a.b", 4)])) == 4


def test_candidate_module_lookup_helpers_use_pass_dict() -> None:
    """Candidate helper lookups resolve root and pass-qualified module calls."""

    call = _synthetic_call("block", 2, ["x"])
    root = _synthetic_call("self", 1, ["root"])
    trace = SyntheticTrace([call, root], {})
    assert _candidate_module_call_for(trace, "block", 2) is call
    assert _candidate_root_module(trace) is root


def test_compare_module_output_grads_reports_mismatched_bucket() -> None:
    """Failed allclose comparisons are mismatched, not covered."""

    grad = torch.ones(2, 2)
    trace = SyntheticTrace(
        [_synthetic_call("linear", 1, ["linear_out"])],
        {"linear_out": _synthetic_layer("linear_out", grad)},
    )
    report = _compare_module_output_grads(trace, {("linear", 1): grad + 1}, set())
    assert report.coverage["linear:1"] == "mismatched"
    assert report.mismatched_count == 1
    assert not report.overall_passed


def test_overall_passed_requires_skipped_no_grad_zero() -> None:
    """One skipped-no-grad entry fails both 1+4 and 4+1 synthetic shapes."""

    for covered, skipped in ((1, 4), (4, 1)):
        calls = []
        layers = {}
        stock = {}
        for index in range(covered + skipped):
            address = f"m{index}"
            label = f"layer{index}"
            grad = torch.ones(1) if index < covered else None
            calls.append(_synthetic_call(address, 1, [label]))
            layers[label] = _synthetic_layer(label, grad)
            if grad is not None:
                stock[(address, 1)] = grad.clone()
        report = _compare_module_output_grads(SyntheticTrace(calls, layers), stock, set())
        assert report.covered_count == covered
        assert report.skipped_no_grad_count == skipped
        assert not report.overall_passed


def test_compare_excludes_root_and_identity_from_denominator() -> None:
    """Root and identity outputs are classified but do not block passing."""

    grad = torch.ones(1)
    trace = SyntheticTrace(
        [
            _synthetic_call("self", 1, ["root"]),
            _synthetic_call("id", 1, ["id_out"]),
            _synthetic_call("linear", 1, ["linear_out"]),
        ],
        {
            "root": _synthetic_layer("root", grad),
            "id_out": _synthetic_layer("id_out", grad),
            "linear_out": _synthetic_layer("linear_out", grad),
        },
    )
    report = _compare_module_output_grads(
        trace,
        {("linear", 1): grad},
        {("id", 1)},
    )
    assert report.coverage["self:1"] == "skipped_root_module"
    assert report.coverage["id:1"] == "skipped_identity_output"
    assert report.overall_passed


def test_compare_counts_no_first_leaf() -> None:
    """Module calls with no output layer receive the no-first-leaf bucket."""

    report = _compare_module_output_grads(
        SyntheticTrace([_synthetic_call("empty", 1, [])], {}),
        {},
        set(),
    )
    assert report.skipped_no_first_leaf_count == 1
    assert report.coverage["empty:1"] == "skipped_no_first_leaf"


def test_compare_counts_module_less_layers_diagnostically() -> None:
    """Module-less layers are tracked separately from module-call coverage."""

    grad = torch.ones(1)
    trace = SyntheticTrace(
        [_synthetic_call("linear", 1, ["linear_out"])],
        {
            "linear_out": _synthetic_layer("linear_out", grad, ["linear:1"]),
            "top": _synthetic_layer("top", grad, []),
        },
    )
    report = _compare_module_output_grads(trace, {("linear", 1): grad}, set())
    assert report.skipped_module_less_count == 1
    assert report.overall_passed


def test_per_module_output_oracle_basic() -> None:
    """TinyMLP passes the PATH E oracle with required coverage."""

    torch.manual_seed(0)
    report = backward_validation._validate_layer_grads(
        TinyMLP(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    _assert_acceptance(report)


def test_validate_backward_pass_validate_layer_grads_public_flag() -> None:
    """The public backward validator can include the PATH E oracle."""

    torch.manual_seed(0)
    assert backward_validation.validate_backward_pass(
        TinyMLP(),
        torch.randn(2, 3),
        random_seed=42,
        validate_layer_grads=True,
    )


@pytest.mark.smoke
def test_layer_grad_default_off_no_overhead(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default public validator does not run PATH E."""

    def fail_if_called(*args: Any, **kwargs: Any) -> LayerGradReport:
        """Fail if the optional layer-grad path is invoked."""

        raise AssertionError("layer grad oracle should be opt-in")

    monkeypatch.setattr(backward_validation, "_validate_layer_grads", fail_if_called)
    assert backward_validation.validate_backward_pass(TinyMLP(), torch.randn(2, 3), random_seed=42)


def test_nested_module_parent_and_child_outputs_are_covered() -> None:
    """Nested child and parent module calls both appear in PATH E coverage."""

    model = nn.Sequential(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), nn.Linear(4, 2))
    report = backward_validation._validate_layer_grads(
        model,
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    assert report.coverage["0.1:1"] == "covered"
    assert report.coverage["0:1"] == "covered"
    _assert_acceptance(report)


def test_identity_output_modules_are_skipped() -> None:
    """Identity-output modules are skipped instead of falsely covered."""

    report = backward_validation._validate_layer_grads(
        IdentityWrapper(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    assert report.coverage["identity:1"] == "skipped_identity_output"
    assert report.skipped_identity_output_count == 1
    assert report.overall_passed


def test_weight_tied_module_call_indices_are_separate() -> None:
    """Repeated calls to the same module address are compared separately."""

    class Tied(nn.Module):
        """Use one module instance twice."""

        def __init__(self) -> None:
            """Initialize shared layer."""

            super().__init__()
            self.shared = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the shared layer twice."""

            return self.shared(self.shared(x))

    report = backward_validation._validate_layer_grads(
        Tied(),
        torch.randn(2, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    assert report.coverage["shared:1"] == "covered"
    assert report.coverage["shared:2"] == "covered"
    _assert_acceptance(report)


def test_oracle_rnn_3_step() -> None:
    """Three-step RNN fixture passes the PATH E oracle."""

    report = backward_validation._validate_layer_grads(
        TinyRNN(),
        torch.randn(2, 3, 3),
        {},
        _loss,
        atol=1e-5,
        rtol=1e-4,
        random_seed=42,
    )
    _assert_acceptance(report)


@pytest.mark.slow
def test_oracle_resnet50_eval() -> None:
    """ResNet-style eval fixture passes the PATH E oracle."""

    try:
        from torchvision.models import resnet50
    except Exception:
        model = TinyResNet().eval()
        x = torch.randn(2, 4)
    else:
        model = resnet50(weights=None).eval()
        x = torch.randn(1, 3, 32, 32)
    report = backward_validation._validate_layer_grads(
        model,
        x,
        {},
        lambda output: output.float().sum(),
        atol=1e-4,
        rtol=1e-3,
        random_seed=42,
    )
    _assert_acceptance(report)


@pytest.mark.slow
def test_oracle_gpt2_small_forward_backward() -> None:
    """GPT-2-small-style fixture passes the PATH E oracle when available."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_positions=8,
        n_ctx=8,
        vocab_size=32,
    )
    model = transformers.GPT2Model(config).eval()
    input_ids = torch.randint(0, 32, (1, 4))

    def gpt2_loss(output: Any) -> torch.Tensor:
        """Return a scalar GPT-2 loss from stock or reconstructed output."""

        if isinstance(output, list):
            return output[0].float().sum()
        return output.last_hidden_state.float().sum()

    report = backward_validation._validate_layer_grads(
        model,
        input_ids,
        {},
        gpt2_loss,
        atol=1e-4,
        rtol=1e-3,
        random_seed=42,
    )
    _assert_acceptance(report)


def test_per_operation_oracle_deferred_per_ad_50() -> None:
    """Shipping code and tests do not contain PATH B implementation symbols."""

    repo = Path(__file__).resolve().parents[1]
    deleted_identifiers = [
        "_Stock" + "GradCaptureMode",
        "align" + "_stock_to_candidate",
    ]
    for identifier in deleted_identifiers:
        result = subprocess.run(
            ["rg", "-w", identifier, "torchlens", "tests"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stdout
    stock_side_result = subprocess.run(
        ["rg", "-w", "_normalize" + "_func_name", "torchlens/validation"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stock_side_result.returncode == 1, stock_side_result.stdout


def test_path_e_module_exports_expected_surface() -> None:
    """The PATH E helper and report modules expose the expected names."""

    import torchlens.validation._stock_layer_grads as stock_module

    assert hasattr(stock_module, "_StockModuleGradCollector")
    assert hasattr(stock_module, "_first_leaf_tensor")
    assert hasattr(stock_module, "_stock_layer_grads")
    assert LayerGradReport(
        mode="module_output",
        overall_passed=True,
        coverage={},
        covered_count=1,
        skipped_no_first_leaf_count=0,
        skipped_module_less_count=0,
        skipped_no_grad_count=0,
        skipped_identity_output_count=0,
        skipped_root_module_count=0,
        mismatched_count=0,
        unexpected_count=0,
        candidate_grad_count=1,
        atol=1e-5,
        rtol=1e-4,
    )
