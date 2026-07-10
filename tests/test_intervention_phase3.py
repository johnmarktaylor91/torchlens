"""Smoke tests for intervention Phase 3 hook contracts and helpers."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import (
    HelperMountError,
    HookSignatureError,
    HookSiteCoverageError,
    HookValueError,
    SpliceModuleDtypeError,
)
from torchlens.intervention.hooks import HookContext, make_hook_context, normalize_hook_plan
from torchlens.intervention import runtime as intervention_runtime
from torchlens.intervention.runtime import _execute_hook
from torchlens.intervention.sites import sites
from torchlens.intervention.types import HelperSpec
from torchlens.validation import check_metadata_invariants


def _good_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
    """Return the out unchanged.

    Parameters
    ----------
    out:
        Activation tensor.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Input out.
    """

    return out


def _context() -> HookContext:
    """Return a representative hook context.

    Returns
    -------
    HookContext
        Context with mapping-proxy layer metadata.
    """

    return make_hook_context(
        name="test",
        layer_log={
            "layer_label": "relu_1_1",
            "layer_type": "relu",
            "shape": (2, 3),
            "dtype": torch.float32,
            "tensor_device": torch.device("cpu"),
            "address": "block",
            "call_index": 1,
        },
    )


def _context_with_args(*args: Any) -> HookContext:
    """Return a representative hook context with captured call inputs.

    Parameters
    ----------
    *args:
        Positional call inputs to expose to input-routed helpers.

    Returns
    -------
    HookContext
        Context with call inputs.
    """

    return make_hook_context(
        name="test",
        layer_log={
            "layer_label": "linear_1_1",
            "layer_type": "linear",
            "shape": (2, 3),
            "dtype": torch.float32,
            "tensor_device": torch.device("cpu"),
            "address": "fc",
            "call_index": 1,
        },
        args=args,
    )


@pytest.mark.smoke
def test_phase3_helpers_import_and_return_specs() -> None:
    """All Phase 3 helpers are importable from the top-level namespace."""

    helper_specs = [
        tl.zero_ablate(),
        tl.mean_ablate(),
        tl.resample_ablate(),
        tl.steer(torch.ones(3), feature_axis=-1),
        tl.scale(2.0),
        tl.clamp(min=-1.0, max=1.0),
        tl.noise(0.1, seed=1),
        tl.project_onto(torch.ones(3), feature_axis=-1),
        tl.project_off(torch.ones(3), feature_axis=-1),
        tl.swap_with(torch.ones(2, 3)),
        tl.splice_module(torch.nn.Identity()),
        tl.bwd_hook(_good_hook),
        tl.grad_zero(),
        tl.grad_scale(0.5),
    ]

    assert all(isinstance(spec, HelperSpec) for spec in helper_specs)
    assert helper_specs[0].name == "zero_ablate"
    assert helper_specs[-1].kind == "backward"
    assert dict(helper_specs[-1].metadata)["live_rerun_only"] is True


@pytest.mark.smoke
def test_project_onto_uses_batch_independent_feature_axis_projection() -> None:
    """Projecting onto a vector direction computes one coefficient per sample."""

    out = torch.tensor(
        [
            [[2.0, 4.0, 0.0], [1.0, 3.0, 5.0]],
            [[6.0, 8.0, 0.0], [7.0, 9.0, 11.0]],
        ]
    )
    direction = torch.tensor([1.0, 1.0, 0.0])
    hook = tl.project_onto(direction, feature_axis=-1)()

    projected = hook(out, hook=_context())

    expected = torch.tensor(
        [
            [[3.0, 3.0, 0.0], [2.0, 2.0, 0.0]],
            [[7.0, 7.0, 0.0], [8.0, 8.0, 0.0]],
        ]
    )
    assert torch.allclose(projected, expected)

    changed = out.clone()
    changed[1] = torch.tensor([[100.0, 102.0, 0.0], [104.0, 106.0, 108.0]])
    changed_projected = hook(changed, hook=_context())

    assert torch.allclose(changed_projected[0], expected[0])
    assert not torch.allclose(changed_projected[1], expected[1])


@pytest.mark.smoke
def test_hook_context_uses_mapping_proxy_and_frozen_fields() -> None:
    """HookContext exposes metadata as snapshots rather than live logs."""

    context = _context()

    assert isinstance(context.layer_log, MappingProxyType)
    assert context.layer_log["layer_label"] == "relu_1_1"
    with pytest.raises(TypeError):
        context.layer_log["layer_label"] = "mutated"  # type: ignore[index]
    with pytest.raises(AttributeError):
        context.name = "mutated"  # type: ignore[misc]


@pytest.mark.smoke
def test_normalizer_accepts_supported_shapes_in_order() -> None:
    """Hook normalization covers callable, helper, mapping, list, and pair shapes."""

    default_entries = normalize_hook_plan(_good_hook, default_site_target=tl.label("x"))
    helper_entries = normalize_hook_plan(tl.zero_ablate(), default_site_target=tl.label("x"))
    mapping_entries = normalize_hook_plan({tl.label("x"): _good_hook, tl.func("relu"): tl.scale(2)})
    list_entries = normalize_hook_plan([(tl.label("x"), _good_hook), (tl.label("y"), tl.scale(2))])
    pair_entries = normalize_hook_plan((tl.label("x"), _good_hook))

    assert len(default_entries) == 1
    assert helper_entries[0].helper_spec is not None
    assert [entry.metadata["attach_order"] for entry in mapping_entries] == [0, 1]
    assert [entry.metadata["attach_order"] for entry in list_entries] == [0, 1]
    assert pair_entries[0].site_target == tl.label("x")


def test_helper_direction_override_contradiction_raises() -> None:
    """Backward-only helpers cannot be mounted as forward hooks."""

    with pytest.raises(HelperMountError, match="intrinsically 'backward'"):
        normalize_hook_plan(tl.func("relu"), tl.grad_zero(), direction="forward")


def test_sites_callable_predicate_is_preserved_with_ops_filter() -> None:
    """Callable site predicates remain part of composed ``sites(..., ops=...)`` selectors."""

    class _Relu(torch.nn.Module):
        """Small model with a relu op."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply a relu.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                ReLU output.
            """

            return torch.relu(x)

    log = tl.trace(_Relu(), torch.randn(1, 3), intervention_ready=True)

    selected = log.find_sites(sites(lambda ctx: False, ops=["relu"]).entries[0].selector)

    assert selected.labels() == ()


@pytest.mark.smoke
def test_normalizer_rejects_missing_site_and_bad_signature() -> None:
    """Normalizer fails closed on ambiguous bare hooks and bad signatures."""

    def bad_hook(out: torch.Tensor) -> torch.Tensor:
        """Return out with a deliberately invalid signature."""

        return out

    with pytest.raises(HookSiteCoverageError):
        normalize_hook_plan(_good_hook)
    with pytest.raises(HookSignatureError, match="hook"):
        normalize_hook_plan((tl.label("x"), bad_hook))


@pytest.mark.smoke
def test_execute_hook_rejects_none_type_shape_dtype_and_device() -> None:
    """Hook execution validates return values by default."""

    out = torch.ones(2, 3)
    context = _context()

    def none_hook(out: torch.Tensor, *, hook: HookContext) -> None:
        """Return None, which Phase 3 rejects."""

        return None

    def list_hook(out: torch.Tensor, *, hook: HookContext) -> list[torch.Tensor]:
        """Return the wrong type."""

        return [out]

    def shape_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return the wrong shape."""

        return torch.ones(3, 2)

    def dtype_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return the wrong dtype."""

        return out.to(torch.float64)

    def device_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return a tensor on the wrong device."""

        return torch.empty(out.shape, dtype=out.dtype, device="meta")

    with pytest.raises(HookValueError, match="None"):
        _execute_hook(none_hook, out, context)
    with pytest.raises(HookValueError, match="list"):
        _execute_hook(list_hook, out, context)
    with pytest.raises(HookValueError, match="shape"):
        _execute_hook(shape_hook, out, context)
    with pytest.raises(HookValueError, match="dtype"):
        _execute_hook(dtype_hook, out, context)
    with pytest.raises(HookValueError, match="device"):
        _execute_hook(device_hook, out, context)


@pytest.mark.smoke
def test_force_shape_change_allows_metadata_changes() -> None:
    """The escape hatch byops dtype/device/shape checks."""

    out = torch.ones(2, 3)

    def shape_hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Return a shape-changed tensor."""

        return torch.ones(3, 2, dtype=torch.float64)

    result = _execute_hook(shape_hook, out, _context(), force_shape_change=True)

    assert result.shape == (3, 2)
    assert result.dtype == torch.float64


@pytest.mark.smoke
def test_seeded_noise_is_deterministic_and_unseeded_records_note() -> None:
    """Stochastic helper RNG follows the Phase 3 policy."""

    out = torch.zeros(2, 3)
    context = _context()

    seeded_a = tl.noise(1.0, seed=123)()
    seeded_b = tl.noise(1.0, seed=123)()
    unseeded = tl.noise(1.0)()

    assert torch.equal(seeded_a(out, hook=context), seeded_b(out, hook=context))
    _ = unseeded(out, hook=context)
    assert any("noise used unseeded" in note for note in context.run_ctx["ledger_notes"])


@pytest.mark.smoke
def test_splice_module_dtype_error_is_specific() -> None:
    """splice_module reports dtype mismatches with its specific error type."""

    class _DoubleModule(torch.nn.Module):
        """Module that deliberately changes dtype."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a float64 tensor.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Converted tensor.
            """

            return x.to(torch.float64)

    hook = tl.splice_module(_DoubleModule())()

    with pytest.raises(SpliceModuleDtypeError):
        hook(torch.ones(2, 3), hook=_context_with_args(torch.ones(2, 3)))


@pytest.mark.smoke
def test_splice_module_default_replaces_with_module_on_input() -> None:
    """splice_module defaults to documented input-splice semantics."""

    class _HundredModule(torch.nn.Module):
        """Replacement module that makes input-vs-output routing observable."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``100 * x``."""

            return 100 * x

    hook = tl.splice_module(_HundredModule())()
    original_input = torch.tensor([[1.0, 2.0, 3.0]])
    original_output = 2 * original_input

    result = hook(original_output, hook=_context_with_args(original_input))

    assert torch.equal(result, 100 * original_input)
    assert not torch.equal(result, 100 * original_output)


def test_splice_module_forwards_full_input_signature() -> None:
    """Input-splice helpers receive all positional and keyword inputs."""

    class _SubtractModule(torch.nn.Module):
        """Replacement that requires both captured inputs."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Return ``a - b``."""

            return a - b

    hook = tl.splice_module(_SubtractModule())()
    a = torch.tensor([[5.0, 7.0]])
    b = torch.tensor([[2.0, 3.0]])

    result = hook(a + b, hook=_context_with_args(a, b))

    assert torch.equal(result, a - b)


def test_splice_module_input_splices_multi_input_module_end_to_end() -> None:
    """Module-scoped input splice receives the module call's full input structure."""

    class _AddBlock(torch.nn.Module):
        """Two-input module used as the splice target."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Return a visible function of both inputs."""

            return torch.add(a, b) * 3

    class _SubtractReplacement(torch.nn.Module):
        """Replacement requiring both original module inputs."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Return ``a - b``."""

            return a - b

    class _Model(torch.nn.Module):
        """Model with one named multi-input block."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.block = _AddBlock()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Run the block."""

            return self.block(a, b)

    a = torch.tensor([[5.0, 7.0]])
    b = torch.tensor([[2.0, 3.0]])

    log = tl.trace(
        _Model().eval(),
        (a, b),
        intervention_ready=True,
        hooks={tl.in_module("block"): tl.splice_module(_SubtractReplacement())},
    )

    assert torch.equal(log[log.output_layers[0]].out, a - b)


def test_splice_module_input_splices_module_scope_once_end_to_end() -> None:
    """Module-scoped input splice applies once at the module-call boundary."""

    class _HundredModule(torch.nn.Module):
        """Replacement that exposes repeated application."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``100 * x``."""

            return 100 * x

    class _Model(torch.nn.Module):
        """Model with a two-op named block."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.block = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the block."""

            return self.block(x)

    x = torch.tensor([[1.0, 2.0, 3.0]])

    log = tl.trace(
        _Model().eval(),
        x,
        intervention_ready=True,
        hooks={tl.in_module("block"): tl.splice_module(_HundredModule())},
    )

    assert torch.equal(log[log.output_layers[0]].out, 100 * x)


@pytest.mark.timeout(5)
def test_module_scoped_splice_records_replacement_parent_and_fire_record(tmp_path: Path) -> None:
    """A module-boundary input splice records the value consumed downstream."""

    class _AddOneBlock(torch.nn.Module):
        """Produce a distinguishable pre-splice module output."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add one to the input."""

            return x + 1

    class _DoubleReplacement(torch.nn.Module):
        """Double the original module input."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return twice the input."""

            return x * 2

    class _Model(torch.nn.Module):
        """Consume the spliced module output in a downstream sigmoid."""

        def __init__(self) -> None:
            """Initialize the named splice target."""

            super().__init__()
            self.block = _AddOneBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the block and its downstream consumer."""

            return torch.sigmoid(self.block(x))

    x = torch.tensor([1.0, 2.0])
    log = tl.trace(
        _Model(),
        x,
        intervention_ready=True,
        hooks={tl.module("block"): tl.splice_module(_DoubleReplacement())},
    )
    sigmoid = next(layer for layer in log.layer_list if layer.func_name == "sigmoid")
    replacement = log[sigmoid.parents[0]]

    assert torch.equal(replacement.out, x * 2)
    assert torch.equal(sigmoid.out, torch.sigmoid(replacement.out))
    assert replacement.intervention_replaced is True
    assert replacement.interventions[-1].replaced is True
    assert replacement.interventions[-1].helper_name == "splice_module"
    assert check_metadata_invariants(log) is True
    for module_call_label, module_call in log.module_calls.items():
        assert module_call.call_parent != module_call_label
        assert module_call_label not in module_call.call_children
    rendered = log.draw(
        collapse="none",
        vis_outpath=str(tmp_path / "module_splice"),
        vis_fileformat="svg",
        vis_save_only=True,
    )
    assert rendered is not None


def test_splice_module_input_rejects_module_scoped_op_granularity() -> None:
    """Input splice rejects ambiguous op-level selectors inside module scopes."""

    class _IdentityReplacement(torch.nn.Module):
        """Replacement that would be ambiguous over multiple ops."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the input."""

            return x

    class _Model(torch.nn.Module):
        """Model with a multi-op named block."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.block = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the block."""

            return self.block(x)

    with pytest.raises(HookValueError, match="module-scoped op selector is ambiguous"):
        tl.trace(
            _Model().eval(),
            torch.ones(1, 3),
            intervention_ready=True,
            hooks={
                tl.in_module("block") & tl.func("relu"): tl.splice_module(_IdentityReplacement())
            },
        )


def test_splice_module_input_preserves_common_op_arg_orders() -> None:
    """Op-level input splice forwards captured args for nontrivial torch signatures."""

    class _AddReplacement(torch.nn.Module):
        """Replacement for ``torch.add``."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Return ``a - b``."""

            return a - b

    class _WhereReplacement(torch.nn.Module):
        """Replacement for ``torch.where``."""

        def forward(
            self, condition: torch.Tensor, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            """Return a shifted where result."""

            return torch.where(condition, a, b) + 10

    class _AddmmReplacement(torch.nn.Module):
        """Replacement for ``torch.addmm``."""

        def forward(
            self, bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            """Return a shifted addmm result."""

            return torch.addmm(bias, mat1, mat2) + 10

    class _CatReplacement(torch.nn.Module):
        """Replacement for ``torch.cat``."""

        def forward(self, tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
            """Concatenate tensors in reverse order."""

            return torch.cat(tuple(reversed(tensors)), dim=dim)

    class _Ops(torch.nn.Module):
        """Model containing representative argument orders."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Run representative ops."""

            condition = torch.tensor([[True, False]], device=a.device)
            bias = torch.ones(1, 2, device=a.device)
            mat2 = torch.eye(2, device=a.device)
            return (
                torch.add(a, b),
                torch.where(condition, a, b),
                torch.addmm(bias, a, mat2),
                torch.cat([a, b], dim=1),
            )

    a = torch.tensor([[5.0, 7.0]])
    b = torch.tensor([[2.0, 3.0]])
    log = tl.trace(
        _Ops().eval(),
        (a, b),
        intervention_ready=True,
        hooks={
            tl.func("add"): tl.splice_module(_AddReplacement()),
            tl.func("where"): tl.splice_module(_WhereReplacement()),
            tl.func("addmm"): tl.splice_module(_AddmmReplacement()),
            tl.func("cat"): tl.splice_module(_CatReplacement()),
        },
    )
    outputs = [log[label].out for label in log.output_layers]

    assert any(torch.equal(out, a - b) for out in outputs)
    assert any(torch.equal(out, torch.tensor([[15.0, 13.0]])) for out in outputs)
    assert any(torch.equal(out, torch.tensor([[16.0, 18.0]])) for out in outputs)
    assert any(torch.equal(out, torch.tensor([[2.0, 3.0, 5.0, 7.0]])) for out in outputs)


def test_intervene_input_splice_on_inplace_relu_receives_pre_mutation_values() -> None:
    """Input-routed post-hook intervention on ``relu_`` receives semantic inputs."""

    class _DoubleAndRecord(torch.nn.Module):
        """Replacement module that records the input values it receives."""

        def __init__(self) -> None:
            """Initialize the recorder."""

            super().__init__()
            self.seen: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Record and double ``x``."""

            self.seen.append(x.detach().clone())
            return x * 2

    class _InplaceReluModel(torch.nn.Module):
        """Model with a downstream consumer after an in-place relu."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run an in-place relu and consume its replacement."""

            y = x * 1
            y = torch.relu_(y)
            return y + 1

    replacement = _DoubleAndRecord()
    x = torch.tensor([[-2.0, 3.0]])

    log = tl.trace(
        _InplaceReluModel().eval(),
        x,
        intervene=tl.when(tl.func("relu_"), tl.splice_module(replacement)),
    )

    assert len(replacement.seen) == 1
    assert torch.equal(replacement.seen[0], x)
    assert torch.equal(log[log.output_layers[0]].out, x * 2 + 1)


def test_intervene_input_splice_on_relu_matches_inplace_semantics() -> None:
    """Input-routed post-hook intervention on ``relu`` matches ``relu_`` semantics."""

    class _DoubleAndRecord(torch.nn.Module):
        """Replacement module that records the input values it receives."""

        def __init__(self) -> None:
            """Initialize the recorder."""

            super().__init__()
            self.seen: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Record and double ``x``."""

            self.seen.append(x.detach().clone())
            return x * 2

    class _ReluModel(torch.nn.Module):
        """Model with a downstream consumer after an out-of-place relu."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run an out-of-place relu and consume its replacement."""

            y = x * 1
            y = torch.relu(y)
            return y + 1

    replacement = _DoubleAndRecord()
    x = torch.tensor([[-2.0, 3.0]])

    log = tl.trace(
        _ReluModel().eval(),
        x,
        intervene=tl.when(tl.func("relu"), tl.splice_module(replacement)),
    )

    assert len(replacement.seen) == 1
    assert torch.equal(replacement.seen[0], x)
    assert torch.equal(log[log.output_layers[0]].out, x * 2 + 1)


def test_plain_trace_of_inplace_relu_does_not_snapshot_intervention_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain traces of in-place ops do not clone intervention input snapshots."""

    class _InplaceReluModel(torch.nn.Module):
        """Model with an in-place relu."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run an in-place relu."""

            y = x * 1
            return torch.relu_(y)

    calls = 0
    original = intervention_runtime.copy_arg_tree

    def _counting_copy_arg_tree(arg: Any) -> Any:
        """Count intervention snapshot clones."""

        nonlocal calls
        calls += 1
        return original(arg)

    monkeypatch.setattr(intervention_runtime, "copy_arg_tree", _counting_copy_arg_tree)

    tl.trace(_InplaceReluModel().eval(), torch.tensor([[-2.0, 3.0]]))

    assert calls == 0


def test_intervention_on_different_op_does_not_snapshot_inplace_relu_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input-routed intervention on another op leaves in-place relu unsnapshotted."""

    class _Double(torch.nn.Module):
        """Replacement module that doubles its input."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Double ``x``."""

            return x * 2

    class _InplaceReluThenSigmoidModel(torch.nn.Module):
        """Model with an in-place op and a separately targeted op."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run relu_ followed by sigmoid."""

            y = x * 1
            y = torch.relu_(y)
            return torch.sigmoid(y)

    calls = 0
    original = intervention_runtime.copy_arg_tree

    def _counting_copy_arg_tree(arg: Any) -> Any:
        """Count intervention snapshot clones."""

        nonlocal calls
        calls += 1
        return original(arg)

    monkeypatch.setattr(intervention_runtime, "copy_arg_tree", _counting_copy_arg_tree)

    tl.trace(
        _InplaceReluThenSigmoidModel().eval(),
        torch.tensor([[-2.0, 3.0]]),
        intervene=tl.when(tl.func("sigmoid"), tl.splice_module(_Double())),
    )

    assert calls == 0


def test_inplace_snapshot_pregate_overapproximates_where_predicates() -> None:
    """A provisional ``tl.where`` failure conservatively snapshots ``relu_`` inputs."""

    class _Recorder(torch.nn.Module):
        """Record the input supplied by an input-routed intervention."""

        def __init__(self) -> None:
            """Initialize the recorded input list."""

            super().__init__()
            self.seen: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Record and return the intervention input.

            Parameters
            ----------
            x:
                Input routed from the matched operation.

            Returns
            -------
            torch.Tensor
                Unchanged input.
            """

            self.seen.append(x.detach().clone())
            return x

    class _Model(torch.nn.Module):
        """Apply an in-place ReLU."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the in-place operation.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                ReLU result.
            """

            return torch.relu_(x * 1)

    recorder = _Recorder()
    x = torch.tensor([[-2.0, 3.0]])

    tl.trace(
        _Model().eval(),
        x,
        intervene=tl.when(
            tl.where(lambda site: site.shape[0] >= 1) & tl.func("relu_"),
            tl.splice_module(recorder),
        ),
    )

    assert len(recorder.seen) == 1
    assert torch.equal(recorder.seen[0], x)


def test_inplace_snapshot_pregate_populates_module_context() -> None:
    """An ``in_module`` conjunct snapshots pre-mutation inputs for ``relu_``."""

    class _Recorder(torch.nn.Module):
        """Record the input supplied by an input-routed intervention."""

        def __init__(self) -> None:
            """Initialize the recorded input list."""

            super().__init__()
            self.seen: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Record and return the intervention input.

            Parameters
            ----------
            x:
                Input routed from the matched operation.

            Returns
            -------
            torch.Tensor
                Unchanged input.
            """

            self.seen.append(x.detach().clone())
            return x

    class _Encoder(torch.nn.Module):
        """Named module containing an in-place operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the in-place operation.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                ReLU result.
            """

            return torch.relu_(x * 1)

    class _Model(torch.nn.Module):
        """Expose the named encoder module."""

        def __init__(self) -> None:
            """Initialize the encoder."""

            super().__init__()
            self.encoder = _Encoder()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the encoder.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Encoder result.
            """

            return self.encoder(x)

    recorder = _Recorder()
    x = torch.tensor([[-2.0, 3.0]])

    tl.trace(
        _Model().eval(),
        x,
        intervene=tl.when(tl.func("relu_") & tl.in_module("encoder"), tl.splice_module(recorder)),
    )

    assert len(recorder.seen) == 1
    assert torch.equal(recorder.seen[0], x)


def test_noninplace_dunder_does_not_take_intervention_input_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-in-place dunder op does not enter the in-place snapshot pre-gate."""

    class _Double(torch.nn.Module):
        """Replacement module used only to activate an unrelated intervention."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the input unchanged.

            Parameters
            ----------
            x:
                Intervention input.

            Returns
            -------
            torch.Tensor
                Unchanged input.
            """

            return x

    class _Model(torch.nn.Module):
        """Run a dunder multiplication without any in-place operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Multiply the input.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Product tensor.
            """

            return x.__mul__(2)

    calls = 0
    original = intervention_runtime.copy_arg_tree

    def _counting_copy_arg_tree(arg: Any) -> Any:
        """Count intervention snapshot copies.

        Parameters
        ----------
        arg:
            Value passed to the snapshot copier.

        Returns
        -------
        Any
            Copied value.
        """

        nonlocal calls
        calls += 1
        return original(arg)

    monkeypatch.setattr(intervention_runtime, "copy_arg_tree", _counting_copy_arg_tree)

    tl.trace(
        _Model().eval(),
        torch.tensor([[2.0]]),
        intervene=tl.when(tl.func("relu_"), tl.splice_module(_Double())),
    )

    assert calls == 0


def test_input_splice_warns_for_out_aliasing_without_snapshot() -> None:
    """An input-routed ``out=`` aliasing intervention emits the live-ref warning."""

    class _Identity(torch.nn.Module):
        """Replacement module that leaves the routed input unchanged."""

        def forward(self, x: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
            """Return the input.

            Parameters
            ----------
            x:
                Routed input.
            *_args:
                Additional routed positional inputs.
            **_kwargs:
                Additional routed keyword inputs.

            Returns
            -------
            torch.Tensor
                Unchanged input.
            """

            return x

    class _OutAliasModel(torch.nn.Module):
        """Write an addition output into its first input."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Run the aliased addition.

            Parameters
            ----------
            a:
                Destination tensor.
            b:
                Addend tensor.

            Returns
            -------
            torch.Tensor
                Aliased addition result.
            """

            return torch.add(a, b, out=a)

    with pytest.warns(UserWarning, match="post-mutation inputs"):
        tl.trace(
            _OutAliasModel().eval(),
            (torch.tensor([[1.0]]), torch.tensor([[2.0]])),
            intervene=tl.when(tl.func("add"), tl.splice_module(_Identity())),
        )
