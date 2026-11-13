"""Live interventions and halt on the MLX backend.

The lifted contract: ``trace(intervene=tl.when(selector, action))`` and
``trace(halt=selector)`` genuinely dispatch — the replacement flows into
downstream ops, the frontier op ends a halted partial trace — while raw
decision callables, backward decisions, unsupported helpers, and recipe specs
keep typed refusals (value-dependent conditions are covered by
``test_mlx_parity_value_edits.py``). The validation oracle treats declared interventions
as genuine user substitutions (replay compares against hook(raw)), and
stripping that declaration from a replay record FAILS instead of mispresenting
an intervened value as captured-native.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.errors import BackendUnsupportedError, InvalidArgumentError  # noqa: E402


class _TwoLayerMLP(nn.Module):
    """Two-layer MLX MLP with a relu boundary to intervene on.

    Weights are deterministic and positive so the relu output is provably
    nonzero on the all-ones input — random init occasionally lands an
    all-negative hidden layer, which would make an ablation indistinguishable
    from the baseline.
    """

    def __init__(self) -> None:
        """Initialize two linear layers with deterministic positive weights."""

        super().__init__()
        self.l1 = nn.Linear(4, 3)
        self.l2 = nn.Linear(3, 2)
        self.l1.weight = mx.full((3, 4), 0.5)
        self.l1.bias = mx.full((3,), 0.1)
        self.l2.weight = mx.full((2, 3), 0.25)
        self.l2.bias = mx.full((2,), -0.2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass."""

        return self.l2(nn.relu(self.l1(x)))


def _input() -> mx.array:
    """Return a deterministic non-negative test input."""

    return mx.ones((1, 4))


def test_mlx_module_exit_changes_output_and_validates() -> None:
    """A module site edits its output before the downstream layer consumes it."""

    model = _TwoLayerMLP()
    trace = tl.trace(
        model, _input(), backend="mlx", intervene=tl.when(tl.module("l1"), tl.zero_ablate())
    )
    producer = next(op for op in trace._layer_list if op.type == "linear")
    assert producer.is_edited
    assert producer.edits[0]["backend"] == "mlx"
    assert producer.edits[0]["site_label"] == producer._label_raw
    assert np.allclose(np.asarray(trace._layer_list[-1].out), np.asarray(model.l2.bias))
    assert MLXBackend().validate_trace(trace).passed


@pytest.mark.parametrize("dtype", [mx.bool_, mx.int32, mx.float32])
def test_mlx_module_identity_output_preserves_dtype(dtype: Any) -> None:
    """The boundary copy preserves boolean and numeric identity outputs."""

    class Block(nn.Module):
        def __call__(self, x: Any) -> Any:
            return x

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = Block()

        def __call__(self, x: Any) -> Any:
            return self.block(x)

    trace = tl.trace(
        Model(),
        mx.array([1, 0], dtype=dtype),
        backend="mlx",
        intervene=tl.when(tl.module("block"), tl.zero_ablate()),
    )
    out = trace._layer_list[-1].out
    assert out.dtype == dtype
    assert np.array_equal(np.asarray(out), np.zeros(2))
    assert MLXBackend().validate_trace(trace).passed


@pytest.mark.parametrize("second, factor", [("l1", 1.0), ("l2", 0.5)])
def test_mlx_module_union_fires_once_per_call(second: str, factor: float) -> None:
    """An overlapping union applies a non-idempotent action only once."""

    model = _TwoLayerMLP()
    trace = tl.trace(
        model,
        _input(),
        backend="mlx",
        intervene=tl.when(tl.module("l1") | tl.module(second), tl.scale(0.5)),
    )
    expected = np.full((1, 2), (2.1 * 0.5 * 3 * 0.25 - 0.2) * factor)
    assert np.allclose(np.asarray(trace._layer_list[-1].out), expected)
    assert MLXBackend().validate_trace(trace).passed


@pytest.mark.parametrize("address, expected", [("block", 9.0), ("block:2", 13.5)])
def test_mlx_module_exit_nested_leaves_aliases_and_repeated_calls(
    address: str, expected: float
) -> None:
    """Edit boundary leaves, preserving earlier consumers and call-qualified sites."""

    class Leaves(nn.Module):
        def __call__(self, x: Any) -> Any:
            first = x + 1
            return {"main": first, "alias": first, "other": [first * 2, x], "tag": "kept"}

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = Leaves()

        def __call__(self, x: Any) -> Any:
            a, b = self.block(x), self.block(x)
            assert a["tag"] == b["tag"] == "kept"
            return (
                a["main"] + a["alias"] + sum(a["other"]) + b["main"] + b["alias"] + sum(b["other"])
            )

    trace = tl.trace(
        Model(), _input(), backend="mlx", intervene=tl.when(tl.module(address), tl.scale(0.5))
    )
    assert np.allclose(np.asarray(trace._layer_list[-1].out), expected)
    assert MLXBackend().validate_trace(trace).passed


@pytest.mark.parametrize("root", [False, True])
def test_mlx_module_exit_inside_opaque_parent_validates(root: bool) -> None:
    """A selected attention projection remains editable through its parent."""

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiHeadAttention(4, 2, bias=False)
            for name in ("query_proj", "key_proj", "value_proj", "out_proj"):
                getattr(self.attn, name).weight = mx.full((4, 4), 0.25)

        def __call__(self, x: Any) -> Any:
            return self.attn(x, x, x)

    x = mx.arange(8).reshape(1, 2, 4).astype(mx.float32) / 8
    model = Model()
    expected = np.asarray(model(x)) * 0.5
    address = "value_proj" if root else "attn.value_proj"
    trace = tl.trace(
        model.attn if root else model,
        (x, x, x) if root else x,
        backend="mlx",
        intervene=tl.when(tl.module(address), tl.scale(0.5)),
    )
    assert np.allclose(np.asarray(trace._layer_list[-1].out), expected)
    assert MLXBackend().validate_trace(trace).passed


@pytest.mark.parametrize("kind", ["and", "or", "not"])
def test_mlx_mixed_module_selector_refuses_before_forward(kind: str) -> None:
    """Unsupported module trees fail preflight rather than silently matching nothing."""

    calls = []
    module = tl.module("l1")
    selector = {
        "and": module & tl.func("linear"),
        "or": module | tl.func("linear"),
        "not": ~module,
    }[kind]

    def forward(x: Any) -> Any:
        calls.append(True)
        return x + 1

    with pytest.raises(BackendUnsupportedError, match="(?i)mlx"):
        tl.trace(forward, _input(), backend="mlx", intervene=tl.when(selector, tl.zero_ablate()))
    assert calls == []


def test_mlx_zero_ablate_substitutes_downstream_and_validates() -> None:
    """Ablating the relu changes downstream values; validation still passes."""

    model = _TwoLayerMLP()
    x = _input()
    baseline = tl.trace(model, x, backend="mlx")
    ablated = tl.trace(
        model,
        x,
        backend="mlx",
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )

    relu_op = next(op for op in ablated._layer_list if op.type == "relu")
    assert relu_op.is_edited is True
    assert np.allclose(np.asarray(relu_op.out), 0.0)

    # The substituted zeros flowed into the downstream linear: its output is
    # the bias alone and differs from the baseline forward.
    final_ablated = np.asarray(ablated._layer_list[-1].out)
    final_baseline = np.asarray(baseline._layer_list[-1].out)
    bias = np.asarray(model.l2.bias)
    assert np.allclose(final_ablated, bias, atol=1e-5)
    assert not np.allclose(final_ablated, final_baseline)

    non_intervened = [op for op in ablated._layer_list if op.type != "relu" and not op.is_input]
    assert all(not op.is_edited for op in non_intervened)

    assert MLXBackend().validate_trace(ablated).passed is True


def test_mlx_scale_and_callable_transform_apply() -> None:
    """tl.scale and a user mx-callable transform both dispatch for real."""

    model = _TwoLayerMLP()
    x = _input()
    baseline = tl.trace(model, x, backend="mlx")
    relu_baseline = np.asarray(next(op for op in baseline._layer_list if op.type == "relu").out)

    scaled = tl.trace(model, x, backend="mlx", intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
    relu_scaled = np.asarray(next(op for op in scaled._layer_list if op.type == "relu").out)
    assert np.allclose(relu_scaled, relu_baseline * 0.5, atol=1e-5)
    assert MLXBackend().validate_trace(scaled).passed is True

    def _negate(out: mx.array, *, ctx: object) -> mx.array:
        """Return the negated activation."""

        del ctx
        return out * -1.0

    negated = tl.trace(model, x, backend="mlx", intervene=tl.when(tl.func("relu"), _negate))
    relu_negated = np.asarray(next(op for op in negated._layer_list if op.type == "relu").out)
    assert np.allclose(relu_negated, -relu_baseline, atol=1e-5)
    assert MLXBackend().validate_trace(negated).passed is True


def test_mlx_halt_returns_partial_trace_at_frontier() -> None:
    """halt= stops the forward at the matched op; the partial trace ends there."""

    model = _TwoLayerMLP()
    trace = tl.trace(model, _input(), backend="mlx", stop_after=tl.func("relu"))

    assert trace.is_stopped is True
    assert str(trace.stop_reason).startswith("relu")
    labels = [op._label_raw for op in trace._layer_list]
    # Save-then-halt: the relu frontier is captured; the second linear is not.
    assert any(label.startswith("relu") for label in labels)
    assert sum(label.startswith("linear") for label in labels) == 1
    assert MLXBackend().validate_trace(trace).passed is True


def test_mlx_unfired_stop_after_refuses_like_torch() -> None:
    """A stop_after selector that never fires refuses typed, with torch's code."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(_TwoLayerMLP(), _input(), backend="mlx", stop_after=tl.func("nosuchopzzz"))
    assert excinfo.value.fields["code"] == "stop_after_never_fired"


def test_mlx_intervened_record_stripped_of_declaration_fails_validation() -> None:
    """Mispresenting an intervened value as captured-native FAILS the oracle."""

    import dataclasses

    trace = tl.trace(
        _TwoLayerMLP(),
        _input(),
        backend="mlx",
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    captures = trace._mlx_op_captures
    index = next(i for i, capture in enumerate(captures) if capture.interventions)
    captures[index] = dataclasses.replace(captures[index], interventions=(), appliers=())

    assert MLXBackend().validate_trace(trace).state == "failed"


def test_mlx_intervened_record_with_forged_declaration_fails_validation() -> None:
    """Forging an intervention claim on a plain capture FAILS coverage."""

    import dataclasses

    trace = tl.trace(_TwoLayerMLP(), _input(), backend="mlx")
    captures = trace._mlx_op_captures
    index = next(i for i, capture in enumerate(captures) if capture.op_name == "relu")
    captures[index] = dataclasses.replace(
        captures[index],
        interventions=((0, "helper:zero_ablate"),),
        appliers=((0, lambda out: mx.zeros_like(out)),),
    )

    assert MLXBackend().validate_trace(trace).state == "failed"


def test_mlx_raw_decision_intervene_refuses_typed() -> None:
    """A raw decision callable (not the tl.when shape) keeps the typed refusal,
    and so does a module selector mixed with a value-dependent condition."""

    model = _TwoLayerMLP()
    with pytest.raises(BackendUnsupportedError, match="one-rule"):
        tl.trace(
            model,
            _input(),
            backend="mlx",
            intervene=lambda ctx: None,
        )
    with pytest.raises(BackendUnsupportedError, match="module"):
        tl.trace(
            model,
            _input(),
            backend="mlx",
            intervene=tl.when(tl.module("l1") & tl.where(lambda ctx: True), tl.zero_ablate()),
        )


def test_mlx_backward_direction_intervene_refuses_typed() -> None:
    """A tl.grad site without grad_options refuses: MLX edits gradients only in the derived run."""

    with pytest.raises(BackendUnsupportedError, match="grad_options") as info:
        tl.trace(
            _TwoLayerMLP(),
            _input(),
            backend="mlx",
            intervene=tl.when(tl.grad(tl.func("relu")), tl.zero_ablate()),
        )
    assert info.value.code == "intervention_with_derived_grads_unsupported"


def test_mlx_unsupported_helper_refuses_typed() -> None:
    """Helpers without an MLX-native application refuse by name."""

    with pytest.raises(BackendUnsupportedError, match="scramble_elements"):
        tl.trace(
            _TwoLayerMLP(),
            _input(),
            backend="mlx",
            intervene=tl.when(tl.func("relu"), tl.intervention.scramble_elements(mx.ones((3,)))),
        )


def test_mlx_shape_changing_hook_refuses_typed() -> None:
    """A fired hook that changes shape refuses instead of mis-broadcasting."""

    def _grow(out: mx.array, *, ctx: object) -> mx.array:
        """Return a wrong-shape replacement."""

        del ctx
        return mx.zeros((7, 7))

    with pytest.raises(BackendUnsupportedError, match="shape") as info:
        tl.trace(
            _TwoLayerMLP(),
            _input(),
            backend="mlx",
            intervene=tl.when(tl.func("relu"), _grow),
        )
    assert info.value.code == "intervention_result_invalid"


def test_mlx_recipes_snapshot_dispatches() -> None:
    """facets= attaches a per-trace facet registry snapshot; non-callables refuse typed."""

    def recipe(record: Any) -> dict[str, Any]:
        return {"is_relu": "relu" in str(getattr(record, "label", ""))}

    trace = tl.trace(_TwoLayerMLP(), _input(), backend="mlx", facets=[recipe])
    snapshot = trace.facet_functions
    assert snapshot is not None
    assert any("recipe" in entry.public.name for entry in snapshot.functions)

    with pytest.raises(BackendUnsupportedError, match="recipe callables") as info:
        tl.trace(_TwoLayerMLP(), _input(), backend="mlx", facets=object())
    assert info.value.code == "trace_option_unsupported"


def test_mlx_halt_non_selector_refuses_typed() -> None:
    """A stop site that is neither a selector nor a callable keeps the typed refusal."""

    with pytest.raises(BackendUnsupportedError, match="stop_after"):
        tl.trace(_TwoLayerMLP(), _input(), backend="mlx", stop_after=object())


def test_mlx_grad_options_with_module_exit_intervene_refuses_typed() -> None:
    """grad_options cannot combine with a tl.module edit (applied outside any wrapped call).

    Op-level edits with grad_options are covered by
    ``test_parity_b_mlx_int_edit_grads.py``.
    """

    from torchlens.backends.mlx import GradOptions

    model = _TwoLayerMLP()
    with pytest.raises(BackendUnsupportedError, match="grad_options") as info:
        MLXBackend().capture_trace(
            model,
            _input(),
            grad_options=GradOptions(loss_fn=lambda out: out.sum()),
            intervene=tl.when(tl.module("l1"), tl.zero_ablate()),
        )
    assert info.value.code == "intervention_with_derived_grads_unsupported"


def test_mlx_interventions_flag_false_refuses_the_surface() -> None:
    """Review probe, reverse direction: flipping interventions=False in place
    refuses trace(intervene=) typed — the table stays load-bearing on MLX."""

    spec = tl.backends.spec("mlx")
    object.__setattr__(spec.capabilities, "can_intervene", False)
    try:
        with pytest.raises(BackendUnsupportedError, match="can_intervene=False"):
            tl.trace(
                _TwoLayerMLP(),
                _input(),
                backend="mlx",
                intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
            )
    finally:
        object.__setattr__(spec.capabilities, "can_intervene", True)


def test_mlx_in_module_intervention_site() -> None:
    """tl.in_module static sites resolve through the object-module stack."""

    class _Wrapped(nn.Module):
        """Model with a named encoder submodule."""

        def __init__(self) -> None:
            """Initialize the encoder and head with deterministic weights."""

            super().__init__()
            self.encoder = nn.Linear(4, 3)
            self.head = nn.Linear(3, 2)
            self.encoder.weight = mx.full((3, 4), 0.5)
            self.encoder.bias = mx.full((3,), 0.1)
            self.head.weight = mx.full((2, 3), 0.25)
            self.head.bias = mx.full((2,), -0.2)

        def __call__(self, x: mx.array) -> mx.array:
            """Run encoder then head."""

            return self.head(self.encoder(x))

    model = _Wrapped()
    x = _input()
    trace = tl.trace(
        model,
        x,
        backend="mlx",
        intervene=tl.when(tl.in_module("encoder") & tl.func("linear"), tl.zero_ablate()),
    )
    encoder_op = next(op for op in trace._layer_list if op.type == "linear")
    assert encoder_op.is_edited is True
    assert np.allclose(np.asarray(encoder_op.out), 0.0)
    head_ops = [op for op in trace._layer_list if op.type == "linear"]
    assert not head_ops[-1].is_edited
    assert MLXBackend().validate_trace(trace).passed is True
