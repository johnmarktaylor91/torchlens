"""Static-label live interventions and halt on the technical-preview MLX backend.

The lifted contract: ``trace(intervene=tl.when(static_selector, action))`` and
``trace(halt=static_selector)`` genuinely dispatch — the replacement flows into
downstream ops, the frontier op ends a halted partial trace — while
value-dependent predicates, backward decisions, unsupported helpers, and recipe
specs keep typed refusals. The validation oracle treats declared interventions
as genuine user substitutions (replay compares against hook(raw)), and
stripping that declaration from a replay record FAILS instead of mispresenting
an intervened value as captured-native.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.backend_mlx

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import BackendUnsupportedError  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402


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

    relu_op = next(op for op in ablated.layer_list if op.layer_type == "relu")
    assert relu_op.intervention_replaced is True
    assert np.allclose(np.asarray(relu_op.out), 0.0)

    # The substituted zeros flowed into the downstream linear: its output is
    # the bias alone and differs from the baseline forward.
    final_ablated = np.asarray(ablated.layer_list[-1].out)
    final_baseline = np.asarray(baseline.layer_list[-1].out)
    bias = np.asarray(model.l2.bias)
    assert np.allclose(final_ablated, bias, atol=1e-5)
    assert not np.allclose(final_ablated, final_baseline)

    non_intervened = [
        op for op in ablated.layer_list if op.layer_type != "relu" and not op.is_input
    ]
    assert all(not op.intervention_replaced for op in non_intervened)

    assert MLXBackend().validate_trace(ablated) is True


def test_mlx_scale_and_callable_transform_apply() -> None:
    """tl.scale and a user mx-callable transform both dispatch for real."""

    model = _TwoLayerMLP()
    x = _input()
    baseline = tl.trace(model, x, backend="mlx")
    relu_baseline = np.asarray(
        next(op for op in baseline.layer_list if op.layer_type == "relu").out
    )

    scaled = tl.trace(model, x, backend="mlx", intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
    relu_scaled = np.asarray(next(op for op in scaled.layer_list if op.layer_type == "relu").out)
    assert np.allclose(relu_scaled, relu_baseline * 0.5, atol=1e-5)
    assert MLXBackend().validate_trace(scaled) is True

    def _negate(out: mx.array, *, hook: object) -> mx.array:
        """Return the negated activation."""

        del hook
        return out * -1.0

    negated = tl.trace(model, x, backend="mlx", intervene=tl.when(tl.func("relu"), _negate))
    relu_negated = np.asarray(next(op for op in negated.layer_list if op.layer_type == "relu").out)
    assert np.allclose(relu_negated, -relu_baseline, atol=1e-5)
    assert MLXBackend().validate_trace(negated) is True


def test_mlx_halt_returns_partial_trace_at_frontier() -> None:
    """halt= stops the forward at the matched op; the partial trace ends there."""

    model = _TwoLayerMLP()
    trace = tl.trace(model, _input(), backend="mlx", halt=tl.func("relu"))

    assert trace.halted is True
    assert str(trace.halt_reason).startswith("relu")
    labels = [op._label_raw for op in trace.layer_list]
    # Save-then-halt: the relu frontier is captured; the second linear is not.
    assert any(label.startswith("relu") for label in labels)
    assert sum(label.startswith("linear") for label in labels) == 1
    assert MLXBackend().validate_trace(trace) is True


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

    assert MLXBackend().validate_trace(trace) is False


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

    assert MLXBackend().validate_trace(trace) is False


def test_mlx_value_dependent_intervene_refuses_typed() -> None:
    """Non-static intervention predicates keep the typed refusal."""

    model = _TwoLayerMLP()
    with pytest.raises(BackendUnsupportedError, match="static"):
        tl.trace(
            model,
            _input(),
            backend="mlx",
            intervene=lambda ctx: None,
        )
    with pytest.raises(BackendUnsupportedError, match="selector kind"):
        tl.trace(
            model,
            _input(),
            backend="mlx",
            intervene=tl.when(tl.where(lambda ctx: True), tl.zero_ablate()),
        )


def test_mlx_backward_direction_intervene_refuses_typed() -> None:
    """Backward-direction decisions refuse: MLX has no backward capture."""

    with pytest.raises(BackendUnsupportedError, match="forward"):
        tl.trace(
            _TwoLayerMLP(),
            _input(),
            backend="mlx",
            intervene=tl.when(tl.func("relu"), tl.grad_zero()),
        )


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

    def _grow(out: mx.array, *, hook: object) -> mx.array:
        """Return a wrong-shape replacement."""

        del hook
        return mx.zeros((7, 7))

    with pytest.raises(BackendUnsupportedError, match="shape"):
        tl.trace(
            _TwoLayerMLP(),
            _input(),
            backend="mlx",
            intervene=tl.when(tl.func("relu"), _grow),
        )


def test_mlx_recipes_refuses_typed() -> None:
    """Recipe specs stay refused; only tl.when static predicates dispatch."""

    with pytest.raises(BackendUnsupportedError, match="recipe"):
        tl.trace(_TwoLayerMLP(), _input(), backend="mlx", recipes=object())


def test_mlx_halt_non_selector_refuses_typed() -> None:
    """Value-dependent halt predicates keep the typed refusal."""

    with pytest.raises(BackendUnsupportedError, match="static-label"):
        tl.trace(_TwoLayerMLP(), _input(), backend="mlx", halt=lambda ctx: True)


def test_mlx_grad_options_with_intervene_refuses_typed() -> None:
    """grad_options cannot combine with interventions (divergent replay)."""

    from torchlens.backends.mlx import GradOptions

    model = _TwoLayerMLP()
    with pytest.raises(BackendUnsupportedError, match="grad_options"):
        MLXBackend().capture_trace(
            model,
            _input(),
            grad_options=GradOptions(loss_fn=lambda out: out.sum()),
            intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        )


def test_mlx_interventions_flag_false_refuses_the_surface() -> None:
    """Review probe, reverse direction: flipping interventions=False in place
    refuses trace(intervene=) typed — the table stays load-bearing on MLX."""

    from torchlens.backends import get_backend_spec

    spec = get_backend_spec("mlx")
    object.__setattr__(spec.capabilities, "interventions", False)
    try:
        with pytest.raises(BackendUnsupportedError, match="interventions=False"):
            tl.trace(
                _TwoLayerMLP(),
                _input(),
                backend="mlx",
                intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
            )
    finally:
        object.__setattr__(spec.capabilities, "interventions", True)


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
    encoder_op = next(op for op in trace.layer_list if op.layer_type == "linear")
    assert encoder_op.intervention_replaced is True
    assert np.allclose(np.asarray(encoder_op.out), 0.0)
    head_ops = [op for op in trace.layer_list if op.layer_type == "linear"]
    assert not head_ops[-1].intervention_replaced
    assert MLXBackend().validate_trace(trace) is True
