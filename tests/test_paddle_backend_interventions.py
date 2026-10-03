"""Live intervention and halt coverage for the technical-preview Paddle backend.

The ``interventions`` capability lift is honest only with this coverage: the
mechanism demonstrably works end to end (E2E ablation changes downstream
values AND replay validation passes), halt genuinely stops execution at the
frontier, unsupported shapes refuse typed, and the validation oracle FAILS
when an intervened value is mispresented as captured-native (tamper tests).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import (  # noqa: E402
    BackendUnsupportedError,
    get_backend_spec,
    require_capability_implementation,
)
from torchlens.backends.paddle import PaddleBackend  # noqa: E402

pytestmark = pytest.mark.backend_paddle


def _inputs() -> tuple[Any, Any, Any, Any, Any]:
    """Return deterministic explicit-parameter MLP inputs.

    Returns
    -------
    tuple[Any, Any, Any, Any, Any]
        Input, first weight, first bias, second weight, second bias tensors.
    """

    paddle.seed(0)
    x = paddle.arange(8, dtype="float32").reshape([2, 4]) / 8.0
    w1 = paddle.arange(32, dtype="float32").reshape([4, 8]) / 16.0
    b1 = paddle.arange(8, dtype="float32") / 10.0
    w2 = paddle.arange(16, dtype="float32").reshape([8, 2]) / 12.0
    b2 = paddle.arange(2, dtype="float32") / 7.0
    return x, w1, b1, w2, b2


def _mlp(x: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
    """Run a two-layer MLP with explicit parameter tensors.

    Parameters
    ----------
    x, w1, b1, w2, b2
        Input and parameter tensors.

    Returns
    -------
    Any
        MLP output tensor.
    """

    hidden = paddle.nn.functional.linear(x, w1, b1)
    hidden = paddle.nn.functional.relu(hidden)
    return paddle.nn.functional.linear(hidden, w2, b2)


def _output_op(trace: Any) -> Any:
    """Return the output-parent op of a trace."""

    return next(op for op in trace.layer_list if op.is_output_parent)


def _relu_op(trace: Any) -> Any:
    """Return the relu op of a trace."""

    return next(op for op in trace.layer_list if "relu" in op.label)


def _ablated_trace() -> Any:
    """Capture the MLP with the relu output zero-ablated."""

    return tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), tl.zero_ablate()),
    )


def test_paddle_capability_lift_is_real() -> None:
    """The lifted flag binds a resolvable implementing surface."""

    spec = get_backend_spec("paddle")
    assert spec.capabilities.interventions is True
    implementation = require_capability_implementation(spec, "interventions")
    assert implementation is not None


def test_paddle_zero_ablate_e2e_changes_downstream_and_validates() -> None:
    """Ablating relu changes downstream values and replay validation passes."""

    plain = tl.trace(_mlp, _inputs(), backend="paddle")
    ablated = _ablated_trace()

    relu = _relu_op(ablated)
    assert float(relu.out.abs().max()) == 0.0
    assert relu.intervention_replaced is True
    fire_records = list(relu.interventions)
    assert fire_records and fire_records[0].replaced is True
    assert fire_records[0].helper_name == "zero_ablate"

    downstream_delta = np.abs(_output_op(plain).out.numpy() - _output_op(ablated).out.numpy()).max()
    assert downstream_delta > 1.0

    backend = PaddleBackend()
    assert backend.validate_trace(ablated) is True
    assert backend.validate_trace(plain) is True


def test_paddle_intervention_helper_roster() -> None:
    """scale/add/replace_with adapters apply their documented semantics."""

    plain = tl.trace(_mlp, _inputs(), backend="paddle")
    relu_plain = _relu_op(plain).out.numpy()

    scaled = tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), tl.scale(0.5)),
    )
    np.testing.assert_allclose(_relu_op(scaled).out.numpy(), relu_plain * 0.5, rtol=1e-6)

    shifted = tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), tl.add(1.0)),
    )
    np.testing.assert_allclose(_relu_op(shifted).out.numpy(), relu_plain + 1.0, rtol=1e-6)

    replacement = paddle.ones_like(paddle.to_tensor(relu_plain)) * 7.0
    replaced = tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        intervene=tl.when(
            tl.func("functional.relu"),
            tl.replace_with(replacement),
        ),
    )
    np.testing.assert_allclose(_relu_op(replaced).out.numpy(), relu_plain * 0.0 + 7.0)

    assert PaddleBackend().validate_trace(scaled) is True
    assert PaddleBackend().validate_trace(shifted) is True
    assert PaddleBackend().validate_trace(replaced) is True


def test_paddle_plain_callable_hook_applies() -> None:
    """A raw callable hook transforms the matched output tensor."""

    plain = tl.trace(_mlp, _inputs(), backend="paddle")
    negated = tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), lambda out: -out),
    )
    np.testing.assert_allclose(_relu_op(negated).out.numpy(), -_relu_op(plain).out.numpy())
    assert PaddleBackend().validate_trace(negated) is True


def test_paddle_value_dependent_predicates_see_real_values() -> None:
    """Callable predicates read real eager context values (stretch surface)."""

    seen: list[tuple[str, Any]] = []

    def predicate(ctx: Any) -> Any:
        seen.append((str(ctx.func_name), ctx.tensor_requires_grad))
        if ctx.func_name == "functional.relu" and ctx.tensor_requires_grad is False:
            return tl.zero_ablate()
        return None

    trace = tl.trace(_mlp, _inputs(), backend="paddle", intervene=predicate)
    assert float(_relu_op(trace).out.abs().max()) == 0.0
    assert all(flag in (True, False) for _name, flag in seen)
    assert PaddleBackend().validate_trace(trace) is True


def test_paddle_halt_stops_execution_at_frontier() -> None:
    """halt= stops the forward mid-flight; post-frontier user code never runs."""

    executed: list[str] = []

    def mlp_with_flag(x: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
        hidden = paddle.nn.functional.linear(x, w1, b1)
        hidden = paddle.nn.functional.relu(hidden)
        executed.append("post-frontier")
        return paddle.nn.functional.linear(hidden, w2, b2)

    trace = tl.trace(mlp_with_flag, _inputs(), backend="paddle", halt=tl.func("functional.relu"))
    assert executed == []
    assert trace.halted is True
    assert "relu" in str(trace.halt_reason)
    assert trace.halt_frontier == trace.halt_reason
    assert not any("linear_2" in label for label in trace.layer_labels)
    assert any("relu" in label for label in trace.layer_labels)
    assert PaddleBackend().validate_trace(trace) is True


def test_paddle_halt_value_dependent_predicate() -> None:
    """A callable halt predicate reading context metadata stops capture."""

    trace = tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        halt=lambda ctx: ctx.func_name == "functional.relu",
    )
    assert trace.halted is True


def test_paddle_halt_non_bool_predicate_refuses_typed() -> None:
    """halt predicates must return bool, mirroring the torch contract."""

    from torchlens.fastlog.exceptions import PredicateError

    with pytest.raises(PredicateError, match="must return bool"):
        tl.trace(_mlp, _inputs(), backend="paddle", halt=lambda ctx: 1)


def test_paddle_tamper_stripped_evidence_fails_validation() -> None:
    """Mispresenting an intervened value as captured-native FAILS the oracle."""

    trace = _ablated_trace()
    captures = trace._paddle_op_captures
    index = next(i for i, capture in enumerate(captures) if capture.intervention is not None)
    captures[index] = dataclasses.replace(captures[index], intervention=None)
    relu = _relu_op(trace)
    relu.intervention_replaced = False
    relu.interventions = []
    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_tamper_disarmed_spec_fails_validation() -> None:
    """A capture-side intervention record without an armed spec is refused."""

    trace = _ablated_trace()
    trace._intervention_spec = None
    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_tamper_stripped_op_fire_records_fails_validation() -> None:
    """Op-level fire stamps are load-bearing corroboration, not decoration."""

    trace = _ablated_trace()
    relu = _relu_op(trace)
    relu.intervention_replaced = False
    relu.interventions = []
    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_unknown_helper_refuses_typed() -> None:
    """Helpers without a Paddle adapter refuse typed (never silently skip)."""

    with pytest.raises(BackendUnsupportedError, match="no adapter"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            intervene=tl.when(tl.func("functional.relu"), tl.noise(0.1)),
        )


def test_paddle_backward_direction_refuses_typed() -> None:
    """Backward-direction decisions refuse: paddle has no backward capture."""

    with pytest.raises(BackendUnsupportedError, match="forward interventions only"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            intervene=tl.when(
                tl.func("functional.relu"),
                tl.zero_ablate(),
                direction="backward",
            ),
        )


def test_paddle_non_tensor_hook_return_refuses_typed() -> None:
    """Hooks must return paddle tensors; other returns refuse typed."""

    with pytest.raises(BackendUnsupportedError, match="must return a"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            intervene=tl.when(tl.func("functional.relu"), lambda out: "nonsense"),
        )


def test_paddle_torch_tensor_helper_arg_refuses_typed() -> None:
    """torch.Tensor helper arguments refuse instead of failing confusingly."""

    import torch

    with pytest.raises(BackendUnsupportedError, match="torch.Tensor argument"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            intervene=tl.when(
                tl.func("functional.relu"),
                tl.add(torch.ones(1)),
            ),
        )


def test_paddle_zero_match_selector_warns() -> None:
    """A selector-conditioned predicate matching no sites warns, torch-style."""

    with pytest.warns(UserWarning, match="matched zero sites"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            intervene=tl.when(tl.func("does_not_exist"), tl.zero_ablate()),
        )


def test_paddle_intervene_selector_match_emits_no_warning() -> None:
    """A firing selector does not emit the zero-match warning."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _ablated_trace()


def test_paddle_grad_options_with_intervene_refuses_typed() -> None:
    """Derived-grad replay cannot honestly re-run an intervened forward."""

    grad_options = tl.backends.paddle.GradOptions(
        loss_fn=lambda output: paddle.sum(output * output),
    )
    with pytest.raises(BackendUnsupportedError, match="grad_options"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            grad_options=grad_options,
            intervene=tl.when(tl.func("functional.relu"), tl.zero_ablate()),
        )
    with pytest.raises(BackendUnsupportedError, match="grad_options"):
        tl.trace(
            _mlp,
            _inputs(),
            backend="paddle",
            grad_options=grad_options,
            halt=tl.func("functional.relu"),
        )


def test_paddle_recipes_snapshot_dispatches() -> None:
    """recipes= attaches a per-trace facet registry snapshot."""

    def recipe(record: Any) -> dict[str, Any]:
        return {"is_relu": "relu" in str(getattr(record, "label", ""))}

    trace = tl.trace(_mlp, _inputs(), backend="paddle", recipes=[recipe])
    snapshot = trace.facet_registry_snapshot
    assert snapshot is not None
    assert any("recipe" in entry.public.recipe_name for entry in snapshot.recipes)

    with pytest.raises(BackendUnsupportedError, match="recipe callables"):
        tl.trace(_mlp, _inputs(), backend="paddle", recipes=["not-callable"])


def test_paddle_streaming_still_refuses_typed() -> None:
    """The unlifted gated options keep their typed refusals."""

    with pytest.raises(BackendUnsupportedError):
        tl.trace(_mlp, _inputs(), backend="paddle", storage=tl.to_disk("unused.tlspec"))


def test_paddle_intervene_and_save_predicates_compose() -> None:
    """save= filtering and intervene= apply independently."""

    trace = tl.trace(
        _mlp,
        _inputs(),
        backend="paddle",
        save=tl.func("functional.relu"),
        intervene=tl.when(tl.func("functional.relu"), tl.zero_ablate()),
    )
    relu = _relu_op(trace)
    assert relu.has_saved_activation is True
    assert float(relu.out.abs().max()) == 0.0
    linear_ops = [op for op in trace.layer_list if "linear" in op.label]
    assert all(not op.has_saved_activation for op in linear_ops)


def test_paddle_intervened_trace_round_trips_metadata(tmp_path: Any) -> None:
    """Portable saves keep the intervention stamps; loads stay honest."""

    trace = _ablated_trace()
    path = tmp_path / "paddle_intervened.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    relu = _relu_op(loaded)
    assert relu.intervention_replaced is True
    assert any(record.replaced for record in relu.interventions)
    assert float(relu.out.abs().max()) == 0.0
