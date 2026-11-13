"""Paddle edit helpers: live captures, finished-trace edits and typed refusals.

Each of the six built-in helpers Paddle supports (``zero_ablate``, ``scale``,
``add``, ``clamp``, ``mean_ablate``, ``replace_with``) must give a live
``trace(intervene=...)`` capture that passes validation, and the same edit on a
fork of a clean capture (``trace.fork().do(...)``) must pass the replay check
and equal the live capture. An edit whose replay reaches an op the Paddle
adapter cannot replay (a training-mode dropout) refuses naming that op.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle import _edit_helpers, interventions  # noqa: E402
from torchlens.backends.paddle.backend import PaddleBackend  # noqa: E402
from torchlens.intervention._backend_replay import verify_backend_replay  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

_X = np.asarray([[-1.5, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]], dtype=np.float32)


def _chain(v: Any) -> Any:
    a = v + 1.0
    return paddle.nn.functional.relu(a * a - 2.0) + a


def _helpers() -> dict[str, Any]:
    return {
        "zero_ablate": tl.zero_ablate(),
        "scale": tl.scale(3.0),
        "add": tl.add(0.5),
        "clamp": tl.clamp(low=0.25, high=1.5),
        "mean_ablate": tl.mean_ablate(dim=0),
        "replace_with": tl.replace_with(paddle.full([2, 4], 0.75)),
    }


def _output(trace: Any) -> np.ndarray:
    outputs = [op for op in trace._layer_list if getattr(op, "is_output", False)]
    return np.asarray((outputs or list(trace._layer_list))[-1].out.numpy())


def _relu_label(trace: Any) -> str:
    return next(op.label for op in trace._layer_list if "relu" in str(op.func_name))


def test_helper_tables_agree() -> None:
    assert interventions._PADDLE_SUPPORTED_HELPER_NAMES == (
        _edit_helpers.PADDLE_SUPPORTED_HELPER_NAMES
    )
    assert sorted(_helpers()) == list(_edit_helpers.PADDLE_SUPPORTED_HELPER_NAMES)


@pytest.mark.parametrize("name", sorted(_helpers()))
def test_each_helper_live_capture_validates(name: str) -> None:
    trace = tl.trace(
        _chain,
        paddle.to_tensor(_X),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), _helpers()[name]),
    )
    assert PaddleBackend().validate_trace(trace) is True
    assert not np.array_equal(_output(trace), _output(tl.trace(_chain, paddle.to_tensor(_X))))


@pytest.mark.parametrize("name", sorted(_helpers()))
def test_each_helper_on_a_fork_passes_replay_and_equals_live(name: str) -> None:
    helper = _helpers()[name]
    clean = tl.trace(_chain, paddle.to_tensor(_X), backend="paddle")
    edited = clean.fork().do(_relu_label(clean), helper)
    assert verify_backend_replay(edited) == []
    assert edited.is_pristine is False
    live = tl.trace(
        _chain,
        paddle.to_tensor(_X),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), helper),
    )
    np.testing.assert_allclose(_output(edited), _output(live), rtol=1e-6, atol=1e-6)


def test_mean_ablate_with_a_source_refuses_typed() -> None:
    from torchlens.errors import BackendUnsupportedError

    with pytest.raises(BackendUnsupportedError, match="self form"):
        tl.trace(
            _chain,
            paddle.to_tensor(_X),
            backend="paddle",
            intervene=tl.when(
                tl.func("functional.relu"), tl.mean_ablate(paddle.ones([2, 4]), dim=0)
            ),
        )


def _with_dropout(v: Any) -> Any:
    a = v + 1.0
    s = paddle.nn.functional.relu(a * a - 2.0)
    return paddle.nn.functional.dropout(s, p=0.5, training=True) + a


def test_fork_edit_reaching_an_unreplayable_op_refuses_by_name() -> None:
    paddle.seed(0)
    clean = tl.trace(_with_dropout, paddle.to_tensor(_X), backend="paddle")
    dropout = next(op for op in clean._layer_list if "dropout" in str(op.func_name))
    with pytest.raises(Exception, match="cannot be replayed") as caught:
        clean.fork().do(_relu_label(clean), tl.scale(2.0))
    assert str(dropout._label_raw) in str(caught.value) or dropout.label in str(caught.value)


def _apply(helper: Any) -> Callable[[Any], Any]:
    return _edit_helpers.resolve_paddle_helper(helper, paddle)


def test_resolved_helpers_compute_their_documented_values() -> None:
    out = paddle.to_tensor(_X)
    np.testing.assert_allclose(_apply(tl.scale(2.0))(out).numpy(), _X * 2.0)
    np.testing.assert_allclose(_apply(tl.add(1.0))(out).numpy(), _X + 1.0)
    np.testing.assert_allclose(
        _apply(tl.clamp(low=-1.0, high=1.0))(out).numpy(), np.clip(_X, -1.0, 1.0)
    )
    np.testing.assert_allclose(
        _apply(tl.mean_ablate(dim=0))(out).numpy(),
        np.broadcast_to(_X.mean(axis=0, keepdims=True), _X.shape),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _apply(tl.mean_ablate(dim=None))(out).numpy(), np.full_like(_X, _X.mean()), rtol=1e-6
    )
    assert not _apply(tl.zero_ablate())(out).numpy().any()
