"""MLX edit helpers on every path: live capture and finished-trace fork.

Each built-in helper with an MLX application is applied at the rectifier of a
small chain, once live (``intervene=``) and once on a fork of a clean capture
(the replay adapter's ``apply_helper``). Both results must equal the chain
run by hand with the helper's math written in ``mlx.core``, and both edited
traces must pass their validators.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

mx = pytest.importorskip("mlx.core")

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import MLXBackend  # noqa: E402
from torchlens.backends.mlx._mlx_helpers import MLX_SUPPORTED_HELPER_NAMES  # noqa: E402
from torchlens.errors import BackendUnsupportedError, InvalidArgumentError  # noqa: E402
from torchlens.intervention._backend_replay import verify_backend_replay  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_X = np.asarray([[-1.5, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]], dtype=np.float32)
_DIRECTION = [1.0, 0.0, -1.0, 0.5]
_PLANE = [1.0, 1.0, 0.0, 0.0]


def _chain(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return ``v -> edit(relu((v + 1)^2 - 2)) + (v + 1)``."""

    def chain(v: Any) -> Any:
        a = v + 1.0
        return edit(mx.maximum(a * a - 2.0, 0.0)) + a

    return chain


def _project(out: Any, direction: Any) -> Any:
    """Component of ``out`` along ``direction`` over axis 1, by hand."""

    d = direction.reshape(1, -1)
    return d * (mx.sum(out * d, axis=1, keepdims=True) / mx.sum(d * d, axis=1, keepdims=True))


def _clip(out: Any, max_norm: float) -> Any:
    """Torch ``clip_norm`` branch, by hand (evaluates the norm)."""

    norm = float(np.linalg.norm(np.asarray(out).reshape(-1)))
    return out if norm <= max_norm else out * (max_norm / norm)


def _cases() -> list[tuple[str, Callable[[], Any], Callable[[Any], Any]]]:
    """``(id, helper factory, hand edit)`` for every helper in the MLX table."""

    direction = mx.array(_DIRECTION)
    plane = mx.array(_PLANE)
    return [
        ("zero_ablate", tl.zero_ablate, mx.zeros_like),
        ("scale", lambda: tl.scale(2.0), lambda o: o * 2.0),
        ("add", lambda: tl.add(0.5), lambda o: o + 0.5),
        ("clamp", lambda: tl.clamp(low=0.1, high=0.6), lambda o: mx.clip(o, 0.1, 0.6)),
        ("mean_ablate", lambda: tl.mean_ablate(dim=None), lambda o: mx.zeros_like(o) + mx.mean(o)),
        (
            "replace_with",
            lambda: tl.replace_with(mx.full((2, 4), 0.3)),
            lambda o: mx.full((2, 4), 0.3),
        ),
        (
            "steer",
            lambda: tl.steer(direction, strength=0.5, dim=1),
            lambda o: o + direction.reshape(1, -1) * 0.5,
        ),
        ("project_onto", lambda: tl.project_onto(plane, dim=1), lambda o: _project(o, plane)),
        ("project_out", lambda: tl.project_out(plane, dim=1), lambda o: o - _project(o, plane)),
        ("clip_norm", lambda: tl.clip_norm(0.5), lambda o: _clip(o, 0.5)),
    ]


_CASES = _cases()
_IDS = [case[0] for case in _CASES]


def _expected(edit: Callable[[Any], Any]) -> np.ndarray:
    """The chain run by hand with ``edit`` at the rectifier."""

    out = _chain(edit)(mx.array(_X))
    mx.eval(out)
    return np.asarray(out)


def _rectifier_label(trace: Any) -> str:
    """Label of the chain's ``maximum`` op."""

    return next(op.label for op in trace._layer_list if op.func_name == "maximum")


def _final_out(trace: Any) -> np.ndarray:
    """Host copy of the trace's model output."""

    outputs = [op for op in trace._layer_list if getattr(op, "is_output", False)]
    value = (outputs or list(trace._layer_list))[-1].out
    mx.eval(value)
    return np.asarray(value)


def test_case_list_covers_the_mlx_helper_table() -> None:
    """Every helper in the MLX table has a case here, and the table literal matches."""

    assert set(_IDS) == set(MLX_SUPPORTED_HELPER_NAMES)
    tree = ast.parse((_REPO / "torchlens/backends/mlx/interventions.py").read_text("utf-8"))
    literal = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_MLX_SUPPORTED_HELPER_NAMES" for t in node.targets)
    )
    assert isinstance(literal, ast.Tuple)
    assert {element.value for element in literal.elts} == set(MLX_SUPPORTED_HELPER_NAMES)


@pytest.mark.parametrize(("name", "helper", "hand"), _CASES, ids=_IDS)
def test_live_helper_equals_hand_edit_and_validates(
    name: str, helper: Callable[[], Any], hand: Callable[[Any], Any]
) -> None:
    """``intervene=tl.when(site, helper)`` equals the hand edit and passes validation."""

    trace = tl.trace(
        _chain(lambda o: o),
        mx.array(_X),
        backend="mlx",
        intervene=tl.when(tl.func("maximum"), helper()),
    )
    np.testing.assert_allclose(_final_out(trace), _expected(hand), rtol=1e-5, atol=1e-6)
    assert MLXBackend().validate_trace(trace).passed, name


@pytest.mark.parametrize(("name", "helper", "hand"), _CASES, ids=_IDS)
def test_fork_helper_equals_hand_edit_and_validates(
    name: str, helper: Callable[[], Any], hand: Callable[[Any], Any]
) -> None:
    """A fork edited with the helper equals the hand edit and passes the replay check."""

    clean = tl.trace(_chain(lambda o: o), mx.array(_X), backend="mlx")
    edited = clean.fork().do(_rectifier_label(clean), helper())
    np.testing.assert_allclose(_final_out(edited), _expected(hand), rtol=1e-5, atol=1e-6)
    assert verify_backend_replay(edited) == [], name


def test_fork_helper_outside_the_table_refuses_typed() -> None:
    """A helper with no MLX application refuses on a fork with the fork driver's code."""

    clean = tl.trace(_chain(lambda o: o), mx.array(_X), backend="mlx")
    with pytest.raises(Exception, match="no replay application") as info:
        clean.fork().do(_rectifier_label(clean), tl.add_noise(0.1))
    assert info.value.fields["code"] == "backend_replay_edit_helper_unsupported"


def test_live_helper_outside_the_table_refuses_typed() -> None:
    """A helper with no MLX application refuses at capture, naming the helper."""

    with pytest.raises(BackendUnsupportedError, match="add_noise") as info:
        tl.trace(
            _chain(lambda o: o),
            mx.array(_X),
            backend="mlx",
            intervene=tl.when(tl.func("maximum"), tl.add_noise(0.1)),
        )
    assert info.value.code == "interventions_unsupported"


def test_projection_with_zero_direction_refuses() -> None:
    """A zero-norm projection direction refuses before the capture runs."""

    with pytest.raises(Exception, match="zero norm") as info:
        tl.trace(
            _chain(lambda o: o),
            mx.array(_X),
            backend="mlx",
            intervene=tl.when(tl.func("maximum"), tl.project_onto(mx.zeros((4,)), dim=1)),
        )
    assert info.value.fields["code"] == "intervention_helper_arguments_invalid"


def test_forward_signal_edit_at_a_grad_site_refuses() -> None:
    """An edit that reads forward state cannot edit a gradient (torch's mount rule)."""

    from torchlens.backends.mlx import GradOptions

    donor = tl.trace(_chain(lambda o: o), mx.array(_X), backend="mlx")
    with pytest.raises(InvalidArgumentError) as info:
        MLXBackend().capture_trace(
            _chain(lambda o: o),
            mx.array(_X),
            grad_options=GradOptions(input_grad_argnums=(0,), loss_fn=mx.sum),
            intervene=tl.when(tl.grad(tl.func("maximum")), tl.patch_from(donor)),
        )
    assert info.value.fields["code"] == "edit_signal_mismatch"
