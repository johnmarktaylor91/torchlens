"""MLX random draws replay to their saved values; the caller's random state is untouched."""

from __future__ import annotations

import pickle
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from torchlens.backends._rng_capture import RngDraw, rng_module  # noqa: E402

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

rng = rng_module("mlx")


def _np(value: object) -> np.ndarray:
    """Evaluate an MLX array to NumPy."""

    return np.array(value)


def _forward(x: object) -> tuple[object, ...]:
    """Two global draws, an explicit-key draw and a dropout."""

    a = mx.random.normal((4,))
    b = mx.random.uniform(-1.0, 1.0, (3,))
    c = mx.random.normal((2,), key=mx.random.key(7))
    drop = nn.Dropout(0.5)
    drop.train()
    return a, b, c, drop(x)


def _replay_all(session: object, outputs: tuple[object, ...], x: object) -> None:
    """Every logged draw replays bit for bit, also after a data round trip."""

    log = session._torchlens_rng_log
    assert log.finished and len(log.draws) == 4
    assert rng.draw_for(session, outputs[0]) is log.draws[0]
    for draw in log.draws:
        assert rng.can_replay_draw(draw) is None
        live = log.output_of(draw.ordinal)
        np.testing.assert_array_equal(_np(rng.replay_draw(draw, None)[0]), _np(live))
        loaded = RngDraw.from_data(pickle.loads(pickle.dumps(draw.to_data())))
        recipe = types.SimpleNamespace(rng=loaded.to_data())
        np.testing.assert_array_equal(_np(rng.replay_draw(recipe, None)[0]), _np(live))
    bern = log.draws[3]
    assert bern.op_name == "bernoulli"
    np.testing.assert_array_equal(_np(outputs[3]) != 0, _np(rng.replay_draw(bern, None)[0]))


@pytest.mark.parametrize("seed", [0, 123])
def test_draws_replay_and_caller_state_is_unchanged(seed: int) -> None:
    """Two draws, an explicit key and dropout replay under two seeds."""

    x = mx.ones((16,))
    mx.random.seed(seed)
    session = types.SimpleNamespace()
    with rng.rng_scope(session):
        outputs = _forward(x)
    mx.eval(*outputs)
    after_forward = _np(mx.random.state[0])
    mx.random.seed(seed)
    reference = _forward(x)
    np.testing.assert_array_equal(_np(outputs[0]), _np(reference[0]))
    np.testing.assert_array_equal(after_forward, _np(mx.random.state[0]))
    _replay_all(session, outputs, x)
    np.testing.assert_array_equal(after_forward, _np(mx.random.state[0]))


def test_changed_seed_changes_draws_and_replay_follows() -> None:
    """Different seeds record different keys and each replays its own values."""

    values = []
    for seed in (1, 2):
        mx.random.seed(seed)
        session = types.SimpleNamespace()
        with rng.rng_scope(session) as log:
            out = mx.random.normal((5,))
        np.testing.assert_array_equal(_np(rng.replay_draw(log.draws[0], None)[0]), _np(out))
        values.append(_np(out))
    assert not np.array_equal(values[0], values[1])


def test_positional_key_and_array_parent_replay() -> None:
    """A positional key and an array probability parent are recorded correctly."""

    p = mx.array([0.1, 0.9, 0.5])
    with rng.rng_scope(None) as log:
        out = mx.random.bernoulli(p, (3,), mx.random.key(3))
        other = mx.random.bernoulli(p)
    assert log.draws[0].kind == "explicit_key" and log.draws[1].kind == "global_state"
    np.testing.assert_array_equal(_np(rng.replay_draw(log.draws[0], (p,))[0]), _np(out))
    np.testing.assert_array_equal(_np(rng.replay_draw(log.draws[1], [p])[0]), _np(other))


def test_exception_inside_scope_restores_functions_and_marks_log_unfinished() -> None:
    """A raising forward leaves no wrapper behind and an unfinished log."""

    originals = {name: getattr(mx.random, name) for name, _ in rng.KEY_POSITIONS}
    with pytest.raises(RuntimeError):
        with rng.rng_scope(None) as log:
            mx.random.normal((2,))
            raise RuntimeError("forward failed")
    assert {name: getattr(mx.random, name) for name in originals} == originals
    assert not log.finished and len(log.draws) == 1


def test_failed_replay_leaves_caller_state_unchanged() -> None:
    """A replay that raises (missing parent) does not move the global key."""

    with rng.rng_scope(None) as log:
        mx.random.bernoulli(mx.array([0.5, 0.5]), (2,))
    before = _np(mx.random.state[0])
    big = log.draws[0]
    stripped = RngDraw(**{**big.__dict__, "recorded_parents": None})
    with pytest.raises(ValueError):
        rng.replay_draw(stripped, None)
    np.testing.assert_array_equal(before, _np(mx.random.state[0]))
