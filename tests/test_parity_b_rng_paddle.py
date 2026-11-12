"""Paddle random draws replay to their saved values; the caller's generator state is untouched."""

from __future__ import annotations

import pickle
import types

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

from torchlens.backends._rng_capture import RngDraw, rng_module  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

rng = rng_module("paddle")


def _forward(x: object) -> tuple[object, ...]:
    """Two global draws, an explicitly seeded draw, functional and layer dropout."""

    a = paddle.randn([4])
    b = paddle.rand([3])
    c = paddle.uniform([3], min=-1.0, max=1.0, seed=5)
    d = paddle.nn.functional.dropout(x, 0.5, training=True)
    layer = paddle.nn.Dropout(0.5)
    layer.train()
    return a, b, c, d, layer(x)


def _next_values() -> np.ndarray:
    """The caller's next draw, to compare sequences."""

    return paddle.rand([5]).numpy()


@pytest.mark.parametrize("seed", [0, 123])
def test_draws_replay_and_caller_state_is_unchanged(seed: int) -> None:
    """Two draws, a seeded draw and both dropouts replay under two seeds."""

    x = paddle.ones([16])
    paddle.seed(seed)
    session = types.SimpleNamespace()
    with rng.rng_scope(session) as log:
        outputs = _forward(x)
    paddle.seed(seed)
    reference = _forward(x)
    expected_next = _next_values()
    paddle.seed(seed)
    _forward(x)
    assert log.finished and len(log.draws) == 5
    assert [d.kind for d in log.draws][:3] == ["global_state", "global_state", "explicit_seed"]
    for draw, out, ref in zip(log.draws, outputs, reference):
        assert rng.can_replay_draw(draw) is None
        np.testing.assert_array_equal(out.numpy(), ref.numpy())
        assert rng.draw_for(session, out) is draw
        np.testing.assert_array_equal(
            rng.replay_draw(draw, (x,) * len(draw.recorded_parents or ()))[0].numpy(), out.numpy()
        )
        in_memory = types.SimpleNamespace(rng=RngDraw.from_data(draw.to_data()))
        np.testing.assert_array_equal(rng.replay_draw(in_memory, None)[0].numpy(), out.numpy())
    np.testing.assert_array_equal(_next_values(), expected_next)


def test_changed_seed_changes_draws_and_replay_follows() -> None:
    """Different seeds record different states and each replays its own values."""

    values = []
    for seed in (1, 2):
        paddle.seed(seed)
        with rng.rng_scope(None) as log:
            out = paddle.randn([5])
        np.testing.assert_array_equal(rng.replay_draw(log.draws[0], None)[0].numpy(), out.numpy())
        values.append(out.numpy())
    assert not np.array_equal(values[0], values[1])


def test_saved_data_replays_seeded_draws_and_names_the_generator_limit() -> None:
    """A seeded draw survives pickling; a generator-state draw states why it cannot."""

    with rng.rng_scope(None) as log:
        seeded = paddle.uniform([3], min=0.0, max=1.0, seed=9)
        drawn = paddle.rand([3])
    loaded = RngDraw.from_data(pickle.loads(pickle.dumps(log.draws[0].to_data())))
    assert rng.data_reason(loaded) is None
    np.testing.assert_array_equal(rng.replay_draw(loaded, None)[0].numpy(), seeded.numpy())
    assert rng.data_reason(log.draws[1]) == rng.REASON_STATE_NOT_DATA
    np.testing.assert_array_equal(rng.replay_draw(log.draws[1], None)[0].numpy(), drawn.numpy())


def test_exception_inside_scope_restores_functions_and_marks_log_unfinished() -> None:
    """A raising forward leaves no wrapper behind and an unfinished log."""

    before = (paddle.randn, paddle.nn.functional.dropout)
    with pytest.raises(RuntimeError):
        with rng.rng_scope(None) as log:
            paddle.randn([2])
            raise RuntimeError("forward failed")
    assert (paddle.randn, paddle.nn.functional.dropout) == before
    assert not log.finished and len(log.draws) == 1


def test_failed_replay_restores_the_caller_state() -> None:
    """A replay that raises inside the draw still restores the caller's generator."""

    paddle.seed(4)
    with rng.rng_scope(None) as log:
        paddle.randint(0, 9, [3])
    paddle.seed(4)
    expected_next = _next_values()
    paddle.seed(4)
    broken = RngDraw(**{**log.draws[0].__dict__, "kwargs": (("bogus_argument", 1),)})
    with pytest.raises(TypeError):
        rng.replay_draw(broken, None)
    np.testing.assert_array_equal(_next_values(), expected_next)
