"""TensorFlow random kernels replay to their saved values; the caller's random state is untouched."""

from __future__ import annotations

import pickle
import types

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from torchlens.backends._rng_capture import RngDraw, rng_module  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

rng = rng_module("tf")


def _forward(x: object, generator: object) -> tuple[object, ...]:
    """Legacy seeded (op seed and global-only), stateless, generator and dropout draws."""

    a = tf.random.normal([4], seed=1)
    b = tf.random.normal([3], seed=1)
    c = tf.random.uniform([3])
    d = tf.random.stateless_normal([3], seed=[1, 2])
    e = generator.normal([4])
    f = tf.nn.dropout(x, 0.5, seed=3)
    return a, b, c, d, e, f


def _replay_ok(log: object) -> list[object]:
    """Replay every draw from its data round trip; return the replayable ones."""

    replayed = []
    for draw in log.draws:
        if rng.can_replay_draw(draw) is not None:
            continue
        live = log.output_of(draw.ordinal)
        loaded = RngDraw.from_data(pickle.loads(pickle.dumps(draw.to_data())))
        again = rng.replay_draw(types.SimpleNamespace(rng=loaded.to_data()), None)[0]
        np.testing.assert_array_equal(again.numpy(), live.numpy())
        replayed.append(draw)
    return replayed


@pytest.mark.parametrize("seed", [5, 123])
def test_draws_replay_and_caller_sequence_is_unchanged(seed: int) -> None:
    """Every family replays; the caller's next legacy and generator draws are unchanged."""

    x = tf.ones([16])
    tf.random.set_seed(seed)
    generator = tf.random.Generator.from_seed(seed)
    session = types.SimpleNamespace()
    with rng.rng_scope(session) as log:
        outputs = _forward(x, generator)
    after_state = generator.state.numpy().copy()
    kinds = {d.op_name: d.kind for d in log.draws}
    assert kinds["RandomStandardNormal"] == "legacy_seeded"
    assert kinds["StatelessRandomNormalV2"] == "stateless"
    assert "generator_state" in kinds.values()
    replayed = _replay_ok(log)
    names = [d.op_name for d in replayed]
    assert names.count("RandomStandardNormal") >= 2 and "RandomUniform" in names
    assert any(d.kind == "generator_state" for d in replayed)
    # Draws are logged per kernel: tf.random.normal scales the kernel output, so the
    # capture lane looks a draw up by the RandomStandardNormal output it records.
    assert rng.draw_for(session, log.output_of(0)) is log.draws[0]
    assert rng.draw_for(session, outputs[0]) is None
    np.testing.assert_array_equal(generator.state.numpy(), after_state)
    tf.random.set_seed(seed)
    reference_generator = tf.random.Generator.from_seed(seed)
    _forward(x, reference_generator)
    expected = (tf.random.normal([4], seed=1).numpy(), reference_generator.normal([2]).numpy())
    tf.random.set_seed(seed)
    generator2 = tf.random.Generator.from_seed(seed)
    with rng.rng_scope(None) as log2:
        _forward(x, generator2)
    _replay_ok(log2)
    got = (tf.random.normal([4], seed=1).numpy(), generator2.normal([2]).numpy())
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


def test_changed_seed_changes_draws_and_replay_follows() -> None:
    """Different global seeds give different legacy draws, each replayed."""

    values = []
    for seed in (1, 2):
        tf.random.set_seed(seed)
        with rng.rng_scope(None) as log:
            out = tf.random.normal([5])
        draw = log.draws[0]
        assert rng.can_replay_draw(draw) is None
        np.testing.assert_array_equal(rng.replay_draw(draw, None)[0].numpy(), out.numpy())
        values.append(out.numpy())
    assert not np.array_equal(values[0], values[1])


def test_kernel_used_before_the_scope_is_refused_not_guessed() -> None:
    """A seeded kernel that already ran outside the scope is marked unreplayable."""

    tf.random.set_seed(8)
    tf.random.normal([4], seed=21)
    with rng.rng_scope(None) as log:
        tf.random.normal([4], seed=21)
    assert rng.can_replay_draw(log.draws[0]) == rng.REASON_POSITION


def test_stateless_replay_follows_parent_values() -> None:
    """A stateless draw replays from the given parents, so a changed key changes it."""

    with rng.rng_scope(None) as log:
        out = tf.random.stateless_uniform([3], seed=[4, 5])
    draw = next(d for d in log.draws if d.op_name.startswith("StatelessRandomUniform"))
    parents = list(draw.recorded_parents)
    np.testing.assert_array_equal(rng.replay_draw(draw, parents)[0].numpy(), out.numpy())
    key_slot = 1
    parents[key_slot] = parents[key_slot] + 1
    assert not np.array_equal(rng.replay_draw(draw, parents)[0].numpy(), out.numpy())


def test_exception_inside_scope_removes_hooks_and_marks_log_unfinished() -> None:
    """A raising forward removes the callback and wrappers."""

    from tensorflow.python.ops import gen_stateful_random_ops

    before = gen_stateful_random_ops.stateful_standard_normal_v2
    with pytest.raises(RuntimeError), rng.rng_scope(None) as log:
        tf.random.normal([2], seed=4)
        raise RuntimeError("forward failed")
    assert gen_stateful_random_ops.stateful_standard_normal_v2 is before
    assert not log.finished
    count = len(log.draws)
    tf.random.normal([2], seed=4)
    assert len(log.draws) == count


def test_unseeded_legacy_draw_states_the_limit() -> None:
    """Without any seed TensorFlow draws an OS seed; the draw says so."""

    tf.random.set_seed(None)
    with rng.rng_scope(None) as log:
        tf.random.normal([2])
    assert rng.can_replay_draw(log.draws[0]) == rng.REASON_UNSEEDED
