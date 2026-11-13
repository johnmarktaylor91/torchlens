"""tinygrad random draws replay to their saved values without touching the caller's counter."""

from __future__ import annotations

import pickle
import types

import numpy as np
import pytest

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor

from torchlens.backends._rng_capture import RngDraw, rng_module  # noqa: E402

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

rng = rng_module("tinygrad")


def _forward(x: Tensor) -> tuple[Tensor, ...]:
    """Two draws, a normal draw and a dropout."""

    with Tensor.train():
        return Tensor.rand(4), Tensor.rand(3), Tensor.randn(2), x.dropout(0.5)


def _replay_all(log: object, outputs: tuple[Tensor, ...], x: Tensor) -> None:
    """Every draw replays bit for bit, also after a data round trip."""

    assert len(log.draws) == len(outputs)
    for draw, out in zip(log.draws, outputs):
        assert rng.can_replay_draw(draw) is None, draw.reason
        parents = (x,) if draw.op_name == "dropout" else ()
        loaded = RngDraw.from_data(pickle.loads(pickle.dumps(draw.to_data())))
        with Tensor.train():
            again = rng.replay_draw(types.SimpleNamespace(rng=loaded.to_data()), parents)[0]
        np.testing.assert_array_equal(again.numpy(), out.numpy())


@pytest.mark.parametrize("seed", [0, 42])
def test_draws_replay_and_caller_counter_is_unchanged(seed: int) -> None:
    """Fresh tables: draws resolve at scope exit and replay; the next draw is unchanged."""

    x = Tensor.ones(16).contiguous().realize()
    Tensor.manual_seed(seed)
    reference = [t.numpy() for t in _forward(x)]
    expected_next = Tensor.rand(5).numpy()
    Tensor.manual_seed(seed)
    session = types.SimpleNamespace()
    with rng.rng_scope(session) as log:
        outputs = _forward(x)
    Tensor.realize(*outputs)
    for out, ref in zip(outputs, reference):
        np.testing.assert_array_equal(out.numpy(), ref)
    _replay_all(log, outputs, x)
    assert rng.draw_for(session, outputs[0]) is log.draws[0]
    np.testing.assert_array_equal(Tensor.rand(5).numpy(), expected_next)


def test_pending_counter_is_never_read_early_and_resolves_after_realize() -> None:
    """A lazy pre-scope weight keeps its eager value; draws resolve after the realize."""

    x = Tensor.ones(8).contiguous().realize()
    Tensor.manual_seed(7)
    Tensor.rand(2).realize()
    weight_ref = Tensor.rand(6).numpy()
    draws_ref = [t.numpy() for t in _forward(x)]
    Tensor.manual_seed(7)
    Tensor.rand(2).realize()
    weight = Tensor.rand(6)
    with rng.rng_scope(None) as log:
        outputs = _forward(x)
    assert all(d.reason == rng.REASON_COUNTER_PENDING for d in log.draws)
    Tensor.realize(weight, *outputs)
    np.testing.assert_array_equal(weight.numpy(), weight_ref)
    for out, ref in zip(outputs, draws_ref):
        np.testing.assert_array_equal(out.numpy(), ref)
    rng.finalize_counters(log)
    _replay_all(log, outputs, x)


def test_changed_seed_changes_draws_and_replay_follows() -> None:
    """Different seeds record different seeds and each replays its own values."""

    values = []
    for seed in (1, 2):
        Tensor.manual_seed(seed)
        with rng.rng_scope(None) as log:
            out = Tensor.rand(5)
        np.testing.assert_array_equal(rng.replay_draw(log.draws[0], ())[0].numpy(), out.numpy())
        values.append(out.numpy())
    assert not np.array_equal(values[0], values[1])


def test_exception_inside_scope_restores_methods_and_marks_log_unfinished() -> None:
    """A raising forward leaves no wrapper behind and an unfinished log."""

    before = {
        n: Tensor.__dict__[n]
        for n in (*rng.TENSOR_RANDOM_NAMES, "_next_counter")
        if n in Tensor.__dict__
    }
    with pytest.raises(RuntimeError), rng.rng_scope(None) as log:
        Tensor.rand(2)
        raise RuntimeError("forward failed")
    assert {n: Tensor.__dict__[n] for n in before} == before
    assert not log.finished


def test_failed_replay_restores_the_caller_tables() -> None:
    """A replay that raises puts the caller's seed and counter objects back."""

    Tensor.manual_seed(3)
    with rng.rng_scope(None) as log:
        Tensor.rand(3)
    device = log.draws[0].state[0]
    seeds, counters = Tensor._device_seeds.get(device), Tensor._device_rng_counters.get(device)
    broken = RngDraw(**{**log.draws[0].__dict__, "kwargs": (("bogus_argument", 1),)})
    with pytest.raises(TypeError):
        rng.replay_draw(broken, ())
    assert Tensor._device_seeds.get(device) is seeds
    assert Tensor._device_rng_counters.get(device) is counters
