"""Framework-free checks of the random-state capture package (no backend runtime)."""

from __future__ import annotations

import pickle
import types

import numpy as np
import pytest

from torchlens.backends._rng_capture import (
    ParentSlot,
    RngDraw,
    RngLog,
    decode_call,
    draw_from_recipe,
    encode_call,
    patched_attributes,
    rng_module,
)

pytestmark = [pytest.mark.smoke]


def _is_array(value: object) -> bool:
    """NumPy arrays stand in for framework arrays."""

    return isinstance(value, np.ndarray)


def test_encode_decode_round_trip_replaces_arrays_with_parent_slots() -> None:
    """Arrays become ordered slots; literals stay; decode substitutes parents."""

    a, b = np.ones(2), np.zeros(3)
    encoded = encode_call((a, 3, (1, 2)), {"p": b, "flag": True}, is_array=_is_array)
    assert encoded is not None
    args, kwargs, arrays = encoded
    assert args == (ParentSlot(0), 3, (1, 2))
    assert dict(kwargs) == {"flag": True, "p": ParentSlot(1)}
    assert arrays[0] is a and arrays[1] is b
    draw = RngDraw("x", "global_state", "f", 0, args=args, kwargs=kwargs)
    out_args, out_kwargs = decode_call(draw, ("A", "B"))
    assert out_args == ["A", 3, (1, 2)] and out_kwargs == {"flag": True, "p": "B"}


def test_encode_refuses_an_object_without_a_data_form() -> None:
    """An argument that is not data or an array makes the call unencodable."""

    assert encode_call((object(),), {}, is_array=_is_array) is None


def test_decode_without_enough_parents_raises() -> None:
    """A missing parent is an error, never a silent default."""

    draw = RngDraw("x", "global_state", "f", 0, args=(ParentSlot(1),))
    with pytest.raises(ValueError, match="array argument 1"):
        decode_call(draw, (np.ones(1),))


def test_draw_data_round_trip_survives_pickle() -> None:
    """``to_data`` is plain data: it pickles and rebuilds an equal draw."""

    draw = RngDraw(
        "mlx",
        "global_state",
        "normal",
        2,
        (np.arange(2, dtype=np.uint32),),
        ((3,),),
        (("dtype", 1),),
    )
    data = pickle.loads(pickle.dumps(draw.to_data()))
    again = RngDraw.from_data(data)
    assert again.kind == draw.kind and again.args == draw.args and again.kwargs == draw.kwargs
    np.testing.assert_array_equal(again.state[0], draw.state[0])
    assert draw_from_recipe(types.SimpleNamespace(rng=data)).op_name == "normal"
    assert draw_from_recipe(draw) is draw


def test_patched_attributes_restores_on_exception_and_spares_later_wrappers() -> None:
    """Restoration runs when the block raises; a foreign later wrapper is kept."""

    owner = types.SimpleNamespace(f=lambda: "orig", g=lambda: "orig-g")
    original_f, original_g = owner.f, owner.g
    patched = patched_attributes([(owner, "f"), (owner, "missing")], lambda n, o: lambda: "wrapped")
    with pytest.raises(RuntimeError), patched:
        assert owner.f() == "wrapped"
        raise RuntimeError("boom")
    assert owner.f is original_f
    foreign = lambda: "foreign"  # noqa: E731
    with patched_attributes([(owner, "g")], lambda n, o: lambda: "wrapped"):
        owner.g = foreign
    assert owner.g is foreign
    owner.g = original_g


def test_patched_attributes_keeps_staticmethod_descriptors() -> None:
    """A class staticmethod is wrapped as a staticmethod and restored exactly."""

    class Holder:
        @staticmethod
        def make(x: int) -> int:
            return x

    descriptor = Holder.__dict__["make"]
    with patched_attributes([(Holder, "make")], lambda n, o: lambda x: o(x) + 1):
        assert isinstance(Holder.__dict__["make"], staticmethod)
        assert Holder.make(1) == 2 and Holder().make(1) == 2
    assert Holder.__dict__["make"] is descriptor


def test_log_indexes_draws_by_output_identity_and_nesting() -> None:
    """Draws are found by output object; only the outermost call logs."""

    log = RngLog("x")
    out = np.ones(1)
    with log.outermost() as first, log.outermost() as second:
        assert first and not second
    log.add(RngDraw("x", "global_state", "f", 99), out)
    assert log.draw_for(out) is log.draws[0] and log.draws[0].ordinal == 0
    assert log.draw_for(np.ones(1)) is None and log.output_of(0) is out


def test_rng_module_lookup_is_closed() -> None:
    """Only the four backends resolve."""

    with pytest.raises(KeyError):
        rng_module("jax")
