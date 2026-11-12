"""Replay session over the NumPy fake adapter: repeats, groups, edits, cleanup."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.support import replay_fake_adapter as fake
from torchlens.backends import _replay_adapters as ra


class CountingAdapter:
    """Wrap the fake adapter and count ``forward`` calls per op label."""

    def __init__(self, arity_bug: str | None = None) -> None:
        self.forward_calls: list[str] = []
        self.arity_bug = arity_bug

    def __getattr__(self, name: str) -> Any:
        return getattr(fake, name)

    def forward(self, trace: Any, op: Any, parent_values: Any) -> tuple[Any, ...]:
        self.forward_calls.append(op.label)
        out = fake.forward(trace, op, parent_values)
        return out[:-1] if op.label == self.arity_bug else out


def _trace() -> fake.FakeTrace:
    """x -> sq = x*x -> (lo, hi) = split2(sq) -> m = 3*hi -> p = lo*lo ... s = sum(m)."""

    ops = [
        fake.FakeOp("x", "input", (), ("x",)),
        fake.FakeOp("sq", "mul", ("x", "x"), ("sq",)),
        fake.FakeOp("lo", "split2", ("sq",), ("lo", "hi"), attrs={"index": 2}),
        fake.FakeOp("m", "scale", ("hi",), ("m",), literals=((("factor",), 3.0),)),
        fake.FakeOp("p", "mul", ("lo", "lo"), ("p",)),
        fake.FakeOp("s", "sum", ("m",), ("s",), attrs={"axis": 0}),
        fake.FakeOp("t", "sum", ("p",), ("t",), attrs={"axis": 0}),
        fake.FakeOp("u", "add", ("s", "t"), ("u",)),
    ]
    return fake.build_trace(ops, {"x": np.arange(1.0, 6.0)})


def _steps(trace: fake.FakeTrace, skip_inputs: bool = True) -> list[ra.ReplayStep]:
    steps = []
    for op in trace.ops.values():
        if skip_inputs and op.op_identity == "input":
            continue
        rec = fake.recipe(trace, op)
        assert rec is not None
        steps.append(ra.ReplayStep.from_recipe(op, rec))
    return steps


def _session(trace: fake.FakeTrace, adapter: Any = fake, **kw: Any) -> ra.ReplaySession:
    kw.setdefault("initial", {"x": trace.payloads["x"]})
    return ra.ReplaySession(adapter, trace, _steps(trace), **kw)


def test_replay_reproduces_every_payload_exactly() -> None:
    trace = _trace()
    values = _session(trace).run()
    assert set(values) == set(trace.payloads)
    for label, payload in trace.payloads.items():
        assert fake.close(payload, values[label]), label


def test_repeated_parent_is_passed_and_counted_per_occurrence() -> None:
    trace = _trace()
    seen: list[int] = []

    class Spy(CountingAdapter):
        def forward(self, trace: Any, op: Any, parent_values: Any) -> tuple[Any, ...]:
            if op.label == "sq":
                seen.append(len(parent_values))
            return super().forward(trace, op, parent_values)

    session = _session(trace, Spy(), keep={"u"})
    assert session.remaining_uses("x") == 2
    assert session.remaining_uses("lo") == 2
    session.step()
    assert seen == [2]
    assert session.remaining_uses("x") == 0


def test_multi_output_group_commits_atomically() -> None:
    trace = _trace()
    adapter = CountingAdapter(arity_bug="lo")
    session = _session(trace, adapter)
    session.step()  # sq
    before = session.values()
    with pytest.raises(ra.ReplaySessionError) as info:
        session.step()  # split2 returns one value for two slots
    assert info.value.reason == "arity"
    assert info.value.op_label == "lo"
    after = session.values()
    assert set(after) == set(before) and "lo" not in after and "hi" not in after
    assert not session.done


def test_last_use_releases_values_unless_kept() -> None:
    trace = _trace()
    session = _session(trace, keep={"u"})
    session.step()  # sq consumed x twice
    assert "x" not in session.values()
    session.step()  # split2 consumed sq
    assert set(session.values()) == {"lo", "hi"}
    values = session.run()
    assert set(values) == {"u"}
    assert fake.close(trace.payloads["u"], values["u"])


def test_no_op_substitution_reproduces_the_trace_exactly() -> None:
    trace = _trace()
    values = _session(trace, substitutions={"hi": trace.payloads["hi"].copy()}).run()
    for label, payload in trace.payloads.items():
        assert fake.close(payload, values[label]), label


def test_substitution_of_one_slot_of_a_group_flows_downstream() -> None:
    trace = _trace()
    edited_hi = np.array([10.0, 20.0, 30.0])
    values = _session(trace, substitutions={"hi": edited_hi}).run()
    x = trace.payloads["x"]
    lo = (x * x)[:2]
    expected_u = np.sum(3.0 * edited_hi) + np.sum(lo * lo)
    assert values["hi"] is edited_hi
    assert fake.close(lo, values["lo"])
    assert fake.close(expected_u, values["u"])


def test_fully_substituted_step_does_not_run() -> None:
    trace = _trace()
    adapter = CountingAdapter()
    subs = {"lo": np.zeros(2), "hi": np.ones(3)}
    values = _session(trace, adapter, substitutions=subs).run()
    assert "lo" not in adapter.forward_calls
    assert fake.close(np.float64(9.0), values["u"])


def test_substituted_source_reads_everywhere() -> None:
    trace = _trace()
    new_x = np.full(5, 2.0)
    values = _session(trace, substitutions={"x": new_x}).run()
    sq = new_x * new_x
    assert fake.close(np.sum(3.0 * sq[2:]) + np.sum(sq[:2] ** 2), values["u"])


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"initial": {}}, "missing_parent"),
        ({"substitutions": {"nowhere": np.zeros(1)}}, "substitution"),
        ({"initial": {"x": np.zeros(5), "m": np.zeros(3)}}, "duplicate_slot"),
    ],
)
def test_bad_plans_are_refused_before_any_forward(kwargs: dict[str, Any], reason: str) -> None:
    trace = _trace()
    adapter = CountingAdapter()
    with pytest.raises(ra.ReplaySessionError) as info:
        _session(trace, adapter, **kwargs)
    assert info.value.reason == reason
    assert adapter.forward_calls == []


def test_unreplayable_op_late_in_the_plan_refuses_before_any_forward() -> None:
    trace = _trace()
    opaque = fake.FakeOp("z", "opaque", ("u",), ("z",))
    trace.ops["z"] = opaque
    steps = _steps(trace)[:-1] + [ra.ReplayStep(opaque, (ra.InputOccurrence("u", (0,)),), ("z",))]
    adapter = CountingAdapter()
    with pytest.raises(ra.ReplaySessionError) as info:
        ra.ReplaySession(adapter, trace, steps, initial={"x": trace.payloads["x"]})
    assert info.value.reason == "cannot_replay"
    assert info.value.op_label == "z"
    assert adapter.forward_calls == []


def test_out_of_order_plan_is_refused() -> None:
    trace = _trace()
    steps = _steps(trace)
    steps[0], steps[1] = steps[1], steps[0]
    with pytest.raises(ra.ReplaySessionError) as info:
        ra.ReplaySession(fake, trace, steps, initial={"x": trace.payloads["x"]})
    assert info.value.reason == "missing_parent"


def test_close_drops_values_and_refuses_reads() -> None:
    trace = _trace()
    with _session(trace) as session:
        session.run()
        assert session.values()
    with pytest.raises(ra.ReplaySessionError) as info:
        session.value("u")
    assert info.value.reason == "closed"
    with pytest.raises(ra.ReplaySessionError):
        session.step()


def test_reverse_pass_with_pullback_matches_the_hand_gradient() -> None:
    trace = _trace()
    steps = _steps(trace)
    session = _session(trace)  # keep=None retains every value for the pullback
    values = session.run()
    grads: dict[str, Any] = {"u": np.float64(1.0)}
    for step in reversed(steps):
        cts = [grads.get(slot) for slot in step.output_slots]
        if all(c is None for c in cts):
            continue
        parents = [values[occ.parent_label] for occ in step.inputs]
        ra.pullback_step(fake, trace, step, parents, cts, grads)
    x = trace.payloads["x"]
    # u = 3*sum(x[2:]**2) + sum(x[:2]**4)
    expected = np.concatenate([4.0 * x[:2] ** 3, 6.0 * x[2:]])
    assert np.allclose(grads["x"], expected, rtol=0, atol=1e-12)
    assert fake.close(grads["lo"], 2.0 * values["lo"])


def test_pullback_refuses_a_wrong_cotangent_count() -> None:
    trace = _trace()
    step = _steps(trace)[0]  # sq = x * x, two occurrences

    class Bad(CountingAdapter):
        def vjp(self, *args: Any) -> tuple[Any, ...]:
            return (np.ones(5),)

    with pytest.raises(ra.ReplaySessionError) as info:
        ra.pullback_step(Bad(), trace, step, [np.ones(5)] * 2, [np.ones(5)], {})
    assert info.value.reason == "arity"
