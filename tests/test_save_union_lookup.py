"""A ``save=`` union of plain ``tl.module`` selectors is answered by one set lookup.

``tl.module(a) | tl.module(b) | ...`` reads only an event's
``output_of_module_calls``, so capture compiles it once into a set of module
addresses instead of walking the selector tree for every recorded op. These tests
pin that the lookup is a pure cost change:

1. the per-event save decisions (in order), the saved payloads, ``saved`` flags,
   ``by_address`` index and selector fire counts equal the selector walker's, for
   ``tl.record`` and ``tl.trace``, over unions of 1, 2, 6 and 24 addresses, a nested
   module, a module called twice, and tuple- and dict-output modules;
2. any union with another selector kind, ``&``, ``~``, a label or a callable keeps
   the walker;
3. with the lookup, selector walking no longer scales with the number of ops;
4. a pass-qualified address (``"shared:2"``) still selects only that call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.capture.predicates as predicates_mod
import torchlens.ir.selector_eval as selector_eval
import torchlens.user_funcs as user_funcs_mod

_WIDTH = 6
_DEPTH = 24


class _Layer(nn.Module):
    """One stack layer with a nested child module."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_WIDTH, _WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _Wrap(nn.Module):
    """Returns its child's output unchanged, so one op is the output of both."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(_WIDTH, _WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class _TupleOut(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 2, x + 1


class _DictOut(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"a": torch.tanh(x), "b": x.sum(-1)}


class _Stack(nn.Module):
    """24 nested layers, a module called twice, a pass-through wrapper, tuple and dict outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(_DEPTH))
        self.shared = nn.Linear(_WIDTH, _WIDTH)
        self.wrap = _Wrap()
        self.tup = _TupleOut()
        self.dct = _DictOut()
        self.head = nn.Linear(_WIDTH, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.shared(x)
        x = self.wrap(self.shared(torch.sigmoid(x)))
        a, b = self.tup(x)
        d = self.dct(a * b)
        return self.head(d["a"] + d["b"].unsqueeze(-1))


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Stack().eval(), torch.randn(3, _WIDTH)


def _union(*addresses: str) -> Any:
    selector = tl.module(addresses[0])
    for address in addresses[1:]:
        selector = selector | tl.module(address)
    return selector


_ALL_LAYERS = tuple(f"layers.{index}" for index in range(_DEPTH))

UNIONS: dict[str, Callable[[], Any]] = {
    "one": lambda: _union("layers.3"),
    "two_with_nested": lambda: _union("layers.0", "layers.5.lin"),
    "six": lambda: _union(*(f"layers.{index}" for index in (1, 4, 9, 13, 17, 22))),
    "twenty_four": lambda: _union(*_ALL_LAYERS),
    "called_twice": lambda: _union("shared"),
    "second_pass_only": lambda: _union("shared:2"),
    "tuple_and_dict": lambda: _union("tup", "dct"),
    "pass_through_wrapper": lambda: _union("wrap", "wrap.inner"),
    "everything_kind": lambda: _union("layers.2", "shared", "tup", "dct", "head"),
}

CONTROLS: dict[str, Callable[[], Any]] = {
    "module_or_func": lambda: tl.module("layers.3") | tl.func("linear"),
    "module_and_module": lambda: tl.module("wrap") & tl.module("wrap.inner"),
    "not_module": lambda: ~tl.module("layers.3"),
    "module_or_label": lambda: tl.module("layers.3") | tl.label("relu_2"),
    "callable": lambda: lambda ctx: ctx.kind == "op" and ctx.func_name == "relu",
}


@contextmanager
def _walker_forced(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Disable the set lookup so every save decision walks the selector tree."""

    with monkeypatch.context() as patch:
        patch.setattr(predicates_mod, "_plain_module_union", lambda _predicate: None)
        yield


@contextmanager
def _decisions(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[Any, ...]]]:
    """Record every save decision in the order capture made it."""

    seen: list[tuple[Any, ...]] = []
    real = predicates_mod._normalize_capture_decision

    def recording(result: Any, ctx: Any, default: Any) -> Any:
        spec = real(result, ctx, default)
        seen.append(
            (
                ctx.kind,
                ctx.raw_label,
                ctx.pass_index,
                tuple(ctx.output_of_module_calls),
                getattr(spec, "save_out", None),
                getattr(spec, "save_metadata", None),
            )
        )
        return spec

    with monkeypatch.context() as patch:
        patch.setattr(predicates_mod, "_normalize_capture_decision", recording)
        yield seen


@contextmanager
def _fire_counts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Read ``tl.trace``'s save-selector fire count before capture discards it."""

    counts: list[Any] = []
    real = user_funcs_mod._warn_zero_match_capture_selectors

    def spying(trace: Any, **kwargs: Any) -> Any:
        counts.append(trace.__dict__.get("_tl_save_selector_fire_count"))
        return real(trace, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(user_funcs_mod, "_warn_zero_match_capture_selectors", spying)
        yield counts


@contextmanager
def _call_counts(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Count selector-tree walks and selector evaluations during one capture."""

    counts = {"walk_selector": 0, "_evaluate_subject": 0}
    real_walk = selector_eval.walk_selector
    real_eval = selector_eval._evaluate_subject

    def counting_walk(*args: Any, **kwargs: Any) -> Any:
        counts["walk_selector"] += 1
        return real_walk(*args, **kwargs)

    def counting_eval(*args: Any, **kwargs: Any) -> Any:
        counts["_evaluate_subject"] += 1
        return real_eval(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(selector_eval, "walk_selector", counting_walk)
        patch.setattr(selector_eval, "_evaluate_subject", counting_eval)
        yield counts


def _payload_bytes(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, torch.Tensor):
        tensor = payload.detach().cpu().contiguous()
        return (str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes())
    return repr(payload)


def _recording_fingerprint(recording: Any) -> tuple[Any, ...]:
    rows = [
        (
            record.ctx.kind,
            record.ctx.raw_label,
            record.ctx.pass_index,
            record.ctx.address,
            tuple(record.ctx.output_of_module_calls),
            record.spec.save_out,
            record.spec.save_metadata,
            _payload_bytes(record.ram_payload),
            _payload_bytes(record.transformed_ram_payload),
        )
        for record in recording.records
    ]
    by_address = {key: list(value) for key, value in recording.by_address.items()}
    by_label = {key: list(value) for key, value in recording.by_label.items()}
    return rows, by_address, by_label, len(recording.records), recording.n_ops


def _trace_fingerprint(trace: Any) -> tuple[Any, ...]:
    rows = [
        (
            op.layer_label,
            op.has_saved_activation,
            tuple(op.shape or ()),
            str(op.dtype),
            tuple(op.output_of_module_calls or ()),
            tuple(op.parents),
            tuple(op.children),
            _payload_bytes(op.out) if op.has_saved_activation else None,
        )
        for op in trace.layer_list
    ]
    return rows, trace.num_saved_module_calls


def _record_both_ways(
    make_selector: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    model, x = _model_and_input()
    with _decisions(monkeypatch) as fast_decisions:
        fast = _recording_fingerprint(tl.record(model, x, save=make_selector()))
    with _walker_forced(monkeypatch), _decisions(monkeypatch) as walker_decisions:
        walker = _recording_fingerprint(tl.record(model, x, save=make_selector()))
    return (fast, fast_decisions), (walker, walker_decisions)


def _trace_both_ways(
    make_selector: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    model, x = _model_and_input()
    with _decisions(monkeypatch) as fast_decisions, _fire_counts(monkeypatch) as fast_fires:
        fast_trace = tl.trace(model, x, save=make_selector())
        fast = _trace_fingerprint(fast_trace)
        fast_valid = fast_trace.check_metadata_invariants()
    with (
        _walker_forced(monkeypatch),
        _decisions(monkeypatch) as walker_decisions,
        _fire_counts(monkeypatch) as walker_fires,
    ):
        walker_trace = tl.trace(model, x, save=make_selector())
        walker = _trace_fingerprint(walker_trace)
        walker_valid = walker_trace.check_metadata_invariants()
    return (
        (fast, fast_decisions, fast_fires, fast_valid),
        (walker, walker_decisions, walker_fires, walker_valid),
    )


# ---------------------------------------------------------------------------
# 1. Equivalence with the walker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(UNIONS))
def test_union_detector_returns_addresses(name: str) -> None:
    """Every pure module union is recognised, with its addresses in tree order."""

    addresses = selector_eval.module_union_addresses(UNIONS[name]())
    assert addresses is not None
    assert len(addresses) >= 1


@pytest.mark.parametrize("name", sorted(UNIONS))
def test_record_lookup_equals_walker(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``tl.record``: same decisions in the same order, same records, payloads and index."""

    (fast, fast_decisions), (walker, walker_decisions) = _record_both_ways(
        UNIONS[name], monkeypatch
    )
    assert fast_decisions == walker_decisions
    assert any(decision[4] for decision in fast_decisions), f"{name}: nothing was saved"
    assert fast == walker


@pytest.mark.parametrize("name", sorted(UNIONS))
def test_trace_lookup_equals_walker(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``tl.trace``: same decisions, saved flags, payloads, fire count and invariants."""

    fast, walker = _trace_both_ways(UNIONS[name], monkeypatch)
    assert fast[1] == walker[1]
    assert fast[0] == walker[0]
    assert fast[2] == walker[2]
    assert fast[2] and fast[2][0], f"{name}: the save selector never fired"
    assert fast[3] == walker[3] is True


@pytest.mark.smoke
def test_twenty_four_site_union_equals_walker_on_both_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The many-site caching shape: record and trace both match the walker exactly."""

    (fast, fast_decisions), (walker, walker_decisions) = _record_both_ways(
        UNIONS["twenty_four"], monkeypatch
    )
    assert fast_decisions == walker_decisions
    assert fast == walker
    assert len(fast[0]) == _DEPTH  # each layer's output: its relu
    trace_fast, trace_walker = _trace_both_ways(UNIONS["twenty_four"], monkeypatch)
    assert trace_fast[:3] == trace_walker[:3]


# ---------------------------------------------------------------------------
# 2. Controls keep the walker
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_mixed_selectors_keep_the_walker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another kind, ``&``, ``~``, a label or a callable is not a plain module union."""

    for name, make_selector in sorted(CONTROLS.items()):
        selector = make_selector()
        assert selector_eval.module_union_addresses(selector) is None, name
        assert predicates_mod._plain_module_union(selector) is None, name
        (fast, fast_decisions), (walker, walker_decisions) = _record_both_ways(
            make_selector, monkeypatch
        )
        assert fast_decisions == walker_decisions, name
        assert fast == walker, name


# ---------------------------------------------------------------------------
# 3. Selector work no longer scales with the op count
# ---------------------------------------------------------------------------

#: Upper bounds on selector-tree walks and selector evaluations for one whole
#: ``tl.record`` of the 24-site union with the lookup on. They are per capture,
#: not per op: the capture makes about 70 save decisions.
_MAX_WALKS_PER_CAPTURE = 64
_MAX_EVALUATIONS_PER_CAPTURE = 0


def test_twenty_four_site_union_walks_once_per_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the lookup, walks and evaluations are a per-capture constant, not per op."""

    model, x = _model_and_input()
    selector = UNIONS["twenty_four"]()
    with _decisions(monkeypatch) as decisions, _call_counts(monkeypatch) as counts:
        tl.record(model, x, save=selector)
    assert len(decisions) > 2 * _DEPTH
    assert counts["_evaluate_subject"] <= _MAX_EVALUATIONS_PER_CAPTURE, counts
    assert counts["walk_selector"] <= _MAX_WALKS_PER_CAPTURE, counts

    with _walker_forced(monkeypatch), _call_counts(monkeypatch) as walker_counts:
        tl.record(model, x, save=UNIONS["twenty_four"]())
    # The walker evaluates every union member for every non-matching event.
    assert walker_counts["_evaluate_subject"] > _DEPTH * len(decisions) // 2, walker_counts


def test_mixed_union_counts_equal_the_walker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed union does the walker's work; the only extra walk is the one-time check."""

    model, x = _model_and_input()
    with _call_counts(monkeypatch) as counts:
        tl.record(model, x, save=CONTROLS["module_or_func"]())
    with _walker_forced(monkeypatch), _call_counts(monkeypatch) as walker_counts:
        tl.record(model, x, save=CONTROLS["module_or_func"]())
    assert counts["_evaluate_subject"] == walker_counts["_evaluate_subject"]
    assert 0 <= counts["walk_selector"] - walker_counts["walk_selector"] <= 1


# ---------------------------------------------------------------------------
# 4. Pass-qualified addresses
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_pass_qualified_address_saves_only_that_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tl.module("shared:2")`` saves the second call only, with and without the lookup."""

    model, x = _model_and_input()
    for forced in (False, True):
        with monkeypatch.context() as patch:
            if forced:
                patch.setattr(predicates_mod, "_plain_module_union", lambda _predicate: None)
            recording = tl.record(model, x, save=tl.module("shared:2"))
            trace = tl.trace(model, x, save=tl.module("shared:2"))
        record_calls = {
            call for record in recording.records for call in record.ctx.output_of_module_calls
        }
        assert "shared:2" in record_calls
        assert "shared:1" not in record_calls
        assert len(recording.records) == 1
        saved = [op for op in trace.layer_list if op.has_saved_activation]
        saved_calls = {call for op in saved for call in (op.output_of_module_calls or ())}
        assert "shared:2" in saved_calls
        assert "shared:1" not in saved_calls
