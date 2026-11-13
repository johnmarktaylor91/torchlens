"""Neutral input-shape search: controller, bounds and diagnostics with a fake prior.

The fake prior uses NumPy arrays as its "framework", so these tests need no
optional backend. They pin the controller contract the Paddle, tinygrad and
later priors rely on: determined versus working versus ambiguous messages, the
linear and spatial corrections, dtype correction, rank and size bounds, the
probe budget, lazy refusal, state restoration on every exit and the
``on_failure`` contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from torch import nn

from torchlens._errors import ShapeInferenceError
from torchlens.debug import guess_input_shape
from torchlens.debug._shape_priors import (
    ProbeError,
    SearchRequest,
    ShapeCandidate,
    framework_of,
    parse_feature_mismatch,
    resolve_priors,
    walk_references,
)
from torchlens.debug._shape_search import (
    MAX_CANDIDATE_ELEMENTS,
    guess_with_priors,
    within_bounds,
)

_UNIFORM = ("uniform", 0.0, 1.0)


def _fake_priors(
    model_candidates: Sequence[ShapeCandidate],
    *,
    lazy: str | None = None,
    log: list[str] | None = None,
    candidates_error: Exception | None = None,
) -> SimpleNamespace:
    """A prior module over NumPy arrays whose errors carry torch-style matmul text."""

    events = log if log is not None else []

    @contextmanager
    def state_scope(model: Any) -> Iterator[None]:
        del model
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def candidates(model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
        del model, request
        if candidates_error is not None:
            raise candidates_error
        return model_candidates

    def make_input(shape: tuple[int, ...], dtype: Any, value_range: Any, seed: int) -> Any:
        del value_range
        return np.random.default_rng(seed).uniform(size=shape).astype(dtype)

    def forward(model: Any, example: Any) -> Any:
        events.append(f"forward{example.shape}")
        return model(example)

    def classify_error(exc: Exception) -> ProbeError:
        got, target = parse_feature_mismatch(str(exc))
        if isinstance(exc, TypeError):
            return ProbeError("dtype", str(exc), suggested_dtype="int64")
        return ProbeError("shape", str(exc), got_features=got, target_features=target)

    return SimpleNamespace(
        handles=lambda model: False,
        lazy_reason=lambda model: lazy,
        candidates=candidates,
        make_input=make_input,
        forward=forward,
        classify_error=classify_error,
        executed_constraint=lambda trace: None,
        state_scope=state_scope,
    )


def _trace_ok(model: Any, example: Any) -> Any:
    """A stand-in verification trace that always succeeds."""

    del model, example
    return object()


def _needs_features(width: int) -> Callable[[Any], Any]:
    """A model whose first matmul needs ``width`` trailing features."""

    def model(x: Any) -> Any:
        if x.shape[-1] != width:
            raise ValueError(
                f"mat1 and mat2 shapes cannot be multiplied (1x{x.shape[-1]} and {width}x3)"
            )
        return x[..., :3]

    return model


def _candidate(shape: tuple[int, ...], **fields: Any) -> ShapeCandidate:
    """A float32 candidate with defaults for the rest."""

    fields.setdefault("layout", "NC")
    kind = fields.pop("kind", "linear")
    return ShapeCandidate(kind, shape, "float32", _UNIFORM, **fields)


def test_determined_linear_prior_reports_unique_shape() -> None:
    """A weight-pinned candidate that runs is reported as uniquely determined."""

    priors = _fake_priors([_candidate((1, 5), determined=True, constraining_module="fc")])
    report = guess_with_priors(priors, _needs_features(5), _trace_ok)
    assert report.found and report.shape == (1, 5)
    assert "uniquely determined by fc" in report.message
    assert report.constraining_module == "fc"


def test_linear_feature_correction_reads_the_mismatch() -> None:
    """A wrong feature count is corrected from the mismatch text in one probe."""

    priors = _fake_priors([_candidate((1, 7), determined=True)])
    report = guess_with_priors(priors, _needs_features(5), _trace_ok)
    assert report.found and report.shape == (1, 5)
    assert report.strategy == "executed_op_linear"
    assert len(report.attempts) == 2


def test_any_shape_function_reports_ambiguity() -> None:
    """A function every candidate runs on is reported as ambiguous."""

    ladder = [_candidate(shape, kind="unknown") for shape in ((1, 8), (8,), (1, 2, 4))]
    report = guess_with_priors(_fake_priors(ladder), lambda x: x * 2.0, _trace_ok)
    assert report.found and report.shape == (1, 8)
    assert "ambiguous: (8,) also runs" in report.message


def test_lone_working_candidate_is_not_called_determined() -> None:
    """An undetermined candidate that alone runs is a working candidate, not unique."""

    def needs_eight(x: Any) -> Any:
        if x.size != 8 or x.ndim != 2:
            raise ValueError("cannot reshape")
        return x.reshape(4, 2)

    ladder = [_candidate(shape, kind="unknown") for shape in ((1, 8), (8,), (1, 4))]
    report = guess_with_priors(_fake_priors(ladder), needs_eight, _trace_ok)
    assert report.found and report.shape == (1, 8)
    assert "a working candidate, not uniquely determined" in report.message


def test_spatial_side_search_uses_measured_features() -> None:
    """A conv candidate finds the side whose flattened features fit the head."""

    def cnn(x: Any) -> Any:
        features = x.shape[2] * x.shape[3]
        if features != 36:
            raise ValueError(f"mat1 and mat2 shapes cannot be multiplied (1x{features} and 36x3)")
        return x

    candidate = _candidate(
        (1, 1, 32, 32), kind="conv", layout="NCHW", spatial_axes=(2, 3), determined=True
    )
    report = guess_with_priors(_fake_priors([candidate]), cnn, _trace_ok, max_probes=64)
    assert report.found and report.shape == (1, 1, 6, 6)
    assert report.strategy in {"binary_search", "probe_success"}
    assert "working candidate" in report.message


def test_flexible_axes_are_confirmed_by_a_grown_probe() -> None:
    """Claimed flexible axes stay only when a grown shape also runs."""

    candidate = _candidate((1, 4, 8), layout="NLC", flexible_dims=(1,), determined=True)
    report = guess_with_priors(_fake_priors([candidate]), _needs_features(8), _trace_ok)
    assert report.flexible_dims == (1,)
    assert "up to flexible dims (1,)" in report.message


def test_dtype_correction_switches_to_integer_values() -> None:
    """A dtype error with a suggested integer dtype retries with integer values."""

    def wants_int(x: Any) -> Any:
        if x.dtype != np.int64:
            raise TypeError("expected int64 indices")
        return x

    report = guess_with_priors(
        _fake_priors([_candidate((1, 4), determined=True)]), wants_int, _trace_ok
    )
    assert report.found and report.strategy == "dtype_corrected"
    assert report.value_range is not None and report.value_range[0] == "randint"


def test_rank_and_size_bounds_skip_without_probing() -> None:
    """Out-of-bounds candidates are recorded as skipped and never allocated."""

    log: list[str] = []
    huge = (1, MAX_CANDIDATE_ELEMENTS + 1)
    deep = (1,) * 7
    priors = _fake_priors([_candidate(huge), _candidate(deep), _candidate((1, 5))], log=log)
    report = guess_with_priors(priors, _needs_features(5), _trace_ok)
    assert report.found and report.shape == (1, 5)
    skipped = [outcome for _shape, outcome in report.attempts if outcome.startswith("skipped")]
    assert len(skipped) == 2
    assert not any(str(huge) in event for event in log)
    assert not within_bounds(deep) and not within_bounds(huge) and within_bounds((2, 3))


def test_probe_budget_is_a_typed_failure() -> None:
    """An exhausted budget returns budget_exhausted, never more probes."""

    def never(x: Any) -> Any:
        raise ValueError(f"bad shape {x.shape}")

    ladder = [_candidate((1, width)) for width in (1, 2, 3, 4)]
    report = guess_with_priors(_fake_priors(ladder), never, _trace_ok, max_probes=2)
    assert not report.found and report.reason == "budget_exhausted"
    assert len(report.attempts) == 2


def test_lazy_model_is_refused_without_a_forward() -> None:
    """A prior's lazy reason refuses before any probe; the model is left untouched."""

    log: list[str] = []
    priors = _fake_priors([_candidate((1, 5))], lazy="Unbuilt layers.", log=log)
    report = guess_with_priors(priors, _needs_features(5), _trace_ok)
    assert report.reason == "lazy_uninitialized" and "left untouched" in report.message
    assert log == []


def test_state_scope_exits_on_verification_failure_and_internal_error() -> None:
    """State is restored when verification fails and when the prior itself errors."""

    log: list[str] = []

    def bad_trace(model: Any, example: Any) -> Any:
        raise RuntimeError("trace broke")

    report = guess_with_priors(
        _fake_priors([_candidate((1, 5))], log=log), _needs_features(5), bad_trace
    )
    assert report.reason == "verification_failed" and log[0] == "enter" and log[-1] == "exit"
    log.clear()
    broken = _fake_priors([], log=log, candidates_error=RuntimeError("boom"))
    report = guess_with_priors(broken, _needs_features(5), _trace_ok)
    assert report.reason == "non_shape_blocker" and log == ["enter", "exit"]
    with pytest.raises(ShapeInferenceError, match="boom"):
        guess_with_priors(broken, _needs_features(5), _trace_ok, on_failure="raise")


def test_arguments_are_validated_and_device_is_torch_only() -> None:
    """Invalid arguments and ``device=`` raise regardless of ``on_failure``."""

    priors = _fake_priors([_candidate((1, 5))])
    with pytest.raises(ShapeInferenceError, match="batch_size"):
        guess_with_priors(priors, _needs_features(5), _trace_ok, batch_size=0)
    with pytest.raises(ShapeInferenceError, match="device"):
        guess_with_priors(priors, _needs_features(5), _trace_ok, device="cpu")


def test_no_candidates_is_unknown_entry() -> None:
    """A prior with no candidates reports unknown_entry."""

    report = guess_with_priors(_fake_priors([]), _needs_features(5), _trace_ok)
    assert report.reason == "unknown_entry"


def test_mismatch_parser_reads_each_framework() -> None:
    """Feature counts parse from torch, Paddle and tinygrad mismatch texts."""

    assert parse_feature_mismatch("shapes cannot be multiplied (1x7 and 4x3)") == (7, 4)
    assert parse_feature_mismatch("but received K_lhs:32 != K_rhs:18.") == (32, 18)
    assert parse_feature_mismatch("dim_a.width_:7 != dim_b.height_:4") == (7, 4)
    assert parse_feature_mismatch("cannot dot (1, 32) and (18, 3)") == (32, 18)
    assert parse_feature_mismatch("no numbers here") == (None, None)


def test_walk_references_is_cycle_safe_and_reads_closures() -> None:
    """The walker follows closures and attributes, stops on cycles and tensors."""

    holder = SimpleNamespace(weight=np.zeros((4, 3)))
    holder.me = holder
    captured = [holder]

    def fn(x: Any) -> Any:
        return captured[0].weight @ x

    paths = [path for path, _obj in walk_references(fn)]
    assert "captured.0.weight" in paths
    assert sum(1 for _path, obj in walk_references(fn) if obj is holder) == 1
    assert framework_of(np.zeros(1)) == "numpy" and framework_of(np) == "numpy"


def test_torch_modules_and_unknown_objects_keep_their_paths() -> None:
    """Torch modules still use the torch search; an unknown object has no prior."""

    report = guess_input_shape(nn.Sequential(nn.Linear(8, 4), nn.ReLU()))
    assert report.found and report.shape is not None and report.shape[-1] == 8
    assert resolve_priors(object()) is None
    with pytest.raises(ShapeInferenceError, match="torch.nn.Module"):
        guess_input_shape(object())
