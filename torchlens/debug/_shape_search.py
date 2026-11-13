"""Neutral input-shape search: the controller, its bounds and its diagnostics.

:func:`torchlens.debug.guess_input_shape` sends torch modules to the torch
search in :mod:`torchlens.debug._infer_input_shape` and every other model to
:func:`guess_with_priors` here, which drives one framework prior module (see
:mod:`torchlens.debug._shape_priors`). The pieces both searches share live here
too: argument validation, failure results, the ``on_failure`` contract, the
spatial side ladder, the rank-inflation test and the terminal failure order.

The neutral search is bounded (candidate rank, element count, candidate count
and probes), never initializes lazy layers in the caller's object, restores
the model's mode flags and the framework RNG through the prior's
``state_scope``, and tells a working candidate apart from a uniquely
determined shape in the report message.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .._errors import ShapeInferenceError
from ..backends._backend_names import TORCH_BACKEND_NAME
from ._infer_failure_classify import classify_nongeometric_failure
from ._infer_report import GuessInputShapeReport
from ._shape_priors import (
    ExecutedConstraint,
    ProbeError,
    SearchRequest,
    ShapeCandidate,
    ShapePriors,
    ValueRange,
    framework_of,
    imported_priors,
    resolve_priors,
    walk_references,
)

FailureReason = Literal[
    "non_shape_blocker",
    "exact_size_unreachable",
    "multi_input_required",
    "multi_input_unsupported",
    "budget_exhausted",
    "unknown_entry",
    "lazy_uninitialized",
    "verification_failed",
    "device_mismatch",
    "rank_undetermined",
]

Attempt = tuple[tuple[int, ...] | None, str]

#: Largest candidate rank the neutral search probes.
MAX_CANDIDATE_RANK = 6
#: Largest candidate element count the neutral search allocates.
MAX_CANDIDATE_ELEMENTS = 1 << 24
#: Most prior candidates the neutral search considers.
MAX_CANDIDATES = 16
#: Most extra probes spent looking for a second working shape.
MAX_AMBIGUITY_PROBES = 3
#: Integer value range used when a dtype correction moves to an integer dtype.
_INTEGER_RANGE: ValueRange = ("randint", 0.0, 2.0)


def failure_result(
    reason: FailureReason, attempts: Sequence[Attempt], message: str
) -> GuessInputShapeReport:
    """Build a standardized failed inference result."""

    return GuessInputShapeReport(
        found=False,
        shape=None,
        shapes=None,
        dtype=None,
        value_range=None,
        flexible_dims=(),
        constraining_module=None,
        constraining_op=None,
        source_line=None,
        example_input=None,
        strategy="none",
        reason=reason,
        attempts=tuple(attempts),
        trace=None,
        message=message,
    )


def maybe_raise(result: GuessInputShapeReport, on_failure: Literal["return", "raise"]) -> None:
    """Raise :class:`ShapeInferenceError` for a failed result when requested.

    Parameters
    ----------
    result:
        Inference result.
    on_failure:
        Failure behavior.
    """

    if not result.found and on_failure == "raise":
        raise ShapeInferenceError(result.message)


def validate_search_arguments(
    *,
    batch_size: int,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    max_size: int,
    max_probes: int,
) -> None:
    """Validate caller-supplied search arguments with typed errors.

    Invalid arguments are caller bugs, so they raise :class:`ShapeInferenceError`
    regardless of ``on_failure`` instead of producing degenerate probes such as
    a ``(0, N)`` "success" or a raw negative-dimension error.

    Parameters
    ----------
    batch_size, channels, spatial_rank, seq_len, min_size, max_size, max_probes:
        See :func:`torchlens.debug.guess_input_shape`.
    """

    if batch_size < 1:
        raise ShapeInferenceError(f"batch_size must be >= 1, got {batch_size}.")
    if channels is not None and channels < 1:
        raise ShapeInferenceError(f"channels must be >= 1 when provided, got {channels}.")
    if spatial_rank != "auto" and int(spatial_rank) < 1:
        raise ShapeInferenceError(f"spatial_rank must be 'auto' or >= 1, got {spatial_rank!r}.")
    if seq_len is not None and seq_len < 1:
        raise ShapeInferenceError(f"seq_len must be >= 1 when provided, got {seq_len}.")
    if min_size < 1:
        raise ShapeInferenceError(f"min_size must be >= 1, got {min_size}.")
    if max_size < min_size:
        raise ShapeInferenceError(f"max_size ({max_size}) must be >= min_size ({min_size}).")
    if max_probes < 1:
        raise ShapeInferenceError(f"max_probes must be >= 1, got {max_probes}.")


def entry_precheck(*, input_specs: Any, square: bool) -> FailureReason | None:
    """Return the failure reason for unsupported entry arguments (``input_specs``, non-square), or ``None``."""

    if input_specs is not None:
        return "multi_input_unsupported"
    if not square:
        return "exact_size_unreachable"
    return None


PRECHECK_MESSAGES: dict[str, str] = {
    "multi_input_unsupported": (
        "input_specs multi-input inference is not implemented yet; pass a single-input module."
    ),
    "exact_size_unreachable": (
        "Non-square spatial inference requires an explicit aspect-ratio hint, which is not supported."
    ),
}


def candidate_sides(
    start: int, min_size: int, max_size: int, preferred_sizes: Sequence[int]
) -> list[int]:
    """Return unique spatial side candidates in probe order, inside ``[min_size, max_size]``."""

    values = [start, *preferred_sizes, 32, 28, 16, 8, min_size, max_size]
    return [
        side
        for index, side in enumerate(values)
        if min_size <= side <= max_size and side not in values[:index]
    ]


def is_rank_inflated(
    kind: str, constraint_shape: tuple[int, ...] | None, example_shape: tuple[int, ...]
) -> bool:
    """Return whether a success only ran by broadcasting the input to a higher rank.

    The tell is that the first executed input-consuming constraint saw a
    higher-rank tensor with more non-singleton dimensions than the probe input;
    rank-preserving reshapes such as ``unsqueeze`` only add singleton
    dimensions and stay legal.
    """

    if constraint_shape is None or kind not in {"conv", "linear", "matmul"}:
        return False
    if len(constraint_shape) <= len(example_shape):
        return False
    example_wide = sum(1 for dim in example_shape if int(dim) > 1)
    constraint_wide = sum(1 for dim in constraint_shape if int(dim) > 1)
    return constraint_wide > example_wide


@dataclass
class SearchLedger:
    """Probe budget and blocker diary shared by a search run."""

    attempts: list[Attempt]
    max_probes: int
    probes: int = 0
    device_blockers: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    rank_rejections: list[str] = field(default_factory=list)

    def budget_left(self) -> bool:
        """Whether another probe fits in the budget."""

        return self.probes < self.max_probes


def terminal_failure(
    ledger: SearchLedger, is_skippable: Callable[[str], bool]
) -> GuessInputShapeReport:
    """Build the failure for a search that ran out of candidates.

    The order is fixed: budget, device, rank, other blockers, the evidence-led
    non-geometric classifier, then the square-size fallback.
    """

    attempts = ledger.attempts
    if ledger.probes >= ledger.max_probes:
        return failure_result(
            "budget_exhausted",
            attempts,
            f"No valid shape was found within {ledger.max_probes} probes.",
        )
    if ledger.device_blockers:
        return failure_result(
            "device_mismatch",
            attempts,
            "Shape inference was blocked by a device mismatch, not an input-shape problem: "
            f"{ledger.device_blockers[-1]} Move the model to a single device or pass a matching "
            "device= argument.",
        )
    if ledger.rank_rejections:
        return failure_result(
            "rank_undetermined",
            attempts,
            "Probes ran only after in-model broadcasting of a lower-rank input (rejected: "
            f"{', '.join(ledger.rank_rejections)}); the true input rank is under-determined. Pass "
            "seq_len= or explicit channels/spatial_rank hints.",
        )
    if ledger.blockers:
        return failure_result(
            "non_shape_blocker",
            attempts,
            f"Shape inference was blocked by an unsupported forward error: {ledger.blockers[-1]}",
        )
    classified = classify_nongeometric_failure(attempts, is_skippable)
    if classified is not None:
        return failure_result(cast("FailureReason", classified[0]), attempts, classified[1])
    return failure_result(
        "exact_size_unreachable",
        attempts,
        "No square valid input was found; the model may need a rectangular (non-square) input, "
        "whose exact aspect ratio is not searched, or sides outside min_size/max_size.",
    )


def within_bounds(shape: tuple[int, ...]) -> bool:
    """Whether ``shape`` respects the neutral search's rank and size bounds."""

    if not shape or len(shape) > MAX_CANDIDATE_RANK or any(int(dim) < 1 for dim in shape):
        return False
    return math.prod(shape) <= MAX_CANDIDATE_ELEMENTS


def _is_integer_dtype(dtype: Any) -> bool:
    """Whether a native dtype object names an integer dtype (any framework)."""

    text = str(dtype).lower().rsplit(".", 1)[-1]
    return text.startswith(("int", "uint", "long"))


def _shape_error_outcome(outcome: str) -> bool:
    """Whether a recorded probe outcome is an ordinary shape mismatch."""

    return outcome.startswith("shape:")


@dataclass(frozen=True)
class _Win:
    """A probe that ran, with the facts the report needs."""

    candidate: ShapeCandidate
    example: Any
    shape: tuple[int, ...]
    dtype: Any
    value_range: ValueRange
    strategy: str
    searched: bool


class NeutralSearch:
    """One bounded search over a framework prior's candidates."""

    def __init__(
        self,
        priors: ShapePriors,
        model: Any,
        tracer: Callable[[Any, Any], Any],
        settings: SearchSettings,
    ) -> None:
        self.priors = priors
        self.model = model
        self.tracer = tracer
        self.request = SearchRequest(
            settings.batch_size,
            settings.input_dtype,
            settings.channels,
            settings.spatial_rank,
            settings.seq_len,
            settings.min_size,
            settings.max_size,
            tuple(settings.preferred_sizes),
        )
        self.ledger = SearchLedger(attempts=[], max_probes=settings.max_probes)
        self.seed = settings.seed
        self.return_trace = settings._return_trace
        self.tried: set[tuple[tuple[int, ...], str]] = set()

    def run(self) -> GuessInputShapeReport:
        """Probe candidates best first and return a verified result or a failure."""

        candidates = list(self.priors.candidates(self.model, self.request))[:MAX_CANDIDATES]
        if not candidates:
            return failure_result(
                "unknown_entry",
                self.ledger.attempts,
                "No first layer or closure weight gives a shape prior for this model.",
            )
        for index, candidate in enumerate(candidates):
            if not self.ledger.budget_left():
                break
            result = self._try_candidate(candidate, candidates[index + 1 :])
            if result is not None:
                return result
        return terminal_failure(self.ledger, _shape_error_outcome)

    def probe(
        self, shape: tuple[int, ...], dtype: Any, value_range: ValueRange
    ) -> tuple[Any, ProbeError | None] | None:
        """Run one bounded forward probe; ``None`` when the shape is out of bounds or seen."""

        key = (shape, str(dtype))
        if key in self.tried or not self.ledger.budget_left():
            return None
        self.tried.add(key)
        if not within_bounds(shape):
            self.ledger.attempts.append((shape, "skipped: outside the rank or size bound"))
            return None
        example = self.priors.make_input(shape, dtype, value_range, self.seed)
        self.ledger.probes += 1
        try:
            self.priors.forward(self.model, example)
        except Exception as exc:  # noqa: BLE001 - every forward error is classified, never raised.
            error = self.priors.classify_error(exc)
            self.ledger.attempts.append((shape, f"{error.kind}: {error.message}"))
            return example, error
        self.ledger.attempts.append((shape, "ok"))
        return example, None

    def _try_candidate(
        self, candidate: ShapeCandidate, rest: Sequence[ShapeCandidate]
    ) -> GuessInputShapeReport | None:
        """Probe one candidate and its corrections; a result when one verifies."""

        outcome = self.probe(candidate.shape, candidate.dtype, candidate.value_range)
        if outcome is None:
            return None
        example, error = outcome
        if error is None:
            win = _Win(
                candidate,
                example,
                candidate.shape,
                candidate.dtype,
                candidate.value_range,
                candidate.strategy,
                False,
            )
            return self.finalize(win, rest)
        win_or_none = self._correct(candidate, error)
        if win_or_none is not None:
            return self.finalize(win_or_none, rest)
        self._record_blocker(error)
        return None

    def _record_blocker(self, error: ProbeError) -> None:
        """File a non-shape error in the ledger."""

        if error.kind == "device":
            self.ledger.device_blockers.append(error.message)
        elif error.kind == "blocker":
            self.ledger.blockers.append(error.message)

    def _correct(self, candidate: ShapeCandidate, error: ProbeError) -> _Win | None:
        """Try the dtype, feature-count and spatial corrections for a failed probe."""

        if error.kind == "dtype" and error.suggested_dtype is not None:
            value_range = (
                _INTEGER_RANGE
                if _is_integer_dtype(error.suggested_dtype)
                else candidate.value_range
            )
            outcome = self.probe(candidate.shape, error.suggested_dtype, value_range)
            if outcome is not None and outcome[1] is None:
                return _Win(
                    candidate,
                    outcome[0],
                    candidate.shape,
                    error.suggested_dtype,
                    value_range,
                    "dtype_corrected",
                    False,
                )
            return None
        if error.kind != "shape":
            return None
        if candidate.kind == "linear" and error.target_features is not None:
            shape = (*candidate.shape[:-1], int(error.target_features))
            outcome = self.probe(shape, candidate.dtype, candidate.value_range)
            if outcome is not None and outcome[1] is None:
                return _Win(
                    candidate,
                    outcome[0],
                    shape,
                    candidate.dtype,
                    candidate.value_range,
                    "executed_op_linear",
                    False,
                )
        if candidate.spatial_axes:
            return SideSearch(self, candidate, error).run()
        return None

    def finalize(self, win: _Win, rest: Sequence[ShapeCandidate]) -> GuessInputShapeReport | None:
        """Verify a working probe with a trace and build the report."""

        try:
            trace_obj = self.tracer(self.model, win.example)
        except Exception as exc:  # noqa: BLE001 - honor the on_failure contract for any trace error.
            return failure_result(
                "verification_failed",
                self.ledger.attempts,
                f"Forward probes succeeded with shape {win.shape}, but the TorchLens "
                f"verification trace failed: {exc}",
            )
        constraint = self.priors.executed_constraint(trace_obj)
        if constraint is not None and is_rank_inflated(
            constraint.kind, constraint.input_shape, win.shape
        ):
            self.ledger.attempts.append((win.shape, "ok_but_rank_inflated"))
            self.ledger.rank_rejections.append(str(win.shape))
            return None
        flexible = self._confirmed_flexible(win)
        return build_success(
            win,
            constraint,
            flexible,
            self._determination(win, flexible, rest),
            (trace_obj if self.return_trace else None, self.ledger.attempts),
        )

    def _confirmed_flexible(self, win: _Win) -> tuple[int, ...]:
        """Keep the candidate's flexible axes only if a grown probe also runs."""

        flexible = tuple(axis for axis in win.candidate.flexible_dims if 0 < axis < len(win.shape))
        if not flexible:
            return ()
        grown = tuple(dim + 4 if index in flexible else dim for index, dim in enumerate(win.shape))
        outcome = self.probe(grown, win.dtype, win.value_range)
        return flexible if outcome is not None and outcome[1] is None else ()

    def _determination(
        self, win: _Win, flexible: tuple[int, ...], rest: Sequence[ShapeCandidate]
    ) -> str:
        """Message clause saying whether the shape is determined or one working candidate."""

        # A guessed axis the grown probe could not confirm as flexible is not pinned.
        pinned_all = set(flexible) >= set(win.candidate.flexible_dims)
        if win.candidate.determined and not win.searched and pinned_all:
            pinned = (
                win.candidate.constraining_module
                or win.candidate.constraining_op
                or "a weight shape"
            )
            flex = f" up to flexible dims {flexible}" if flexible else ""
            return f"uniquely determined by {pinned}{flex}"
        alternative = self._second_working_shape(win, rest)
        if alternative is not None:
            return (
                f"ambiguous: {alternative} also runs, so the model does not determine its input "
                "shape; pass the intended shape explicitly"
            )
        return "a working candidate, not uniquely determined (no weight pins every axis)"

    def _second_working_shape(
        self, win: _Win, rest: Sequence[ShapeCandidate]
    ) -> tuple[int, ...] | None:
        """Look for another shape that also runs among the remaining candidates."""

        spent = 0
        for other in rest:
            if spent >= MAX_AMBIGUITY_PROBES:
                break
            if other.shape == win.shape:
                continue
            outcome = self.probe(other.shape, other.dtype, other.value_range)
            if outcome is None:
                continue
            spent += 1
            if outcome[1] is None:
                return other.shape
        return None


class SideSearch:
    """Square spatial side search for a convolution candidate."""

    def __init__(
        self, search: NeutralSearch, candidate: ShapeCandidate, first_error: ProbeError
    ) -> None:
        self.search = search
        self.candidate = candidate
        self.measured = (
            first_error.got_features is not None and first_error.target_features is not None
        )
        self.target = first_error.target_features
        request = search.request
        self.low_bound = max(request.min_size, candidate.min_side, 1)
        self.high_bound = request.max_size

    def shape_for(self, side: int) -> tuple[int, ...]:
        """The candidate shape with every spatial axis set to ``side``."""

        axes = set(self.candidate.spatial_axes)
        return tuple(
            side if index in axes else dim for index, dim in enumerate(self.candidate.shape)
        )

    def try_side(self, side: int) -> tuple[_Win | None, ProbeError | None]:
        """Probe one side; the win, or the error (``None`` when skipped)."""

        shape = self.shape_for(side)
        outcome = self.search.probe(shape, self.candidate.dtype, self.candidate.value_range)
        if outcome is None:
            return None, None
        example, error = outcome
        if error is None:
            win = _Win(
                self.candidate,
                example,
                shape,
                self.candidate.dtype,
                self.candidate.value_range,
                "probe_success",
                True,
            )
            return win, None
        if error.got_features is not None and error.target_features is not None:
            self.measured = True
            self.target = error.target_features
        return None, error

    def run(self) -> _Win | None:
        """Ladder sides, then a measured binary search, then a small-side sweep."""

        start = self.candidate.shape[self.candidate.spatial_axes[0]]
        for side in candidate_sides(
            start, self.low_bound, self.high_bound, self.search.request.preferred_sizes
        ):
            win, error = self.try_side(side)
            if win is not None:
                return win
            if error is not None and error.kind != "shape":
                self.search._record_blocker(error)
                return None
        if self.measured:
            win = self._binary()
            if win is not None:
                return win
        for side in range(self.low_bound, min(self.high_bound, 16) + 1):
            win, _error = self.try_side(side)
            if win is not None:
                return win
        return None

    def _binary(self) -> _Win | None:
        """Binary search on the measured feature count, then a near sweep."""

        low, high = self.low_bound, self.high_bound
        while low <= high and self.search.ledger.budget_left():
            side = (low + high) // 2
            win, error = self.try_side(side)
            if win is not None:
                return replace_strategy(win, "binary_search")
            if error is None or error.got_features is None or self.target is None:
                low = side + 1
                continue
            if error.got_features < self.target:
                low = side + 1
            else:
                high = side - 1
        for side in range(max(self.low_bound, low - 4), min(self.high_bound, low + 4) + 1):
            win, _error = self.try_side(side)
            if win is not None:
                return replace_strategy(win, "binary_search")
        return None


def replace_strategy(win: _Win, strategy: str) -> _Win:
    """A copy of ``win`` with another strategy label."""

    return _Win(
        win.candidate, win.example, win.shape, win.dtype, win.value_range, strategy, win.searched
    )


def build_success(
    win: _Win,
    constraint: ExecutedConstraint | None,
    flexible: tuple[int, ...],
    determination: str,
    trace_and_attempts: tuple[Any, Sequence[Attempt]],
) -> GuessInputShapeReport:
    """Build the verified report for a winning probe (the trace is ``None`` unless asked)."""

    trace_obj, attempts = trace_and_attempts

    module = (
        constraint.module if constraint is not None else None
    ) or win.candidate.constraining_module
    op = (constraint.op if constraint is not None else None) or win.candidate.constraining_op
    return GuessInputShapeReport(
        found=True,
        shape=win.shape,
        shapes=None,
        dtype=win.dtype,
        value_range=win.value_range,
        flexible_dims=flexible,
        constraining_module=module,
        constraining_op=op,
        source_line=constraint.source_line if constraint is not None else None,
        example_input=win.example,
        strategy=win.strategy,
        reason=None,
        attempts=tuple(attempts),
        trace=trace_obj,
        message=(
            f"Found valid input shape {win.shape} in {win.candidate.layout} layout "
            f"(strategy: {win.strategy}); {determination}."
        ),
    )


def dispatch_non_torch(
    model: Any, kwargs: dict[str, Any], tracer: Callable[[Any, Any], Any]
) -> GuessInputShapeReport | None:
    """The neutral search's report for a non-torch model, or ``None`` for torch's path.

    A model a prior handles goes to that prior. A callable that names no
    framework (it only calls tensor methods) goes to the one prior whose
    framework is imported; with several imported the framework is ambiguous
    and the report says so. Anything touching torch stays on torch's path.
    """

    resolved = resolve_priors(model)
    if resolved is not None:
        return guess_with_priors(resolved[1], model, tracer, **kwargs)
    if not callable(model) or isinstance(model, type):
        return None
    if any(framework_of(obj) == TORCH_BACKEND_NAME for _path, obj in walk_references(model)):
        return None
    options = imported_priors()
    if len(options) == 1:
        return guess_with_priors(options[0][1], model, tracer, **kwargs)
    if not options:
        return None
    names = ", ".join(name for name, _priors in options)
    result = failure_result(
        "unknown_entry",
        [],
        f"The callable names no framework and several are imported ({names}), so which "
        "framework's tensors it takes is ambiguous. Pass a model object or a function that "
        "uses its framework's layers or functions.",
    )
    maybe_raise(result, kwargs.get("on_failure", "return"))
    return result


@dataclass(frozen=True)
class SearchSettings:
    """The keyword arguments of :func:`torchlens.debug.guess_input_shape`, with its defaults.

    Constructing it from the caller's keywords rejects an unknown keyword with
    the usual ``TypeError``, as the torch path's signature does.
    """

    batch_size: int = 1
    input_dtype: Any = None
    channels: int | None = None
    spatial_rank: int | Literal["auto"] = "auto"
    seq_len: int | None = None
    square: bool = True
    min_size: int = 1
    max_size: int = 512
    preferred_sizes: Sequence[int] = (224, 256, 384, 299, 128, 96, 64, 32, 28)
    max_probes: int = 64
    device: Any = None
    seed: int = 0
    _return_trace: bool = False
    on_failure: Literal["return", "raise"] = "return"
    input_specs: Any = None


def guess_with_priors(
    priors: ShapePriors, model: Any, tracer: Callable[[Any, Any], Any], **options: Any
) -> GuessInputShapeReport:
    """Run the neutral search for a non-torch model with the same contract as torch.

    ``options`` are :class:`SearchSettings` fields and mean what they mean for
    :func:`torchlens.debug.guess_input_shape`; ``device`` is torch-only.
    ``tracer(model, example)`` runs the verification ``tl.trace`` (injected to
    keep this module out of the package import cycle).
    """

    settings = SearchSettings(**options)
    validate_search_arguments(
        batch_size=settings.batch_size,
        channels=settings.channels,
        spatial_rank=settings.spatial_rank,
        seq_len=settings.seq_len,
        min_size=settings.min_size,
        max_size=settings.max_size,
        max_probes=settings.max_probes,
    )
    if settings.device is not None:  # An argument error raises whatever on_failure says.
        refusal = "device= applies to torch modules only; leave it None."
        maybe_raise(failure_result("unknown_entry", [], refusal), "raise")
    search = NeutralSearch(priors, model, tracer, settings)
    result = _guarded_search(search, settings.input_specs, settings.square)
    maybe_raise(result, settings.on_failure)
    return result


def _guarded_search(search: NeutralSearch, input_specs: Any, square: bool) -> GuessInputShapeReport:
    """Prechecks, then the search inside the prior's state scope and the error net."""

    reason = entry_precheck(input_specs=input_specs, square=square)
    if reason is not None:
        return failure_result(reason, [], PRECHECK_MESSAGES[reason])
    lazy = search.priors.lazy_reason(search.model)
    if lazy is not None:
        return failure_result("lazy_uninitialized", [], f"{lazy} The model was left untouched.")
    try:
        with search.priors.state_scope(search.model):
            return search.run()
    except Exception as exc:  # noqa: BLE001 - the on_failure contract admits no raw escape.
        message = f"Shape inference aborted on an unexpected internal error: {exc!r}"
        return failure_result("non_shape_blocker", search.ledger.attempts, message)
