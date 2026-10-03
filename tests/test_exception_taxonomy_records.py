"""Typed-door tests for the r3 data_classes refusal conversions.

Every case provokes a public refusal converted from a raw builtin raise in
``torchlens/data_classes/`` and asserts the error-refusal contract: the
exception is a ``TorchLensError`` subclass retaining its historical builtin
base, carries a stable ``fields["code"]``, and names a concrete remedy in
both ``fields["remedy"]`` and the message tail.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import errors
from torchlens._errors import PayloadUnavailableError, RecordBindingError


@pytest.fixture(scope="module")
def small_trace() -> Iterator[Any]:
    """Capture one tiny fully-saved trace shared by the door provocations."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _assert_contract(exc: BaseException, expected_code: str, builtin: type[BaseException]) -> None:
    """Assert the shared refusal contract for one provoked door."""

    assert isinstance(exc, errors.TorchLensError)
    assert isinstance(exc, builtin)
    fields = exc.fields  # type: ignore[attr-defined]
    assert fields["code"] == expected_code
    remedy = fields.get("remedy")
    assert isinstance(remedy, str) and remedy
    assert "Remedy:" in str(exc)
    assert remedy in str(exc)


RECORD_DOOR_CASES: tuple[tuple[str, type[BaseException], Callable[[Any], object]], ...] = (
    (
        "op_lookup_not_found",
        ValueError,
        lambda log: log["no_such_layer_xyz_123"],
    ),
    (
        "op_lookup_index_out_of_range",
        ValueError,
        lambda log: log[10_000],
    ),
    (
        "annotation_payload_missing",
        ValueError,
        lambda log: log.annotate(log.op_labels[0]),
    ),
    (
        "annotation_not_json_serializable",
        ValueError,
        lambda log: log.annotate(tl.func("relu"), data={"bad": object()}),
    ),
    (
        "collapse_mode_invalid",
        ValueError,
        lambda log: log.collapse_plan(mode="bogus"),  # type: ignore[arg-type]
    ),
    (
        "collapse_level_invalid",
        ValueError,
        lambda log: log.collapse_plan(mode=1.5),
    ),
    (
        "stack_selector_no_match",
        ValueError,
        lambda log: log.stack("no_such_selector_xyz"),
    ),
    (
        "intervention_direction_invalid",
        ValueError,
        lambda log: log.set(log.op_labels[0], torch.zeros(1), direction="sideways"),
    ),
    (
        "run_legacy_arguments_conflict",
        TypeError,
        lambda log: log.run(object(), torch.randn(1, 4), inputs=torch.randn(1, 4)),
    ),
    (
        "derived_field_assignment_invalid",
        ValueError,
        lambda log: setattr(log[0], "raw_label", "forged_label"),
    ),
    (
        "relation_assignment_type_invalid",
        TypeError,
        lambda log: setattr(log[0], "parents", 42),
    ),
    (
        "top_n_invalid",
        ValueError,
        lambda log: log.output_table(top_n=0),
    ),
)


@pytest.mark.smoke_cells(
    "test_record_refusals_carry_codes_and_remedies[annotation_not_json_serializable]",
    "test_record_refusals_carry_codes_and_remedies[intervention_direction_invalid]",
)
@pytest.mark.parametrize(
    ("expected_code", "builtin", "trigger"),
    RECORD_DOOR_CASES,
    ids=[case[0] for case in RECORD_DOOR_CASES],
)
def test_record_refusals_carry_codes_and_remedies(
    small_trace: Any,
    expected_code: str,
    builtin: type[BaseException],
    trigger: Callable[[Any], object],
) -> None:
    """Converted data_classes doors expose stable codes and concrete remedies."""

    with pytest.raises(errors.TorchLensError) as exc_info:
        trigger(small_trace)
    _assert_contract(exc_info.value, expected_code, builtin)


def test_unsaved_activation_read_is_typed() -> None:
    """A predicate-selective capture refuses unsaved payload reads typed."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.randn(1, 4), save=tl.func("relu"))
    unsaved = [label for label in log.op_labels if "linear" in label]
    assert unsaved
    with pytest.raises(PayloadUnavailableError) as exc_info:
        _ = log[unsaved[0]].out
    _assert_contract(exc_info.value, "activation_not_saved", ValueError)


def test_new_record_classes_resolve_lazily_and_pickle() -> None:
    """The r3 classes resolve from ``torchlens.errors`` and survive pickling."""

    assert errors.PayloadUnavailableError is PayloadUnavailableError
    assert errors.RecordBindingError is RecordBindingError
    assert issubclass(PayloadUnavailableError, errors.CaptureError)
    assert issubclass(PayloadUnavailableError, ValueError)
    assert issubclass(RecordBindingError, errors.CaptureError)
    assert issubclass(RecordBindingError, RuntimeError)

    original = RecordBindingError(
        "ModuleCall not bound to a Trace",
        code="record_not_bound",
        remedy="keep the owning Trace alive and read records through it",
    )
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, RecordBindingError)
    assert restored.fields == original.fields
    assert str(restored) == str(original)


FASTLOG_DOOR_CASES: tuple[tuple[str, type[BaseException], dict[str, Any]], ...] = (
    ("history_size_invalid", ValueError, {"history_size": -1}),
    ("lookback_invalid", ValueError, {"lookback": 4096}),
    ("lookback_payload_policy_invalid", ValueError, {"lookback_payload_policy": "bogus"}),
    ("recording_intervention_predicate_type_invalid", ValueError, {"intervene": "not-callable"}),
    ("recording_halt_predicate_type_invalid", ValueError, {"halt": "not-callable"}),
    ("max_predicate_failures_invalid", ValueError, {"max_predicate_failures": -3}),
    ("on_predicate_error_invalid", ValueError, {"on_predicate_error": "explode"}),
    ("on_forward_error_invalid", ValueError, {"on_forward_error": "shrug"}),
    ("recording_option_type_invalid", ValueError, {"activation_transform": "not-callable"}),
    ("recording_option_type_invalid", ValueError, {"save_raw_activations": "yes"}),
)


@pytest.mark.parametrize(
    ("expected_code", "builtin", "kwargs"),
    FASTLOG_DOOR_CASES,
    ids=[f"{case[0]}-{next(iter(case[2]))}" for case in FASTLOG_DOOR_CASES],
)
def test_recorder_option_refusals_carry_codes_and_remedies(
    expected_code: str,
    builtin: type[BaseException],
    kwargs: dict[str, Any],
) -> None:
    """Converted fastlog recorder option doors expose codes and remedies."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    with pytest.raises(errors.TorchLensError) as exc_info:
        tl.record(model, torch.randn(1, 4), save=tl.func("relu"), **kwargs)
    _assert_contract(exc_info.value, expected_code, builtin)


TRACE_PREDICATE_DOOR_CASES: tuple[tuple[str, type[BaseException], dict[str, Any]], ...] = (
    ("intervention_predicate_type_invalid", TypeError, {"intervene": "not-callable"}),
    ("halt_predicate_type_invalid", TypeError, {"halt": "not-callable"}),
)


@pytest.mark.parametrize(
    ("expected_code", "builtin", "kwargs"),
    TRACE_PREDICATE_DOOR_CASES,
    ids=[case[0] for case in TRACE_PREDICATE_DOOR_CASES],
)
def test_trace_predicate_type_doors_carry_codes_and_remedies(
    expected_code: str,
    builtin: type[BaseException],
    kwargs: dict[str, Any],
) -> None:
    """The tl.trace() non-callable intervene/halt doors expose codes and remedies.

    Trace-side twins of the ``recording_*_predicate_type_invalid`` fastlog
    doors above (new vocabulary must ship provoked).
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    with pytest.raises(errors.TorchLensError) as exc_info:
        tl.trace(model, torch.randn(1, 4), **kwargs)
    _assert_contract(exc_info.value, expected_code, builtin)


@pytest.mark.smoke
def test_intervention_doors_carry_codes_and_remedies() -> None:
    """Converted intervention-family doors expose codes and remedies."""

    from torchlens.intervention.errors import ReplayPreconditionError
    from torchlens.intervention.resolver import resolve_import_ref
    from torchlens.intervention.save import _validate_format_version

    with pytest.raises(errors.TorchLensError) as exc_info:
        ReplayPreconditionError("message", selector="both")  # args + fields conflict
    _assert_contract(exc_info.value, "error_constructor_args_conflict", TypeError)

    with pytest.raises(errors.TorchLensError) as exc_info:
        resolve_import_ref("no-colon-here")
    _assert_contract(exc_info.value, "import_path_invalid", ValueError)

    with pytest.raises(errors.TorchLensError) as exc_info:
        _validate_format_version("999.0")
    _assert_contract(exc_info.value, "spec_format_version_unsupported", ValueError)


@pytest.mark.smoke
def test_visualization_doors_carry_codes_and_remedies(small_trace: Any) -> None:
    """Converted visualization option doors expose codes and remedies."""

    from torchlens.visualization.overlays import normalize_overlay_name
    from torchlens.visualization.themes import resolve_theme

    with pytest.raises(errors.TorchLensError) as exc_info:
        resolve_theme("bogus_theme")
    _assert_contract(exc_info.value, "visualization_theme_invalid", ValueError)

    with pytest.raises(errors.TorchLensError) as exc_info:
        normalize_overlay_name("bogus_overlay")
    _assert_contract(exc_info.value, "node_overlay_invalid", ValueError)

    with pytest.raises(errors.TorchLensError) as exc_info:
        small_trace.summary(level="bogus_level")
    _assert_contract(exc_info.value, "summary_level_invalid", ValueError)

    with pytest.raises(errors.TorchLensError) as exc_info:
        small_trace.collapse_order(mode="bogus")
    _assert_contract(exc_info.value, "collapse_mode_invalid", ValueError)


@pytest.mark.smoke
def test_trace_stack_shape_mismatch_is_typed(small_trace: Any) -> None:
    """Stacking differently-shaped saved outs refuses with the stack code."""

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    log = tl.trace(model, torch.randn(1, 4))
    with pytest.raises(errors.TorchLensError) as exc_info:
        log.stack(tl.func("linear"))
    _assert_contract(exc_info.value, "stack_shape_mismatch", ValueError)
