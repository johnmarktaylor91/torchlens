"""Provocations for the coded public-entry refusals (``torchlens._entry_refusals``).

Public entry points once raised plain ``ValueError`` / ``TypeError`` / ``RuntimeError``.
Each now raises a typed TorchLens error with a stable ``code`` and a concrete ``remedy``,
and the class still subclasses the builtin the site raised before, so existing
``except ValueError`` callers keep working. Two layers:

1. ``HELPER_CASES``: every refusal helper, provoked directly, asserting the code, a
   non-empty remedy, the TorchLens base class and the builtin lineage.
2. ``test_public_entry_*``: the cheap public doors provoked end to end, so the code is
   proven reachable from the user-facing call, not only from the helper.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from torch import nn

import torchlens as tl
from torchlens import _entry_refusals as refusals, _errors as base_refusals
from torchlens.backends import _entry_refusals as backend_refusals
from torchlens.errors import TorchLensError
from torchlens.recording.exceptions import refuse_recorder_uninitialized

_HERE = Path(__file__)

# (case id, thunk that must raise, builtin the class must keep, stable code)
HELPER_CASES: tuple[tuple[str, Callable[[], Any], type[BaseException], str], ...] = (
    (
        "contract_parent",
        refusals.refuse_contract_parent_missing,
        RuntimeError,
        "contract_parent_missing",
    ),
    ("bound_root", lambda: refusals.refuse_bound_root(len), TypeError, "bound_root_method_invalid"),
    (
        "foreign_workspace",
        lambda: refusals.refuse_if_foreign_workspace(object(), object()),
        RuntimeError,
        "raw_graph_workspace_foreign",
    ),
    (
        "tensorbase",
        lambda: refusals.require_tensorbase_class(None),
        RuntimeError,
        "torch_tensorbase_unavailable",
    ),
    (
        "exemption_budget",
        lambda: refusals.refuse_if_exemptions_over_budget(3, 2),
        RuntimeError,
        "escape_exemption_budget_exceeded",
    ),
    ("module_unprepared", refusals.refuse_module_unprepared, ValueError, "module_metadata_missing"),
    (
        "stack_underflow",
        lambda: refusals.refuse_module_stack_underflow("block:1"),
        RuntimeError,
        "module_stack_underflow",
    ),
    (
        "stack_mismatch",
        lambda: refusals.refuse_module_stack_mismatch("block:1", "head:1"),
        RuntimeError,
        "module_stack_mismatch",
    ),
    (
        "source_kind",
        lambda: refusals.refuse_source_kind("weights", "input", "buffer"),
        ValueError,
        "source_kind_invalid",
    ),
    (
        "brain_score_scalar",
        lambda: refusals.refuse_brain_score_not_scalar([1.0, 2.0]),
        TypeError,
        "bridge_brain_score_not_scalar",
    ),
    (
        "brain_score_benchmark",
        lambda: refusals.refuse_brain_score_benchmark("benchmark"),
        TypeError,
        "bridge_brain_score_benchmark_invalid",
    ),
    (
        "captum_method",
        lambda: refusals.refuse_captum_method(object()),
        TypeError,
        "bridge_captum_method_invalid",
    ),
    (
        "sae_encoder",
        lambda: refusals.refuse_sae_encoder(object()),
        TypeError,
        "sae_interface_invalid",
    ),
    (
        "sae_reconstructor",
        lambda: refusals.refuse_sae_reconstructor(object()),
        TypeError,
        "sae_interface_invalid",
    ),
    ("hf_processor", refusals.refuse_processor_unresolved, ValueError, "hf_processor_unresolved"),
    (
        "bundle_index",
        lambda: refusals.require_member_name(0),
        TypeError,
        "bundle_index_type_invalid",
    ),
    ("torchextractor", refusals.refuse_torchextractor_layers, ValueError, "compat_layers_required"),
    ("ilg", refusals.refuse_ilg_return_layers, ValueError, "compat_layers_required"),
    ("lovely", refusals.refuse_lovely_formatter, RuntimeError, "lovely_formatter_unavailable"),
    (
        "attribute_walk",
        lambda: refusals.refuse_attribute_walk_segment("bad["),
        ValueError,
        "attribute_walk_segment_invalid",
    ),
    ("auto_env", refusals.refuse_auto_environment, RuntimeError, "auto_environment_unsupported"),
    (
        "auto_every",
        lambda: refusals.refuse_if_nonpositive_every(0),
        ValueError,
        "auto_capture_every_invalid",
    ),
    (
        "xarray_nothing_saved",
        lambda: refusals.require_xarray_presentations([], None),
        ValueError,
        "activation_not_saved",
    ),
    (
        "injected_rows",
        refusals.refuse_unfinalized_injected_rows,
        RuntimeError,
        "injected_rows_unfinalized",
    ),
    (
        "edit_factory",
        lambda: refusals.require_hook_factory("zero_ablate", None),
        TypeError,
        "edit_hook_factory_missing",
    ),
    (
        "signal",
        lambda: refusals.refuse_if_bad_signal("sideways"),
        ValueError,
        "observer_signal_invalid",
    ),
    (
        "log_value",
        refusals.refuse_log_value_outside_trace,
        RuntimeError,
        "capture_context_required",
    ),
    (
        "projection",
        lambda: base_refusals.require_recording_projection(None),
        RuntimeError,
        "recording_projection_uninitialized",
    ),
    (
        "replay_events",
        lambda: base_refusals.refuse_if_no_replay_events(None),
        RuntimeError,
        "recording_events_not_retained",
    ),
    (
        "tlspec_not_dir",
        lambda: refusals.refuse_if_not_directory(_HERE),
        ValueError,
        "tlspec_path_not_directory",
    ),
    (
        "manifest_unparseable",
        lambda: refusals.refuse_manifest_unparseable(_HERE, ValueError("truncated")),
        ValueError,
        "manifest_unreadable",
    ),
    (
        "manifest_not_object",
        lambda: refusals.refuse_if_manifest_not_object(_HERE, []),
        ValueError,
        "manifest_not_json_object",
    ),
    (
        "tlspec_format",
        lambda: refusals.refuse_if_unknown_tlspec_format(_HERE, "mystery"),
        ValueError,
        "tlspec_format_unrecognized",
    ),
    (
        "validation_scope",
        lambda: refusals.refuse_validation_scope("everything", {"forward", "backward"}),
        ValueError,
        "validation_scope_invalid",
    ),
    ("status_not_bool", refusals.refuse_status_not_bool, TypeError, "validation_status_not_run"),
    (
        "region_provenance",
        lambda: refusals.refuse_region_provenance_missing("linear_1_1"),
        ValueError,
        "region_annotation_provenance_missing",
    ),
    ("viz_below", lambda: refusals.refuse_if_below("max_n", 0, 1), ValueError, "viz_limit_invalid"),
    (
        "viz_canvas",
        lambda: refusals.refuse_if_canvas_too_small(100, 60),
        ValueError,
        "viz_limit_invalid",
    ),
    (
        "viz_shape",
        lambda: refusals.refuse_viz_shape("expects 2D"),
        ValueError,
        "viz_array_shape_invalid",
    ),
    (
        "viz_empty",
        lambda: refusals.refuse_viz_empty("nothing to draw"),
        ValueError,
        "viz_input_empty",
    ),
    (
        "viz_nonfinite",
        lambda: refusals.refuse_viz_nonfinite("values contain NaN"),
        ValueError,
        "viz_values_not_finite",
    ),
    (
        "viz_images",
        lambda: refusals.refuse_if_too_few_images([object()], 2),
        ValueError,
        "viz_images_too_few",
    ),
    (
        "feature_map_site",
        lambda: refusals.refuse_feature_map_site("conv1 has 2 passes", KeyError("conv1")),
        ValueError,
        "feature_maps_site_invalid",
    ),
    (
        "feature_maps_spatial",
        lambda: refusals.refuse_feature_maps_not_spatial("linear_1_1 [1, 4]"),
        ValueError,
        "feature_maps_spatial_missing",
    ),
    (
        "max_intermediate_grads",
        lambda: backend_refusals.refuse_if_bad_max_intermediate_grads(0),
        ValueError,
        "grad_options_max_intermediate_invalid",
    ),
    (
        "jax_control_flow_mode",
        lambda: backend_refusals.refuse_if_bad_jax_control_flow("loop", 4),
        ValueError,
        "jax_control_flow_invalid",
    ),
    (
        "jax_unroll_type",
        lambda: backend_refusals.refuse_if_bad_jax_control_flow("unroll", "4"),
        ValueError,
        "jax_unroll_type_invalid",
    ),
    (
        "jax_unroll_range",
        lambda: backend_refusals.refuse_if_bad_jax_control_flow("unroll", 0),
        ValueError,
        "jax_unroll_range_invalid",
    ),
    (
        "jax_reject",
        lambda: backend_refusals.refuse_if_control_flow_rejected("reject", "while"),
        ValueError,
        "jax_control_flow_rejected",
    ),
    (
        "jax_nested_primitive",
        lambda: backend_refusals.refuse_nested_primitive("custom_vjp_call"),
        ValueError,
        "jax_nested_primitive_unsupported",
    ),
    (
        "jax_nested_jaxpr",
        lambda: backend_refusals.refuse_nested_jaxpr("custom_vjp_call"),
        ValueError,
        "jax_nested_primitive_unsupported",
    ),
    (
        "jax_equation_effects",
        lambda: backend_refusals.refuse_equation_effects("debug_callback", {"io"}),
        ValueError,
        "jax_effects_unsupported",
    ),
    (
        "jax_effect_primitive",
        lambda: backend_refusals.refuse_effect_primitive("debug_callback"),
        ValueError,
        "jax_effects_unsupported",
    ),
    (
        "mlx_policy",
        lambda: backend_refusals.refuse_mlx_unwrapped_policy("silent", {"warn", "error"}),
        ValueError,
        "mlx_unwrapped_policy_invalid",
    ),
    (
        "backend_registered",
        lambda: backend_refusals.refuse_backend_registered("torch"),
        ValueError,
        "backend_already_registered",
    ),
    (
        "tf_no_leaves",
        backend_refusals.refuse_tf_no_grad_leaves,
        ValueError,
        "tf_derived_grads_no_leaves",
    ),
    (
        "tf_loss",
        backend_refusals.refuse_tf_loss_not_scalar,
        ValueError,
        "tf_derived_grads_loss_not_scalar",
    ),
    (
        "tf_replay",
        backend_refusals.refuse_tf_replay_diverged,
        ValueError,
        "tf_derived_grads_replay_diverged",
    ),
    (
        "recorder_state",
        lambda: refuse_recorder_uninitialized("Recorder state was not initialized"),
        RuntimeError,
        "recorder_state_uninitialized",
    ),
)


def _assert_coded(exc: BaseException, builtin: type[BaseException], code: str) -> None:
    """Assert one refusal carries ``code``, a concrete remedy, and its builtin lineage."""

    assert isinstance(exc, TorchLensError)
    assert isinstance(exc, builtin)
    assert exc.fields["code"] == code
    remedy = exc.fields["remedy"]
    assert isinstance(remedy, str) and remedy.strip()


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("thunk", "builtin", "code"),
    [case[1:] for case in HELPER_CASES],
    ids=[case[0] for case in HELPER_CASES],
)
def test_entry_refusal_helper_is_coded(
    thunk: Callable[[], Any], builtin: type[BaseException], code: str
) -> None:
    """Every public-entry refusal helper raises its stable code with a remedy."""

    with pytest.raises(builtin) as exc_info:
        thunk()
    _assert_coded(exc_info.value, builtin, code)


@pytest.mark.smoke
def test_internal_invariant_class_keeps_both_builtins() -> None:
    """The internal-invariant class answers both former ``except`` spellings."""

    with pytest.raises(ValueError) as as_value:
        refusals.refuse_module_unprepared()
    with pytest.raises(RuntimeError) as as_runtime:
        refusals.refuse_contract_parent_missing()
    assert type(as_value.value) is type(as_runtime.value)


@pytest.mark.smoke
def test_public_entry_tap_signal() -> None:
    """``observers.tap`` refuses an unknown signal with ``observer_signal_invalid``."""

    with pytest.raises(ValueError) as exc_info:
        tl.observers.tap("linear_1_1", signal="sideways")
    _assert_coded(exc_info.value, ValueError, "observer_signal_invalid")


@pytest.mark.smoke
def test_public_entry_log_value_outside_trace() -> None:
    """``observers.log_value`` outside a capture refuses ``capture_context_required``."""

    with pytest.raises(RuntimeError) as exc_info:
        tl.observers.log_value("loss", 1.0)
    _assert_coded(exc_info.value, RuntimeError, "capture_context_required")


@pytest.mark.smoke
def test_public_entry_validate_tlspec_file(tmp_path: Path) -> None:
    """``validate_tlspec`` on a file refuses ``tlspec_path_not_directory``."""

    from torchlens.validation import validate_tlspec

    target = tmp_path / "artifact.tlspec"
    target.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        validate_tlspec(target)
    _assert_coded(exc_info.value, ValueError, "tlspec_path_not_directory")


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("manifest_text", "code"),
    [("{", "manifest_unreadable"), (json.dumps([1, 2]), "manifest_not_json_object")],
    ids=["truncated", "list"],
)
def test_public_entry_validate_tlspec_manifest(
    tmp_path: Path, manifest_text: str, code: str
) -> None:
    """``validate_tlspec`` refuses a broken manifest with its manifest code."""

    from torchlens.validation import validate_tlspec

    target = tmp_path / "artifact.tlspec"
    target.mkdir()
    (target / "manifest.json").write_text(manifest_text, encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        validate_tlspec(target)
    _assert_coded(exc_info.value, ValueError, code)


@pytest.mark.smoke
def test_public_entry_attribute_walk_segment() -> None:
    """``experimental.attribute_walk`` refuses a malformed segment."""

    from torchlens.experimental import attribute_walk

    with pytest.raises(ValueError) as exc_info:
        attribute_walk(nn.Sequential(nn.Linear(2, 2)), "bad[")
    _assert_coded(exc_info.value, ValueError, "attribute_walk_segment_invalid")


@pytest.mark.smoke
def test_public_entry_auto_capture_every(monkeypatch: pytest.MonkeyPatch) -> None:
    """``experimental.auto_capture`` refuses a non-positive ``every``."""

    from torchlens.experimental import auto_capture

    monkeypatch.delenv("TORCHLENS_AUTO", raising=False)
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError) as exc_info:
        with auto_capture(model, every=0):
            pass
    _assert_coded(exc_info.value, ValueError, "auto_capture_every_invalid")


@pytest.mark.smoke
def test_public_entry_text_table_limit() -> None:
    """``viz.batch_summary.text_table`` refuses ``max_n`` below one."""

    from torchlens.viz.batch_summary import text_table

    with pytest.raises(ValueError) as exc_info:
        text_table(["a"], 0)
    _assert_coded(exc_info.value, ValueError, "viz_limit_invalid")


@pytest.mark.smoke
def test_public_entry_backend_register_duplicate() -> None:
    """``backends.register`` refuses a name that is already registered."""

    spec = tl.backends.specs()[0]
    with pytest.raises(ValueError) as exc_info:
        tl.backends.register(spec)
    _assert_coded(exc_info.value, ValueError, "backend_already_registered")
