"""Typed, coded refusals raised by public entry points.

Each ``refuse_*`` helper raises one stable code from the error refusal contract
(``docs/reference/error_refusal_contract.md``) with a concrete remedy; a
``refuse_if_*`` helper checks its condition first. Every class used here subclasses the
builtin the entry point raised before it was coded, so ``except ValueError`` /
``except TypeError`` / ``except RuntimeError`` callers keep working.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, NoReturn, TypeVar, cast

from ._errors import (
    ArgumentTypeError,
    CaptureNotActiveError,
    InvalidArgumentError,
    _actionable_message,
    _ActionableErrorMixin,
    _InternalInvariantError,
)
from .errors._base import CompatibilityError

_FactoryT = TypeVar("_FactoryT")

BUG_REMEDY = "report this as a TorchLens bug with the full traceback"


class _DependencyIncompatibleError(_ActionableErrorMixin, CompatibilityError, RuntimeError):
    """Raised when an installed dependency lacks a capability TorchLens needs.

    Examples
    --------
    >>> issubclass(_DependencyIncompatibleError, RuntimeError)
    True
    """

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize a dependency-compatibility refusal with its code and remedy."""

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


# --- torch backend ---------------------------------------------------------------------


def refuse_contract_parent_missing() -> NoReturn:
    """Raise ``contract_parent_missing``: a contract parent is absent from the call args."""

    raise _InternalInvariantError(
        "Parent layer not found in function arguments",
        code="contract_parent_missing",
        remedy=BUG_REMEDY,
    )


def refuse_bound_root(method: object) -> NoReturn:
    """Raise ``bound_root_method_invalid`` for a root that is not a bound module method."""

    raise ArgumentTypeError(
        f"TLBoundMethodRoot wraps a bound method of an nn.Module; got {type(method).__name__}",
        code="bound_root_method_invalid",
        remedy="pass a bound method of an nn.Module instance, such as model.forward",
        received_type=type(method).__name__,
    )


def refuse_if_foreign_workspace(owner: object, trace_state: object) -> None:
    """Raise ``raw_graph_workspace_foreign`` unless ``owner`` is ``trace_state``."""

    if owner is not trace_state:
        raise _InternalInvariantError(
            "Torch backend received a foreign raw-graph workspace owner",
            code="raw_graph_workspace_foreign",
            remedy=BUG_REMEDY,
        )


def require_tensorbase_class(tensorbase_cls: type[Any] | None) -> type[Any]:
    """Return ``tensorbase_cls``; raise ``torch_tensorbase_unavailable`` when it is ``None``."""

    if tensorbase_cls is None:
        raise _DependencyIncompatibleError(
            "TorchLens requires torch._C.TensorBase / _TensorBase (torch >= 2.1) for the "
            "completeness witness originals; this torch build exposes neither",
            code="torch_tensorbase_unavailable",
            remedy="install torch 2.1 or newer",
        )
    return tensorbase_cls


def refuse_if_exemptions_over_budget(count: int, budget: int) -> None:
    """Raise ``escape_exemption_budget_exceeded`` when ``count`` exceeds ``budget``."""

    if count > budget:
        raise _InternalInvariantError(
            "TorchLens callable escape exemption budget exceeded",
            code="escape_exemption_budget_exceeded",
            remedy=(
                "remove an entry from AUDITED_ESCAPE_EXEMPTIONS, or raise "
                "MAX_AUDITED_EXEMPTIONS in the change that adds the exemption's regression test"
            ),
            count=count,
            budget=budget,
        )


def refuse_module_unprepared() -> NoReturn:
    """Raise ``module_metadata_missing`` for a module call TorchLens did not prepare."""

    raise _InternalInvariantError(
        "Module is missing TorchLens metadata; was it prepared?",
        code="module_metadata_missing",
        remedy=(
            "capture through tl.trace(model, ...), which prepares every submodule; "
            "if this came from tl.trace, report it as a TorchLens bug"
        ),
    )


def refuse_module_stack_underflow(frame: object) -> NoReturn:
    """Raise ``module_stack_underflow`` for a pop on an empty module stack."""

    raise _InternalInvariantError(
        f"pop_frame called on empty module stack; expected frame {frame!r}",
        code="module_stack_underflow",
        remedy=BUG_REMEDY,
    )


def refuse_module_stack_mismatch(frame: object, top: object) -> NoReturn:
    """Raise ``module_stack_mismatch`` when the popped frame is not the stack top."""

    raise _InternalInvariantError(
        f"module stack top mismatch: expected {frame!r}, got {top!r}",
        code="module_stack_mismatch",
        remedy=BUG_REMEDY,
    )


def refuse_source_kind(source: object, *allowed: str) -> NoReturn:
    """Raise ``source_kind_invalid`` for a source-tensor kind outside ``allowed``."""

    kinds = ", ".join(repr(kind) for kind in allowed)
    raise InvalidArgumentError(
        f"source must be one of {kinds}",
        code="source_kind_invalid",
        remedy=f"pass source as one of {kinds}",
        source=repr(source),
    )


# --- bridges -----------------------------------------------------------------------------


def refuse_brain_score_not_scalar(score: object) -> NoReturn:
    """Raise ``bridge_brain_score_not_scalar`` for a score with no single value."""

    raise ArgumentTypeError(
        f"score_parts needs a single-number score; got {type(score).__name__} without one value",
        code="bridge_brain_score_not_scalar",
        remedy="pass a score that holds exactly one number (a scalar or one-element array)",
        received_type=type(score).__name__,
    )


def refuse_brain_score_benchmark(benchmark: object) -> NoReturn:
    """Raise ``bridge_brain_score_benchmark_invalid`` for a non-callable benchmark."""

    raise ArgumentTypeError(
        "Brain-Score bridge currently requires a callable offline benchmark",
        code="bridge_brain_score_benchmark_invalid",
        remedy="pass a callable that takes the candidate model and returns a score",
        received_type=type(benchmark).__name__,
    )


def refuse_captum_method(method: object) -> NoReturn:
    """Raise ``bridge_captum_method_invalid`` for an object without ``attribute(...)``."""

    raise ArgumentTypeError(
        "Captum bridge expected a method object with an attribute(...) method",
        code="bridge_captum_method_invalid",
        remedy="pass a Captum attribution object, such as captum.attr.IntegratedGradients(model)",
        received_type=type(method).__name__,
    )


def refuse_sae_encoder(sae: object) -> NoReturn:
    """Raise ``sae_interface_invalid`` for an SAE that cannot encode."""

    raise ArgumentTypeError(
        "SAE Lens bridge expected an SAE with encode(...) or a callable SAE",
        code="sae_interface_invalid",
        remedy="pass an SAE with encode(...), or a callable that maps activations to latents",
        received_type=type(sae).__name__,
    )


def refuse_sae_reconstructor(sae: object) -> NoReturn:
    """Raise ``sae_interface_invalid`` for an SAE that cannot encode and decode."""

    raise ArgumentTypeError(
        "SAE Lens bridge expected an SAE with encode(...) and decode(...) or a callable SAE",
        code="sae_interface_invalid",
        remedy=(
            "pass an SAE with encode(...) and decode(...), or a callable that returns "
            "the reconstruction"
        ),
        received_type=type(sae).__name__,
    )


def refuse_processor_unresolved() -> NoReturn:
    """Raise ``hf_processor_unresolved`` when no checkpoint names the model's processor."""

    raise InvalidArgumentError(
        "Could not auto-resolve AutoProcessor for this model",
        code="hf_processor_unresolved",
        remedy=(
            "load the model with from_pretrained(...) so its config names a checkpoint, "
            "or run the processor yourself and call tl.trace"
        ),
    )


# --- bundle, compat, experimental, export ------------------------------------------------


def require_member_name(name: object) -> str:
    """Return ``name``, or raise ``bundle_index_type_invalid`` unless it is a string."""

    if not isinstance(name, str):
        raise ArgumentTypeError(
            f"Bundle indices must be member names, got {type(name).__name__}",
            code="bundle_index_type_invalid",
            remedy="index the Bundle by a member name string; bundle.names lists them",
            received_type=type(name).__name__,
        )
    return name


def refuse_torchextractor_layers() -> NoReturn:
    """Raise ``compat_layers_required`` for ``from_torchextractor`` without layers."""

    raise InvalidArgumentError(
        "from_torchextractor requires explicit layers or an extractor-like object",
        code="compat_layers_required",
        remedy="pass layers=[...] with the module addresses to extract",
    )


def refuse_ilg_return_layers() -> NoReturn:
    """Raise ``compat_layers_required`` for ``from_ilg`` without ``return_layers``."""

    raise InvalidArgumentError(
        "from_ilg requires a return_layers mapping",
        code="compat_layers_required",
        remedy="pass return_layers={'module.address': 'output_name'}",
    )


def refuse_lovely_formatter() -> NoReturn:
    """Raise ``lovely_formatter_unavailable`` for a lovely-tensors without ``lovely()``."""

    raise _DependencyIncompatibleError(
        "The installed lovely_tensors does not expose a lovely() formatter, and "
        "torchlens.compat.lovely will not call the global monkey_patch() to work around this",
        code="lovely_formatter_unavailable",
        remedy=(
            "upgrade lovely-tensors to a version exposing lovely(), or call "
            "lovely_tensors directly if you want global patching"
        ),
    )


def refuse_attribute_walk_segment(part: str) -> NoReturn:
    """Raise ``attribute_walk_segment_invalid`` for a malformed address segment."""

    raise InvalidArgumentError(
        f"Malformed attribute-walk segment: {part!r}",
        code="attribute_walk_segment_invalid",
        remedy="use dotted attribute names with optional integer indexes, such as 'blocks[0].attn'",
        segment=part,
    )


def refuse_auto_environment() -> NoReturn:
    """Raise ``auto_environment_unsupported`` while ``TORCHLENS_AUTO`` is set."""

    raise CaptureNotActiveError(
        "TORCHLENS_AUTO is intentionally unsupported; use auto_capture()",
        code="auto_environment_unsupported",
        remedy="unset TORCHLENS_AUTO and call auto_capture() explicitly",
    )


def refuse_if_nonpositive_every(every: int) -> None:
    """Raise ``auto_capture_every_invalid`` unless ``every`` is positive."""

    if every <= 0:
        raise InvalidArgumentError(
            "every must be positive",
            code="auto_capture_every_invalid",
            remedy="pass every as a positive integer (capture one call in every N)",
            every=every,
        )


def require_xarray_presentations(arrays: Collection[Any], presentation_count: int | None) -> int:
    """Return the presentation count, or raise ``activation_not_saved`` with nothing to export."""

    if not arrays or presentation_count is None:
        raise InvalidArgumentError(
            "No saved tensor outs are available for xarray export",
            code="activation_not_saved",
            remedy="capture with save= covering the layers to export, then call xarray again",
        )
    return presentation_count


# --- intervention, observers, recording ---------------------------------------------------


def refuse_unfinalized_injected_rows() -> NoReturn:
    """Raise ``injected_rows_unfinalized``: a load path skipped ``rehydrate_trace``."""

    raise _InternalInvariantError(
        "loaded injected-op rows were split but never finalized; this artifact was "
        "loaded through a path that skipped rehydrate_trace",
        code="injected_rows_unfinalized",
        remedy="load the artifact with tl.load(path); if it was, report it as a TorchLens bug",
    )


def require_hook_factory(name: str, factory: _FactoryT | None) -> _FactoryT:
    """Return the edit's hook factory, or raise ``edit_hook_factory_missing`` without one."""

    if factory is None:
        raise ArgumentTypeError(
            f"Edit {name!r} has no hook factory",
            code="edit_hook_factory_missing",
            remedy="build the edit with a TorchLens edit helper, which attaches its hook factory",
            edit=name,
        )
    return factory


def refuse_if_bad_signal(signal: object) -> None:
    """Raise ``observer_signal_invalid`` unless ``signal`` is a known observer signal."""

    if signal not in {"forward", "backward", "both"}:
        raise InvalidArgumentError(
            f"signal must be 'forward', 'backward', or 'both'; got {signal!r}",
            code="observer_signal_invalid",
            remedy="pass signal='forward', 'backward', or 'both'",
        )


def refuse_log_value_outside_trace() -> NoReturn:
    """Raise ``capture_context_required`` for ``log_value`` outside a capture."""

    raise CaptureNotActiveError(
        "torchlens.observers.log_value() must be called during trace",
        code="capture_context_required",
        remedy="call log_value from inside the forward pass that tl.trace runs",
    )


# --- validation ------------------------------------------------------------------------


def refuse_if_not_directory(path: Any) -> None:
    """Raise ``tlspec_path_not_directory`` unless ``path`` is a directory."""

    if not path.is_dir():
        raise InvalidArgumentError(
            f"TorchLens .tlspec path must be a directory: {path}",
            code="tlspec_path_not_directory",
            remedy="pass the .tlspec directory that trace.save(path) wrote, not a file inside it",
            path=str(path),
        )


def refuse_manifest_unparseable(path: object, cause: BaseException) -> NoReturn:
    """Raise ``manifest_unreadable`` for a manifest that is not valid JSON."""

    raise InvalidArgumentError(
        f"Failed to parse .tlspec manifest JSON at {path}",
        code="manifest_unreadable",
        remedy="re-save the artifact with trace.save(path); the manifest is truncated or edited",
        path=str(path),
    ) from cause


def refuse_if_manifest_not_object(path: object, manifest: object) -> None:
    """Raise ``manifest_not_json_object`` unless the parsed manifest is a JSON object."""

    if not isinstance(manifest, dict):
        raise InvalidArgumentError(
            f".tlspec manifest at {path} must be a JSON object",
            code="manifest_not_json_object",
            remedy="re-save the artifact with trace.save(path); do not hand-edit the manifest",
            path=str(path),
        )


def refuse_if_unknown_tlspec_format(path: object, tlspec_format: object) -> None:
    """Raise ``tlspec_format_unrecognized`` unless the format is the current unified layout."""

    if tlspec_format != "v2.0_unified":
        raise InvalidArgumentError(
            f"Unrecognized TorchLens .tlspec format at {path}",
            code="tlspec_format_unrecognized",
            remedy="pass a .tlspec written by trace.save(path) in this TorchLens release",
            path=str(path),
            tlspec_format=str(tlspec_format),
        )


def refuse_validation_scope(scope: str, valid_scopes: Collection[str]) -> NoReturn:
    """Raise ``validation_scope_invalid`` for a scope outside ``valid_scopes``."""

    raise InvalidArgumentError(
        f"scope must be one of {sorted(valid_scopes)!r}",
        code="validation_scope_invalid",
        remedy="pass scope as one of " + ", ".join(repr(name) for name in sorted(valid_scopes)),
        scope=scope,
    )


def refuse_status_not_bool() -> NoReturn:
    """Raise ``validation_status_not_run`` for ``bool()`` of a status that never ran."""

    raise ArgumentTypeError(
        "ValidationStatus is not a boolean pass/fail result unless replay validation "
        "has actually run",
        code="validation_status_not_run",
        remedy="read status.state, or run replay validation first and then read status.passed",
    )


def refuse_region_provenance_missing(label: object) -> NoReturn:
    """Raise ``region_annotation_provenance_missing`` for an unattributed region annotation."""

    raise InvalidArgumentError(
        f"Region replay annotation on {label!r} is missing importer provenance",
        code="region_annotation_provenance_missing",
        remedy="re-import the model with the TorchLens importer that wrote the region annotation",
        label=str(label),
    )


# --- visualization -----------------------------------------------------------------------


def refuse_if_below(name: str, value: int, minimum: int) -> None:
    """Raise ``viz_limit_invalid`` when a plotting limit is below ``minimum``."""

    if value < minimum:
        raise InvalidArgumentError(
            f"{name} must be at least {minimum}",
            code="viz_limit_invalid",
            remedy=f"pass {name} as an integer of at least {minimum}",
            argument=name,
            value=value,
        )


def refuse_if_canvas_too_small(canvas_size: int, thumbnail_size: int) -> None:
    """Raise ``viz_limit_invalid`` unless the canvas exceeds twice the thumbnail size."""

    if canvas_size <= thumbnail_size * 2:
        raise InvalidArgumentError(
            "canvas_size must be larger than twice thumbnail_size",
            code="viz_limit_invalid",
            remedy=f"pass canvas_size greater than {thumbnail_size * 2}, or a smaller thumbnail_size",
            argument="canvas_size",
            value=canvas_size,
        )


def refuse_viz_shape(problem: str) -> NoReturn:
    """Raise ``viz_array_shape_invalid`` for plotting data of the wrong shape."""

    raise InvalidArgumentError(
        problem,
        code="viz_array_shape_invalid",
        remedy="pass an array of the shape the message names (check .shape before plotting)",
    )


def refuse_viz_empty(problem: str) -> NoReturn:
    """Raise ``viz_input_empty`` for plotting input with nothing to draw."""

    raise InvalidArgumentError(
        problem,
        code="viz_input_empty",
        remedy="pass at least one item to draw (an empty selection plots nothing)",
    )


def refuse_viz_nonfinite(problem: str) -> NoReturn:
    """Raise ``viz_values_not_finite`` for plotting values with no finite point to draw."""

    raise InvalidArgumentError(
        problem,
        code="viz_values_not_finite",
        remedy="replace or drop NaN and infinite values (np.nan_to_num) before plotting",
    )


def refuse_if_too_few_images(images: Collection[Any] | None, needed: int) -> None:
    """Raise ``viz_images_too_few`` when fewer images than drawn points are given."""

    if images is not None and len(images) < needed:
        raise InvalidArgumentError(
            "images must contain at least the number of drawn coordinates",
            code="viz_images_too_few",
            remedy=f"pass at least {needed} images, or lower max_items",
            needed=needed,
            received=len(images),
        )


def refuse_feature_map_site(message: str, cause: BaseException) -> NoReturn:
    """Raise ``feature_maps_site_invalid`` with the site refusal recast for feature maps."""

    raise InvalidArgumentError(
        message,
        code="feature_maps_site_invalid",
        remedy=(
            "pick saved single-pass layers, or name a recurrent layer by pass "
            "(for example 'conv1:2')"
        ),
    ) from cause


def refuse_feature_maps_not_spatial(shape_text: str) -> NoReturn:
    """Raise ``feature_maps_spatial_missing`` when no saved activation is spatial."""

    raise InvalidArgumentError(
        f"feature_map_evolution requires spatial [N, C, H, W] activations; saw {shape_text}",
        code="feature_maps_spatial_missing",
        remedy="capture with save= covering the conv layers before calling feature_map_evolution",
        seen=shape_text,
    )
