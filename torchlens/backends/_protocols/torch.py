"""Torch's ``BackendProtocol`` object: the reference, by delegation.

Delegates to the registered torch spec and the public validation entry. It
changes nothing about torch capture; the capture-pipeline rebuild replaces
this adapter with a native implementation and the conformance suite's torch
cells are that rebuild's acceptance test.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from ..._utils._numpy_bridge import tensor_to_numpy
from ...options import CaptureOptions
from .._backend_protocol import FrameworkRequirement, SpecBackedProtocol
from .._errors import BackendUnsupportedError
from .._trace_options import TORCH_TRACE_OPTIONS

TORCH_FRAMEWORK = FrameworkRequirement(
    distributions=("torch",),
    minimum="2.1",
    below=None,
    tested_through="2.13.0",
    pinned="2.13.0",
    extra="",
)


def replay_ready_capture() -> Any:
    """The ``CaptureOptions`` the protocol's capture uses when the caller passes none.

    Torch's replay validation re-executes each op from its saved arguments
    and RNG state, which a default ``tl.trace`` does not record (the other
    backends record what their replay needs by default).
    """

    return CaptureOptions(save_args=True, save_rng=True)


class TorchProtocol(SpecBackedProtocol):
    """The reference backend's protocol object (delegating adapter)."""

    def __init__(self) -> None:
        super().__init__(
            name="torch",
            framework=TORCH_FRAMEWORK,
            materialization="eager",
            trace_options=TORCH_TRACE_OPTIONS,
            refusal_codes=(
                "trace_option_unsupported",
                "value_dependent_branch_unsupported",
                "followed_by_unsupported",
                "torchscript_boundary",
                "scripted_function_not_captured",
                "foreign_kernel_boundary",
                "unsupported_tensor_variant",
                "output_unsupported_tensor_variant",
                "subclass_ctor_under_dispatch_mode_unsupported",
                "predicate_return_invalid",
                "structure_only_substrate_mismatch",
                "lookback_payload_policy_conflict",
                "legacy_constructor_uncaptured",
                "input_rung_requires_module_root",
                "held_tensor_scan_truncated",
                "meta_kernel_unavailable",
                "emit_nvtx_unavailable",
                "dlpack_call_not_captured",
                "output_attribution_failed",
                "wrappers_removed_before_capture",
                "unwrap_during_active_capture",
                "release_during_active_capture",
            ),
        )

    def runtime_available(self) -> str | None:
        """Torch is a base dependency; only the range check applies."""

        return super().runtime_available()

    def capture(self, model: Any, args: Any, kwargs: dict[str, Any] | None, **options: Any) -> Any:
        """Capture through the public ``tl.trace`` entry with ``backend="torch"``.

        Forward kwargs travel only in ``kwargs`` (``tl.trace``'s
        ``input_kwargs``). Without a caller ``capture=``, the capture records
        what :meth:`validate` replays from (:func:`replay_ready_capture`). ``options`` are trace keywords; one that is not a
        named ``tl.trace`` parameter would otherwise fall into ``tl.trace``'s
        forward-kwargs catch-all and reach the model silently, so it refuses
        typed (``trace_option_unsupported``) before anything runs.

        Raises
        ------
        BackendUnsupportedError
            ``trace_option_unsupported`` for an unknown trace keyword.
        """

        from ..._user_funcs import trace  # deferred: cycle _user_funcs

        unknown = sorted(set(options) - _trace_keyword_names())
        if unknown:
            raise BackendUnsupportedError(
                f"The torch backend protocol does not recognise trace option(s) "
                f"{unknown}: they are not tl.trace parameters, and passing them "
                "on would forward them to the model.",
                code="trace_option_unsupported",
                remedy="put model keyword arguments in kwargs; use documented tl.trace options",
                backend="torch",
                option=unknown[0],
            )
        if "capture" not in options:
            options = {**options, "capture": replay_ready_capture()}
        return trace(model, args, input_kwargs=kwargs, backend="torch", **options)

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay-validate through ``Trace.validate_forward_pass``; return its status.

        ``validate_forward_pass`` keeps every entry refusal and the full
        tripwire; for torch it returns a bool and records the status object,
        which this method returns (the protocol never returns a bare bool).

        Parameters
        ----------
        trace:
            A torch ``Trace``.
        ground_truth:
            The reference outputs (a tensor or a nested container of tensors,
            as :meth:`reference_output` returns them).

        Returns
        -------
        ValidationStatus
            The replay status from this validation run.
        """

        trace.validate_forward_pass(_ground_truth_tensors(ground_truth))
        return trace.validation_replay_status

    def reference_output(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> Any:
        """Run the model on pristine torch: wrappers removed for the call."""

        from ...validation._pristine import pristine_torch_oracle

        with pristine_torch_oracle():
            return super().reference_output(model, args, kwargs)

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is a ``torch.Tensor``."""

        import torch

        return isinstance(value, torch.Tensor)

    def host_copy(self, value: Any) -> Any:
        """Detached CPU NumPy copy; bfloat16 widens to float32 (NumPy has no bfloat16)."""

        import torch

        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor_to_numpy(tensor).copy()

    def realize_once(self, values: Sequence[Any]) -> None:
        """Eager framework: no evaluation step."""

        return None


def _trace_keyword_names() -> frozenset[str]:
    """Named ``tl.trace`` parameters other than the model, its inputs and ``backend``."""

    from ..._user_funcs import trace  # deferred: cycle _user_funcs

    parameters = inspect.signature(trace).parameters.values()
    return frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        and parameter.name not in {"model", "inputs", "input_kwargs", "backend"}
    )


def _ground_truth_tensors(ground_truth: Any) -> list[Any]:
    """Detached clones of the reference outputs' tensor leaves, deduplicated.

    Uses the capture-side output walker (the one forward validation uses for
    its own ground truth), so leaf order and deduplication match how the trace
    recorded its outputs.
    """

    from ..torch._ops_containers import _walk_output_tensors_with_paths

    tensors: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for tensor, path, _spec in _walk_output_tensors_with_paths(ground_truth):
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        tensors.append(tensor.detach().clone())
    return tensors


def protocol() -> TorchProtocol:
    """Factory bound to ``BackendSpec.protocol`` for torch."""

    return TorchProtocol()
