"""Technical-preview MLX implementation of the capture backend Protocol."""

from __future__ import annotations

import random
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ... import _state
from ..._trace_core.relation_views import freeze_trace_relation_views
from ...backends import (
    BackendName,
    BackendUnsupportedError,
    get_backend_spec,
    require_capability_implementation,
)
from ...capture.outcome import StopRequest, stamp_backend_finalized
from ...data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from ...data_classes.param import Param, ParamAccessor
from ...data_classes.trace import Trace
from ...fastlog._halt import HaltSignal
from ...fastlog.types import CaptureSpec
from ...ir.capture_events import CaptureEvents
from ...ir.events import (
    ArgTemplateRef,
    FunctionCallRef,
    ModuleFrame,
    OpEvent,
    OutputRef,
    ParentEdge,
)
from ...ir.intervention import FireResult, FunctionEventInput
from ...ir.predicate import _DEFERRED_VALUE, RecordContext
from ...ir.refs import DeviceRef, DtypeRef, ReservedLabel, TensorRef
from ...ir.semantics import BackendSemantics, CapturePolicy
from ...ir.workspaces import RawGraphWorkspace
from ...postprocess._materialize import materialize_from_events
from ...quantities import Duration
from ...validation.status import ValidationReplaySource, ValidationReplayStatus  # noqa: TC001
from .._finalize import (
    attach_function_root_module,
    attach_module_owned_op_params,
    attach_object_module_logs,
    finalize_single_pass_trace,
    join_module_address as _join_module_address,
    mark_output_label,
    mirror_param_derived_grads,
    nearest_metadata_parent,
    normalize_op_module_calls,
    numel_from_shape as _numel,
    session_callable_identity as _callable_identity,
    value_nbytes as _nbytes,
)
from .._options import MLX_PREVIEW_TRACE_OPTION_POLICY, reject_unsupported_trace_options
from .._validation_shared import float_replay_tolerances
from . import capabilities
from .model_prep import (
    MLXModuleTree,
    cleanup_model_session,
    discover_mlx_module_tree,
    prepare_model_once,
    prepare_model_session,
)
from .tensor_store import MLXTensorLabelStore
from .validation import MLXOpCapture, build_capture_template
from .wrappers import is_mlx_wrapped, mlx_tap_observer, unwrap_mlx, wrap_mlx


@dataclass(frozen=True)
class MLXParameterCandidate:
    """One flattened MLX parameter candidate.

    Parameters
    ----------
    address
        Dotted parameter address.
    value
        MLX array value.
    owner
        Owning primary module address.
    """

    address: str
    value: Any
    owner: str


@dataclass(frozen=True)
class GradOptions:
    """MLX derived-gradient preview options.

    Parameters
    ----------
    params
        Optional MLX parameter tree to pass as explicit AD argument 0. When
        omitted for ``mlx.nn.Module`` models, ``model.parameters()`` is used.
    loss_fn
        Optional callable mapping raw function output to a scalar loss. Required
        unless the raw traced output is already scalar.
    input_grad_argnums
        Positional input argument indexes to differentiate in addition to params.
    intermediate_grads
        Whether to run the opt-in custom-VJP tap replay for exact op-level
        intermediate derived gradients.
    max_intermediate_grads
        Hard cap on saved op boundaries processed by the intermediate-gradient
        producer and oracle.
    """

    params: Any | None = None
    loss_fn: Callable[[Any], Any] | None = None
    input_grad_argnums: tuple[int, ...] = ()
    intermediate_grads: bool = False
    max_intermediate_grads: int = 64

    def __init__(
        self,
        *,
        params: Any | None = None,
        loss_fn: Callable[[Any], Any] | None = None,
        input_grad_argnums: Sequence[int] = (),
        intermediate_grads: bool = False,
        max_intermediate_grads: int = 64,
    ) -> None:
        """Initialize MLX derived-gradient options.

        Parameters
        ----------
        params
            Explicit MLX parameter tree, or ``None`` to use module parameters
            when the captured model exposes ``parameters()``.
        loss_fn
            Callable mapping raw output to scalar loss, or ``None`` for scalar
            raw outputs.
        input_grad_argnums
            Input-relative argnums to differentiate.
        intermediate_grads
            Whether to expose exact op-level intermediate derived gradients.
        max_intermediate_grads
            Hard cap on saved op boundaries for the tap producer and oracle.
        """

        if not isinstance(max_intermediate_grads, int) or max_intermediate_grads < 1:
            raise ValueError("max_intermediate_grads must be an integer >= 1")
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "loss_fn", loss_fn)
        object.__setattr__(self, "input_grad_argnums", tuple(input_grad_argnums))
        object.__setattr__(self, "intermediate_grads", bool(intermediate_grads))
        object.__setattr__(self, "max_intermediate_grads", max_intermediate_grads)


@dataclass(frozen=True)
class _MLXIntermediateSignature:
    """Conservative MLX replay/trace signature for one op output.

    Parameters
    ----------
    op_name
        Wrapped MLX operation name.
    call_ordinal
        Function-call ordinal from wrapper execution, shared by multi-output leaves.
    parent_labels
        Raw TorchLens labels for data parents observed at the boundary.
    shape
        Output shape.
    dtype
        Output dtype string.
    module_calls
        Normalized module call labels active for the trace-side op.
    """

    op_name: str
    call_ordinal: int
    parent_labels: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    module_calls: tuple[str, ...]


@dataclass
class _MLXIntermediateCandidate:
    """One observed MLX intermediate tap candidate.

    Parameters
    ----------
    raw_label
        Replay raw label generated with TorchLens label-counter semantics.
    signature
        Conservative grouped signature used for exact 1:1 attachment.
    value
        Boundary primal value from the AD replay.
    grad
        Cotangent received by the custom VJP tap, once AD runs.
    """

    raw_label: str
    signature: _MLXIntermediateSignature
    value: Any
    grad: Any | None = None


class _MLXIntermediateTapObserver:
    """Observe wrapped MLX calls and inject custom-VJP identity taps."""

    def __init__(self, backend: MLXBackend, trace: Trace) -> None:
        """Initialize a tap observer aligned to an existing MLX trace.

        Parameters
        ----------
        backend
            Active MLX backend instance.
        trace
            Finalized trace whose input/source labels seed replay labeling.
        """

        self.backend = backend
        self.mx = backend.mx
        self.raw_layer_counter = _max_mlx_input_raw_index(trace)
        self.type_counters: dict[str, int] = {}
        self.func_call_id = 0
        self.depth = 0
        self.labels_by_id: dict[int, str] = {}
        self.candidates: list[_MLXIntermediateCandidate] = []
        self._label_trace_inputs(trace)

    def _label_trace_inputs(self, trace: Trace) -> None:
        """Seed replay labels from finalized source/input ops.

        Parameters
        ----------
        trace
            Finalized trace containing input operations.
        """

        for op in trace.layer_list:
            if not op.is_input or op.out is None:
                continue
            self.labels_by_id[id(op.out)] = str(op._label_raw)

    def label_call_inputs(self, args: Sequence[Any], kwargs: Mapping[Any, Any]) -> None:
        """Label top-level replay model inputs with stable source labels.

        Parameters
        ----------
        args
            Positional model arguments passed during replay.
        kwargs
            Keyword model arguments passed during replay.
        """

        for index, arg in enumerate(args):
            if self.backend.is_tensor(arg):
                self.labels_by_id[id(arg)] = f"input.arg_{index}"
        for key, value in kwargs.items():
            if self.backend.is_tensor(value):
                self.labels_by_id[id(value)] = f"input.{key}"

    def call(
        self,
        original: Callable[..., Any],
        op_name: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a wrapped MLX callable and tap eligible outputs.

        Parameters
        ----------
        original
            Original unwrapped MLX callable.
        op_name
            TorchLens operation name, or ``None`` for stack-only module wrappers.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.

        Returns
        -------
        Any
            Original output with eligible array leaves replaced by identity taps.
        """

        if op_name is None or self.depth > 0:
            return original(*args, **kwargs)
        self.depth += 1
        try:
            output = original(*args, **kwargs)
        finally:
            self.depth -= 1
        return self._tap_output(op_name, args, kwargs, output)

    def _tap_output(
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        """Tap all eligible array leaves in one wrapped-call output.

        Parameters
        ----------
        op_name
            TorchLens operation name.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.
        output
            Raw callable output.

        Returns
        -------
        Any
            Output tree with tapped leaves.
        """

        leaves = list(self.backend._iter_arrays(output))
        if not leaves:
            return output
        self.func_call_id += 1
        parent_labels = _mlx_parent_replay_labels(self.backend, self.labels_by_id, args, kwargs)
        replacements: dict[int, Any] = {}
        for leaf in leaves:
            self.raw_layer_counter += 1
            type_counter = self.type_counters.get(op_name, 0) + 1
            self.type_counters[op_name] = type_counter
            raw_label = f"{op_name}_{type_counter}_{self.raw_layer_counter}_raw"
            tapped = leaf
            if _is_float_mlx_array(leaf):
                signature = _MLXIntermediateSignature(
                    op_name=op_name,
                    call_ordinal=self.func_call_id,
                    parent_labels=parent_labels,
                    shape=tuple(getattr(leaf, "shape", ())),
                    dtype=str(getattr(leaf, "dtype", "")),
                    module_calls=(),
                )
                candidate = _MLXIntermediateCandidate(
                    raw_label=raw_label,
                    signature=signature,
                    value=leaf,
                )
                tapped = _mlx_identity_tap(self.mx, candidate, leaf)
                self.candidates.append(candidate)
            self.labels_by_id[id(leaf)] = raw_label
            self.labels_by_id[id(tapped)] = raw_label
            replacements[id(leaf)] = tapped
        return _replace_mlx_array_leaves(self.backend, output, replacements)


class _MLXBoundaryReplacementObserver:
    """Replace one observed MLX boundary during an oracle replay."""

    def __init__(
        self,
        backend: MLXBackend,
        trace: Trace,
        target_signature: _MLXIntermediateSignature,
        replacement: Any,
    ) -> None:
        """Initialize a boundary replacement observer.

        Parameters
        ----------
        backend
            Active MLX backend instance.
        trace
            Finalized trace whose input labels seed replay labeling.
        target_signature
            Signature of the boundary to replace.
        replacement
            Replacement array supplied by the oracle AD transform.
        """

        self.backend = backend
        self.raw_layer_counter = _max_mlx_input_raw_index(trace)
        self.type_counters: dict[str, int] = {}
        self.func_call_id = 0
        self.depth = 0
        self.labels_by_id: dict[int, str] = {}
        self.target_signature = target_signature
        self.replacement = replacement
        self.replaced = False

    def label_call_inputs(self, args: Sequence[Any], kwargs: Mapping[Any, Any]) -> None:
        """Label top-level replay model inputs with stable source labels.

        Parameters
        ----------
        args
            Positional model arguments passed during replay.
        kwargs
            Keyword model arguments passed during replay.
        """

        for index, arg in enumerate(args):
            if self.backend.is_tensor(arg):
                self.labels_by_id[id(arg)] = f"input.arg_{index}"
        for key, value in kwargs.items():
            if self.backend.is_tensor(value):
                self.labels_by_id[id(value)] = f"input.{key}"

    def call(
        self,
        original: Callable[..., Any],
        op_name: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a wrapped callable and replace the target boundary.

        Parameters
        ----------
        original
            Original unwrapped MLX callable.
        op_name
            TorchLens operation name, or ``None`` for stack-only module wrappers.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.

        Returns
        -------
        Any
            Output tree with the target boundary replaced when matched.
        """

        if op_name is None or self.depth > 0:
            return original(*args, **kwargs)
        self.depth += 1
        try:
            output = original(*args, **kwargs)
        finally:
            self.depth -= 1
        return self._replace_output(op_name, args, kwargs, output)

    def _replace_output(
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        """Replace matching array leaves in one output tree.

        Parameters
        ----------
        op_name
            TorchLens operation name.
        args
            Positional call arguments.
        kwargs
            Keyword call arguments.
        output
            Raw callable output.

        Returns
        -------
        Any
            Output tree with the target leaf replaced.
        """

        leaves = list(self.backend._iter_arrays(output))
        if not leaves:
            return output
        self.func_call_id += 1
        parent_labels = _mlx_parent_replay_labels(self.backend, self.labels_by_id, args, kwargs)
        replacements: dict[int, Any] = {}
        for leaf in leaves:
            self.raw_layer_counter += 1
            type_counter = self.type_counters.get(op_name, 0) + 1
            self.type_counters[op_name] = type_counter
            raw_label = f"{op_name}_{type_counter}_{self.raw_layer_counter}_raw"
            signature = _MLXIntermediateSignature(
                op_name=op_name,
                call_ordinal=self.func_call_id,
                parent_labels=parent_labels,
                shape=tuple(getattr(leaf, "shape", ())),
                dtype=str(getattr(leaf, "dtype", "")),
                module_calls=(),
            )
            replacement = self.replacement if signature == self.target_signature else leaf
            if signature == self.target_signature:
                self.replaced = True
            self.labels_by_id[id(leaf)] = raw_label
            self.labels_by_id[id(replacement)] = raw_label
            replacements[id(leaf)] = replacement
        return _replace_mlx_array_leaves(self.backend, output, replacements)


def _mlx_probe_identity(value: Any) -> Any:
    """Return ``value`` unchanged; probe body for traced-transform detection.

    Parameters
    ----------
    value
        Probe argument.

    Returns
    -------
    Any
        ``value`` unchanged.
    """

    return value


def _mlx_traced_transform_type(mx: Any) -> type | None:
    """Resolve the MLX traced-transform wrapper type from the runtime itself.

    ``mx.compile``, ``mx.grad``, ``mx.value_and_grad``, and ``mx.vmap`` all
    return the same opaque wrapper type, which replays a traced graph and
    therefore bypasses (or worse, tracer-pollutes) the monkeypatched eager
    capture surface. The type is resolved by compiling a trivial probe so the
    authority is the runtime, never a spoofable ``__module__`` string.

    Parameters
    ----------
    mx
        Imported ``mlx.core`` module.

    Returns
    -------
    type | None
        Exact wrapper type, or ``None`` when the runtime exposes no
        ``compile`` entry (detection then degrades gracefully).
    """

    compile_fn = getattr(mx, "compile", None)
    if not callable(compile_fn):
        return None
    try:
        return type(compile_fn(_mlx_probe_identity))
    except Exception:
        return None


def _find_mlx_compiled_attributes(
    model: object,
    transform_type: type | None,
    *,
    max_depth: int = 8,
) -> tuple[str, ...]:
    """Return dotted paths of traced-transform callables reachable from ``model``.

    The scan is bounded and disclosed: instance ``__dict__`` values of the
    model and nested ``mlx.nn.Module`` children, plus one level inside plain
    ``list``/``tuple``/``dict`` containers. Slots-only holders, values created
    inside ``__call__``, and hot global/free compiled callables remain
    documented residuals.

    Parameters
    ----------
    model
        Capture entry object.
    transform_type
        Exact traced-transform wrapper type from
        :func:`_mlx_traced_transform_type`.
    max_depth
        Recursion bound over nested module attributes.

    Returns
    -------
    tuple[str, ...]
        Sorted dotted attribute paths holding traced-transform wrappers.
    """

    if transform_type is None:
        return ()
    try:
        import mlx.nn as mlx_nn
    except ImportError:
        return ()
    found: set[str] = set()
    seen: set[int] = set()

    def _scan_value(path: str, value: Any, depth: int) -> None:
        """Record ``path`` when ``value`` (or a direct list/dict item) is the transform type."""

        if type(value) is transform_type:
            found.add(path)
            return
        if isinstance(value, mlx_nn.Module):
            _scan_module(path, value, depth)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if type(item) is transform_type:
                    found.add(f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                if type(item) is transform_type:
                    found.add(f"{path}[{key!r}]")

    def _scan_module(prefix: str, module: object, depth: int) -> None:
        """Recurse through one ``mlx.nn.Module``'s attribute and dict surfaces.

        Both surfaces are scanned because ``mlx.nn.Module`` subclasses ``dict``:
        children and arrays live in the dict items while plain Python attributes
        land in ``__dict__``. Bounded by ``max_depth`` and an identity-seen set.
        """

        if depth > max_depth or id(module) in seen:
            return
        seen.add(id(module))
        # mlx.nn.Module subclasses dict: children/arrays live in the dict
        # items while plain Python attributes land in __dict__ — scan both.
        surfaces: list[dict[str, Any]] = []
        attributes = getattr(module, "__dict__", None)
        if isinstance(attributes, dict):
            surfaces.append(attributes)
        if isinstance(module, dict):
            surfaces.append(module)
        for surface in surfaces:
            for name, value in surface.items():
                child_path = f"{prefix}.{name}" if prefix else str(name)
                _scan_value(child_path, value, depth + 1)

    if isinstance(model, mlx_nn.Module):
        _scan_module("", model, 0)
    return tuple(sorted(found))


def _mlx_loaded_replay_unavailable(trace: Trace) -> bool:
    """Return whether a loaded MLX trace lacks live replay artifacts.

    Parameters
    ----------
    trace
        Trace being validated.

    Returns
    -------
    bool
        True when replay must report unavailable instead of a pass/fail bool.
    """

    if not bool(getattr(trace, "_loaded_from_bundle", False)):
        return False
    if not bool(getattr(trace, "_mlx_op_captures", ())):
        return True
    return str(getattr(trace, "payload_load_status", "")).startswith("audit_only")


def _mlx_validation_source(trace: Trace) -> ValidationReplaySource:
    """Return the validation source label for ``trace``.

    Parameters
    ----------
    trace
        Trace being validated.

    Returns
    -------
    ValidationReplaySource
        ``"loaded"`` for bundle-loaded traces, otherwise ``"live"``.
    """

    return "loaded" if getattr(trace, "_loaded_from_bundle", False) else "live"


class MLXBackend:
    """MLX adapter for the backend-neutral capture Protocol."""

    name = "mlx"
    supports_backward_capture = capabilities.supports_backward_capture

    def __init__(self) -> None:
        """Initialize an MLX backend and verify the optional dependency."""

        self.mx, self.nn = self._import_mlx()
        self.tensor_store = MLXTensorLabelStore()

    def wrap(self, value: object, module_tree: MLXModuleTree | None = None) -> object:
        """Install MLX wrappers and return ``value`` unchanged."""

        wrap_mlx(self, module_tree=module_tree)
        return value

    def unwrap(self, value: object) -> object:
        """Remove MLX wrappers and return ``value`` unchanged."""

        unwrap_mlx()
        return value

    def is_wrapped(self, value: object) -> bool:
        """Return whether MLX wrappers are installed."""

        return is_mlx_wrapped()

    def start_session(self, options: object) -> object:
        """Start an MLX capture session.

        Parameters
        ----------
        options:
            Trace-like capture options object.

        Returns
        -------
        object
            The unchanged options object.
        """

        return options

    def prepare_model(self, session: object, model: object) -> object:
        """Apply one-time and per-session MLX model preparation."""

        self.prepare_model_once(model)
        self.prepare_model_session(session, model)
        return model

    def prepare_model_once(self, model: object) -> object:
        """Apply one-time MLX model preparation."""

        return prepare_model_once(model)

    def prepare_model_session(self, session: object, model: object) -> object:
        """Apply per-session MLX model preparation."""

        return prepare_model_session(session, model)

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        """Clean up per-session MLX model preparation."""

        cleanup_model_session(session, prepared_model)

    def cleanup_halted_forward_session(self, session: object, prepared_model: object) -> None:
        """Clean up MLX state after a halted forward capture."""

        self.cleanup_model_session(session, prepared_model)

    def cleanup_failed_forward_session(
        self,
        session: object,
        prepared_model: object,
        exc: Exception,
    ) -> None:
        """Clean up MLX state after a failed forward capture."""

        del exc
        self.cleanup_model_session(session, prepared_model)

    def cleanup_forward_memory(self, session: object) -> None:
        """Release MLX transient forward-memory caches.

        Parameters
        ----------
        session:
            Active trace session, unused by the current MLX preview.

        Returns
        -------
        None
            No memory cleanup is required.
        """

        del session

    def active_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that enables MLX logging."""

        return _state.active_logging(cast(Trace, session))

    def pause_logging(self, session: object) -> AbstractContextManager[None]:
        """Return a context manager that pauses MLX logging."""

        return _state.pause_logging()

    def snapshot_rng(self, session: object) -> object:
        """Return the initial MLX RNG snapshot.

        MLX RNG replay is intentionally unsupported in this milestone, per AD-9.
        """

        return None

    def seed_rng(self, session: object, seed: int) -> None:
        """Seed host RNG engines for an MLX shared-capture session.

        Parameters
        ----------
        session:
            Active trace session, unused by MLX RNG seeding.
        seed:
            Integer seed value.

        Returns
        -------
        None
            Python and NumPy RNG engines are seeded in place.
        """

        del session
        random.seed(seed)
        np.random.seed(seed)

    def set_capture_producer_policy(self, session: object, capture_mode: object) -> None:
        """Install MLX producer policy metadata for a shared-capture session.

        Parameters
        ----------
        session:
            Active trace session, unused by MLX.
        capture_mode:
            Capture mode, unused by MLX.

        Returns
        -------
        None
            No producer policy metadata is required for MLX.
        """

        del session, capture_mode

    def restore_rng(self, session: object, rng_state: object) -> None:
        """Restore an MLX RNG snapshot.

        Parameters
        ----------
        session:
            Active trace session, unused because MLX RNG replay is unsupported.
        rng_state:
            Opaque RNG snapshot returned by :meth:`snapshot_rng`.

        Returns
        -------
        None
            No state is restored for the current MLX preview.
        """

        del session, rng_state

    def inference_context(self, session: object) -> AbstractContextManager[None]:
        """Return the MLX inference-only context for this session.

        Parameters
        ----------
        session:
            Active trace session, unused because MLX has no TorchLens no-grad
            context integration in this preview.

        Returns
        -------
        AbstractContextManager[None]
            A null context.
        """

        del session
        return nullcontext()

    def build_record_context(
        self,
        session: object,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> RecordContext:
        """Build the selector predicate context for one MLX output.

        MLX's lazy backend guarantees shape, dtype, and device metadata at call
        time. Value-dependent fields, specifically ``tensor_requires_grad``,
        ``is_scalar_bool``, and ``bool_value``, are represented by the
        ``_DEFERRED_VALUE`` sentinel so predicates cannot accidentally consume
        silently-wrong values or force per-op ``mx.eval``.
        """

        return RecordContext(
            kind="op",
            label=reserved.label,
            raw_label=reserved.label_raw,
            pass_index=1,
            event_index=reserved.raw_index,
            step_index=None,
            layer_type=reserved.layer_type,
            type_index=reserved.type_index,
            raw_index=reserved.raw_index,
            func_name=func_event_input.func_name,
            address=None,
            module_type=None,
            module_pass_index=None,
            module_stack=func_event_input.module_stack,
            recent_events=(),
            recent_ops=(),
            parent_labels=(),
            input_output_address=None,
            shape=self._shape(output),
            dtype=DtypeRef.from_value(self._dtype(output)),
            tensor_device=DeviceRef.from_value(self._device(output)),
            tensor_requires_grad=_DEFERRED_VALUE,
            output_index=None,
            is_bottom_level_func=func_event_input.is_bottom_level_func,
            time_since_pass_start=0.0,
            sample_id=None,
            label_raw=reserved.label_raw,
            label_prefix=reserved.layer_type,
            func_call_id=func_event_input.func_call_id,
            parent_labels_raw=(),
            is_output_parent=False,
            backend_requires_isolation=False,
            is_scalar_bool=_DEFERRED_VALUE,
            bool_value=_DEFERRED_VALUE,
        )

    def detect_backend_semantics(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        output: object,
    ) -> BackendSemantics:
        """Return MLX backend semantics for one output."""

        return BackendSemantics(
            backend_grad_handle=None,
            grad_fn_class_name=None,
            autograd_memory=None,
            num_autograd_tensors=None,
            mutated_input_positions=(),
            aliased_output_inputs=(),
            unknown_aliasing=False,
            bytes_delta_at_call=None,
            bytes_peak_at_call=None,
        )

    def tensor_ref(
        self,
        session: object,
        value: object,
        payload: object | None,
        policy: CapturePolicy,
    ) -> TensorRef:
        """Build metadata for an MLX array without forcing materialization."""

        if not self.is_tensor(value):
            return TensorRef("", None, None, None, None, None, payload, None, None)
        return TensorRef(
            label_raw=self.tensor_store.get_label(value) or "",
            shape=self._shape(value),
            dtype=self._dtype(value),
            device=self._device(value),
            requires_grad=None,
            memory=self._memory(value),
            payload=payload,
            blob_ref=None,
            backend_handle_id=str(id(value)),
        )

    def set_tensor_label(self, session: object, value: object, label: str) -> None:
        """Set the raw TorchLens label for an MLX array."""

        if self.is_tensor(value):
            self.tensor_store.set_label(value, label)

    def is_tensor(self, value: object) -> bool:
        """Return whether ``value`` is an MLX array."""

        array_type = getattr(self.mx, "array", None)
        return array_type is not None and isinstance(value, array_type)

    def is_parameter(self, value: object) -> bool:
        """Return whether ``value`` is an MLX parameter-like array."""

        return self.is_tensor(value)

    def apply_live_hooks(
        self,
        session: object,
        value: object,
        site: ReservedLabel,
    ) -> tuple[object, tuple[FireResult, ...]]:
        """Return MLX values unchanged because live intervention is out of scope."""

        return value, ()

    def safe_copy(self, session: object, value: object, policy: CapturePolicy) -> object:
        """Return an MLX payload reference for deferred materialization."""

        return value

    def copy_replacement_metadata(self, session: object, src: object, dst: object) -> None:
        """Copy MLX side-table labels between replacement arrays."""

        label = self.tensor_store.get_label(src)
        if label is not None:
            self.tensor_store.set_label(dst, label)

    def emit_function_outputs(
        self,
        session: object,
        func_event_input: FunctionEventInput,
        isolated_output: object,
        output_sites: tuple[object, ...],
        reserved_block: tuple[ReservedLabel, ...],
        *,
        fire_results_by_site: dict[int, tuple[FireResult, ...]] | None = None,
    ) -> tuple[OpEvent, ...]:
        """Emit topology-complete Protocol operation events for MLX outputs.

        MLX eval timing is deliberately split: shape, dtype, device, parent
        edges, and container paths are populated at call time; saved payloads
        remain lazy and are batch-forced in ``finalize_forward_session`` via
        ``mx.eval``; value-dependent predicate fields are deferred with the
        ``_DEFERRED_VALUE`` sentinel and projected to ``None`` for stored event
        metadata.
        """

        events: list[OpEvent] = []
        policy = self._capture_policy(session)
        output_by_site = tuple(output_sites)
        for site_index, (output, reserved) in enumerate(
            zip(output_by_site, reserved_block, strict=True)
        ):
            if not self.is_tensor(output):
                continue
            self.tensor_store.set_label(output, reserved.label_raw)
            if policy.save_payload:
                getattr(session, "_mlx_saved_payloads").append(output)
            parents, parent_arg_positions, edge_uses = self._parent_edges(
                func_event_input.args,
                dict(func_event_input.kwargs),
            )
            fire_results = (fire_results_by_site or {}).get(site_index, ())
            event = self._build_event(
                session=session,
                kind="op",
                reserved=reserved,
                func_event_input=func_event_input,
                output=output,
                parents=parents,
                parent_arg_positions=parent_arg_positions,
                edge_uses=edge_uses,
                policy=policy,
                is_input=False,
                fire_results=fire_results,
            )
            events.append(event)
        return tuple(events)

    def finalize_forward_session(
        self,
        session: object,
        trace_state: RawGraphWorkspace,
    ) -> None:
        """Materialize deferred MLX payloads in a single batch."""

        del trace_state
        payloads = getattr(session, "_mlx_saved_payloads", ())
        if payloads:
            cast(Any, self.mx).eval(*payloads)

    def capture_trace(
        self,
        model: object,
        input_args: object,
        input_kwargs: dict[Any, Any] | None = None,
        *,
        layers_to_save: str | list[Any] | None = "all",
        keep_orphans: bool = False,
        output_device: str = "same",
        activation_transform: object | None = None,
        save_raw_activations: bool = True,
        detach_saved_activations: bool = False,
        save_grads: bool | str | list[Any] | object | None = None,
        random_seed: int | None = None,
        num_context_lines: int = 7,
        save_arg_values: bool = False,
        save_code_context: bool = False,
        save_rng_states: bool = False,
        recurrence_detection: bool = True,
        compute_input_output_distances: bool = True,
        verbose: bool = False,
        backward_ready: bool = False,
        name: str | None = None,
        module_filter: object | None = None,
        transform: object | None = None,
        raw_input: object | None = None,
        save_raw_input: str | bool = "small",
        batch_render: str = "auto",
        output_transform: object | None = None,
        save_raw_output: str | bool = "small",
        layer_visualizers: dict[Any, Any] | None = None,
        save_visualizations: bool = False,
        module_identity_mode: str | None = None,
        grad_options: GradOptions | None = None,
        intervene: object | None = None,
        halt: object | None = None,
    ) -> Trace:
        """Capture an MLX forward pass into a smoke-compatible Trace."""

        reject_unsupported_trace_options(
            {
                "layers_to_save": layers_to_save,
                "activation_transform": activation_transform,
                "detach_saved_activations": detach_saved_activations,
                "save_arg_values": save_arg_values,
                "save_grads": save_grads,
                "save_code_context": save_code_context,
                "save_rng_states": save_rng_states,
                "backward_ready": backward_ready,
                "module_filter": module_filter,
                "transform": transform,
                "output_device": output_device,
                "layer_visualizers": layer_visualizers,
                "save_visualizations": save_visualizations,
            },
            MLX_PREVIEW_TRACE_OPTION_POLICY,
            spec=get_backend_spec("mlx"),
        )
        if random_seed is not None:
            raise BackendUnsupportedError(
                "MLX backend preview does not support random_seed; pass explicit MLX RNG "
                "state through the model/input surface instead."
            )
        intervention_plan = None
        halt_selector = None
        if intervene is not None or halt is not None:
            spec = get_backend_spec("mlx")
            if not spec.capabilities.interventions:
                raise BackendUnsupportedError(
                    "MLX backend capability table declares interventions=False; "
                    "refusing trace(intervene=/halt=) instead of silently "
                    "ignoring the requested behavior."
                )
            # The registered binding IS the dispatch surface: resolution and
            # application come from the capability implementation, so a bound-
            # but-unregistered or in-place-flipped table refuses typed.
            resolve = cast(Any, require_capability_implementation(spec, "interventions"))
            intervention_plan, halt_selector = resolve(intervene, halt, self.mx)
            if grad_options is not None:
                raise BackendUnsupportedError(
                    "MLX backend cannot combine grad_options with intervene=/halt=: "
                    "the derived-gradient replay re-runs the model without the "
                    "intervention surface, so its output-divergence oracle would "
                    "always refuse. Capture the intervened trace and the derived "
                    "gradients in separate calls."
                )
        transform_type = _mlx_traced_transform_type(self.mx)
        if transform_type is not None and type(model) is transform_type:
            raise BackendUnsupportedError(
                "MLX capture entry is an mx.compile/mx.grad/mx.vmap traced-transform "
                "wrapper. Traced replays bypass the eager capture surface, so logging "
                "them would silently under-capture. Trace the eager mlx.nn.Module or "
                "plain Python callable instead of its compiled/transformed wrapper."
            )
        compiled_attribute_paths = _find_mlx_compiled_attributes(model, transform_type)
        module_tree = discover_mlx_module_tree(model)
        use_object_module = _resolve_mlx_module_identity_mode(module_identity_mode, module_tree)
        trace = Trace(
            model_class_name=type(model).__name__,
            output_device=output_device,
            activation_transform=cast("Callable[[Any], Any] | None", activation_transform),
            grad_transform=None,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=True,
            keep_orphans=keep_orphans,
            save_arg_values=save_arg_values,
            save_grads=None,
            detach_saved_activations=detach_saved_activations,
            mark_layer_depths=compute_input_output_distances,
            num_context_lines=num_context_lines,
            optimizer=None,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            module_filter=cast("Callable[[Any], bool] | None", module_filter),
            emit_nvtx=False,
            transform=cast("Callable[[Any], Any] | None", transform),
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=cast("Callable[[Any], Any] | None", output_transform),
            save_raw_output=save_raw_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
        )
        trace.trace_label = name
        trace.backend = cast(BackendName, self.name)
        if compiled_attribute_paths:
            # Conservative ceiling, not a refusal: the attribute may never be
            # called, but a call would bypass wrapper capture (cold trace) or
            # replay a cached graph (warm), so honesty cannot depend on it.
            names = ", ".join(compiled_attribute_paths)
            warnings.warn(
                "MLX model holds mx.compile/traced-transform attribute(s) "
                f"({names}); their interiors are not logged, so this capture "
                "is marked capture_verified=False.",
                UserWarning,
                stacklevel=2,
            )
            trace.capture_verified = False
            trace.capture_verification_reason = "mlx_compiled_attribute_not_logged"
        trace.capture_events = CaptureEvents()
        trace._mlx_saved_payloads = []
        trace._mlx_capture_depth = 0
        trace._mlx_module_stack = []
        trace._mlx_intervention_plan = intervention_plan
        trace._mlx_halt_selector = halt_selector
        trace._pre_forward_rng_states = None
        setattr(
            trace,
            "random_seed",
            cast(int, random_seed) if random_seed is not None else random.randint(1, 4294967294),
        )
        self.tensor_store.clear()
        # R07: the try owns the wrap call itself -- a raise anywhere between
        # wrapper install and the forward (session prep, input normalization,
        # source labeling) used to strand the process-global MLX wrappers
        # because the unwrap-owning finally had not been entered yet.
        # ``unwrap`` on a partially-installed (or empty) registry is safe.
        try:
            self.wrap(model, module_tree if use_object_module else None)
            self.prepare_model_session(trace, model)
            args = self._normalize_input_args(input_args)
            kwargs = {} if input_kwargs is None else dict(input_kwargs)
            self._label_source_arrays(trace, args, kwargs)
            trace.capture_start_time = time.time()
            halt_signal: HaltSignal | None = None
            try:
                with self.active_logging(trace):
                    output = cast(Any, model)(*args, **kwargs)
            except HaltSignal as signal:
                # Save-then-halt ordering: the frontier op's events were
                # appended before the signal, so the partial graph ends at it.
                halt_signal = signal
                output = signal.frontier_output
                trace.halted = True
                trace.halt_reason = signal.reason
                trace.halt_frontier = signal.reason
            # Forward-time-only state: the emit path reads these during the
            # model call; leaving them on the trace breaks the portable-state
            # scrub (they are not in PORTABLE_STATE_SPEC by design).
            del trace._mlx_intervention_plan
            del trace._mlx_halt_selector
            # Zero-match disclosure (preview half of the torch-side fix): a
            # selector matching nothing produced a trace byte-identical to
            # plain capture with no warning and no counter anywhere.
            if intervention_plan is not None and not trace.__dict__.pop(
                "_mlx_intervene_fired", False
            ):
                warnings.warn(
                    "Capture-time intervention selector matched zero sites on the mlx "
                    "forward; no intervention fired (check the selector's op/module name).",
                    UserWarning,
                    stacklevel=2,
                )
            if halt_selector is not None and halt_signal is None:
                warnings.warn(
                    "Capture-time halt selector matched zero sites on the mlx forward; "
                    "the capture ran the full forward and completed without halting.",
                    UserWarning,
                    stacklevel=2,
                )
            trace.forward_duration = Duration(time.time() - trace.capture_start_time)
            if halt_signal is not None:
                trace.raw_output = None
            else:
                trace.raw_output = output_transform(output) if callable(output_transform) else None
            self.finalize_forward_session(trace, trace._raw_graph_ws)
            self._mark_outputs(trace, output)
            materialize_from_events(trace, trace.capture_events)
            delattr(trace, "capture_events")
            if use_object_module and module_tree is not None:
                trace.param_logs = ParamAccessor(mlx_param_logs(module_tree, trace))
                trace.num_param_tensors = len(trace.param_logs)
                trace.num_params = sum(param.num_params for param in trace.param_logs)
                trace.num_params_trainable = sum(
                    param.num_params for param in trace.param_logs if param.is_trainable
                )
                trace.num_params_frozen = trace.num_params - trace.num_params_trainable
                trace.param_source = "native-module"
            else:
                trace.param_logs = ParamAccessor({})
                trace.num_param_tensors = 0
                trace.num_params = 0
                trace.num_params_trainable = 0
                trace.num_params_frozen = 0
                trace.param_source = "none"
            self._finish_trace(trace, module_tree if use_object_module else None)
            if halt_signal is not None:
                self._restrict_halted_param_logs(trace)
            if grad_options is not None:
                self._attach_derived_grads(
                    trace=trace,
                    model=cast(Callable[..., Any], model),
                    args=args,
                    kwargs=kwargs,
                    captured_output=output,
                    grad_options=grad_options,
                )
            if hasattr(trace, "_mlx_module_stack"):
                delattr(trace, "_mlx_module_stack")
            freeze_trace_relation_views(trace)
        finally:
            # Independently-owned resources: a raising session cleanup must
            # not leave the process-global MLX wrappers installed.
            try:
                self.cleanup_model_session(trace, model)
            finally:
                self.unwrap(model)
        # Settlement is the LAST act, after ALL teardown (the path-20 stamp
        # contract): a teardown raise escapes productless -- the object
        # derives UNATTESTED, never carrying a COMPLETE/HALTED stamp.
        stamp_backend_finalized(trace)
        return trace

    def _restrict_halted_param_logs(self, trace: Trace) -> None:
        """Restrict a halted trace's parameter accounting to captured ops.

        A halted partial graph never ran the modules past the frontier, so
        declaring the full model's parameter inventory would break the
        trace/op self-consistency invariant (and overstate what the partial
        capture proved). Only parameters attached to captured ops survive.

        Parameters
        ----------
        trace
            Finalized halted trace.
        """

        attached = {
            barcode
            for op in trace.layer_list
            for barcode in (getattr(op, "_param_barcodes", None) or ())
        }
        kept = {
            address: param for address, param in trace.params.items() if param.barcode in attached
        }
        trace.param_logs = ParamAccessor(kept)
        trace.num_param_tensors = len(kept)
        trace.num_params = sum(param.num_params for param in kept.values())
        trace.num_params_trainable = sum(
            param.num_params for param in kept.values() if param.is_trainable
        )
        trace.num_params_frozen = trace.num_params - trace.num_params_trainable

    def _attach_derived_grads(
        self,
        *,
        trace: Trace,
        model: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[Any, Any],
        captured_output: Any,
        grad_options: GradOptions,
    ) -> None:
        """Compute and attach MLX leaf-level derived gradients.

        Parameters
        ----------
        trace
            Trace receiving derived gradient records.
        model
            Captured MLX callable.
        args
            Positional call arguments used for capture.
        kwargs
            Keyword call arguments used for capture.
        captured_output
            Raw output from the captured forward call.
        grad_options
            MLX derived-gradient options.

        Returns
        -------
        None
            ``trace.derived_grads`` and unambiguous param gradient slots are populated.
        """

        params = grad_options.params
        if params is None:
            parameters = getattr(model, "parameters", None)
            if callable(parameters):
                params = parameters()
        has_params = _has_mlx_array_leaf(params)
        input_grad_argnums = _normalize_mlx_input_grad_argnums(
            grad_options.input_grad_argnums,
            len(args),
        )
        if not has_params and not input_grad_argnums:
            raise ValueError(
                "MLX derived gradients require module params or at least one input argnum."
            )

        value_args: tuple[Any, ...]
        differentiated_argnums: tuple[int, ...]
        param_argnum: int | None
        input_argnum_by_value_argnum: dict[int, int]
        if has_params:
            value_args = (params, *args)
            param_argnum = 0
            input_argnum_by_value_argnum = {index + 1: index for index in input_grad_argnums}
            differentiated_argnums = (0, *(index + 1 for index in input_grad_argnums))
        else:
            value_args = tuple(args)
            param_argnum = None
            input_argnum_by_value_argnum = {index: index for index in input_grad_argnums}
            differentiated_argnums = tuple(input_grad_argnums)

        original_params = params if has_params else None
        tap_observer: _MLXIntermediateTapObserver | None = None

        def value_fn(*value_fn_args: Any) -> tuple[Any, Any]:
            """Return scalar loss plus raw output aux for ``mx.value_and_grad``.

            Parameters
            ----------
            *value_fn_args
                Positional values passed by MLX AD.

            Returns
            -------
            tuple[Any, Any]
                Scalar loss and raw model output.
            """

            offset = 0
            if has_params:
                update = getattr(model, "update", None)
                if not callable(update):
                    raise BackendUnsupportedError(
                        "MLX derived gradients require model.update(params) for parameter "
                        "rebinding."
                    )
                update(value_fn_args[0])
                offset = 1
            if tap_observer is not None:
                tap_observer.label_call_inputs(value_fn_args[offset:], kwargs)
            raw_output = model(*value_fn_args[offset:], **dict(kwargs))
            loss = grad_options.loss_fn(raw_output) if grad_options.loss_fn else raw_output
            if not _is_scalar_mlx_value(loss):
                raise ValueError(
                    "MLX derived gradients require loss_fn(raw_output) to be scalar unless "
                    "the traced output is already scalar."
                )
            return loss, raw_output

        grad_fn = cast(Any, self.mx).value_and_grad(
            value_fn,
            argnums=differentiated_argnums[0]
            if len(differentiated_argnums) == 1
            else differentiated_argnums,
        )
        try:
            if grad_options.intermediate_grads:
                tap_observer = _MLXIntermediateTapObserver(self, trace)
                with mlx_tap_observer(tap_observer):
                    (_loss, aux_output), grads = grad_fn(*value_args)
            else:
                (_loss, aux_output), grads = grad_fn(*value_args)
            _eval_mlx_tree(self.mx, aux_output)
            _eval_mlx_tree(self.mx, captured_output)
            if not _mlx_trees_close(aux_output, captured_output):
                raise ValueError(
                    "MLX derived gradient run raw output diverged from captured raw output; "
                    "refusing to expose trace.derived_grads."
                )
            grad_trees = grads if len(differentiated_argnums) != 1 else (grads,)
            records: dict[str, DerivedGradRecord] = {}
            for value_argnum, grad_tree in zip(differentiated_argnums, grad_trees, strict=True):
                records.update(
                    _records_for_mlx_grad_tree(
                        grad_tree=grad_tree,
                        argnum=value_argnum,
                        param_argnum=param_argnum,
                        input_argnum_by_value_argnum=input_argnum_by_value_argnum,
                        provenance={
                            "backend": "mlx",
                            "kind": "derived_gradient",
                            "mechanism": "mlx_value_and_grad",
                            "loss_fn": _callable_identity(grad_options.loss_fn),
                        },
                    )
                )
            if grad_options.intermediate_grads and tap_observer is not None:
                trace.intermediate_derived_grads = self._records_for_intermediate_mlx_grads(
                    trace=trace,
                    candidates=tap_observer.candidates,
                    value_args=value_args,
                    kwargs=kwargs,
                    value_fn=value_fn,
                    grad_options=grad_options,
                )
        finally:
            if has_params:
                update = getattr(model, "update", None)
                if callable(update):
                    update(original_params)
        trace.derived_grads = DerivedGradAccessor(records)
        mirror_param_derived_grads(trace, records)

    def _records_for_intermediate_mlx_grads(
        self,
        *,
        trace: Trace,
        candidates: Sequence[_MLXIntermediateCandidate],
        value_args: tuple[Any, ...],
        kwargs: Mapping[Any, Any],
        value_fn: Callable[..., tuple[Any, Any]],
        grad_options: GradOptions,
    ) -> IntermediateDerivedGradAccessor:
        """Build exact MLX op-level derived gradient records.

        Parameters
        ----------
        trace
            Finalized trace whose saved ops define the public attachment set.
        candidates
            Tap candidates observed during the auxiliary AD replay.
        value_args
            Arguments passed to ``mx.value_and_grad``.
        kwargs
            Original keyword arguments passed to the model.
        value_fn
            Scalar-loss function used by the leaf-gradient replay.
        grad_options
            MLX derived-gradient options.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Exact, oracle-confirmed records keyed by pass-qualified op label.
        """

        selected_ops = tuple(
            op
            for op in trace.layer_list
            if op.has_saved_activation and not op.is_input and _is_float_mlx_array(op.out)
        )
        if len(selected_ops) > grad_options.max_intermediate_grads:
            raise BackendUnsupportedError(
                "MLX intermediate derived gradients are capped at "
                f"{grad_options.max_intermediate_grads} saved boundaries; got "
                f"{len(selected_ops)}."
            )
        if not selected_ops:
            return IntermediateDerivedGradAccessor()
        _eval_mlx_tree(
            self.mx,
            tuple(candidate.grad for candidate in candidates if candidate.grad is not None),
        )
        trace_groups = _mlx_trace_intermediate_signatures(selected_ops)
        replay_groups: dict[_MLXIntermediateSignature, list[_MLXIntermediateCandidate]] = (
            defaultdict(list)
        )
        for candidate in candidates:
            replay_groups[candidate.signature].append(candidate)

        records: dict[str, IntermediateDerivedGradRecord] = {}
        for signature, ops in trace_groups.items():
            if len(ops) != 1:
                continue
            replay_candidates = replay_groups.get(signature, [])
            if len(replay_candidates) != 1:
                continue
            op = ops[0]
            candidate = replay_candidates[0]
            if candidate.grad is None:
                continue
            if not _mlx_intermediate_oracle_passes(
                backend=self,
                trace=trace,
                signature=signature,
                value=candidate.value,
                producer_grad=candidate.grad,
                value_args=value_args,
                kwargs=kwargs,
                value_fn=value_fn,
            ):
                continue
            records[op.label] = IntermediateDerivedGradRecord(
                op_label=op.label,
                layer_label=op.layer_label,
                aval=(
                    f"array(shape={tuple(getattr(candidate.grad, 'shape', ()))}, "
                    f"dtype={getattr(candidate.grad, 'dtype', None)})"
                ),
                dtype_ref=DtypeRef(backend="mlx", name=str(getattr(candidate.grad, "dtype", ""))),
                grad=candidate.grad,
                provenance={
                    "backend": "mlx",
                    "kind": "intermediate_derived_gradient",
                    "mechanism": "mlx_custom_vjp_tap_value_and_grad",
                    "loss_id": _callable_identity(grad_options.loss_fn),
                    "save_predicate_id": "trace.saved_ops",
                    "status": "exact",
                    "oracle": "producer_custom_vjp+boundary_replacement_grad+perturbation",
                    "max_intermediate_grads": grad_options.max_intermediate_grads,
                },
            )
        return IntermediateDerivedGradAccessor(records)

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Capture then validate an MLX forward pass.

        Parameters
        ----------
        *args, **kwargs
            Public validation arguments forwarded to ``capture_trace``.

        Returns
        -------
        bool
            True when live replay validation passes.
        """

        validate_metadata = bool(kwargs.pop("validate_metadata", True))
        trace = self.capture_trace(*args, **kwargs)
        result = self.validate_trace(trace, validate_metadata=validate_metadata)
        if isinstance(result, ValidationReplayStatus):
            return result.passed
        return result

    def validate_trace(
        self,
        trace: Trace,
        *_args: Any,
        **kwargs: Any,
    ) -> bool | ValidationReplayStatus:
        """Validate an MLX trace with per-op replay and perturbation.

        Parameters
        ----------
        trace
            MLX trace to validate.
        *_args
            Ignored compatibility arguments.
        **kwargs
            Compatibility keyword arguments. ``validate_metadata`` controls
            whether backend-neutral invariant checks run.

        Returns
        -------
        bool or ValidationReplayStatus
            True for a verified live pass, False for a verified live failure,
            or an explicit unavailable status for loaded payload-stripped
            traces.
        """

        status = trace.validation_replay_status
        if not status.available or _mlx_loaded_replay_unavailable(trace):
            status_result = ValidationReplayStatus.unavailable_loaded_runtime_stripped(
                backend=self.name,
                payload_load_status=getattr(trace, "payload_load_status", None),
            )
            setattr(trace, "_validation_replay_status", status_result)
            return status_result
        replayed_count = 0
        failed_count = 0
        try:
            from ...validation.invariants import check_metadata_invariants
            from ...validation.status import count_importer_region_annotations
            from .validation import validate_mlx_captures

            perturbation_gaps: tuple[str, ...] = ()
            if kwargs.get("validate_metadata", True) and not check_metadata_invariants(trace):
                failed_count = 1
            else:
                replayed_count, failed_count, perturbation_gaps = validate_mlx_captures(trace)
            if failed_count == 0 and replayed_count < 1:
                failed_count = 1
            unverified_count = 0
            reason_counts: dict[str, int] | None = None
            if failed_count == 0:
                # Perturbation gaps are honest UNVERIFIED evidence, never a
                # silent pass: a constant producer's parent dependency cannot
                # be perturbation-proven.
                unverified_count = count_importer_region_annotations(trace) + len(perturbation_gaps)
                if perturbation_gaps:
                    reason_counts = {
                        "mlx_perturbation_no_perturbable_input": len(perturbation_gaps)
                    }
            status_result = ValidationReplayStatus.from_replay_counts(
                backend=self.name,
                source=_mlx_validation_source(trace),
                replayed_node_count=replayed_count,
                unverified_node_count=unverified_count,
                failed_node_count=failed_count,
                payload_load_status=getattr(trace, "payload_load_status", None),
                unverified_reason_counts=reason_counts,
            )
        except Exception:
            status_result = ValidationReplayStatus.result(
                passed=False,
                backend=self.name,
                source=_mlx_validation_source(trace),
                payload_load_status=getattr(trace, "payload_load_status", None),
                replayed_node_count=replayed_count,
                failed_node_count=max(1, failed_count),
            )
        setattr(trace, "_validation_replay_status", status_result)
        return status_result if status_result.state == "unverified" else status_result.passed

    def emit_mlx_operation(
        self,
        trace: Trace,
        op_name: str,
        func: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
        *,
        module_stack: tuple[ModuleFrame, ...] | None = None,
    ) -> Any:
        """Append one MLX operation event and return the effective output.

        Returns
        -------
        Any
            The call output the model should consume downstream: the raw
            output, or the tree with intervention-replaced leaves.
        """

        # Container outputs (e.g. mx.split's list of arrays) materialize one
        # op per array leaf; dropping them here would silently disconnect the
        # graph (downstream consumers lose their parents) while keeping the
        # call absent from BOTH sides of the replay inventory, so validation
        # would bless the missing wiring.
        outputs = tuple(self._iter_arrays(output))
        if not outputs:
            return output
        events = getattr(trace, "capture_events", None)
        if events is None:
            events = CaptureEvents()
            trace.capture_events = events
        reserved = events.reserve_label_block(op_name, len(outputs))
        func_call_id = events.func_call_id_counter + 1
        events.func_call_id_counter = func_call_id
        func_event_input = FunctionEventInput(
            func=func,
            func_name=op_name,
            func_qualname=getattr(func, "__qualname__", None),
            args=args,
            kwargs=kwargs,
            raw_output=output,
            arg_copies=None,
            kwarg_copies=None,
            module_stack=module_stack or tuple(getattr(trace, "_mlx_module_stack", ())),
            is_bottom_level_func=True,
            func_call_id=func_call_id,
            expected_output_count=len(outputs),
        )
        # Static-label interventions substitute matched leaves BEFORE labeling
        # and payload capture, so the recorded op output IS the value flowing
        # into downstream ops; halt matches after the frontier op is captured
        # (save-then-halt ordering).
        plan = getattr(trace, "_mlx_intervention_plan", None)
        halt_selector = getattr(trace, "_mlx_halt_selector", None)
        fire_results_by_site: dict[int, tuple[FireResult, ...]] = {}
        intervention_slots: tuple[tuple[int, str], ...] = ()
        intervention_appliers: tuple[tuple[int, Any], ...] = ()
        halt_label: str | None = None
        if plan is not None or halt_selector is not None:
            from .interventions import selector_matches_capture_context

            replacements: dict[int, Any] = {}
            slots: list[tuple[int, str]] = []
            appliers: list[tuple[int, Any]] = []
            for index, (leaf, entry) in enumerate(zip(outputs, reserved, strict=True)):
                record_ctx = self.build_record_context(trace, entry, func_event_input, leaf)
                if plan is not None and selector_matches_capture_context(plan.selector, record_ctx):
                    replacement = self._apply_mlx_intervention(plan, leaf)
                    trace._mlx_intervene_fired = True
                    replacements[id(leaf)] = replacement
                    fire_results_by_site[index] = (
                        FireResult(
                            plan_id="mlx_intervene",
                            site_label=entry.label_raw,
                            fired_at_capture_index=entry.raw_index,
                            pre_hook_shape=self._shape(leaf),
                            post_hook_shape=self._shape(replacement),
                            pre_hook_dtype=self._dtype(leaf),
                            post_hook_dtype=self._dtype(replacement),
                            replaced=True,
                            fire_record=None,
                        ),
                    )
                    slots.append((index, plan.applier.identity))
                    appliers.append((index, plan.applier.apply))
                if (
                    halt_selector is not None
                    and halt_label is None
                    and selector_matches_capture_context(halt_selector, record_ctx)
                ):
                    halt_label = entry.label_raw
            if replacements:
                output = _replace_mlx_array_leaves(self, output, replacements)
                outputs = tuple(self._iter_arrays(output))
                func_event_input = replace(func_event_input, raw_output=output)
                intervention_slots = tuple(slots)
                intervention_appliers = tuple(appliers)
        op_captures = getattr(trace, "_mlx_op_captures", None)
        if op_captures is None:
            op_captures = []
            trace._mlx_op_captures = op_captures
        labels_raw = tuple(entry.label_raw for entry in reserved)
        # Parent labels are recorded per array leaf BEFORE outputs are labeled,
        # so aliasing outputs cannot shadow their own parents.
        arg_leaf_labels = tuple(
            tuple(self.tensor_store.get_label(leaf) for leaf in self._iter_arrays(value))
            for value in args
        )
        kwarg_leaf_labels = {
            key: tuple(self.tensor_store.get_label(leaf) for leaf in self._iter_arrays(value))
            for key, value in kwargs.items()
        }
        # Template retention (paddle's _template_value model): labeled leaves
        # become REPLAY_SLOT sentinels because replay sources them from saved
        # parent payloads; keeping the raw arrays here pinned every
        # intermediate activation for the trace's lifetime.
        op_captures.append(
            MLXOpCapture(
                labels_raw=labels_raw,
                op_name=op_name,
                func=func,
                args=tuple(
                    build_capture_template(
                        value,
                        arg_leaf_labels[index] if index < len(arg_leaf_labels) else (),
                    )
                    for index, value in enumerate(args)
                ),
                kwargs={
                    key: build_capture_template(value, kwarg_leaf_labels.get(key, ()))
                    for key, value in kwargs.items()
                },
                arg_leaf_labels=arg_leaf_labels,
                kwarg_leaf_labels=kwarg_leaf_labels,
                interventions=intervention_slots,
                appliers=intervention_appliers,
            )
        )
        # Independent replay inventory: the validation oracle's denominator.
        # Losing a subset of _mlx_op_captures can then never shrink the
        # expected-coverage set along with the evidence. Each record also
        # snapshots the emit-time per-leaf parent labels, so stripping a
        # capture's recorded provenance (to launder emit-time argument values
        # through replay) fails coverage instead of replaying vacuously.
        # Appended to a list — O(1) per op on the capture hot path; wholesale
        # attribute replacement is coherent reauthoring, outside the threat
        # model, so a tuple rebuild bought no protection.
        inventory = getattr(trace, "_mlx_replay_inventory", None)
        if inventory is None:
            inventory = []
            trace._mlx_replay_inventory = inventory
        # The 5th element pins emit-time intervention declarations, so a
        # post-hoc claim that a value was (or was not) intervened mismatches
        # the immutable fingerprint instead of steering hook re-application.
        inventory.append(
            (
                op_name,
                labels_raw,
                arg_leaf_labels,
                tuple(sorted((key, labels) for key, labels in kwarg_leaf_labels.items())),
                intervention_slots,
            )
        )
        emitted = self.emit_function_outputs(
            trace,
            func_event_input,
            output,
            outputs,
            reserved,
            fire_results_by_site=fire_results_by_site,
        )
        events.extend(emitted)
        if halt_label is not None:
            # F6 latch parity with torch's evaluate_halt: the stop request is
            # latched on the trace BEFORE the signal is raised, so a user
            # broad-except that swallows the HaltSignal can never reach the
            # settlement stamp as a blessable COMPLETE -- the one preview
            # stamp checks the latch at the capture boundary.
            trace.__dict__["_stop_requested"] = StopRequest(
                kind="halt",
                reason=halt_label,
                boundary_label=halt_label,
            )
            raise HaltSignal(halt_label, frontier_output=output)
        return output

    def _apply_mlx_intervention(self, plan: Any, leaf: Any) -> Any:
        """Apply one resolved intervention to a matched output leaf.

        Parameters
        ----------
        plan
            Resolved ``MLXInterventionPlan``.
        leaf
            Matched MLX output array.

        Returns
        -------
        Any
            Replacement array with identical shape and dtype.

        Raises
        ------
        BackendUnsupportedError
            If the hook result is not an MLX array or changes shape/dtype;
            the MLX preview keeps a strict same-shape/same-dtype contract so
            downstream lazy consumers cannot silently mis-broadcast.
        """

        replacement = plan.applier.apply(leaf)
        if not self.is_tensor(replacement):
            raise BackendUnsupportedError(
                f"MLX intervention {plan.applier.identity!r} returned "
                f"{type(replacement).__name__}; hooks must return an mx.array."
            )
        if self._shape(replacement) != self._shape(leaf) or self._dtype(replacement) != self._dtype(
            leaf
        ):
            raise BackendUnsupportedError(
                f"MLX intervention {plan.applier.identity!r} changed the output "
                f"from shape={self._shape(leaf)} dtype={self._dtype(leaf)} to "
                f"shape={self._shape(replacement)} dtype={self._dtype(replacement)}; "
                "the MLX preview supports shape- and dtype-preserving "
                "interventions only."
            )
        return replacement

    @staticmethod
    def _import_mlx() -> tuple[object, object]:
        """Import MLX lazily.

        Returns
        -------
        tuple[object, object]
            ``mlx.core`` and ``mlx.nn`` modules.
        """

        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError as exc:
            raise ImportError("MLX backend requires the optional 'mlx' package.") from exc
        return mx, nn

    @contextmanager
    def _paused(self) -> Any:
        """Temporarily pause MLX logging."""

        with _state.pause_logging():
            yield

    def _normalize_input_args(self, input_args: object) -> list[Any]:
        """Normalize user MLX input arguments to a positional list."""

        if isinstance(input_args, list):
            return input_args
        if isinstance(input_args, tuple):
            return list(input_args)
        return [input_args]

    def _label_source_arrays(self, trace: Trace, args: list[Any], kwargs: dict[Any, Any]) -> None:
        """Emit resolvable input source events for MLX source arrays."""

        for index, arg in enumerate(args):
            if self.is_tensor(arg):
                label = f"input.arg_{index}"
                self.tensor_store.set_label(arg, label)
                raw_index = trace.capture_events.raw_layer_counter + 1
                trace.capture_events.raw_layer_counter = raw_index
                trace.capture_events.append(
                    self._build_source_event(trace, label, arg, raw_index=raw_index)
                )
        for key, value in kwargs.items():
            if self.is_tensor(value):
                label = f"input.{key}"
                self.tensor_store.set_label(value, label)
                raw_index = trace.capture_events.raw_layer_counter + 1
                trace.capture_events.raw_layer_counter = raw_index
                trace.capture_events.append(
                    self._build_source_event(
                        trace,
                        label,
                        value,
                        raw_index=raw_index,
                    )
                )

    def _capture_policy(self, session: object) -> CapturePolicy:
        """Return the MLX capture policy for one event."""

        return CapturePolicy(
            save_payload=bool(getattr(session, "save_raw_activations", True)),
            save_grad=False,
        )

    def _build_source_event(
        self,
        trace: Trace,
        label: str,
        output: object,
        *,
        raw_index: int,
    ) -> OpEvent:
        """Build an MLX source ``OpEvent`` for an input array."""

        reserved = ReservedLabel(
            label=label,
            label_raw=label,
            raw_index=raw_index,
            type_index=raw_index,
            layer_type="input",
            site=label,
        )
        return self._build_event(
            session=trace,
            kind="source",
            reserved=reserved,
            func_event_input=FunctionEventInput(
                func=None,
                func_name="input",
                func_qualname=None,
                args=(),
                kwargs={},
                raw_output=output,
                arg_copies=None,
                kwarg_copies=None,
                module_stack=(),
                is_bottom_level_func=True,
                func_call_id=raw_index,
                expected_output_count=1,
            ),
            output=output,
            parents=(),
            parent_arg_positions={"args": {}, "kwargs": {}},
            edge_uses=(),
            policy=self._capture_policy(trace),
            is_input=True,
        )

    def _build_event(
        self,
        *,
        session: object,
        kind: str,
        reserved: ReservedLabel,
        func_event_input: FunctionEventInput,
        output: object,
        parents: tuple[ParentEdge, ...],
        parent_arg_positions: dict[str, dict[Any, str]],
        edge_uses: tuple[object, ...],
        policy: CapturePolicy,
        is_input: bool,
        fire_results: tuple[FireResult, ...] = (),
    ) -> OpEvent:
        """Build one topology-complete MLX operation event."""

        tensor_ref = self.tensor_ref(
            session,
            output,
            output if policy.save_payload else None,
            policy,
        )
        input_ancestors = frozenset(
            edge.parent_label_raw for edge in parents if edge.parent_label_raw.startswith("input.")
        )
        return OpEvent(
            kind=kind,
            label_raw=reserved.label_raw,
            layer_label_raw=reserved.label_raw,
            layer_type=reserved.layer_type,
            raw_index=reserved.raw_index,
            type_index=reserved.type_index,
            step_index=reserved.raw_index,
            source_trace=session,
            source_trace_id=None,
            tracing_finished=False,
            construction_done=True,
            function=FunctionCallRef(
                func=func_event_input.func,
                func_name=func_event_input.func_name,
                func_qualname=func_event_input.func_qualname,
                func_call_id=func_event_input.func_call_id,
                code_context=(),
                func_duration=None,
                flops_forward=None,
                flops_backward=None,
                func_rng_states=None,
                func_autocast_state=None,
                arg_names=(),
                num_args_total=len(func_event_input.args) + len(func_event_input.kwargs),
                num_pos_args=len(func_event_input.args),
                num_kwargs=len(func_event_input.kwargs),
                non_tensor_pos_args=tuple(
                    arg for arg in func_event_input.args if not self.is_tensor(arg)
                ),
                non_tensor_kwargs=tuple(
                    (key, value)
                    for key, value in func_event_input.kwargs.items()
                    if not self.is_tensor(value)
                ),
                func_non_tensor_args=tuple(
                    arg for arg in func_event_input.args if not self.is_tensor(arg)
                ),
                is_inplace=False,
                func_config=(),
            ),
            output=OutputRef(
                tensor=tensor_ref,
                transformed_tensor=None,
                has_saved_activation=policy.save_payload,
                output_device="same",
                activation_transform=getattr(session, "activation_transform", None),
                detach_saved_activations=bool(getattr(session, "detach_saved_activations", False)),
                visualizer_path=None,
                multi_output_index=None,
                in_multi_output=False,
                container_path=(),
                container_spec=None,
                child_versions=(),
            ),
            templates=ArgTemplateRef(
                saved_args=None,
                saved_kwargs=None,
                args_template=None,
                kwargs_template=None,
                has_saved_args=False,
            ),
            parents=parents,
            parent_arg_positions=parent_arg_positions,
            _edge_uses=edge_uses,
            params=(),
            parent_params=(),
            module_stack=func_event_input.module_stack,
            modules=tuple(
                (frame.address, frame.call_index) for frame in func_event_input.module_stack
            ),
            backend_semantics=self.detect_backend_semantics(session, func_event_input, output),
            policy=policy,
            predicate_matched=policy.save_payload,
            pass_index=1,
            grad_fn_class_qualname=None,
            grad_fn_handle=None,
            equivalence_class=reserved.layer_type,
            is_transform=False,
            transform_kind=None,
            transform_chain=(),
            transform_config={},
            transform_fn_name=None,
            transform_fn_qualname=None,
            transform_fn_source=None,
            unattributed_tensor_args=(),
            dropped_edge_tensor_args=(),
            is_output_parent=False,
            has_internal_source_ancestor=not is_input and not parents,
            internal_source_ancestors=frozenset(),
            input_ancestors=input_ancestors,
            root_ancestors=input_ancestors or frozenset({reserved.label_raw}),
            func_call_id=func_event_input.func_call_id,
            is_bottom_level=func_event_input.is_bottom_level_func,
            is_scalar_bool=None,
            bool_value=None,
            intervention_fired=bool(fire_results),
            intervention_replaced=any(result.replaced for result in fire_results),
            fire_results=fire_results,
            intervention_template_ref=None,
            record_context=self.build_record_context(
                session,
                reserved,
                func_event_input,
                output,
            ),
            capture_spec=CaptureSpec(save_out=policy.save_payload, save_metadata=True),
        )

    def _parent_edges(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[ParentEdge, ...], dict[str, dict[Any, str]], tuple[object, ...]]:
        """Return parent edges and arg-position metadata for MLX inputs."""

        edges: list[ParentEdge] = []
        arg_positions: dict[Any, str] = {}
        kwarg_positions: dict[Any, str] = {}
        edge_uses: list[tuple[str, Any, str]] = []

        def _add(label: str, position: Any, use: str) -> None:
            """Append one unique parent edge."""

            if any(edge.parent_label_raw == label for edge in edges):
                return
            edges.append(ParentEdge(parent_label_raw=label, arg_position=position, edge_use=use))
            edge_uses.append((label, position, use))

        for index, value in enumerate(args):
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None:
                    arg_positions[index] = label
                    _add(label, index, "arg")
        for key, value in kwargs.items():
            for array in self._iter_arrays(value):
                label = self.tensor_store.get_label(array)
                if label is not None:
                    kwarg_positions[key] = label
                    _add(label, key, "kwarg")
        return tuple(edges), {"args": arg_positions, "kwargs": kwarg_positions}, tuple(edge_uses)

    def _mark_outputs(self, trace: Trace, output: object) -> None:
        """Mark final output-parent operations for an MLX trace."""

        for value in self._iter_arrays(output):
            label = self.tensor_store.get_label(value)
            if label is None:
                continue
            mark_output_label(trace, label)

    def _finish_trace(self, trace: Trace, module_tree: MLXModuleTree | None = None) -> None:
        """Finalize a manually captured MLX Trace.

        Parameters
        ----------
        trace
            Trace to finalize.
        module_tree
            Discovered object-module tree, if object-module mode is active.

        Returns
        -------
        None
            Trace accessors are populated in place.
        """

        # The MLX validation sidecars (`_mlx_op_captures`, `_mlx_replay_inventory`)
        # speak RAW label space on both the build and consume sides, and raw
        # labels are never rewritten by recurrence grouping (they stay resolvable
        # through `_label_raw` with recurrence-safe key precedence in
        # `_ops_by_label`), so no relabel hook is needed here.
        finalize_single_pass_trace(
            trace,
            backend_name=self.name,
            module_tree=module_tree,
            attach_function_root_module=attach_function_root_module,
            attach_object_module_logs=self._attach_object_module_logs,
            attach_op_params=attach_module_owned_op_params,
            count_layers_with_attached_params=True,
            recurrence_detection=bool(getattr(trace, "recurrence_detection", False)),
        )

    def _attach_object_module_logs(self, trace: Trace, tree: MLXModuleTree) -> None:
        """Build public module logs for an MLX object-module trace.

        Parameters
        ----------
        trace
            Trace receiving module accessors.
        tree
            Discovered MLX object-module tree.

        Returns
        -------
        None
            ``trace.modules`` is populated by the shared module-log builder.
        """

        attach_object_module_logs(
            trace,
            tree,
            normalize_module_calls=normalize_op_module_calls,
            metadata_top_level=_mlx_metadata_top_level,
            op_top_level=_mlx_op_top_level,
            training_mode=_mlx_training_mode,
        )

    def _iter_arrays(self, value: object) -> list[object]:
        """Return MLX arrays nested inside ``value``."""

        if self.is_tensor(value):
            return [value]
        if isinstance(value, (list, tuple)):
            arrays: list[object] = []
            for item in value:
                arrays.extend(self._iter_arrays(item))
            return arrays
        if isinstance(value, dict):
            arrays = []
            for item in value.values():
                arrays.extend(self._iter_arrays(item))
            return arrays
        return []

    def _shape(self, value: object) -> tuple[int, ...] | None:
        """Return an MLX array shape without materializing data."""

        return tuple(cast(Any, value).shape) if self.is_tensor(value) else None

    def _dtype(self, value: object) -> str | None:
        """Return an MLX array dtype without materializing data."""

        return str(cast(Any, value).dtype) if self.is_tensor(value) else None

    def _device(self, value: object) -> str | None:
        """Return an MLX array device description without materializing data."""

        if not self.is_tensor(value):
            return None
        device = getattr(value, "device", None)
        return str(device) if device is not None else None

    def _memory(self, value: object) -> int | None:
        """Return MLX array memory in bytes without materializing data."""

        if not self.is_tensor(value):
            return None
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        size = getattr(value, "size", None)
        itemsize = getattr(value, "itemsize", None)
        if size is not None and itemsize is not None:
            return int(size) * int(itemsize)
        return None


def mlx_param_logs(tree: MLXModuleTree, trace: Trace) -> dict[str, Param]:
    """Build TorchLens parameter logs from an MLX module parameter tree.

    Parameters
    ----------
    tree
        Discovered MLX object-module tree.
    trace
        Trace receiving the parameter logs.

    Returns
    -------
    dict[str, Param]
        Parameter logs keyed by primary parameter address.
    """

    param_logs: dict[str, Param] = {}
    trainable_ids = {id(value) for _address, value in _iter_trainable_parameter_tree(tree.root)}
    for candidate in _iter_mlx_parameter_candidates(tree):
        existing_address = tree.param_address_by_id.get(id(candidate.value), candidate.address)
        if existing_address in param_logs:
            param = param_logs[existing_address]
            if candidate.address not in param.all_addresses:
                param.all_addresses.append(candidate.address)
            if candidate.address not in param.co_parent_params:
                param.co_parent_params.append(candidate.address)
            for alias in tree.metadata.get(candidate.owner, {}).get(
                "all_addresses", [candidate.owner]
            ):
                if alias not in param.all_module_addresses:
                    param.all_module_addresses.append(alias)
            continue
        shape = tuple(getattr(candidate.value, "shape", ()))
        dtype = str(getattr(candidate.value, "dtype", ""))
        param = Param(
            module_address=candidate.owner,
            name=candidate.address.rsplit(".", 1)[-1],
            shape=shape,
            dtype=cast(Any, dtype),
            num_params=_numel(shape),
            param_memory=_nbytes(candidate.value) or 0,
            trainable=id(candidate.value) in trainable_ids,
            address=existing_address,
            barcode=f"mlx:{existing_address}",
            has_optimizer=None,
        )
        param.dtype_ref = DtypeRef(backend="mlx", name=dtype)
        param.device_ref = DeviceRef.from_value(getattr(candidate.value, "device", None))
        param.backend_address = f"object:{existing_address}"
        param.resolver_status = "resolved"
        param._param_ref = cast(Any, candidate.value)
        param.source_trace = trace
        param.all_module_addresses = list(
            tree.metadata.get(candidate.owner, {}).get("all_addresses", [candidate.owner])
        )
        param_logs[existing_address] = param
    return param_logs


def _iter_mlx_parameter_candidates(tree: MLXModuleTree) -> list[MLXParameterCandidate]:
    """Return flattened MLX parameter candidates with primary owners.

    Parameters
    ----------
    tree
        Discovered MLX object-module tree.

    Returns
    -------
    list[MLXParameterCandidate]
        Parameter candidates.
    """

    alias_to_primary = _alias_to_primary(tree)
    candidates: list[MLXParameterCandidate] = []
    for param_address, value in _iter_parameter_tree(tree.root):
        owner_alias = param_address.rsplit(".", 1)[0] if "." in param_address else "self"
        owner = alias_to_primary.get(owner_alias, owner_alias)
        primary_param_address = _join_module_address(owner, param_address.rsplit(".", 1)[-1])
        candidates.append(
            MLXParameterCandidate(
                address=primary_param_address,
                value=value,
                owner=owner,
            )
        )
    return candidates


def _iter_trainable_parameter_tree(model: object) -> list[tuple[str, Any]]:
    """Return flattened MLX trainable parameter leaves.

    Parameters
    ----------
    model
        MLX module.

    Returns
    -------
    list[tuple[str, Any]]
        Trainable parameter addresses and arrays.
    """

    trainable_parameters = getattr(model, "trainable_parameters", None)
    if not callable(trainable_parameters):
        return []
    return list(_flatten_parameter_tree(trainable_parameters(), ""))


def _iter_parameter_tree(model: object) -> list[tuple[str, Any]]:
    """Return flattened MLX parameter leaves.

    Parameters
    ----------
    model
        MLX module.

    Returns
    -------
    list[tuple[str, Any]]
        Parameter addresses and arrays.
    """

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return []
    return list(_flatten_parameter_tree(parameters(), ""))


def _flatten_parameter_tree(value: object, prefix: str) -> Iterator[tuple[str, Any]]:
    """Yield array leaves from an MLX parameter tree.

    Parameters
    ----------
    value
        Parameter tree value.
    prefix
        Dotted address prefix.

    Yields
    ------
    tuple[str, Any]
        Parameter address and array.
    """

    if _is_mlx_array(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = _join_module_address(prefix, str(key)) if prefix else str(key)
            yield from _flatten_parameter_tree(item, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = _join_module_address(prefix, str(index)) if prefix else str(index)
            yield from _flatten_parameter_tree(item, child_prefix)


def _is_mlx_array(value: object) -> bool:
    """Return whether ``value`` is an MLX array.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when ``value`` is an MLX array.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def _alias_to_primary(tree: MLXModuleTree) -> dict[str, str]:
    """Return module alias to primary address mapping.

    Parameters
    ----------
    tree
        Discovered MLX module tree.

    Returns
    -------
    dict[str, str]
        Alias-to-primary mapping.
    """

    aliases: dict[str, str] = {}
    for primary, metadata in tree.metadata.items():
        for alias in metadata.get("all_addresses", [primary]):
            aliases[str(alias)] = primary
    return aliases


def _mlx_metadata_top_level(
    address: str,
    metadata: dict[str, Any],
    metadata_by_address: dict[str, dict[str, Any]],
) -> bool:
    """Return whether an MLX metadata address is top-level.

    Parameters
    ----------
    address:
        Module address from the discovered MLX module tree.
    metadata:
        Metadata for ``address``, unused by MLX.
    metadata_by_address:
        Complete module metadata mapping.

    Returns
    -------
    bool
        True when the nearest discovered parent is ``"self"``.
    """

    del metadata
    return address != "self" and nearest_metadata_parent(address, metadata_by_address) == "self"


def _mlx_op_top_level(address: str) -> bool:
    """Return whether an MLX op module address is top-level.

    Parameters
    ----------
    address:
        Module address observed in an op call stack.

    Returns
    -------
    bool
        True for every non-root first frame, matching the previous MLX loop.
    """

    return address != "self"


def _mlx_training_mode(metadata: dict[str, Any]) -> bool:
    """Return MLX module training state from metadata.

    Parameters
    ----------
    metadata:
        Module metadata from ``MLXModuleTree``.

    Returns
    -------
    bool
        Stored training-state flag, defaulting to ``False``.
    """

    return bool(metadata.get("training", False))


def _resolve_mlx_module_identity_mode(
    value: str | None,
    module_tree: MLXModuleTree | None,
) -> bool:
    """Return whether MLX should use object-module attribution.

    Parameters
    ----------
    value
        Public ``module_identity_mode`` value after missing normalization.
    module_tree
        Discovered MLX module tree, if any.

    Returns
    -------
    bool
        True when object-module mode should be used.
    """

    if value not in {None, "function_root", "object_module"}:
        raise BackendUnsupportedError(
            "MLX module_identity_mode must be None, 'function_root', or 'object_module'."
        )
    if value == "object_module" and module_tree is None:
        raise BackendUnsupportedError(
            "MLX module_identity_mode='object_module' requires an mlx.nn.Module object. "
            "Raw callables use module_identity_mode='function_root'."
        )
    if value == "function_root":
        return False
    return module_tree is not None


def _normalize_mlx_input_grad_argnums(
    input_grad_argnums: Sequence[int],
    num_args: int,
) -> tuple[int, ...]:
    """Validate MLX input-relative gradient argnums.

    Parameters
    ----------
    input_grad_argnums
        User-supplied positional input indexes.
    num_args
        Number of positional model inputs.

    Returns
    -------
    tuple[int, ...]
        Normalized unique input argnums.
    """

    normalized = tuple(int(index) for index in input_grad_argnums)
    if len(set(normalized)) != len(normalized):
        raise ValueError("MLX derived gradient input_grad_argnums must be unique.")
    for index in normalized:
        if index < 0 or index >= num_args:
            raise ValueError("MLX derived gradient input_grad_argnums are out of range.")
    return normalized


def _has_mlx_array_leaf(value: Any) -> bool:
    """Return whether ``value`` contains at least one MLX array leaf.

    Parameters
    ----------
    value
        Candidate tree.

    Returns
    -------
    bool
        True when an MLX array appears in ``value``.
    """

    return any(True for _path, _leaf in _flatten_mlx_array_tree(value, ""))


def _flatten_mlx_array_tree(value: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    """Yield MLX array leaves from a nested tree.

    Parameters
    ----------
    value
        Tree value to flatten.
    prefix
        Dotted path prefix.

    Yields
    ------
    tuple[str, Any]
        Local dotted path and array leaf.
    """

    if _is_mlx_array(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = _join_module_address(prefix, str(key)) if prefix else str(key)
            yield from _flatten_mlx_array_tree(item, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = _join_module_address(prefix, str(index)) if prefix else str(index)
            yield from _flatten_mlx_array_tree(item, child_prefix)


def _records_for_mlx_grad_tree(
    *,
    grad_tree: Any,
    argnum: int,
    param_argnum: int | None,
    input_argnum_by_value_argnum: Mapping[int, int],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for one differentiated MLX arg tree.

    Parameters
    ----------
    grad_tree
        Gradient tree returned by MLX AD.
    argnum
        Backend positional argnum for this gradient tree.
    param_argnum
        Backend positional argnum containing params, if present.
    input_argnum_by_value_argnum
        Mapping from backend value argnum to user input argnum.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path.
    """

    records: dict[str, DerivedGradRecord] = {}
    for local_path, grad in _flatten_mlx_array_tree(grad_tree, ""):
        if argnum == param_argnum:
            source = "params"
            input_argnum = None
            full_path = f"params.{local_path}" if local_path else "params"
        else:
            source = "inputs"
            input_argnum = input_argnum_by_value_argnum[argnum]
            prefix = f"inputs.{input_argnum}"
            full_path = f"{prefix}.{local_path}" if local_path else prefix
        records[full_path] = DerivedGradRecord(
            path=full_path,
            source=source,
            argnum=argnum,
            input_argnum=input_argnum,
            aval=f"array(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="mlx", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _mlx_trace_intermediate_signatures(
    ops: Sequence[Any],
) -> dict[_MLXIntermediateSignature, list[Any]]:
    """Group finalized MLX ops by conservative intermediate signatures.

    Parameters
    ----------
    ops
        Saved, non-input op logs selected for intermediate derived gradients.

    Returns
    -------
    dict[_MLXIntermediateSignature, list[Any]]
        Trace ops grouped by attachment signature.

    Notes
    -----
    Replay-side signatures speak RAW label space (the tap observer labels
    values with ``_label_raw``), so recurrence-grouped parents (rewritten to
    final labels at finalize -- the pass-qualified ``op.label`` for a
    multi-pass referenced layer, but the BARE ``op.layer_label`` (torch
    parity) for a single-pass one) are resolved back to raw space here. Both
    final spellings must resolve: the pass-qualified key always, and the
    bare key too for single-pass ops (unambiguous there; omitted for
    multi-pass ops, where the bare label would collide across passes and
    ``parents`` never uses it anyway). Without this, every grouped
    intermediate would silently fail to match its replay candidate.
    """

    final_to_raw: dict[str, str] = {}
    for op in ops:
        source_trace = op._source_trace_or_none()
        if source_trace is None:
            continue
        for trace_op in getattr(source_trace, "layer_list", ()):
            label_raw = getattr(trace_op, "_label_raw", None)
            if not isinstance(label_raw, str):
                continue
            label = getattr(trace_op, "label", None)
            if isinstance(label, str):
                final_to_raw[label] = label_raw
            layer_label = getattr(trace_op, "layer_label", None)
            if isinstance(layer_label, str) and int(getattr(trace_op, "num_passes", 1)) == 1:
                final_to_raw[layer_label] = label_raw
        break
    grouped: dict[_MLXIntermediateSignature, list[Any]] = defaultdict(list)
    for op in ops:
        signature = _MLXIntermediateSignature(
            op_name=str(op.func_name or op.layer_type),
            call_ordinal=int(op.func_call_id or 0),
            parent_labels=tuple(
                final_to_raw.get(str(parent), str(parent)) for parent in getattr(op, "parents", ())
            ),
            shape=tuple(getattr(op, "shape", ()) or ()),
            dtype=str(getattr(op, "dtype", "")),
            module_calls=tuple(str(module) for module in getattr(op, "modules", ())),
        )
        grouped[signature].append(op)
    return grouped


def _mlx_intermediate_oracle_passes(
    *,
    backend: MLXBackend,
    trace: Trace,
    signature: _MLXIntermediateSignature,
    value: Any,
    producer_grad: Any,
    value_args: tuple[Any, ...],
    kwargs: Mapping[Any, Any],
    value_fn: Callable[..., tuple[Any, Any]],
) -> bool:
    """Return whether replacement-gradient and perturbation checks pass.

    Parameters
    ----------
    backend
        Active MLX backend instance.
    trace
        Finalized trace used to align replay labels.
    signature
        Boundary signature being validated.
    value
        Boundary value from the tap replay.
    producer_grad
        Candidate cotangent produced by the custom-VJP tap.
    value_args
        Arguments for ``value_fn``.
    kwargs
        Original keyword arguments passed to the model.
    value_fn
        Scalar-loss function used by the derived-gradient replay.

    Returns
    -------
    bool
        True only when independent replacement AD and perturbation agree.
    """

    mx = cast(Any, backend.mx)

    def loss_from_replacement(replacement: Any) -> Any:
        """Return scalar loss with one replay boundary replaced.

        Parameters
        ----------
        replacement
            Replacement boundary value.

        Returns
        -------
        Any
            Scalar MLX loss.
        """

        observer = _MLXBoundaryReplacementObserver(backend, trace, signature, replacement)
        observer.label_call_inputs(_mlx_model_args_from_value_args(value_args, trace), kwargs)
        with mlx_tap_observer(observer):
            loss, _raw_output = value_fn(*value_args)
        if not observer.replaced:
            raise ValueError("MLX intermediate replacement oracle did not find the target.")
        return loss

    try:
        reference_grad = mx.grad(loss_from_replacement)(value)
        mx.eval(reference_grad, producer_grad)
        if not _mlx_values_close(producer_grad, reference_grad):
            return False
        perturbation = _mlx_perturbation_for(value)
        perturbed_grad = mx.grad(loss_from_replacement)(value + perturbation)
        expected_perturbed_grad = mx.grad(loss_from_replacement)(value + perturbation)
        mx.eval(perturbed_grad, expected_perturbed_grad)
    except Exception:
        return False
    if not _mlx_values_close(perturbed_grad, expected_perturbed_grad):
        return False
    return not _mlx_values_close(reference_grad, perturbed_grad)


def _mlx_model_args_from_value_args(value_args: tuple[Any, ...], trace: Trace) -> tuple[Any, ...]:
    """Return model input args from AD value args.

    Parameters
    ----------
    value_args
        Arguments passed to ``mx.value_and_grad``.
    trace
        Finalized trace whose parameter source indicates whether arg 0 is params.

    Returns
    -------
    tuple[Any, ...]
        Positional model input arguments.
    """

    if getattr(trace, "param_source", "none") == "native-module" and len(value_args) > 1:
        return value_args[1:]
    return value_args


def _mlx_perturbation_for(value: Any) -> Any:
    """Return a small dtype-scaled MLX perturbation for a boundary value.

    Parameters
    ----------
    value
        Boundary MLX array.

    Returns
    -------
    Any
        Perturbation array with the same shape as ``value``.
    """

    import mlx.core as mx

    dtype_name = str(getattr(value, "dtype", ""))
    epsilon = 1e-2 if "float16" in dtype_name or "bfloat16" in dtype_name else 1e-3
    return mx.ones_like(value) * epsilon


def _mlx_parent_replay_labels(
    backend: MLXBackend,
    labels_by_id: Mapping[int, str],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return raw parent labels for replay call arguments.

    Parameters
    ----------
    backend
        MLX backend used to find array leaves.
    labels_by_id
        Replay-side array id to raw-label mapping.
    args
        Positional call arguments.
    kwargs
        Keyword call arguments.

    Returns
    -------
    tuple[str, ...]
        Unique parent labels in first-seen order.
    """

    labels: list[str] = []
    for value in args:
        for leaf in backend._iter_arrays(value):
            label = labels_by_id.get(id(leaf))
            if label is not None and label not in labels:
                labels.append(label)
    for value in kwargs.values():
        for leaf in backend._iter_arrays(value):
            label = labels_by_id.get(id(leaf))
            if label is not None and label not in labels:
                labels.append(label)
    return tuple(labels)


def _replace_mlx_array_leaves(
    backend: MLXBackend,
    value: Any,
    replacements: Mapping[int, Any],
) -> Any:
    """Replace MLX array leaves in a nested output tree.

    Parameters
    ----------
    backend
        MLX backend used for tensor checks.
    value
        Output tree.
    replacements
        Mapping from original array ``id`` to replacement array.

    Returns
    -------
    Any
        Output tree with matching array leaves replaced.
    """

    if backend.is_tensor(value):
        return replacements.get(id(value), value)
    if isinstance(value, tuple):
        return tuple(_replace_mlx_array_leaves(backend, item, replacements) for item in value)
    if isinstance(value, list):
        return [_replace_mlx_array_leaves(backend, item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_mlx_array_leaves(backend, item, replacements)
            for key, item in value.items()
        }
    return value


def _mlx_identity_tap(mx: Any, candidate: _MLXIntermediateCandidate, value: Any) -> Any:
    """Return an MLX custom-VJP identity tap for one candidate.

    Parameters
    ----------
    mx
        Imported ``mlx.core`` module.
    candidate
        Candidate receiving the VJP cotangent.
    value
        Boundary value to pass through unchanged.

    Returns
    -------
    Any
        Identity value with a custom VJP side channel.
    """

    @mx.custom_function
    def tap(input_value: Any) -> Any:
        """Return ``input_value`` unchanged."""

        return input_value

    @tap.vjp
    def tap_vjp(primals: tuple[Any, ...], cotangent: Any, output: Any) -> tuple[Any]:
        """Record the cotangent and pass it through unchanged."""

        del primals, output
        candidate.grad = cotangent
        return (cotangent,)

    return tap(value)


def _is_float_mlx_array(value: Any) -> bool:
    """Return whether ``value`` is an MLX floating or complex array.

    Parameters
    ----------
    value
        Candidate value.

    Returns
    -------
    bool
        True for MLX float, bfloat, or complex arrays.
    """

    if not _is_mlx_array(value):
        return False
    dtype_name = str(getattr(value, "dtype", ""))
    return "float" in dtype_name or "bfloat" in dtype_name or "complex" in dtype_name


def _max_mlx_input_raw_index(trace: Trace) -> int:
    """Return the highest raw index consumed by MLX input source ops.

    Parameters
    ----------
    trace
        Finalized MLX trace.

    Returns
    -------
    int
        Maximum source-op step index, or zero when no source input exists.
    """

    indexes = [
        int(getattr(op, "step_index", 0) or 0)
        for op in trace.layer_list
        if getattr(op, "is_input", False)
    ]
    return max(indexes, default=0)


def _is_scalar_mlx_value(value: Any) -> bool:
    """Return whether an MLX value is scalar-shaped.

    Parameters
    ----------
    value
        Candidate MLX value.

    Returns
    -------
    bool
        True when ``value`` has shape ``()``.
    """

    return tuple(getattr(value, "shape", ())) == ()


def _eval_mlx_tree(mx: Any, value: Any) -> None:
    """Force all MLX array leaves in ``value``.

    Parameters
    ----------
    mx
        Imported ``mlx.core`` module.
    value
        Tree whose array leaves should be evaluated.

    Returns
    -------
    None
        MLX evaluation has been requested for all leaves.
    """

    leaves = [leaf for _path, leaf in _flatten_mlx_array_tree(value, "")]
    if leaves:
        mx.eval(*leaves)


def _mlx_trees_close(left: Any, right: Any) -> bool:
    """Return whether two MLX output trees are numerically close.

    Parameters
    ----------
    left
        Left output tree.
    right
        Right output tree.

    Returns
    -------
    bool
        True when both trees have matching structure and close leaves.
    """

    if _is_mlx_array(left) or _is_mlx_array(right):
        if not (_is_mlx_array(left) and _is_mlx_array(right)):
            return False
        return _mlx_values_close(left, right)
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not (isinstance(left, tuple) and isinstance(right, tuple)):
            return False
        return len(left) == len(right) and all(
            _mlx_trees_close(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, list) or isinstance(right, list):
        if not (isinstance(left, list) and isinstance(right, list)):
            return False
        return len(left) == len(right) and all(
            _mlx_trees_close(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return False
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_mlx_trees_close(left[key], right[key]) for key in left)
    return left == right


def _mlx_values_close(left: Any, right: Any) -> bool:
    """Return whether two MLX arrays are numerically close.

    Parameters
    ----------
    left
        Left MLX array.
    right
        Right MLX array.

    Returns
    -------
    bool
        True when arrays have matching shape, dtype, and values.
    """

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if np.issubdtype(left_array.dtype, np.bool_) or np.issubdtype(left_array.dtype, np.integer):
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
        left_array.dtype,
        np.complexfloating,
    ):
        # Per-dtype ULP-derived bands (ported paddle/mlx validation-oracle
        # derivation); ``np.finfo`` reports component precision for complex.
        rtol, atol = float_replay_tolerances(np.finfo(left_array.dtype))
        return bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


__all__ = ["GradOptions", "MLXBackend"]
