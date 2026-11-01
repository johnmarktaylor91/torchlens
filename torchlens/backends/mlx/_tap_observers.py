"""MLX intermediate-gradient tap observers and the array helpers they share.

The custom-VJP tap observer and the boundary-replacement observer drive the
opt-in exact intermediate derived-gradient replay; the module-level helpers
label replay inputs, swap array leaves and classify arrays. Kept apart from
the backend module so the capture Protocol implementation stays readable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._replay_frames import _MLXReplayModuleFrames

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from .backend import MLXBackend


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
        self.module_frames = _MLXReplayModuleFrames()
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
        module_address: str | None = None,
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
        module_address
            Primary address when the call is a discovered module call.

        Returns
        -------
        Any
            Original output with eligible array leaves replaced by identity taps.
        """

        pushed = self.module_frames.push(module_address)
        try:
            if op_name is None or self.depth > 0:
                return original(*args, **kwargs)
            self.depth += 1
            try:
                output = original(*args, **kwargs)
            finally:
                self.depth -= 1
            return self._tap_output(op_name, args, kwargs, output)
        finally:
            if pushed:
                self.module_frames.pop()

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
                    module_calls=self.module_frames.current(),
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
        self.module_frames = _MLXReplayModuleFrames()

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
        module_address: str | None = None,
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
        module_address
            Primary address when the call is a discovered module call.

        Returns
        -------
        Any
            Output tree with the target boundary replaced when matched.
        """

        pushed = self.module_frames.push(module_address)
        try:
            if op_name is None or self.depth > 0:
                return original(*args, **kwargs)
            self.depth += 1
            try:
                output = original(*args, **kwargs)
            finally:
                self.depth -= 1
            return self._replace_output(op_name, args, kwargs, output)
        finally:
            if pushed:
                self.module_frames.pop()

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
                module_calls=self.module_frames.current(),
            )
            replacement = self.replacement if signature == self.target_signature else leaf
            if signature == self.target_signature:
                self.replaced = True
            self.labels_by_id[id(leaf)] = raw_label
            self.labels_by_id[id(replacement)] = raw_label
            replacements[id(leaf)] = replacement
        return _replace_mlx_array_leaves(self.backend, output, replacements)


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
