"""Capture records the tinygrad backend keeps for validation and derived gradients.

Extracted from ``backend`` so that module stays under its size ceiling.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from ..._ir.refs import DtypeRef


@dataclass(frozen=True)
class TinygradUOpCapture:
    """Captured tinygrad UOp metadata for validation.

    Parameters
    ----------
    label_raw
        Raw TorchLens label for the materialized op.
    uop
        Source tinygrad UOp snapshot.
    op_name
        UOp operation name.
    parent_labels
        Raw parent labels captured from UOp source edges.
    parent_arg_positions
        UOp source positions for each raw parent label.
    payload_snapshot
        Realized tinygrad tensor copy saved during capture.
    edit
        For an op a ``trace(intervene=...)`` edit replaced, the edit's lazy
        ``tensor -> tensor`` application: ``uop`` is the unedited op and the
        payload is the edit applied to its value. ``None`` for every other op.
    """

    label_raw: str
    uop: Any
    op_name: str
    parent_labels: tuple[str, ...]
    parent_arg_positions: tuple[tuple[int, str], ...]
    payload_snapshot: Any
    edit: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class TinygradIntermediateCandidate:
    """Live retained tinygrad tensor candidate for one op-level gradient.

    Parameters
    ----------
    signature
        Conservative structural signature used for trace-op matching.
    tensor
        Live tinygrad Tensor retained from the no-realize forward pass.
    aval
        Human-readable abstract-value description.
    dtype_ref
        Backend-neutral dtype reference for the tensor.
    """

    signature: tuple[Any, ...]
    tensor: Any
    aval: str
    dtype_ref: DtypeRef | None


@dataclass
class TinygradCaptureSession:
    """Per-capture state the wave B hook modules read and write.

    One session per ``capture_trace`` call (set by ``begin_capture``). The
    random-state module (``_rng_capture.tinygrad``) attaches its draw log to
    it, and the native backward module (``_backward_native.tinygrad``) reads
    ``loss_fn`` and sets ``tinygrad_backward``. Each hook runs only when its
    owner's module is importable, so without them a capture is unchanged.

    Parameters
    ----------
    backward_ready
        Whether the capture prepares a native backward pass.
    loss_fn
        Loss callable for that pass, or ``None`` for a scalar output.
    raw_output
        The forward's return value, held until the single realize.
    uop_labels
        ``id(uop) -> raw label`` of the emitted capture.
    param_addresses
        ``id(uop) -> parameter address`` of an object-module capture.
    tinygrad_backward
        The backward scope the native backward module set, or ``None``.
    """

    backward_ready: bool = False
    loss_fn: Callable[[Any], Any] | None = None
    raw_output: Any = None
    uop_labels: dict[int, str] = field(default_factory=dict)
    param_addresses: Mapping[int, str] = field(default_factory=dict)
    tinygrad_backward: Any = None


def arm_backward(
    session: TinygradCaptureSession | None, backward_ready: object, grad_options: Any
) -> None:
    """Record whether the capture prepares a backward pass, and for which loss."""

    if session is not None:
        session.backward_ready = bool(backward_ready)
        session.loss_fn = getattr(grad_options, "loss_fn", None) if backward_ready else None


def hook_module(name: str) -> ModuleType | None:
    """Import ``torchlens.backends.<name>`` if that hook module exists, else ``None``.

    Only a missing module (or package) counts as absent; an error raised while
    importing a present module propagates.
    """

    qualified = f"torchlens.backends.{name}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as error:
        parts = qualified.split(".")
        absent = {".".join(parts[:index]) for index in range(3, len(parts) + 1)}
        if error.name in absent:
            return None
        raise


@contextmanager
def capture_scopes(session: TinygradCaptureSession | None) -> Iterator[None]:
    """Run the forward inside the random-state and backward scopes that exist.

    ``rng_scope`` logs every draw; ``backward_ready_scope`` opens only when the
    capture asked for a backward pass.
    """

    with ExitStack() as stack:
        if session is not None:
            rng = hook_module("_rng_capture.tinygrad")
            if rng is not None:
                stack.enter_context(rng.rng_scope(session))
            native = hook_module("_backward_native.tinygrad") if session.backward_ready else None
            if native is not None:
                stack.enter_context(native.backward_ready_scope(session))
        yield


def extra_realize_targets(
    session: TinygradCaptureSession | None, plan: Any, source_uops: Sequence[Any]
) -> tuple[Any, ...]:
    """Tensors a hook adds to the capture's single realize (backward gradients)."""

    if session is None or session.tinygrad_backward is None:
        return ()
    return tuple(session.tinygrad_backward.gradient_targets(plan, source_uops, session.raw_output))


def after_realize(session: TinygradCaptureSession | None) -> None:
    """Resolve logged draws once the single realize has run (``finalize_counters``)."""

    rng = hook_module("_rng_capture.tinygrad") if session is not None else None
    if rng is not None:
        rng.finalize_counters(session)


def fresh_label_map(session: TinygradCaptureSession | None) -> dict[int, str]:
    """A new ``id(uop) -> raw label`` map the emission fills, shared with ``session``."""

    labels: dict[int, str] = {}
    if session is not None:
        session.uop_labels = labels
    return labels


def retain_hook_state(trace: Any, session: TinygradCaptureSession | None, module_tree: Any) -> None:
    """Hand the finished trace to the native backward module and drop the forward refs.

    ``module_tree`` is the object-module tree (its ``param_address_by_uop_id``)
    or ``None`` for a function-root capture.
    """

    if session is None:
        return
    addresses = getattr(module_tree, "param_address_by_uop_id", None) or {}
    session.param_addresses = dict(addresses)
    native = hook_module("_backward_native.tinygrad") if session.tinygrad_backward else None
    if native is not None:
        native.retain_backward_state(trace, session)
    session.raw_output = None
