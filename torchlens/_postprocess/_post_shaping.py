"""Save shaping for finished non-torch captures, through each backend's payload codec.

Torch shapes each payload while it saves it (``backends/torch/_ops_activations``):
the raw copy moves to ``output_device`` and is detached when
``detach_saved_activations`` is set, ``activation_transform`` runs on that copy,
and the raw payload stays public only when ``save_raw_activations`` is true or
no transform is set. A finished non-torch capture holds the same realized
payloads, so the same steps run here, in the same order, on every saved op:

- ``output_device``: a round trip through the backend's payload codec
  (``to_numpy`` then ``from_numpy(map_location=device)``);
- ``detach_saved_activations``: ``.detach()`` on payloads whose arrays carry
  autograd history and expose it (Paddle, tinygrad); JAX, TensorFlow and MLX
  arrays carry no autograd graph and have no ``detach``, so their saved
  payloads already meet torch's documented "detached from autograd history"
  semantics and this is an exact no-op;
- ``activation_transform``: ``apply_transform`` with torch's arguments,
  filling ``transformed_out`` and its shape, dtype and memory.

The caller (``_selective_save.apply_backend_post_plan``) keeps the raw payloads
replay-visible, renders ``layer_visualizers`` with torch's post-capture renderer
and refreshes the saved summary.
"""

from __future__ import annotations

__tl_layer__ = "L3"

from collections.abc import Callable
from typing import Any

from .._data_classes._op_transforms import apply_transform
from .._io.payload_codec import get_payload_codec
from .._quantities import Bytes
from .._utils._record_group_table import record_member


def apply_save_shaping(
    trace: Any, plan: Any, *, backend_name: str, set_field: Callable[[Any, str, Any], None]
) -> None:
    """Apply a plan's payload-shaping options to every saved payload of ``trace``.

    Parameters
    ----------
    trace:
        Finished non-torch trace.
    plan:
        ``BackendPostPlan`` carrying the shaping options.
    backend_name:
        Backend name; selects the payload codec.
    set_field:
        Op field setter that does not mark trace state dirty.

    Returns
    -------
    None
        Payload fields and trace option stamps update in place.
    """

    if reshapes_payloads(plan):
        codec = get_payload_codec(backend_name)
        for op in trace._layer_list:
            if record_member(op, "has_saved_activation", False) and op.out is not None:
                _shape_op(op, plan, codec, set_field)
    _stamp_trace_options(trace, plan)


def reshapes_payloads(plan: Any) -> bool:
    """Return whether any option changes the saved payloads themselves."""

    return (
        plan.output_device != "same"
        or bool(plan.detach_saved_activations)
        or plan.activation_transform is not None
    )


def _shape_op(op: Any, plan: Any, codec: Any, set_field: Callable[[Any, str, Any], None]) -> None:
    """Move, detach and transform one op's saved payload like torch's save path."""

    raw = op.out
    if plan.output_device != "same":
        raw = _to_device(raw, plan.output_device, codec)
    if plan.detach_saved_activations and callable(getattr(raw, "detach", None)):
        raw = raw.detach()
    set_field(op, "out", raw)
    if plan.activation_transform is None:
        return
    transformed = apply_transform(
        label=record_member(op, "_label_raw", None),
        raw_label=record_member(op, "_label_raw", None),
        func_name=record_member(op, "func_name", None),
        tensor=raw,
        transform=plan.activation_transform,
        transform_kind="activation",
        streaming_active=False,
    )
    set_field(op, "transformed_out", transformed)
    set_field(op, "transformed_out_shape", _shape_of(transformed, codec))
    set_field(op, "transformed_out_dtype", getattr(transformed, "dtype", None))
    set_field(op, "transformed_activation_memory", _memory_of(transformed, codec))
    if not plan.save_raw_activations:
        set_field(op, "out", None)


def _to_device(payload: Any, device: str, codec: Any) -> Any:
    """Return ``payload`` rebuilt on ``device`` through the backend's payload codec."""

    if not codec.can_encode(payload):
        return payload
    encoded = codec.to_numpy(payload)
    entry = codec.manifest_fields(payload, encoded)
    return codec.from_numpy(encoded.array, entry, map_location=device)


def _shape_of(value: Any, codec: Any) -> tuple[int, ...] | None:
    """Return the shape of a framework array, or ``None`` for non-array values."""

    if not codec.can_encode(value):
        return None
    return tuple(int(dim) for dim in value.shape)


def _memory_of(value: Any, codec: Any) -> Bytes | None:
    """Return the host byte size of a framework array, or ``None`` for non-array values."""

    if not codec.can_encode(value):
        return None
    return Bytes(int(codec.to_numpy(value).array.nbytes))


def _stamp_trace_options(trace: Any, plan: Any) -> None:
    """Record the applied shaping options on the trace like a torch capture does."""

    trace._options_activation_transform = plan.activation_transform
    trace._options_save_raw_activations = bool(plan.save_raw_activations)
    trace._options_output_device = plan.output_device
    trace._options_detach_saved_activations = bool(plan.detach_saved_activations)
    if plan.save_visualizations:
        trace.save_visualizations = True
