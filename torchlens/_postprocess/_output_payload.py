"""Whether a model-output row rides its producer's retained payload (HONESTY 13-R1).

The output pseudo-row shares the producer's one retained payload when the
returned tensor equals it, unless the returned tensor gained autograd history
the payload lacks (:func:`output_gained_graph_after_log`).
"""

from __future__ import annotations

from typing import Any

import torch

from .._utils.tensor_utils import tensor_nanequal

__all__ = (
    "output_copy_detaches",
    "output_gained_graph_after_log",
    "output_payload_aliases_parent",
    "output_rides_parent_payload",
)


def output_payload_aliases_parent(returned: Any, retained: Any) -> bool:
    """True iff the returned output tensor IS the producer's retained payload value.

    The output pseudo-row may share the producer's ONE retained payload
    (HONESTY 13-R1) exactly when the tensor the model returned equals the
    producer's retained ``out`` bit-for-bit under the same shape, dtype and
    device (NaN-equal, so a NaN output is not a spurious "difference"). Any
    other case -- unretained producer, meta payloads, a cross-device
    ``output_device`` policy, an in-place mutation after the producer, a view
    whose base shape differs -- keeps the historical physical copy so the
    variation is recorded, never hidden.
    """

    if not (torch.is_tensor(returned) and torch.is_tensor(retained)):
        return False
    if returned.is_meta or retained.is_meta:
        return False
    if (
        tuple(returned.shape) != tuple(retained.shape)
        or returned.dtype != retained.dtype
        or str(returned.device) != str(retained.device)
    ):
        return False
    return bool(tensor_nanequal(returned, retained))


def output_gained_graph_after_log(returned: Any, retained: Any, *, detach_saved: bool) -> bool:
    """True iff the returned tensor has autograd history its producer's payload lacks.

    A producer logged inside an ``autograd.Function`` forward (a custom
    Function, or the kernel of torch's ``map`` operator) runs below autograd,
    so its payload has no ``grad_fn``; the Function sets the history on the
    returned tensor afterwards. Unless the trace detaches saved activations,
    that output keeps its own graph-connected copy, the handle ``backward()``
    differentiates through, instead of riding the payload.

    Parameters
    ----------
    returned:
        Tensor the model returned.
    retained:
        The producer's retained payload.
    detach_saved:
        Whether the trace detaches saved activations.

    Returns
    -------
    bool
        Whether the output row needs its own graph-connected copy.
    """

    return (
        not detach_saved
        and torch.is_tensor(returned)
        and returned.grad_fn is not None
        and torch.is_tensor(retained)
        and retained.grad_fn is None
    )


def output_rides_parent_payload(returned: Any, retained: Any, trace: Any) -> bool:
    """True iff the output row shares its producer's payload instead of its own copy.

    Parameters
    ----------
    returned:
        The output value.
    retained:
        The producer's retained payload.
    trace:
        The trace being built or refreshed.

    Returns
    -------
    bool
        The alias rule, unless the output gained a graph the payload lacks.
    """

    detach_saved = _trace_payloads_detached(trace)
    gained = output_gained_graph_after_log(returned, retained, detach_saved=detach_saved)
    return not gained and output_payload_aliases_parent(returned, retained)


def output_copy_detaches(returned: Any, retained: Any, trace: Any) -> bool:
    """True iff the output row's own copy is detached from the autograd graph.

    The copy inherits the producer payload's detachment, unless the returned
    tensor gained a graph the payload lacks (:func:`output_gained_graph_after_log`).

    Parameters
    ----------
    returned:
        Tensor the model returned.
    retained:
        The producer's retained payload.
    trace:
        The trace being built.

    Returns
    -------
    bool
        Whether the copy is taken detached.
    """

    detach_saved = _trace_payloads_detached(trace)
    if output_gained_graph_after_log(returned, retained, detach_saved=detach_saved):
        return False
    return not (torch.is_tensor(retained) and retained.grad_fn is not None)


def _trace_payloads_detached(trace: Any) -> bool:
    """True iff the trace stores detached saved activations.

    Set by ``detach_saved_activations``, and always for a Trace cooked from a
    Recording, whose sparse capture stores detached payloads.

    Parameters
    ----------
    trace:
        The trace being built or refreshed.

    Returns
    -------
    bool
        Whether its saved activations are detached.
    """

    if trace._options_detach_saved_activations:
        return True
    return trace._cooked_from is not None
