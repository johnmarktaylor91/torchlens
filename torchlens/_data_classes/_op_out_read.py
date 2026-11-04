"""Helpers behind the ``Op.out`` read path: absent-value refusals, the mutation-check state and the synthesized-read warning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._errors import MutatedReferenceError
from .._utils._torch_compat import tensor_version_or_none


def _refuse_read_without_trace(
    op: Any, *, member: str | None = None, out_trace: Any = None
) -> None:
    """Refuse an Op read whose answer needs the owning Trace when that Trace is gone.

    A bound Op whose Trace was garbage-collected lost its saved values (the op
    store evicts payloads with its last owner) and its graph, so reads that need
    them would answer ``None`` or an untyped miss that reads as "no such value";
    a chained temporary such as ``tl.trace(model, x).output_ops[0].out`` loses
    its Trace at once. The refusal is ``Op.trace``'s own typed
    ``trace_reference_collected``. An Op never bound to a Trace (standalone
    pickle) keeps its own ``record_not_bound`` handling.

    Parameters
    ----------
    op:
        The Op being read.
    member:
        A public name whose lookup missed (``Op.__getattr__``): refuse only when
        it names a class member, so a typo keeps its ``AttributeError``.
    out_trace:
        The live Trace of an ``Op.out`` read that found no value: a Trace that
        kept reduced payloads only refuses through ``refuse_reduced_payload_read``.
    """

    if member is not None and not hasattr(type(op), member):
        return
    ref = op._slot("_source_trace_ref")
    if ref is not None and ref() is None:
        _ = op.trace
    if out_trace is not None and op._slot("_tracing_finished"):
        from ..stats._capture_stats import refuse_reduced_payload_read

        refuse_reduced_payload_read(op._slot, out_trace)


def _out_read_state(slot: Callable[..., Any], out: Any) -> dict[str, Any]:
    """The fields the reference-mutation check reads on an ``Op.out`` access."""

    return {
        "out": out,
        "annotations": slot("annotations"),
        "has_saved_activation": slot("has_saved_activation", False),
        "detach_saved_activations": slot("detach_saved_activations"),
        "label": slot("label"),
        "layer_label": slot("layer_label"),
        "_label_raw": slot("_label_raw"),
    }


def _synthesized_out_read_due(source_trace: Any) -> bool:
    """Whether an ``Op.out`` read on this trace owes the synthesized-value warning.

    Quickstart gate: the first raw value read on a trace whose input
    values were synthesized warns once per Trace. ``trace[label]`` returns the
    Op, so the read the quickstart documents (``log[...].out``) lands on
    ``Op.out``; the gate itself dedupes per trace.
    """

    if source_trace is None:
        return False
    # Instance-dict read: a partially built or loaded trace may lack the field,
    # and an AttributeError raised inside ``Op.__getattribute__`` would turn the
    # ``out`` read into a ``__getattr__`` fallback.
    record = source_trace.__dict__.get("_provenance_input_preprocessor")
    return (
        record is not None
        and not getattr(record, "verified", True)
        and getattr(record, "source", "") == "torchlens._quickstart.input_resolver"
    )


def _validate_reference_out_not_mutated(state: dict[str, Any]) -> None:
    """Raise if a reference-mode saved output has changed since capture."""

    annotations = state.get("annotations") or {}
    if annotations.get("save_mode") != "reference":
        return
    out = state.get("out")
    if not isinstance(out, torch.Tensor):
        return
    saved_version = annotations.get("saved_out_version")
    current_version = tensor_version_or_none(out)
    if saved_version is not None and current_version != saved_version:
        label = state.get("label") or state.get("layer_label") or state.get("_label_raw")
        raise MutatedReferenceError(
            f"saved reference out for op {label!r} was mutated after capture "
            f"(saved _version={saved_version}, current _version={current_version}). "
            "Use save_mode='copy' for isolated saved tensors."
        )
