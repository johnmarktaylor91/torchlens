"""Helpers behind the ``Op.out`` read path: the mutation-check state and the synthesized-read warning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._errors import MutatedReferenceError
from .._utils._torch_compat import tensor_version_or_none


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

    Quickstart memo D7 (F17): the first raw value read on a trace whose input
    values were synthesized warns once per Trace. ``trace[label]`` returns the
    Op, so the read the quickstart documents (``log[...].out``) lands on
    ``Op.out``; the gate itself dedupes per trace.
    """

    record = source_trace._provenance_input_preprocessor
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
