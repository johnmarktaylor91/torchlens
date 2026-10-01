"""Transformed-payload memory totals (explorer P4, lane F25).

Module-level read-time derivations over per-op ``transformed_*_memory``
fields. These deliberately live HERE, not as ``Trace`` properties: the
godobject surface oracle pins the legacy loaded surface byte-identical,
and every new public property on a record class changes that surface.
Submodule functions are the sanctioned home for derived reads
(``Do not add new top-level API names casually; use submodules``).

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..quantities import Bytes

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def total_transformed_activation_memory(trace: Trace) -> Bytes:
    """Total bytes of retained TRANSFORMED activation payloads.

    Read-time derivation over per-op ``transformed_activation_memory``
    (never persisted; works on legacy artifacts). Output pseudo-rows
    alias their producer's tensor and contribute nothing, mirroring
    ``Trace.total_activation_memory``'s identity partition.
    """

    return Bytes(
        sum(
            int(entry.transformed_activation_memory or 0)
            for entry in trace.layer_list
            if entry.layer_type != "output"
            and getattr(entry, "transformed_activation_memory", None) is not None
        )
    )


def total_transformed_gradient_memory(trace: Trace) -> Bytes:
    """Total bytes of retained TRANSFORMED gradient payloads.

    Read-time derivation over per-op ``transformed_gradient_memory``;
    never persisted. NOTE: ``transformed_gradient_memory`` is a
    first-pass-only mirror on multi-pass layers, so this total is a
    disclosed lower bound on recurrent graphs.
    """

    return Bytes(
        sum(
            int(entry.transformed_gradient_memory or 0)
            for entry in trace.layer_list
            if entry.layer_type != "output"
            and getattr(entry, "transformed_gradient_memory", None) is not None
        )
    )
