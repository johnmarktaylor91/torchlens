"""Scoped, NON-attaching payload reads for analytics over lazy artifacts.

The interactive ``op.out`` read on a lazily loaded trace MATERIALIZES the
blob and ATTACHES it to the op (user semantics: read once, keep it). An
analytics/agent layer that walks every site through that door turns one
bounded question into a whole-artifact resident set -- the attach hazard the
agent memo's P0 names. This module is the other door: verify, compute,
release. The returned tensor is the caller's only reference; nothing lands on
the op, so sequential site scans hold ONE payload at a time.

Also home to the manifest-declared byte accounting (P0 part 3): payload work
is budgeted BEFORE materialization from ``body_index`` declared bytes --
exact and torch-free, never a post-hoc RSS observation.
"""

from __future__ import annotations

from typing import Any

from .._errors import PayloadUnavailableError


def read_op_payload(op: Any, *, map_location: Any = "cpu") -> Any:
    """Read one op's saved payload WITHOUT attaching it to the op.

    Parameters
    ----------
    op:
        Op/Layer record.
    map_location:
        Target device for a materialized tensor.

    Returns
    -------
    Any
        The resident payload when already materialized, otherwise a freshly
        materialized (sha-verified) tensor that is NOT cached on the op.

    Raises
    ------
    torchlens.errors.PayloadUnavailableError
        ``payload_unavailable`` when the op retains neither a resident
        payload nor a lazy blob ref.
    torchlens._io.TorchLensIOError
        When the referenced blob is missing, corrupt, or checksum-drifted.
    """

    slot = getattr(op, "_slot", None)
    resident = slot("out") if callable(slot) else None
    if resident is not None:
        return resident
    ref = slot("out_ref") if callable(slot) else getattr(op, "out_ref", None)
    if ref is not None:
        # LazyActivationRef.materialize sha256-verifies before decode and
        # closes the blob file; the result is returned WITHOUT _internal_set.
        return ref.materialize(map_location=map_location)
    label = getattr(op, "label", None) or getattr(op, "layer_label", "unknown")
    raise PayloadUnavailableError(
        f"op {label} retains no payload (not saved, or values were stripped "
        "at save time); nothing to read",
        code="payload_unavailable",
        remedy=(
            "re-capture with save= covering this op, or save the artifact with include_outs=True"
        ),
        label=str(label),
    )


#: Torch-free transport dtype byte widths for manifest ``body_index`` entries
#: (dtype strings like ``"torch.float32"``). Unknown dtypes charge 8 bytes per
#: element -- a budget gate must overestimate, never underestimate.
_DTYPE_NBYTES: dict[str, int] = {
    "float64": 8,
    "double": 8,
    "float32": 4,
    "float": 4,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "complex64": 8,
    "complex128": 16,
    "int64": 8,
    "long": 8,
    "int32": 4,
    "int": 4,
    "int16": 2,
    "short": 2,
    "int8": 1,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "bool": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1,
}


def declared_payload_bytes(manifest: Any) -> int:
    """Return the manifest-declared total payload bytes, torch-free.

    Every ``body_index`` entry declares dtype and element count, so the cost
    of materializing an artifact's payloads is computed EXACTLY before any
    blob is opened (P0 part 3). Used by budget gates: an agent-facing door
    must refuse or go lazy from these declared numbers, never discover the
    cost by materializing.

    Parameters
    ----------
    manifest:
        Parsed manifest mapping (or ``Manifest``-like with ``get``).

    Returns
    -------
    int
        Sum over body-index entries of ``num_elements`` times the dtype's
        transport width (0 when the manifest declares no body index; unknown
        dtypes charge 8 bytes per element, an overestimate by design).
    """

    body_index = None
    getter = getattr(manifest, "get", None)
    if callable(getter):
        body_index = getter("body_index")
    if body_index is None:
        body_index = getattr(manifest, "body_index", None)
    total = 0
    for entry in body_index or ():
        if not isinstance(entry, dict):
            continue
        num_elements = entry.get("num_elements")
        if not isinstance(num_elements, int) or isinstance(num_elements, bool):
            continue
        dtype_name = str(entry.get("dtype", "")).rsplit(".", 1)[-1].lower()
        total += num_elements * _DTYPE_NBYTES.get(dtype_name, 8)
    return total


__all__ = ["declared_payload_bytes", "read_op_payload"]
