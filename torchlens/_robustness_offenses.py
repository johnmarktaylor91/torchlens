"""Offense records for the tensor-variant pre-flight check (``_robustness``).

One structured record per refused variant (R67: WHERE, not just WHAT), the
documentation pointer and user-frame lookup its messages cite, and the two
model-state offenses whose text the check builds from a slot name alone.
"""

from __future__ import annotations

from typing import Any

import torch


def _offense_entry(name: str, reason: str, path: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Build one structured offense record (R67: WHERE, not just WHAT).

    Parameters
    ----------
    name:
        Variant-class label (e.g. ``"meta tensor in input"``).
    reason:
        Why the variant is unsupported (may be empty for compact channels).
    path:
        Input-tree location that reaches the offending tensor.
    tensor:
        The offending tensor; shape/dtype reads are guarded because shapeless
        variants raise on ``.shape`` and hostile subclasses may raise anywhere.

    Returns
    -------
    dict[str, Any]
        ``{"name", "reason", "path", "shape", "dtype"}``.
    """

    try:
        shape: tuple[int, ...] | None = tuple(tensor.shape)
    except Exception:  # noqa: BLE001 - shapeless variants raise internal errors here
        shape = None
    try:
        dtype = str(tensor.dtype)
    except Exception:  # noqa: BLE001
        dtype = None
    return {"name": name, "reason": reason, "path": path, "shape": shape, "dtype": dtype}


def _docs_pointer(section: str | None = None) -> str:
    """Human-readable pointer to the limitations documentation.

    Parameters
    ----------
    section:
        Optional exact section heading of ``docs/reference/limitations.md``
        to cite; ``None`` points at the catalog as a whole.

    Returns
    -------
    str
        Pointer sentence naming a real documentation location.
    """
    if section is None:
        return "See docs/reference/limitations.md for supported alternatives."
    return (
        f"See the {section!r} section of docs/reference/limitations.md for supported alternatives."
    )


def _first_user_frame() -> tuple[str | None, int | None]:
    """Return the first non-torchlens frame (the user's capture callsite).

    Teaching-enrichment helper (L7a): names WHERE the refused capture was
    requested. Best-effort; ``(None, None)`` when no external frame is
    visible.
    """

    import inspect
    from pathlib import Path

    torchlens_root = Path(__file__).resolve().parent
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = Path(frame.f_code.co_filename).resolve()
            try:
                filename.relative_to(torchlens_root)
            except ValueError:
                return str(filename), frame.f_lineno
            frame = frame.f_back
    finally:
        del frame
    return None, None


def _meta_state_offense(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Offense record for a meta tensor registered as model state at ``model.{name}``."""

    return _offense_entry(
        "meta tensor among model parameters/buffers",
        "Meta-init models (e.g. HuggingFace device_map='meta') must be "
        "materialized on a real device before logging.",
        f"model.{name}",
        tensor,
    )


def _sparse_state_offense(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Offense record for a sparse tensor registered as model state at ``model.{name}``.

    A registered sparse tensor has no dense storage for the state-write journal to
    snapshot (``buffer_writes._whole_storage_uint8`` raises torch's bare
    NotImplementedError at session prep): the check refuses typed at entry, naming
    the slot, instead of failing mid-prep (stress case 1).
    """

    return _offense_entry(
        f"sparse tensor ({tensor.layout}) among model parameters/buffers",
        "TorchLens snapshots the storage of every registered parameter "
        "and buffer for its state-write journal, and sparse layouts have "
        "no dense storage to snapshot. Register the dense form "
        "(tensor.to_dense()) and convert it inside forward "
        "(.to_sparse()) where the sparse kernel is needed.",
        f"model.{name}",
        tensor,
    )
