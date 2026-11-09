"""Value helpers behind the label-joined ``tl.debug.compare`` rows.

Moved out of ``_graph`` so the comparison front end stays under the file cap; the
cross-backend module comparison reuses :func:`_align_for_compare` and
:func:`_compare_unavailable_reason`.
"""

from __future__ import annotations

from typing import Any

import torch

from ._common import _tensor_unavailable_reason


def _activation_row(
    label: str, status: str, reason: str, source_line: str | None = None
) -> dict[str, Any]:
    """Build a minimal activation-unavailable comparison row.

    Parameters
    ----------
    label:
        Op label.
    status:
        Presence status.
    reason:
        Unavailable reason.
    source_line:
        ``"file:line"`` of the user code that ran the op, when known.

    Returns
    -------
    dict[str, Any]
        Row dictionary.
    """

    return {
        "op": label,
        "status": status,
        "shape_match": None,
        "dtype_match": None,
        "max_abs": None,
        "mean_abs": None,
        "allclose": None,
        "reason": reason,
        "source_line": source_line,
    }


def _align_for_compare(
    out_a: torch.Tensor, out_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """Bring two activations onto one device and dtype so they can be compared.

    A CPU run against a GPU run, or a float32 run against a float64 run, is
    exactly the comparison a user wants to see; the values are moved to the
    CPU and compared in the wider dtype (``torch.promote_types``). Only a
    shape mismatch truly cannot be compared.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, str] | None
        The aligned pair and a note saying what was aligned (``""`` when
        nothing was), or ``None`` when the shapes differ.
    """

    if out_a.shape != out_b.shape:
        return None
    notes = []
    a, b = out_a.detach(), out_b.detach()
    if a.device != b.device:
        notes.append(f"devices differ ({a.device} vs {b.device}): compared on cpu")
        a, b = a.cpu(), b.cpu()
    if a.dtype != b.dtype:
        common = torch.promote_types(a.dtype, b.dtype)
        notes.append(
            f"dtypes differ ({str(a.dtype).removeprefix('torch.')} vs "
            f"{str(b.dtype).removeprefix('torch.')}): compared in "
            f"{str(common).removeprefix('torch.')}"
        )
        a, b = a.to(common), b.to(common)
    return a, b, "; ".join(notes)


def _compare_unavailable_reason(value: Any) -> str | None:
    """Return why a payload cannot be value-compared, or ``None``.

    Wider than :func:`_tensor_unavailable_reason`: integer, boolean and complex
    tensors compare too (integers and booleans exactly), so an index or mask op
    where two runs part ways is reported instead of skipped.
    """

    reason = _tensor_unavailable_reason(value)
    if reason in ("non-floating", "complex"):
        return None
    return reason


def _value_delta(
    cmp_a: torch.Tensor, cmp_b: torch.Tensor, *, rtol: float, atol: float
) -> tuple[float, float, bool]:
    """Return ``(max_abs, mean_abs, allclose)`` for an aligned pair.

    Floating and complex payloads use ``torch.allclose``; integer and boolean
    payloads must match exactly (a tolerance on an index is meaningless).
    """

    exact = not (torch.is_floating_point(cmp_a) or torch.is_complex(cmp_a))
    wide = torch.complex128 if torch.is_complex(cmp_a) else torch.float64
    delta = torch.abs(cmp_a.to(wide) - cmp_b.to(wide)) if exact else torch.abs(cmp_a - cmp_b)
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.mean().item()) if delta.numel() else 0.0
    if exact:
        return max_abs, mean_abs, bool(torch.equal(cmp_a, cmp_b))
    return max_abs, mean_abs, bool(torch.allclose(cmp_a, cmp_b, rtol=rtol, atol=atol))


def _compare_coverage(rows: list[dict[str, Any]]) -> str:
    """Return the compare card's coverage line: compared vs skipped, and why."""

    compared = sum(1 for row in rows if row["allclose"] is not None)
    aligned = sum(1 for row in rows if row["allclose"] is not None and row["reason"])
    skipped: dict[str, int] = {}
    for row in rows:
        if row["allclose"] is None:
            reason = str(row["reason"] or row["status"])
            skipped[reason] = skipped.get(reason, 0) + 1
    text = f"{compared} of {len(rows)} ops compared"
    if aligned:
        notes = sorted(
            {
                str(note)
                for row in rows
                if row["allclose"] is not None and row["reason"]
                for note in str(row["reason"]).split("; ")
            }
        )
        text += f" ({aligned} after alignment: {'; '.join(notes)})"
    if skipped:
        detail = ", ".join(f"{reason}: {count}" for reason, count in skipped.items())
        text += f"; {sum(skipped.values())} not compared ({detail})"
    return text
