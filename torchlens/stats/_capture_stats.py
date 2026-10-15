"""Reduce-and-discard capture: ``tl.trace(model, x, save="stats")``.

``save="stats"`` folds every site's out into a ``Spine`` as the forward runs
and keeps no activation payload. It rides the existing summary-transform
seam (a reducer with the declared ``summary`` role and
``save_raw_activations=False``), so the capture makes no raw clone; each op
retains one fixed 16-slot float64 vector (128 bytes) instead of its out,
and that vector is all the save budget is charged. ``site_spines(trace)``
decodes those vectors back into live ``Spine`` accumulators equal to
``Spine().update(out)`` on the same out (see :func:`lean_spine` for which
fields are bit-equal). Reading an op's payload on such a trace refuses
(``activation_reduced_to_stats``).

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .._errors import InvalidArgumentError, PayloadUnavailableError
from ..observability._kernels import Spine, _as_reduction_tensor

__all__ = [
    "STATS_REDUCER_REPR",
    "STATS_SAVE_LEVEL",
    "is_stats_capture",
    "lean_spine",
    "refuse_reduced_payload_read",
    "site_spines",
    "stats_save_options",
]

#: The ``save=`` value that selects reduce-and-discard capture.
STATS_SAVE_LEVEL = "stats"

#: The reducer's fixed repr; a Trace records it as its transform repr, which
#: survives save and load, so a reloaded stats trace is still recognised.
STATS_REDUCER_REPR = "tl.stats save='stats' spine reducer"

#: Packed vector slots, in ``Spine`` attribute order, then the dtype code.
_FIELDS = (
    "_count_total",
    "_count_finite",
    "_count_zero",
    "_count_negative",
    "_count_nan",
    "_count_posinf",
    "_count_neginf",
    "_finite_min",
    "_finite_max",
    "_finite_absmax",
    "_sum",
    "_sum_squares",
    "_sum_abs",
    "_mean",
    "_m2",
)
_INT_FIELDS = frozenset(_FIELDS[:7])
_OPTIONAL_FIELDS = frozenset({"_finite_min", "_finite_max", "_finite_absmax"})
_DTYPE_CODES = {None: 0.0, "float32": 1.0, "float64": 2.0}
_DTYPE_NAMES = {code: name for name, code in _DTYPE_CODES.items()}
#: Packed width: the Spine fields plus one dtype code.
PACKED_SLOTS = len(_FIELDS) + 1


class _StatsReducer:
    """Summary-role reducer: one op out to its packed ``Spine`` state.

    The declared ``summary`` role makes the capture skip the raw clone; the
    reducer folds with :func:`lean_spine`, which reproduces ``Spine.update``
    without its full-size float64 working copies. Total by design: an out
    the kernel refuses (a complex tensor) packs to the empty vector and
    ``site_spines`` leaves that site out.
    """

    _tl_transform_role = "summary"
    _tl_stats_reducer = True

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fold one detached out into a fresh ``Spine`` and pack its state."""

        try:
            spine = lean_spine(tensor)
        except Exception:  # noqa: BLE001 -- total reducer: refused dtypes become "no spine"
            return torch.empty(0, dtype=torch.float64)
        return pack_spine(spine)

    def __repr__(self) -> str:
        """Fixed repr: the trace records it, and save/load keeps it."""

        return STATS_REDUCER_REPR


#: Values per float64 working chunk for the squared, absolute and centred
#: sums (8 MiB of float64): bounds the reducer's transient memory per op.
_CHUNK = 1 << 20


def lean_spine(value: Any) -> Spine:
    """Fold one tensor into a fresh ``Spine`` with bounded working memory.

    Produces the state ``Spine().update(value)`` produces. Counts, extrema,
    the sum and the mean are computed over the whole tensor exactly as
    ``Spine.update`` does (bit-equal for float64 outs; for float32 outs the
    sum accumulates in float64 without a converted copy, so it can differ
    in the last bits). The sum of squares, sum of absolute values and M2
    are summed over float64 chunks of ``_CHUNK`` values: bit-equal for outs
    up to that size, otherwise equal within float64 reduction-order rounding.
    ``Spine.update`` instead materializes several full-size float64 copies,
    which would make a reduce-and-discard capture peak above a full one.
    """

    tensor, dtype_tag = _as_reduction_tensor("Spine", value)
    spine = Spine()
    spine._reduction_dtype = dtype_tag
    n = int(tensor.numel())
    if n == 0:
        return spine
    n_nan = int(torch.isnan(tensor).sum().item())
    n_posinf = int((tensor == math.inf).sum().item())
    n_neginf = int((tensor == -math.inf).sum().item())
    finite = tensor[torch.isfinite(tensor)] if (n_nan or n_posinf or n_neginf) else tensor
    n_finite = int(finite.numel())
    spine._count_total, spine._count_finite = n, n_finite
    spine._count_nan, spine._count_posinf, spine._count_neginf = n_nan, n_posinf, n_neginf
    if n_finite == 0:
        return spine
    spine._count_zero = int((finite == 0).sum().item())
    spine._count_negative = int((finite < 0).sum().item())
    low, high = torch.aminmax(finite)
    spine._finite_min, spine._finite_max = float(low.item()), float(high.item())
    spine._finite_absmax = max(abs(spine._finite_min), abs(spine._finite_max))
    total = finite.sum() if finite.dtype == torch.float64 else finite.sum(dtype=torch.float64)
    spine._sum = float(total.item())
    spine._mean = spine._sum / n_finite
    squares = absolutes = centred = 0.0
    for start in range(0, n_finite, _CHUNK):
        chunk = finite[start : start + _CHUNK].to(torch.float64)
        squares += float((chunk * chunk).sum().item())
        absolutes += float(chunk.abs().sum().item())
        centred += float(((chunk - spine._mean) ** 2).sum().item())
    spine._sum_squares, spine._sum_abs, spine._m2 = squares, absolutes, centred
    return spine


def stats_save_options() -> Any:
    """Return the ``SaveOptions`` that ``save="stats"`` stands for.

    The summary-role reducer as the activation transform with raw outs
    dropped: the capture's existing reduce-only path, so no raw clone is made
    and the save budget is charged only the packed vectors.
    """

    from ..options import SaveOptions

    return SaveOptions(activation_transform=_StatsReducer(), save_raw_activations=False)


def refuse_reduced_payload_read(slot: Any, source_trace: Any) -> None:
    """Refuse an op payload read on a ``save="stats"`` trace (reduce and discard).

    Such a trace folded every out into a ``Spine`` and kept no payload, so a
    silent ``None`` would read as "this op produced nothing".
    """

    if not is_stats_capture(source_trace):
        return
    label = slot("label") or slot("layer_label") or slot("_label_raw")
    raise PayloadUnavailableError(
        f"op {label} was captured with save='stats': its out was folded into a Spine "
        "and no payload was kept",
        code="activation_reduced_to_stats",
        remedy="read the statistics with tl.stats.site_spines(trace)[label], or capture "
        "again keeping payloads with tl.trace(model, x, save='all')",
        label=str(label),
    )


def pack_spine(spine: Spine) -> torch.Tensor:
    """Pack a ``Spine``'s state into one float64 vector (counts exact to 2^53)."""

    cells = [math.nan if getattr(spine, f) is None else float(getattr(spine, f)) for f in _FIELDS]
    cells.append(_DTYPE_CODES.get(spine._reduction_dtype, 0.0))
    return torch.tensor(cells, dtype=torch.float64)


def unpack_spine(vector: torch.Tensor, *, name: str | None = None) -> Spine:
    """Rebuild a live ``Spine`` from a packed vector (exact round trip)."""

    cells = vector.detach().to("cpu", torch.float64).tolist()
    spine = Spine(name=name)
    for field, cell in zip(_FIELDS, cells, strict=False):
        if field in _INT_FIELDS:
            setattr(spine, field, int(cell))
        elif field in _OPTIONAL_FIELDS and math.isnan(cell):
            setattr(spine, field, None)
        else:
            setattr(spine, field, cell)
    spine._reduction_dtype = _DTYPE_NAMES.get(cells[len(_FIELDS)])
    return spine


def is_stats_capture(trace: Any) -> bool:
    """Whether ``trace`` was captured with ``save="stats"`` (live or reloaded)."""

    if getattr(trace, "save_raw_activations", True):
        return False
    transform = getattr(trace, "activation_transform", None)
    if getattr(transform, "_tl_stats_reducer", False):
        return True
    return getattr(trace, "_activation_transform_repr", None) == STATS_REDUCER_REPR


def site_spines(trace: Any) -> dict[str, Spine]:
    """Return ``{op label: Spine}`` for a ``save="stats"`` trace, in capture order.

    Parameters
    ----------
    trace:
        A trace captured with ``tl.trace(model, x, save="stats")``.

    Returns
    -------
    dict[str, Spine]
        One live accumulator per op site (pass-qualified labels for repeated
        layers). Each keeps folding with ``update``/``merge`` and saves with
        ``tl.stats.save_fitted``. Sites whose out the kernel refused (complex
        dtypes) are absent.
    """

    if not is_stats_capture(trace):
        raise InvalidArgumentError(
            "tl.stats.site_spines(trace) reads the per-site statistics of a reduce-and-discard "
            "capture, but this trace was not captured with save='stats'",
            code="stats_capture_not_reduced",
            remedy="capture with tl.trace(model, x, save='stats'), then call "
            "tl.stats.site_spines(trace)",
            argument="trace",
        )
    spines: dict[str, Spine] = {}
    for op in trace.ops:
        vector = op.transformed_out
        if not isinstance(vector, torch.Tensor) or vector.numel() != PACKED_SLOTS:
            continue
        label = str(op.label)
        spines[label] = unpack_spine(vector, name=label)
    return spines
