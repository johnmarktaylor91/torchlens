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


#: Values per working chunk (512 KiB as float64, 64 KiB per boolean mask):
#: bounds the reducer's transient memory per op whatever the out's size.
_CHUNK = 1 << 16


def lean_spine(value: Any) -> Spine:
    """Fold one tensor into a fresh ``Spine`` with bounded working memory.

    Produces the state ``Spine().update(value)`` produces. ``Spine.update``
    holds full-size boolean masks and several full-size float64 copies at
    once, which would make a reduce-and-discard capture peak above a full
    one; here every count, extremum and moment is folded over chunks of
    ``_CHUNK`` values, so no working tensor is larger than one chunk. Counts
    and extrema are exact. The sum is one whole-tensor reduction, as in
    ``Spine.update``, so for a float64 out the sum and the mean are
    bit-equal (a float32 out's sum accumulates in float64 without a
    converted copy and can differ in the last bits). The sum of squares, the
    sum of absolute values and M2 are bit-equal for outs up to one chunk and
    otherwise equal within float64 reduction-order rounding.
    """

    tensor, dtype_tag = _as_reduction_tensor("Spine", value)
    spine = Spine()
    spine._reduction_dtype = dtype_tag
    n = int(tensor.numel())
    if n == 0:
        return spine
    nan = posinf = neginf = 0
    for start in range(0, n, _CHUNK):
        chunk = tensor[start : start + _CHUNK]
        nan += int(torch.isnan(chunk).sum().item())
        posinf += int((chunk == math.inf).sum().item())
        neginf += int((chunk == -math.inf).sum().item())
    finite = tensor[torch.isfinite(tensor)] if (nan or posinf or neginf) else tensor
    n_finite = int(finite.numel())
    spine._count_total, spine._count_finite = n, n_finite
    spine._count_nan, spine._count_posinf, spine._count_neginf = nan, posinf, neginf
    if n_finite == 0:
        return spine
    _fold_finite(spine, finite)
    return spine


def _fold_finite(spine: Spine, finite: torch.Tensor) -> None:
    """Fill a fresh spine's finite-value fields from a flat all-finite tensor."""

    n_finite = int(finite.numel())
    total = finite.sum() if finite.dtype == torch.float64 else finite.sum(dtype=torch.float64)
    spine._sum = float(total.item())
    spine._mean = spine._sum / n_finite
    zero = negative = 0
    low, high = math.inf, -math.inf
    squares = absolutes = centred = 0.0
    for start in range(0, n_finite, _CHUNK):
        chunk = finite[start : start + _CHUNK]
        zero += int((chunk == 0).sum().item())
        negative += int((chunk < 0).sum().item())
        chunk_low, chunk_high = torch.aminmax(chunk)
        low, high = min(low, float(chunk_low.item())), max(high, float(chunk_high.item()))
        chunk64 = chunk.to(torch.float64)
        squares += float((chunk64 * chunk64).sum().item())
        absolutes += float(chunk64.abs().sum().item())
        centred += float(((chunk64 - spine._mean) ** 2).sum().item())
    spine._count_zero, spine._count_negative = zero, negative
    spine._finite_min, spine._finite_max = low, high
    spine._finite_absmax = max(abs(low), abs(high))
    spine._sum_squares, spine._sum_abs, spine._m2 = squares, absolutes, centred


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
