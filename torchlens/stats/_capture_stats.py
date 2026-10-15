"""Reduce-and-discard capture: ``tl.trace(model, x, save="stats")``.

``save="stats"`` folds every site's out into a ``Spine`` as the forward runs
and keeps no activation payload. It rides the existing summary-transform
seam (a reducer with the declared ``summary`` role and
``save_raw_activations=False``), so the capture makes no raw clone and
admits nothing to the save budget; each op retains one fixed 16-slot
float64 vector (128 bytes) instead of its out. ``site_spines(trace)``
decodes those vectors back into live ``Spine`` accumulators, bit-equal to
``Spine().update(out)`` on the same out because the reducer runs that very
kernel. Reading an op's payload on such a trace refuses
(``activation_reduced_to_stats``).

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .._errors import InvalidArgumentError
from ..observability._kernels import Spine

__all__ = ["STATS_REDUCER_REPR", "STATS_SAVE_LEVEL", "is_stats_capture", "site_spines"]

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
    reducer runs ``Spine.update`` itself, so the folded statistics are the
    exact ones a ``Spine`` computes from the full out. Total by design: an
    out the kernel refuses (a complex tensor) packs to the empty vector and
    ``site_spines`` leaves that site out.
    """

    _tl_transform_role = "summary"
    _tl_stats_reducer = True

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fold one detached out into a fresh ``Spine`` and pack its state."""

        spine = Spine()
        try:
            spine.update(tensor)
        except Exception:  # noqa: BLE001 -- total reducer: refused dtypes become "no spine"
            return torch.empty(0, dtype=torch.float64)
        return pack_spine(spine)

    def __repr__(self) -> str:
        """Fixed repr: the trace records it, and save/load keeps it."""

        return STATS_REDUCER_REPR


def stats_reducer() -> _StatsReducer:
    """Return the reducer ``save="stats"`` installs as the activation transform."""

    return _StatsReducer()


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
