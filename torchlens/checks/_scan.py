"""The ONE shared batched tensor-scan kernel (checks memo item 1 / 4.4).

One kernel, two spellings: ``tl.debug.audit_params`` shapes rows into a
:class:`~torchlens.checks.ParamAudit`, and the registry's scheduled scan
consumes the same rows with report shaping skipped. Engineering contract
(memo 4.4): batched per ``(device, dtype)`` group with ~1 host sync per
group, integer counts exact, extrema carried as float64 SCALARS only --
NEVER a whole-tensor float64 widening (the fp16 peak-allocated-bytes
regression test pins this; the known bad widen-everything helper is in the
tree and will be tempting). fp8 payloads widen exactly through the
sanctioned ``fp8_widen_for_numeric_ops`` chokepoint; meta / sparse /
quantized tensors are skipped WITH a reason, never silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .. import _state
from ..utils.tensor_utils import fp8_widen_for_numeric_ops
from ._errors import CheckConfigError

__tl_layer__ = "L5"

#: Number of float64 scalar stats packed per tensor into the group sync.
_STATS_PER_TENSOR = 11


@dataclass(frozen=True)
class ScanRow:
    """Health facts for one unique tensor (checks memo 4.4).

    ``name`` is the canonical qualified name; ``aliases`` preserves every
    other name bound to the SAME tensor object (tied weights, double-
    registered buffers) -- deduped by identity, no alias lost. ``audited``
    is False exactly when ``skip_reason`` says why.
    """

    name: str
    aliases: tuple[str, ...]
    kind: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    n_nan: int
    n_posinf: int
    n_neginf: int
    zero_fraction: float | None
    all_same: bool | None
    finite_min: float | None
    finite_max: float | None
    abs_max: float | None
    dtype_max: float | None
    subnormal_fraction: float | None
    bounds_below: int | None
    bounds_above: int | None
    audited: bool
    skip_reason: str | None

    @property
    def n_nonfinite(self) -> int:
        """Total nonfinite elements (NaN + signed infinities)."""

        return self.n_nan + self.n_posinf + self.n_neginf

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable payload for this row."""

        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "numel": self.numel,
            "n_nan": self.n_nan,
            "n_posinf": self.n_posinf,
            "n_neginf": self.n_neginf,
            "zero_fraction": self.zero_fraction,
            "all_same": self.all_same,
            "finite_min": self.finite_min,
            "finite_max": self.finite_max,
            "abs_max": self.abs_max,
            "dtype_max": self.dtype_max,
            "subnormal_fraction": self.subnormal_fraction,
            "bounds_below": self.bounds_below,
            "bounds_above": self.bounds_above,
            "audited": self.audited,
            "skip_reason": self.skip_reason,
        }


def _skip_reason(tensor: torch.Tensor) -> str | None:
    """Return why a tensor cannot be scanned, or ``None`` when it can."""

    if tensor.device.type == "meta":
        return "meta tensor holds no values"
    if tensor.layout != torch.strided:
        return f"unsupported layout {tensor.layout}"
    if tensor.is_quantized:
        return "quantized tensors are not scannable; audit the dequantized view"
    return None


def _skipped_row(
    name: str, aliases: tuple[str, ...], kind: str, tensor: Any, reason: str
) -> ScanRow:
    """Build the skipped-with-reason row for an unscannable entry."""

    shape: tuple[int, ...] = ()
    dtype = type(tensor).__name__
    device = "unknown"
    numel = 0
    if isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype).removeprefix("torch.")
        device = str(tensor.device)
        numel = tensor.numel()
    return ScanRow(
        name=name,
        aliases=aliases,
        kind=kind,
        shape=shape,
        dtype=dtype,
        device=device,
        numel=numel,
        n_nan=0,
        n_posinf=0,
        n_neginf=0,
        zero_fraction=None,
        all_same=None,
        finite_min=None,
        finite_max=None,
        abs_max=None,
        dtype_max=None,
        subnormal_fraction=None,
        bounds_below=None,
        bounds_above=None,
        audited=False,
        skip_reason=reason,
    )


def _dedup_by_identity(
    entries: list[tuple[str, str, Any]],
) -> list[tuple[str, tuple[str, ...], str, Any]]:
    """Collapse identical tensor objects into one canonical row.

    Tied tensors dedup by OBJECT identity with every alias preserved
    (memo 4.4); the first-seen name is canonical, matching iteration order
    of ``named_parameters``.
    """

    seen: dict[int, int] = {}
    deduped: list[tuple[str, list[str], str, Any]] = []
    for name, kind, tensor in entries:
        key = id(tensor)
        if isinstance(tensor, torch.Tensor) and key in seen:
            deduped[seen[key]][1].append(name)
            continue
        if isinstance(tensor, torch.Tensor):
            seen[key] = len(deduped)
        deduped.append((name, [], kind, tensor))
    return [(name, tuple(aliases), kind, tensor) for name, aliases, kind, tensor in deduped]


def _device_stats(
    tensor: torch.Tensor,
    bounds: tuple[float | None, float | None] | None,
) -> list[torch.Tensor]:
    """Compute the on-device scalar stats for one scannable tensor.

    Every returned tensor is 0-dim on the input's device; the caller stacks
    them per group and performs the ONE host sync. Casting the 0-dim stats
    to float64 is exact and allocates 8 bytes each -- the whole-tensor
    widening the kernel forbids never happens here.
    """

    values = fp8_widen_for_numeric_ops(tensor.detach())
    numel = values.numel()
    zero64 = torch.zeros((), dtype=torch.float64, device=values.device)
    if numel == 0:
        return [zero64.clone() for _ in range(_STATS_PER_TENSOR)]
    flat = values.reshape(-1)
    is_floating = values.is_floating_point()
    if is_floating:
        n_nan = torch.isnan(flat).sum()
        n_posinf = (flat == float("inf")).sum()
        n_neginf = (flat == float("-inf")).sum()
        finite_mask = torch.isfinite(flat)
        # Masked extrema: fill with +/-inf sentinels; a group with zero
        # finite elements is detected from the counts, never the sentinel.
        finite_min = torch.where(finite_mask, flat, torch.full_like(flat, float("inf"))).min()
        finite_max = torch.where(finite_mask, flat, torch.full_like(flat, float("-inf"))).max()
        abs_flat = flat.abs()
        abs_max = torch.where(finite_mask, abs_flat, torch.zeros_like(flat)).max()
        smallest_normal = torch.finfo(values.dtype).smallest_normal
        n_subnormal = ((abs_flat < smallest_normal) & (flat != 0)).sum()
    else:
        n_nan = zero64.clone()
        n_posinf = zero64.clone()
        n_neginf = zero64.clone()
        finite_min = flat.min()
        finite_max = flat.max()
        abs_max = flat.abs().max() if values.dtype != torch.bool else flat.max()
        n_subnormal = zero64.clone()
    n_zero = (flat == 0).sum()
    all_same = (flat == flat[0]).all()
    if bounds is not None:
        low, high = bounds
        n_below = (flat < low).sum() if low is not None else zero64.clone()
        n_above = (flat > high).sum() if high is not None else zero64.clone()
    else:
        n_below = zero64.clone()
        n_above = zero64.clone()
    stats = [
        n_nan,
        n_posinf,
        n_neginf,
        n_zero,
        all_same,
        finite_min,
        finite_max,
        abs_max,
        n_subnormal,
        n_below,
        n_above,
    ]
    return [stat.to(torch.float64) for stat in stats]


def scan_named_tensors(
    entries: list[tuple[str, str, Any]],
    *,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> list[ScanRow]:
    """Scan named tensors for numeric-health facts, batched per group.

    Parameters
    ----------
    entries:
        ``(name, kind, tensor)`` triples in caller order; ``kind`` is a
        free label such as ``"parameter"`` / ``"buffer"`` / ``"mapping"``.
    bounds:
        Optional exact-name bounds mapping ``name -> (low, high)`` with
        either side ``None`` for one-sided bounds. Unknown names REFUSE
        (memo 4.4) at the caller (:func:`torchlens.checks.audit_params`),
        which validates against the full inventory before dedup.

    Returns
    -------
    list[ScanRow]
        One row per UNIQUE tensor object (tied aliases preserved), rows in
        first-seen order; unscannable entries carry ``skip_reason``.

    Notes
    -----
    All arithmetic runs under ``torch.no_grad()`` + ``pause_logging()`` so a
    concurrent TorchLens capture never records scan traffic and no autograd
    state is touched (memo 4.2). One host sync per ``(device, dtype)``
    group: every per-tensor stat stays a 0-dim device tensor until the
    group's single stacked ``.tolist()``.
    """

    deduped = _dedup_by_identity(entries)
    rows: list[ScanRow | None] = [None] * len(deduped)
    groups: dict[tuple[str, str], list[int]] = {}
    with torch.no_grad(), _state.pause_logging():
        stats_by_index: dict[int, list[torch.Tensor]] = {}
        for index, (name, aliases, kind, tensor) in enumerate(deduped):
            if not isinstance(tensor, torch.Tensor):
                rows[index] = _skipped_row(
                    name, aliases, kind, tensor, f"not a tensor: {type(tensor).__name__}"
                )
                continue
            reason = _skip_reason(tensor)
            if reason is not None:
                rows[index] = _skipped_row(name, aliases, kind, tensor, reason)
                continue
            tensor_bounds = bounds.get(name) if bounds else None
            stats_by_index[index] = _device_stats(tensor, tensor_bounds)
            groups.setdefault((str(tensor.device), str(tensor.dtype)), []).append(index)

        for group_indexes in groups.values():
            stacked = torch.stack(
                [stat for index in group_indexes for stat in stats_by_index[index]]
            )
            flat_values = stacked.cpu().tolist()  # the ONE sync for this group
            for position, index in enumerate(group_indexes):
                name, aliases, kind, tensor = deduped[index]
                offset = position * _STATS_PER_TENSOR
                rows[index] = _row_from_stats(
                    name,
                    aliases,
                    kind,
                    tensor,
                    flat_values[offset : offset + _STATS_PER_TENSOR],
                    had_bounds=bool(bounds and name in bounds),
                )
    return [row for row in rows if row is not None]


def _row_from_stats(  # noqa: PLR0913 -- one row's identity + measured stats; the shared-kernel row shape is the memo-item-1 contract
    name: str,
    aliases: tuple[str, ...],
    kind: str,
    tensor: torch.Tensor,
    stats: list[float],
    *,
    had_bounds: bool,
) -> ScanRow:
    """Shape one tensor's synced scalar stats into a :class:`ScanRow`."""

    (
        n_nan,
        n_posinf,
        n_neginf,
        n_zero,
        all_same,
        finite_min,
        finite_max,
        abs_max,
        n_subnormal,
        n_below,
        n_above,
    ) = stats
    numel = tensor.numel()
    finite_count = numel - int(n_nan) - int(n_posinf) - int(n_neginf)
    is_floating = tensor.is_floating_point()
    # Headroom is judged against the tensor's OWN dtype ceiling (fp8 included:
    # torch.finfo supports the float8 dtypes); the fp32 widen is only for the
    # predicate ops fp8 does not implement, never for the limits.
    dtype_max = float(torch.finfo(tensor.dtype).max) if is_floating else None
    return ScanRow(
        name=name,
        aliases=aliases,
        kind=kind,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
        numel=numel,
        n_nan=int(n_nan),
        n_posinf=int(n_posinf),
        n_neginf=int(n_neginf),
        zero_fraction=(int(n_zero) / numel) if numel else None,
        all_same=bool(all_same) if numel else None,
        finite_min=finite_min if numel and finite_count else None,
        finite_max=finite_max if numel and finite_count else None,
        abs_max=abs_max if numel and finite_count else None,
        dtype_max=dtype_max,
        subnormal_fraction=(int(n_subnormal) / numel) if numel and is_floating else None,
        bounds_below=int(n_below) if had_bounds else None,
        bounds_above=int(n_above) if had_bounds else None,
        audited=True,
        skip_reason=None,
    )


def tensor_digest(tensor: torch.Tensor) -> str:
    """Return a deterministic content digest of a tensor's bytes.

    The frozen-check digest tier (memo D5): digest scalars differing PROVES
    change (one-sided soundness -- a false raise is impossible); equal
    digests are labeled probabilistic evidence and may NEVER emit an exact
    pass. Runs under ``pause_logging`` and syncs once (the ``.cpu()`` copy).
    """

    with torch.no_grad(), _state.pause_logging():
        flat = tensor.detach().reshape(-1).contiguous()
        payload = b"" if flat.numel() == 0 else flat.view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()


def named_entries_from_target(
    target: Any,
    *,
    include_buffers: bool = True,
) -> list[tuple[str, str, Any]]:
    """Normalize an ``nn.Module`` or name->tensor Mapping into scan entries.

    The Mapping door is the declared extension point (memo D15): live
    gradients, flattened optimizer state, EMA weights, checkpoints, and
    shards are all the same recipe through the same kernel; the caller
    names the mapping and therefore owns the phase label.
    """

    if isinstance(target, torch.nn.Module):
        # remove_duplicate=False so tied tensors surface EVERY alias; the
        # kernel dedups by identity and keeps the alias names (memo 4.4).
        entries: list[tuple[str, str, Any]] = [
            (name, "parameter", parameter)
            for name, parameter in target.named_parameters(remove_duplicate=False)
        ]
        if include_buffers:
            entries.extend(
                (name, "buffer", buffer)
                for name, buffer in target.named_buffers(remove_duplicate=False)
            )
        return entries
    if isinstance(target, Mapping):
        return [(str(name), "mapping", value) for name, value in target.items()]
    raise CheckConfigError(
        f"audit_params target must be an nn.Module or a name->tensor Mapping, "
        f"got {type(target).__name__}. state_dicts, flattened optimizer "
        "state, EMA shadows, and loaded checkpoints all enter through the "
        "Mapping door.",
        code="check_target_invalid",
        remedy="Pass the module itself, or dict(model.state_dict()) / any name->tensor mapping.",
    )


__all__ = [
    "ScanRow",
    "named_entries_from_target",
    "scan_named_tensors",
    "tensor_digest",
]
