"""Word-based value summaries for implicit record displays.

The explicit glance renderer keeps its histogram grammar. Record displays use
these words over the same versioned statistics, computing only small saved CPU
activations. Live parameters and device tensors may only reuse existing evidence.
"""

from __future__ import annotations

from typing import Any

import torch

from .._utils._display_numbers import format_exact_count, format_sig
from ._stats_kernel import SAMPLE_CAP
from ._stats_render import format_sampled
from ._tensor_stats import _STATS_CACHE, _cache_key, _tensor_version, tensor_stats


def record_stats(tensor: Any, *, compute: bool = False, identity: str | None = None) -> Any:
    """Read valid cached evidence, or summarize a small saved CPU tensor once."""

    if not isinstance(tensor, torch.Tensor):
        return None
    version = _tensor_version(tensor)
    cached = _STATS_CACHE.get(id(tensor))
    if cached is not None and cached[0]() is tensor and cached[1] == _cache_key(tensor, version):
        return cached[2]
    if compute and tensor.device.type == "cpu" and tensor.numel() <= SAMPLE_CAP:
        return tensor_stats(tensor, _identity=identity)
    return None


def _number(value: float, *, integer: bool = False) -> str:
    """Retain exact integers and the four-significant-digit float law."""

    if integer and float(value).is_integer():
        return str(int(value))
    return format_sig(value)


def _moment(name: str, value: Any, evidence: Any, se: Any) -> str:
    """One moment with its own sampled or unavailable disclosure."""

    if value is None:
        return f"{name} not computed"
    if evidence.is_sampled:
        return f"{name} ~{format_sampled(value, se).removeprefix('~')}"
    return f"{name} {format_sig(value)}"


def value_health(stats: Any) -> str:
    """Critical evidence that survives the repr width ladder."""

    if stats is None:
        return ""
    if stats.nonfinite_evidence.reason and stats.nonfinite_evidence.policy == "unavailable":
        return str(stats.nonfinite_evidence.reason)
    if stats.nonfinite_evidence.policy == "verdict":
        return "nonfinite present"
    facts = [
        f"{name} {format_exact_count(count)}"
        for name, count in (
            ("NaN", stats.num_nan),
            ("+Inf", stats.num_posinf),
            ("-Inf", stats.num_neginf),
        )
        if count
    ]
    if stats.nonfinite_evidence.policy == "unavailable":
        facts.append("NaN/Inf not checked")
    return ", ".join(facts)


def _summary(stats: Any, *, separator: str, moments: bool) -> str:
    """Render value words without a histogram, preserving evidence and special states."""

    if stats is None:
        return "not computed"
    if stats.nonfinite_evidence.reason and stats.nonfinite_evidence.policy == "unavailable":
        return ""
    if stats.is_empty:
        return "empty"
    if stats.has_no_finite_values:
        return "no finite values"
    if stats.is_all_true:
        return "all true"
    if stats.is_all_false:
        return "all false"
    if stats.is_all_zero:
        return "all zero"
    integer = stats.dtype.startswith(("i8", "i16", "i32", "i64", "u8"))
    prefix = "magnitude " if stats.is_magnitude_based else ""
    parts = [
        f"{name} {_number(value, integer=integer)}"
        for name, value in (("min", stats.finite_min), ("max", stats.finite_max))
        if value is not None
    ]
    if moments:
        parts.extend(
            (
                _moment("mean", stats.mean, stats.mean_evidence, stats.mean_se),
                _moment("sd", stats.std, stats.std_evidence, stats.std_se),
            )
        )
    return prefix + separator.join(parts)


def render_values(stats: Any, *, separator: str = "  ", moments: bool = True) -> str:
    """Full value row, including nonfinite evidence when it is available."""

    summary = _summary(stats, separator=separator, moments=moments)
    health = value_health(stats)
    return separator.join(part for part in (summary, health) if part)
