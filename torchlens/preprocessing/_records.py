"""Neutral preprocessing declaration + resolution records (tvscope B2/B3).

The comparable-field schema is the ONE vocabulary every authority adapter
normalizes into and the audit compares over. Every field is optional:
``None`` means the authority (or the applied pipeline) DECLARED NOTHING for
that field, and the audit reports ``unknown`` with a reason rather than
guessing -- a real shipped model family (torchvision detection presets)
declares nothing at all, so the honest-unknown path is load-bearing, not a
corner.

Every spelling in this package is DOCUMENTED-UNSTABLE pending the naming
sprint; the SEPARATIONS (authority record vs field-level audit vs opt-in
tensor diagnostics, tvscope memo D4) are the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from torchlens.data_classes._preprocessing_provenance import (
    STATUS_AUTHORITATIVE as STATUS_AUTHORITATIVE,
    STATUS_UNKNOWN as STATUS_UNKNOWN,
    STATUS_UNVERIFIED_FALLBACK as STATUS_UNVERIFIED_FALLBACK,
    status_of as status_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchlens.data_classes.trace import ResolvedPreprocessing

__tl_layer__ = "L5"

#: The closed comparable-field vocabulary (tvscope memo B3): resize, crop,
#: interpolation, antialias, channel order, value range, mean AND std --
#: stds can differ when means agree, so the pair is two fields, never one.
COMPARABLE_FIELDS: tuple[str, ...] = (
    "resize_size",
    "crop_size",
    "interpolation",
    "antialias",
    "channel_order",
    "value_range",
    "mean",
    "std",
)

#: Resolver sources that resolved nothing.
_UNKNOWN_SOURCES = frozenset({"unknown", "user_transform", "user_tensors"})


def _normalize_pair(value: Any) -> Any:
    """Normalize size-like and stat-like values for stable comparison.

    Parameters
    ----------
    value:
        Raw adapter value: int, float, list, or tuple.

    Returns
    -------
    Any
        Tuples for sequences, unchanged scalars, ``None`` passed through.
    """

    if isinstance(value, (list, tuple)):
        return tuple(_normalize_pair(item) for item in value)
    return value


@dataclass(frozen=True)
class DeclaredPreprocessing:
    """One authority's (or one applied pipeline's) declared field values.

    Attributes
    ----------
    modality:
        Input modality tag; adapters are modality-tagged from day one so
        non-image authorities can land later without a redesign.
    resize_size:
        Shorter-side int or explicit ``(h, w)`` resize target.
    crop_size:
        Center/final crop size (int or ``(h, w)``).
    interpolation:
        Interpolation mode name, lower-cased (``"bilinear"``, ``"bicubic"``).
    antialias:
        Whether resize applies antialiasing: ``True``/``False``, the literal
        string ``"warn"`` (torchvision's own legacy sentinel -- antialiasing
        differs by input type and torchvision warns once; declared as-is
        rather than coerced into a guessed bool), or ``None`` (undeclared).
    channel_order:
        Channel convention (``"rgb"`` / ``"bgr"``).
    value_range:
        Declared pre-normalization value range, usually ``(0.0, 1.0)``.
    mean:
        Per-channel normalization means.
    std:
        Per-channel normalization stds.
    extras:
        Adapter-specific disclosures that are NOT compared (kept for the
        provenance record, e.g. ``crop_pct``, processor class names).
    """

    modality: str = "image"
    resize_size: Any = None
    crop_size: Any = None
    interpolation: str | None = None
    antialias: bool | str | None = None
    channel_order: str | None = None
    value_range: Any = None
    mean: Any = None
    std: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def comparable_fields(self) -> dict[str, Any]:
        """Return the closed comparable-field mapping for the audit.

        Returns
        -------
        dict[str, Any]
            ``field name -> normalized declared value`` (``None`` = undeclared).
        """

        return {name: _normalize_pair(getattr(self, name)) for name in COMPARABLE_FIELDS}

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-portable config block.

        Returns
        -------
        dict[str, Any]
            Modality, every comparable field, and the extras disclosure.
        """

        config: dict[str, Any] = {"modality": self.modality}
        for name, value in self.comparable_fields().items():
            config[name] = list(value) if isinstance(value, tuple) else value
        if self.extras:
            config["extras"] = dict(self.extras)
        return config


@dataclass(frozen=True)
class Resolution:
    """One authority resolution: provenance record + declared fields + transform.

    Attributes
    ----------
    record:
        The persisted-friendly :class:`~torchlens.data_classes.trace.\
ResolvedPreprocessing` provenance record (source, identifier, verified,
        config, description).
    declared:
        Normalized comparable fields, or ``None`` when the authority declared
        nothing (the record then says so).
    transform:
        The authority's own transform callable when it ships one (torchvision
        preset, timm transform, HF processor); ``None`` for declaration-only
        authorities (explicit dicts) and unknown resolutions.
    """

    record: ResolvedPreprocessing
    declared: DeclaredPreprocessing | None
    transform: Callable[[Any], Any] | None

    @property
    def status(self) -> str:
        """Authority standing of this resolution (closed vocabulary).

        Returns
        -------
        str
            ``"authoritative"`` / ``"unverified_fallback"`` / ``"unknown"``.
        """

        return status_of(self.record)


def unknown_resolution(reason: str, *, fetch: dict[str, Any] | None = None) -> Resolution:
    """Build the honest nothing-resolved record (tvscope memo D8).

    Network failure yields ``unknown`` -- never a lower tier's silent guess --
    and the fetch disclosure rides the config so the failure is inspectable.

    Parameters
    ----------
    reason:
        Human-readable one-line reason nothing resolved.
    fetch:
        Optional fetch disclosure block (attempted / performed / policy /
        error) from a network-gated adapter.

    Returns
    -------
    Resolution
        ``source="unknown"``, ``verified=False``, no declared fields, no
        transform.
    """

    from torchlens.data_classes.trace import ResolvedPreprocessing

    config: dict[str, Any] = {"reason": reason, "resolution_method": "none"}
    if fetch is not None:
        config["fetch"] = dict(fetch)
    return Resolution(
        record=ResolvedPreprocessing(
            source="unknown",
            identifier="unknown",
            verified=False,
            config=config,
            description=f"no preprocessing authority resolved: {reason}",
        ),
        declared=None,
        transform=None,
    )
