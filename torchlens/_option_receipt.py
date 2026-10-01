"""Option explicitness readers, the frozen option-group kit, and the receipt.

Split out of ``torchlens/options.py`` (2026-08-26 P03 fix cycle) along the
honesty seam. Option receipt (compo wave 0, row 0.5; DOCUMENTED-UNSTABLE
spellings pending naming-session ratification): the honesty substrate for the
M3 wrapper x option matrix -- for every public option field, what was
REQUESTED, what is EFFECTIVE, whether it was explicit, and a machine-readable
reason. Effective values equal the constructed field values today; consumers
that FORCE, NORMALIZE, or REFUSE an option feed ``adjustments`` so the forcing
is never silent (FORCED-SILENTLY is banned as a steady state).

The ``_explicit_fields`` / ``_field_is_explicit`` readers over
``_specified_fields`` live here because the receipt is built on them, and the
frozen-group construction kit (``_resolve_option_value`` /
``_set_frozen_fields``) plus :class:`EchoOptions` (lane F28) complete the set;
``options.py`` imports them back (one-directional -- this module never
imports ``options``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

from ._deprecations import MISSING, MissingType
from ._errors import InvalidArgumentError


def _explicit_fields(instance: object) -> frozenset[str]:
    """Return explicitly supplied option fields from an option object.

    Parameters
    ----------
    instance:
        Option object carrying ``_specified_fields``.

    Returns
    -------
    frozenset[str]
        Explicit field names.
    """

    return cast(frozenset[str], getattr(instance, "_specified_fields"))


def _field_is_explicit(instance: object, field_name: str) -> bool:
    """Return whether an option object field was explicitly supplied.

    Parameters
    ----------
    instance:
        Option object carrying ``_specified_fields``.
    field_name:
        Canonical field name to test.

    Returns
    -------
    bool
        Whether the field was explicit.
    """

    return field_name in _explicit_fields(instance)


def _resolve_option_value(
    field_name: str,
    supplied_value: Any,
    default_value: Any,
    specified_fields: set[str],
) -> Any:
    """Resolve an option field while tracking explicit caller presence.

    Parameters
    ----------
    field_name:
        Dataclass field name being resolved.
    supplied_value:
        Value supplied by the caller, or ``MISSING``.
    default_value:
        Public default for the field.
    specified_fields:
        Mutable set populated with fields explicitly supplied by the caller.

    Returns
    -------
    Any
        Resolved field value.
    """

    if supplied_value is MISSING:
        return default_value
    specified_fields.add(field_name)
    return supplied_value


def _set_frozen_fields(
    instance: object, field_names: tuple[str, ...], values: Mapping[str, Any]
) -> None:
    """Set dataclass fields while constructing a frozen options object.

    Parameters
    ----------
    instance:
        Instance under construction.
    field_names:
        Ordered dataclass field names to populate.
    values:
        Resolved field values.
    """

    for field_name in field_names:
        object.__setattr__(instance, field_name, values[field_name])


_ECHO_FIELDS: Final[tuple[str, ...]] = (
    "select",
    "stats",
    "sink",
    "tail_on_error",
    "on_error_only",
    "max_lines",
)


@dataclass(frozen=True, init=False)
class EchoOptions:
    """Grouped live-narration options for ``echo=`` (lane F28, snoop memo D1).

    ``echo=`` on ``tl.trace``, ``tl.record``, and ``fastlog.Recorder``
    narrates capture events -- one line per event -- through one read-only
    observer slot. Narration and retention are fully independent: "narrate
    everything, keep nothing" is legal and is the fast path. All spellings
    are DOCUMENTED-UNSTABLE pending the naming sprint. The public spelling
    stays ``tl.options.EchoOptions``; the class lives beside the frozen-group
    kit it is built on.

    Parameters
    ----------
    select:
        Narration scope: ``True`` (every completed tensor-output event plus
        module structure lines), ``"modules"`` (structure lines only -- a
        cheap flight recorder), or any LIVE-evaluable selector/callable
        (``tl.func``, ``tl.in_module``, boolean combinations, bounded
        ``followed_by``). Selectors needing finalized labels refuse typed
        BEFORE execution; the remedy is post-hoc ``.narrate()``.
    stats:
        Value-stats rung: ``"off"`` (default; metadata only, zero value
        reads, zero device syncs), ``"reuse"`` (only facts another armed
        feature already paid for), ``"sampled"`` (bounded seeded-sample
        moments, ``~``-marked, NEVER a finiteness claim), or ``"exact"``
        (full exact stats; typed refusal above the documented numel budget).
    sink:
        Narration destination: ``None`` (stderr; stdout stays clean), a
        path (per-line flush -- bytes on disk survive a hard process
        death), an open text stream, or ``callable(str)`` (the sole seam to
        trackers/loggers).
    tail_on_error:
        Bound of the always-riding crash deque (metadata-only rendered
        lines). On a forward exception the sink synchronously receives the
        last ``tail_on_error`` selected events, oldest first. ``max_lines``
        never caps this deque.
    on_error_only:
        Narrate NOTHING in the happy path and speak only on failure --
        cheap enough to recommend as a standing training-loop habit.
    max_lines:
        Opt-in live-volume bound. Reaching it emits ONE suppression marker
        with the exact count, keeps counting (the footer discloses the
        total), and never silently samples.

    Examples
    --------
    >>> opts = EchoOptions(select=True, stats="off", tail_on_error=20)
    >>> opts.stats
    'off'
    """

    select: Any = True
    stats: str = "off"
    sink: Any = None
    tail_on_error: int = 20
    on_error_only: bool = False
    max_lines: int | None = None
    _specified_fields: frozenset[str] = dataclasses.field(
        default_factory=frozenset, init=False, repr=False
    )

    def __init__(  # noqa: PLR0913 - an option-group constructor mirrors its public fields
        self,
        select: Any | MissingType = MISSING,
        stats: str | MissingType = MISSING,
        sink: Any | MissingType = MISSING,
        *,
        tail_on_error: int | MissingType = MISSING,
        on_error_only: bool | MissingType = MISSING,
        max_lines: int | None | MissingType = MISSING,
    ) -> None:
        """Initialize a frozen echo option bundle."""

        specified_fields: set[str] = set()
        values: dict[str, Any] = {
            "select": _resolve_option_value("select", select, True, specified_fields),
            "stats": _resolve_option_value("stats", stats, "off", specified_fields),
            "sink": _resolve_option_value("sink", sink, None, specified_fields),
            "tail_on_error": _resolve_option_value(
                "tail_on_error", tail_on_error, 20, specified_fields
            ),
            "on_error_only": _resolve_option_value(
                "on_error_only", on_error_only, False, specified_fields
            ),
            "max_lines": _resolve_option_value("max_lines", max_lines, None, specified_fields),
        }
        _set_frozen_fields(self, _ECHO_FIELDS, values)
        object.__setattr__(self, "_specified_fields", frozenset(specified_fields))

    def as_dict(self) -> dict[str, Any]:
        """Return the option values as a plain dictionary."""

        return {field_name: getattr(self, field_name) for field_name in _ECHO_FIELDS}

    def is_field_explicit(self, field_name: str) -> bool:
        """Return whether a field was explicitly supplied by the caller."""

        return _field_is_explicit(self, field_name)

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        specified_fields: frozenset[str],
    ) -> EchoOptions:
        """Build an instance from already-resolved field values."""

        instance = object.__new__(cls)
        _set_frozen_fields(instance, _ECHO_FIELDS, values)
        object.__setattr__(instance, "_specified_fields", specified_fields)
        return instance


OPTION_RECEIPT_REASONS: Final = ("explicit", "default", "forced", "normalized", "refused")
"""Closed vocabulary of option-receipt reasons."""


@dataclasses.dataclass(frozen=True)
class OptionReceiptEntry:
    """One option's honesty record (requested / effective / explicitness / reason).

    Parameters
    ----------
    name:
        Public option field name.
    requested:
        The value the caller supplied, or ``MISSING`` when the field was not
        explicitly requested.
    effective:
        The value in force after defaulting and any declared adjustment.
    explicit:
        Whether the caller supplied the value.
    reason:
        One of :data:`OPTION_RECEIPT_REASONS` explaining how ``effective``
        came to hold.
    """

    name: str
    requested: Any
    effective: Any
    explicit: bool
    reason: str


def option_receipt(
    options: object,
    *,
    adjustments: Mapping[str, tuple[Any, str]] | None = None,
) -> tuple[OptionReceiptEntry, ...]:
    """Build the per-field option receipt for one options object.

    Parameters
    ----------
    options:
        Any TorchLens options dataclass carrying ``_specified_fields``
        (``CaptureOptions`` and its five siblings).
    adjustments:
        Field name -> ``(effective_value, reason)`` for options a consumer
        forced, normalized, or refused after construction. Reasons must come
        from :data:`OPTION_RECEIPT_REASONS`; unknown field names refuse.

    Returns
    -------
    tuple[OptionReceiptEntry, ...]
        One entry per PUBLIC field, in dataclass field order.
    """

    if not dataclasses.is_dataclass(options) or not hasattr(options, "_specified_fields"):
        raise InvalidArgumentError(
            f"option_receipt() needs a TorchLens options object, got {type(options).__name__!r}.",
            code="option_receipt_not_options",
            remedy="Pass a constructed options dataclass such as tl.options.CaptureOptions(...).",
        )
    field_names = [
        dataclass_field.name
        for dataclass_field in dataclasses.fields(options)
        if not dataclass_field.name.startswith("_")
    ]
    supplied = dict(adjustments or {})
    unknown = set(supplied) - set(field_names)
    if unknown:
        raise InvalidArgumentError(
            f"option_receipt() adjustments name unknown option fields: {sorted(unknown)}.",
            code="option_receipt_unknown_field",
            remedy=f"Use public {type(options).__name__} field names for adjustment keys.",
        )
    entries: list[OptionReceiptEntry] = []
    for name in field_names:
        constructed = getattr(options, name)
        explicit = _field_is_explicit(options, name)
        requested = constructed if explicit else MISSING
        if name in supplied:
            effective, reason = supplied[name]
            if reason not in OPTION_RECEIPT_REASONS[2:]:
                raise InvalidArgumentError(
                    f"option_receipt() adjustment reason {reason!r} for {name!r} is not "
                    f"one of {OPTION_RECEIPT_REASONS[2:]}.",
                    code="option_receipt_reason_invalid",
                    remedy="Adjustment reasons are 'forced', 'normalized', or 'refused'; "
                    "'explicit' and 'default' are derived, never supplied.",
                )
        else:
            effective = constructed
            reason = "explicit" if explicit else "default"
        entries.append(
            OptionReceiptEntry(
                name=name,
                requested=requested,
                effective=effective,
                explicit=explicit,
                reason=reason,
            )
        )
    return tuple(entries)
