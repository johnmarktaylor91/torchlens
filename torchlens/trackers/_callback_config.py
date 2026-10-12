"""Environment configuration for the explicit watch callbacks, with sources (U34.2).

An explicitly constructed callback may read its CONFIGURATION from the
environment (cluster users configure jobs that way); the environment can
still never turn instrumentation ON -- constructing the callback in user
source is the enablement act, and ``TORCHLENS_WATCH_DISABLE`` stays the one
off-only switch. Every setting resolves in one fixed order and the callback
prints where each value came from when it attaches:

1. an explicit constructor argument (``argument``);
2. a ``TORCHLENS_WATCH_*`` environment variable (``env NAME``);
3. a framework-derived value (``every`` only: ``derived: logging_steps=N``);
4. the package default (``default``).

Variables (spellings DOCUMENTED-UNSTABLE): ``TORCHLENS_WATCH_SIGNALS``
(comma list from the closed signal vocabulary), ``TORCHLENS_WATCH_SELECT``
(comma list of module path / parameter prefixes), ``TORCHLENS_WATCH_EVERY``
(positive int), ``TORCHLENS_WATCH_HIST_EVERY`` (positive int, or ``none``).
A set but malformed variable refuses typed, naming the variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from ._errors import WatchConfigError
from ._watch import SIGNALS

__tl_layer__ = "L8"

WATCH_SIGNALS_ENV = "TORCHLENS_WATCH_SIGNALS"
WATCH_SELECT_ENV = "TORCHLENS_WATCH_SELECT"
WATCH_EVERY_ENV = "TORCHLENS_WATCH_EVERY"
WATCH_HIST_EVERY_ENV = "TORCHLENS_WATCH_HIST_EVERY"

#: The configuration variables an explicit callback reads (never activation).
CALLBACK_CONFIG_ENV = (WATCH_SIGNALS_ENV, WATCH_SELECT_ENV, WATCH_EVERY_ENV, WATCH_HIST_EVERY_ENV)

#: Sentinel for "not passed" where ``None`` is itself a meaningful value.
UNSET: Any = object()

_DEFAULT_SIGNALS = ("gradients",)


@dataclass(frozen=True)
class CallbackSettings:
    """The resolved watch settings of one callback and where each came from.

    ``every`` is ``None`` until the framework's cadence is known at attach
    (see :meth:`with_every`).
    """

    signals: tuple[str, ...]
    select: tuple[str, ...] | None
    every: int | None
    hist_every: int | None
    sources: dict[str, str] = field(default_factory=dict)

    def with_every(self, every: int, source: str) -> CallbackSettings:
        """Return these settings with the attach-time cadence filled in."""

        return replace(self, every=every, sources={**self.sources, "every": source})

    def format(self, owner: str) -> str:
        """One plain line per setting: ``name=value  [source]``."""

        values = {
            "signals": self.signals,
            "select": self.select,
            "every": self.every,
            "hist_every": self.hist_every,
        }
        lines = [f"TorchLens watch settings ({owner}):"]
        lines.extend(
            f"  {name}={value!r}  [{self.sources.get(name, 'default')}]"
            for name, value in values.items()
        )
        return "\n".join(lines)


def _raw(name: str) -> str | None:
    """A set, nonempty variable's stripped value, else ``None``."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _refuse(name: str, raw: str, expected: str) -> WatchConfigError:
    """The typed refusal for a malformed configuration variable."""

    return WatchConfigError(
        f"{name}={raw!r} is not {expected}. A malformed configuration variable "
        "refuses rather than silently falling back to a default.",
        code="watch_env_config_invalid",
        variable=name,
        value=raw,
        remedy=f"Set {name} to {expected}, or unset it.",
    )


def _env_list(name: str) -> tuple[str, ...] | None:
    """Parse a comma list variable (``None`` when unset)."""

    raw = _raw(name)
    if raw is None:
        return None
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not items:
        raise _refuse(name, raw, "a nonempty comma-separated list")
    return items


def _env_positive_int(name: str, *, allow_none: bool) -> tuple[bool, int | None]:
    """Parse a positive-int variable: ``(was_set, value)``; ``none`` when allowed."""

    raw = _raw(name)
    if raw is None:
        return (False, None)
    if allow_none and raw.lower() in ("none", "off"):
        return (True, None)
    expected = "a positive integer" + (" or 'none'" if allow_none else "")
    try:
        value = int(raw)
    except ValueError as exc:
        raise _refuse(name, raw, expected) from exc
    if value < 1:
        raise _refuse(name, raw, expected)
    return (True, value)


def _env_signals() -> tuple[str, ...] | None:
    """Read ``TORCHLENS_WATCH_SIGNALS`` against the closed signal vocabulary."""

    signals = _env_list(WATCH_SIGNALS_ENV)
    if signals is not None and any(signal not in SIGNALS for signal in signals):
        raise _refuse(WATCH_SIGNALS_ENV, ",".join(signals), f"a comma list from {SIGNALS}")
    return signals


def _resolve_signals(signals: Any) -> tuple[tuple[str, ...], str]:
    """``signals``: argument, else environment, else the default."""

    if signals is not None:
        return (tuple(signals), "argument")
    env_signals = _env_signals()
    if env_signals is not None:
        return (env_signals, f"env {WATCH_SIGNALS_ENV}")
    return (_DEFAULT_SIGNALS, "default")


def _resolve_select(select: Any) -> tuple[tuple[str, ...] | None, str]:
    """``select``: argument, else environment, else every site."""

    if select is not None:
        return (tuple(select), "argument")
    env_select = _env_list(WATCH_SELECT_ENV)
    if env_select is not None:
        return (env_select, f"env {WATCH_SELECT_ENV}")
    return (None, "default")


def _resolve_every(every: int | None) -> tuple[int | None, str]:
    """``every``: argument, else environment, else derived at attach (``None``)."""

    if every is not None:
        return (every, "argument")
    was_set, env_every = _env_positive_int(WATCH_EVERY_ENV, allow_none=False)
    if was_set:
        return (env_every, f"env {WATCH_EVERY_ENV}")
    return (None, "default")


def _resolve_hist_every(hist_every: Any) -> tuple[int | None, str]:
    """``hist_every``: argument (``None`` included), else environment, else off."""

    if hist_every is not UNSET:
        return (hist_every, "argument")
    was_set, env_hist = _env_positive_int(WATCH_HIST_EVERY_ENV, allow_none=True)
    return (env_hist, f"env {WATCH_HIST_EVERY_ENV}" if was_set else "default")


def resolve_callback_settings(
    *,
    signals: Any,
    select: Any,
    every: int | None,
    hist_every: Any,
) -> CallbackSettings:
    """Resolve each setting: argument, then environment, then default.

    ``signals``/``select``/``every`` treat ``None`` as "not passed";
    ``hist_every`` uses :data:`UNSET` because an explicit ``None`` (no
    histograms) is a real choice.
    """

    resolved_signals, signals_source = _resolve_signals(signals)
    resolved_select, select_source = _resolve_select(select)
    resolved_every, every_source = _resolve_every(every)
    resolved_hist, hist_source = _resolve_hist_every(hist_every)
    return CallbackSettings(
        signals=resolved_signals,
        select=resolved_select,
        every=resolved_every,
        hist_every=resolved_hist,
        sources={
            "signals": signals_source,
            "select": select_source,
            "every": every_source,
            "hist_every": hist_source,
        },
    )


def derived_every_source(logging_steps: int | None, max_steps: int | None) -> str:
    """Name the framework signal :func:`derived_cadence` used for ``every``."""

    if logging_steps is not None and logging_steps > 0:
        return f"derived: logging_steps={logging_steps}"
    if max_steps is not None and max_steps > 0:
        return f"derived: max_steps={max_steps}"
    return "default"


__all__ = [
    "CALLBACK_CONFIG_ENV",
    "UNSET",
    "WATCH_EVERY_ENV",
    "WATCH_HIST_EVERY_ENV",
    "WATCH_SELECT_ENV",
    "WATCH_SIGNALS_ENV",
    "CallbackSettings",
    "derived_every_source",
    "resolve_callback_settings",
]
