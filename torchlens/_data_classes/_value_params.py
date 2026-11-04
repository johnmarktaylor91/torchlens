"""Parameter and buffer records: metadata first, explicit value basis."""

from __future__ import annotations

from typing import Any

from .._utils._display_card import inline_names, render_card
from .._utils._display_numbers import format_exact_count, format_iec_bytes
from .._utils.fail_open import fail_open
from ._value_ops import _names
from ._value_state import METADATA_ONLY, _statistics, metadata_text, value_line, value_words


def _param_facts(param: Any) -> tuple[str, Any, str, Any]:
    """Peek a live handle without scanning it or mutating its reference cache."""

    live = fail_open(lambda: param._peek_live_param(cache=False), lambda _error: None)
    stats = _statistics(live, compute=False, identity=getattr(param, "address", None))
    honesty = "" if live is not None else "(live unavailable)"
    return metadata_text(param, live), stats, honesty, live


def param_repr_line(param: Any) -> str:
    """A parameter never triggers a cold scan, including a tiny CPU parameter."""

    metadata, stats, honesty, _ = _param_facts(param)
    status = "trainable" if getattr(param, "is_trainable", False) else "frozen"
    return value_line(
        f"Param {getattr(param, 'address', '?')}", metadata, stats, honesty, (status,)
    )


def param_rows(param: Any) -> list[tuple[str, str]]:
    """Basis, geometry, ties and gradient facts shared with notebook HTML."""

    _, stats, _, live = _param_facts(param)
    version = (
        fail_open(lambda: int(live._version), lambda _error: None) if live is not None else None
    )
    basis = (
        f"live, version {version}"
        if version is not None
        else (
            "live"
            if live is not None
            else "live tensor unavailable: source model gone or ref released"
        )
    )
    rows = [("module", str(getattr(param, "module_address", "") or "")), ("basis", basis)]
    count = getattr(param, "num_params", None)
    memory = getattr(param, "param_memory", None)
    if count is not None and memory is not None:
        rows.append(
            ("size", f"{format_exact_count(count)} values, {format_iec_bytes(int(memory))}")
        )
    rows.append(("values", value_words(stats)))
    used_by = getattr(param, "used_by_layers", ()) or ()
    if used_by:
        tie = " (tied: feeds multiple sites)" if len(used_by) > 1 else ""
        rows.append(("used by", _names(param, used_by) + tie))
    linked = getattr(param, "co_parent_params", ()) or ()
    if linked:
        rows.append(("linked", inline_names(linked)))
    has_grad = getattr(param, "_has_grad", None)
    rows.append(("gradient", "captured" if has_grad else "none captured"))
    optimizer = getattr(param, "has_optimizer", None)
    if optimizer is not None:
        rows.append(("optimizer", "present" if optimizer else "none detected"))
    return rows


def param_card(param: Any) -> str:
    """Bounded parameter facts and exits, retaining tie and basis disclosure."""

    return render_card(param_repr_line(param), param_rows(param), (".grad", ".to_pandas()"))


def _buffer_facts(buffer: Any) -> tuple[str, Any, str]:
    """Capture snapshot metadata and cached evidence, never a live buffer scan."""

    initial = getattr(buffer, "_initial_value", None)
    if initial is None:
        versions = getattr(buffer, "versions", ()) or ()
        slot = getattr(versions[0], "_slot", None) if versions else None
        initial = slot("out") if callable(slot) else None
    stats = _statistics(initial, compute=True, identity=getattr(buffer, "address", None))
    basis = "snapshot at capture" if initial is not None else METADATA_ONLY
    return metadata_text(buffer, initial), stats, basis


def buffer_repr_line(buffer: Any) -> str:
    """Snapshot-based buffer identity with its version count."""

    metadata, stats, basis = _buffer_facts(buffer)
    versions = len(getattr(buffer, "versions", ()) or ())
    facts = (f"{format_exact_count(versions)} versions",) if versions > 1 else ()
    honesty = METADATA_ONLY if basis == METADATA_ONLY else ""
    return value_line(f"Buffer {getattr(buffer, 'address', '?')}", metadata, stats, honesty, facts)


def buffer_card(buffer: Any) -> str:
    """Snapshot basis and overwrite facts are distinct from current live state."""

    _, stats, basis = _buffer_facts(buffer)
    rows = [
        ("basis", basis),
        ("values", value_words(stats)),
        ("module", str(getattr(buffer, "module_address", "") or "")),
    ]
    overwrites = getattr(buffer, "num_overwrites", None)
    if overwrites:
        rows.append(("overwrites", format_exact_count(overwrites)))
    return render_card(buffer_repr_line(buffer), rows, (".versions", ".to_pandas()"))
