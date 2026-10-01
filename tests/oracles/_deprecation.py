"""The bidirectional deprecation-door gate (oracles D6/D21, build item 5).

Contract: a DeprecationWarning fires at CALL TIME if and only if the registry
says the door is deprecated; a registered door's warning must NAME the
registered replacement, and the replacement must RESOLVE. Both failure
directions go red:

- a door that warns without a registry row is an UNDOCUMENTED deprecation;
- a registered door that does not warn (or whose replacement is gone) is a
  SILENTLY OVERWRITTEN shim.

Behavior predicates root in OBSERVED CALLS, never metadata (D6): at the
panel's SHA, five of seven shims presented the implementation's
module/doc/signature to every introspective route and revealed themselves
only when called. The registry is EMPTY today -- the package is pinned
deprecation-free by tests/test_deprecation_inventory.py (static side; this
gate owns the call-time side) -- and the plants prove both red directions on
a planted registry so emptiness is never mistaken for a dead channel.
"""

from __future__ import annotations

import csv
import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class DeprecatedDoor:
    """One registered deprecated door.

    Parameters
    ----------
    door:
        Dotted path of the deprecated callable (e.g. ``torchlens.old_name``).
    replacement:
        Dotted path of the registered replacement; must resolve.
    since:
        ISO date the deprecation was registered.
    """

    door: str
    replacement: str
    since: str


def load_deprecated_doors(path: Path | None = None) -> tuple[DeprecatedDoor, ...]:
    """Load the deprecation registry.

    Parameters
    ----------
    path:
        Registry path; defaults to the committed ``data/deprecated_doors.tsv``.

    Returns
    -------
    tuple[DeprecatedDoor, ...]
        Registered doors (empty today).
    """

    registry = path if path is not None else DATA_DIR / "deprecated_doors.tsv"
    with registry.open(newline="") as handle:
        return tuple(
            DeprecatedDoor(**record)
            for record in csv.DictReader(
                (line for line in handle if not line.startswith("#")), delimiter="\t", restval=""
            )
        )


def resolve_dotted(dotted: str) -> Any:
    """Resolve ``module.attr`` (or a bare module path) to an object.

    Parameters
    ----------
    dotted:
        Dotted path; the longest importable prefix is imported and the
        remainder resolved by ``getattr``.

    Returns
    -------
    Any
        The resolved object.

    Raises
    ------
    AttributeError, ModuleNotFoundError
        When the path does not resolve -- a registered replacement that fails
        here is a red finding, not a skip.
    """

    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ModuleNotFoundError(dotted)


def call_time_deprecations(invoke: Callable[[], Any]) -> tuple[warnings.WarningMessage, ...]:
    """Invoke one door template and collect its DeprecationWarnings.

    Parameters
    ----------
    invoke:
        Zero-arg callable performing one real invocation of the door.

    Returns
    -------
    tuple[warnings.WarningMessage, ...]
        The DeprecationWarning-category messages the CALL emitted.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        invoke()
    return tuple(
        message
        for message in caught
        if issubclass(message.category, (DeprecationWarning, PendingDeprecationWarning))
    )


def audit_registered_door(
    door: DeprecatedDoor,
    invoke: Callable[[], Any],
) -> tuple[str, ...]:
    """Audit one REGISTERED door bidirectionally; return findings.

    Parameters
    ----------
    door:
        The registry row under audit.
    invoke:
        Zero-arg callable performing one real invocation of the door.

    Returns
    -------
    tuple[str, ...]
        Human-readable findings; empty means the row honors the contract
        (warns at call time, names the replacement, replacement resolves).
    """

    findings: list[str] = []
    try:
        resolve_dotted(door.replacement)
    except (AttributeError, ModuleNotFoundError):
        findings.append(f"{door.door}: registered replacement {door.replacement} does not resolve")
    emitted = call_time_deprecations(invoke)
    if not emitted:
        findings.append(
            f"{door.door}: registered as deprecated but a CALL emits no "
            "DeprecationWarning (silently overwritten shim)"
        )
    else:
        replacement_leaf = door.replacement.rsplit(".", 1)[-1]
        if not any(replacement_leaf in str(message.message) for message in emitted):
            findings.append(
                f"{door.door}: warning does not name the registered replacement {door.replacement}"
            )
    return tuple(findings)
