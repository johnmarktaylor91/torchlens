"""Appliance registry behind the ``trace.using(name)`` accessor door.

An appliance is an analysis toolkit bound to one :class:`~torchlens.Trace`
(for example ``trace.using("mi").head_scores(...)``). Appliances hang off this
single door instead of adding methods to ``Trace``; TorchLens's own appliances
and outside packages register the same way, through
:func:`register`. Spellings are DOCUMENTED-UNSTABLE.

A factory is either a callable taking the trace, or a ``"module:attribute"``
string naming one; the string form defers the import until the first
``trace.using(name)`` call, so registering costs nothing at import time.
The registry is one domain door over the private registry kernel
(``torchlens._registry``), like export targets and transforms.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .._errors import ArgumentConflictError, ArgumentTypeError, InvalidArgumentError
from .._registry import TORCHLENS_PROVIDER, ProviderInfo, Registry, create_registry

ApplianceFactory = Callable[[Any], Any]

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ApplianceEntry:
    """One registered appliance.

    Parameters
    ----------
    name:
        Name passed to ``trace.using``.
    factory:
        Callable taking the trace, or a ``"module:attribute"`` path to one.
    summary:
        One-line description shown by :func:`names`.
    """

    name: str
    factory: ApplianceFactory | str
    summary: str = ""


#: The one appliance registry (kernel-enrolled at import). TorchLens-owned
#: appliances register through :func:`register` like any outside package.
_APPLIANCES: Registry[ApplianceEntry] = create_registry("appliances", kind_label="appliance")


def _validated_name(name: object) -> str:
    """Return ``name`` when it is a usable appliance name, else refuse typed."""

    if not isinstance(name, str):
        raise ArgumentTypeError(
            f"appliance name must be a str; received {type(name).__name__}",
            code="appliance_registration_invalid",
            remedy="pass the appliance name as a string, e.g. trace.using('mi')",
            argument="name",
            received_type=type(name).__name__,
        )
    if not _NAME_PATTERN.match(name):
        raise InvalidArgumentError(
            f"appliance name {name!r} is not a valid name",
            code="appliance_registration_invalid",
            remedy=(
                "use a name that starts with a letter and contains only letters, digits, "
                "'_', '.' or '-', e.g. register('my_pkg', factory)"
            ),
            argument="name",
        )
    return name


def register(
    name: str,
    factory: ApplianceFactory | str,
    *,
    summary: str = "",
    replace: bool = False,
    provider: ProviderInfo | None = None,
) -> None:
    """Register an appliance so ``trace.using(name)`` returns it.

    Parameters
    ----------
    name:
        Appliance name; starts with a letter, then letters, digits, ``_``,
        ``.`` or ``-``.
    factory:
        Callable called with the trace on every ``trace.using(name)``, or a
        ``"module:attribute"`` string naming such a callable (imported lazily).
    summary:
        Optional one-line description.
    replace:
        Allow replacing an existing registration under the same name.
    provider:
        Optional provider identity for out-of-tree packages; defaults to an
        anonymous user provider row.

    Raises
    ------
    ArgumentTypeError
        ``appliance_registration_invalid`` when the name or factory has the
        wrong type.
    InvalidArgumentError
        ``appliance_registration_invalid`` when the name or factory path is
        malformed.
    ArgumentConflictError
        ``appliance_name_taken`` when ``name`` is registered and ``replace``
        is false.
    """

    checked = _validated_name(name)
    if isinstance(factory, str):
        module_path, _, attribute = factory.partition(":")
        if not module_path or not attribute:
            raise InvalidArgumentError(
                f"appliance factory path {factory!r} is not of the form 'module:attribute'",
                code="appliance_registration_invalid",
                remedy="pass a callable, or a path such as 'my_pkg.appliance:make'",
                argument="factory",
            )
    elif not callable(factory):
        raise ArgumentTypeError(
            f"appliance factory must be callable or a 'module:attribute' string; "
            f"received {type(factory).__name__}",
            code="appliance_registration_invalid",
            remedy="pass a callable that takes the trace, e.g. lambda trace: MyTools(trace)",
            argument="factory",
            received_type=type(factory).__name__,
        )
    if checked in _APPLIANCES.snapshot().values and not replace:
        raise ArgumentConflictError(
            f"an appliance named {checked!r} is already registered",
            code="appliance_name_taken",
            remedy=(
                f"choose another name, or pass replace=True to "
                f"register({checked!r}, factory, replace=True)"
            ),
            argument="name",
        )
    _APPLIANCES.register(
        checked,
        ApplianceEntry(checked, factory, str(summary)),
        capabilities={"lazy_import": isinstance(factory, str)},
        provider=provider if provider is not None else ProviderInfo(provider_id="user"),
        replace=replace,
    )


def names() -> dict[str, str]:
    """Return the registered appliance names with their summaries, sorted by name.

    Returns
    -------
    dict[str, str]
        ``name -> summary`` for every registered appliance.
    """

    entries = _APPLIANCES.snapshot().values
    return {name: entries[name].summary for name in sorted(entries)}


def _entry_for(name: object) -> ApplianceEntry:
    """Return the registry entry for ``name`` or refuse typed with the known names."""

    checked = _validated_name(name)
    entries = _APPLIANCES.snapshot().values
    entry = entries.get(checked)
    if entry is not None:
        return entry
    names = sorted(entries)
    if names:
        listed = ", ".join(repr(known) for known in names)
        remedy = f"use one of the registered appliances: trace.using({names[0]!r}); all: {listed}"
    else:
        listed = "none"
        remedy = (
            "no appliance is registered in this process; register one with "
            "torchlens.appliances.register(name, factory) and call trace.using(name)"
        )
    raise InvalidArgumentError(
        f"no appliance named {checked!r}; registered appliances: {listed}",
        code="appliance_unknown",
        remedy=remedy,
        argument="name",
        registered=tuple(names),
    )


def _resolve_factory(entry: ApplianceEntry) -> ApplianceFactory:
    """Return the callable for ``entry``, importing a ``module:attribute`` path on use."""

    factory = entry.factory
    if not isinstance(factory, str):
        return factory
    module_path, _, attribute = factory.partition(":")
    resolved: ApplianceFactory = getattr(importlib.import_module(module_path), attribute)
    return resolved


def open_appliance(trace: Any, name: str) -> Any:
    """Return the appliance ``name`` bound to ``trace`` (the ``trace.using`` body).

    Parameters
    ----------
    trace:
        Trace the appliance reads.
    name:
        Registered appliance name.

    Returns
    -------
    Any
        Whatever the registered factory returns for ``trace``.
    """

    return _resolve_factory(_entry_for(name))(trace)


register(
    "mi",
    "torchlens.mechinterp._appliance:MechInterpAppliance",
    summary="mechinterp kit: head_scores, qk_circuit, ov_circuit",
    provider=TORCHLENS_PROVIDER,
)
