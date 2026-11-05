"""Shared receptive-field rule-registry isolation for the RF test suite.

Every ``tests/test_rf_*.py`` file needs the same process-global hygiene: the
RF rule registry (``torchlens.receptive_field._rules._RF_RULES``) and its
cache-invalidation epoch (``_RF_RULES_EPOCH``) must be snapshotted before a
test mutates them and restored exactly afterwards. Historically each file
hand-rolled one of three fixture variants; this module centralizes all of
them behind one context manager so the semantics stay identical everywhere
(R77 fixture-health round 3).

Variants expressed as arguments:

* ``preserved_rf_registry(bump_epoch=True)`` -- snapshot, clear the registry,
  bump the epoch (invalidates solved-DAG caches for the test body), restore.
* ``preserved_rf_registry(clear=False)`` -- snapshot only; the test keeps the
  currently installed rules and any additions are rolled back on exit.
* ``preserved_rf_registry(install_builtin=True)`` -- snapshot, clear, then
  install the built-in rule pack (captured lazily once per process and
  reused, exactly like the historical per-file ``_PACK`` caches), restore.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager

from torchlens.receptive_field import _rules

_BUILTIN_PACK: dict[str, object] | None = None


def _install_builtin_pack() -> None:
    """Install the built-in RF rule pack into the (already cleared) registry.

    Returns
    -------
    None
        The process-global registry is populated in place.

    Notes
    -----
    The pack is captured lazily ONCE per process: the first caller imports
    ``torchlens.receptive_field.rules`` and, when the registry is still empty
    (the idempotent import-time install already ran earlier in the process),
    reloads each built-in rule module so its registration decorators re-fire.
    Later callers replay the cached snapshot without touching the modules.
    """

    global _BUILTIN_PACK
    if _BUILTIN_PACK is None:
        module = importlib.import_module("torchlens.receptive_field._builtin_rules")
        if not _rules._RF_RULES:
            for name in module.__all__:
                importlib.reload(getattr(module, name))
        _BUILTIN_PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_BUILTIN_PACK)


@contextmanager
def preserved_rf_registry(
    install_builtin: bool = False,
    *,
    clear: bool = True,
    bump_epoch: bool = False,
) -> Iterator[None]:
    """Snapshot the RF rule registry, optionally reshape it, restore on exit.

    Parameters
    ----------
    install_builtin:
        Whether to install the built-in rule pack after clearing (implies
        ``clear=True`` semantics of the historical ``built_in_rule_pack``
        fixtures; the pack snapshot is cached process-wide).
    clear:
        Whether to empty the registry for the managed block. ``False``
        preserves the currently installed rules and only rolls back changes.
    bump_epoch:
        Whether to advance ``_RF_RULES_EPOCH`` after clearing so cached
        solutions keyed on the epoch are invalidated for the block. Only
        meaningful with ``clear=True``.

    Yields
    ------
    None
        Runs the managed block, then restores the exact prior registry
        contents and epoch (also on exception).
    """

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    if clear or install_builtin:
        _rules._RF_RULES.clear()
        if bump_epoch:
            _rules._RF_RULES_EPOCH += 1
    if install_builtin:
        _install_builtin_pack()
    try:
        yield
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(saved_rules)
        _rules._RF_RULES_EPOCH = saved_epoch
