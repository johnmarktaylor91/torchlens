"""Attribute reads for the tensor search that keep property getters' log records quiet."""

from __future__ import annotations

import functools
import logging
import sys
import threading
from typing import Any

from ._log_mute import MUTED_LEVEL, mute, unmute

_PROPERTY_TYPES = (property, functools.cached_property)

_UNMUTED = object()
"""Scope state: inside a search that has not muted log records yet."""

_scope = threading.local()
"""Per-thread state of the innermost :class:`_QuietPropertyLogs` scope.

``_scope.restore`` is absent outside every scope, :data:`_UNMUTED` inside a scope
that has not muted yet, and otherwise the ``logging.disable`` level to restore when
that scope ends (:func:`~torchlens._utils._log_mute.unmute`'s ``previous``).
"""


class _QuietPropertyLogs:
    """One tensor search's mute window for property getters' log records.

    The mute (:mod:`torchlens._utils._log_mute`) has ``logging.disable(WARNING)``'s
    effect without clearing any logger's level cache. Inside this scope the first
    property read mutes and the scope's exit restores the exact level found then,
    so a search mutes at most once. The window opens at the first property read and
    closes when the search returns or raises; it covers only TorchLens's own
    attribute walk, and the user's logging state is as found when it closes.
    """

    __slots__ = ("_outer",)

    def __enter__(self) -> None:
        self._outer = getattr(_scope, "restore", None)
        _scope.restore = _UNMUTED

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        restore = _scope.restore
        if self._outer is None:
            del _scope.restore
        else:
            _scope.restore = self._outer
        if restore is not _UNMUTED:
            unmute(restore)


def read_attr_quietly(item: Any, name: str) -> Any:
    """Return ``getattr(item, name)``, muting log records below ERROR while a property runs.

    The tensor search reads every attribute of an arbitrary object, so it runs property
    getters the user never called. ``get_vars_of_type_from_obj`` already ignores the
    ``warnings`` they raise; a library that announces a deprecated property through
    ``logging`` instead (a generation cache's size properties) would otherwise print the
    notice as if the user's code had read it. Plain attributes are read unchanged.
    Inside a :class:`_QuietPropertyLogs` scope the mute stays on until the scope ends.
    """
    try:
        is_property = isinstance(getattr(type(item), name, None), _PROPERTY_TYPES)
    except Exception:
        is_property = False
    previous = logging.root.manager.disable
    if not is_property or previous >= MUTED_LEVEL:
        return getattr(item, name)
    mute()
    if getattr(_scope, "restore", None) is _UNMUTED:
        _scope.restore = previous
        return getattr(item, name)
    try:
        return getattr(item, name)
    finally:
        unmute(previous)


def class_crawl_names(obj_type: type) -> list[str]:
    """Return the class-level attribute names the tensor search may read on ``obj_type``.

    Normally ``dir(obj_type)``. A ``torch.distributions.Distribution`` keeps its tensor
    state in its instance dictionary (the output-container contract records exactly
    that); its class-level names are derived values (``mean``, ``variance``,
    ``stddev``) and lazy properties (``probs``, ``logits``). Reading one runs torch math
    after the forward (``Exponential.mean`` is ``rate.reciprocal()``), so the search
    would report a fresh tensor no traced op produced as a model output, and a lazy
    property would write a new tensor into the distribution. Its instance attributes
    are still crawled (the caller unions them in).
    """
    module = sys.modules.get("torch.distributions.distribution")
    base = getattr(module, "Distribution", None)
    if isinstance(base, type) and issubclass(obj_type, base):
        return []
    return dir(obj_type)
