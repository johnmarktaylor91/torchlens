"""Attribute reads for the tensor search that keep property getters' log records quiet."""

from __future__ import annotations

import functools
import logging
from typing import Any

_PROPERTY_TYPES = (property, functools.cached_property)


def read_attr_quietly(item: Any, name: str) -> Any:
    """Return ``getattr(item, name)``, muting log records below ERROR while a property runs.

    The tensor search reads every attribute of an arbitrary object, so it runs property
    getters the user never called. ``get_vars_of_type_from_obj`` already ignores the
    ``warnings`` they raise; a library that announces a deprecated property through
    ``logging`` instead (a generation cache's size properties) would otherwise print the
    notice as if the user's code had read it. Plain attributes are read unchanged.
    """
    try:
        is_property = isinstance(getattr(type(item), name, None), _PROPERTY_TYPES)
    except Exception:
        is_property = False
    previous = logging.root.manager.disable
    if not is_property or previous >= logging.WARNING:
        return getattr(item, name)
    logging.disable(logging.WARNING)
    try:
        return getattr(item, name)
    finally:
        logging.disable(previous)
