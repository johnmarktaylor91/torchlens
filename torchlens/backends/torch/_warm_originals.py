"""Show torch's original callables while a derived identity cache is warmed.

Torch memoizes some tables of callables resolved from the torch namespace at
first call (``torch.utils._device._device_constructors``, dynamo's rule
maps). ``wrappers._warm_derived_identity_caches`` warms them before wrappers
are installed so they stay keyed by the ORIGINALS for the whole wrapped epoch
(B8-4, R56). A decoration retry runs that warm while the slots a failed pass
already replaced still hold torchlens wrappers, so the rebuilt table holds
wrappers and ``DeviceContext.__torch_function__``, which receives originals,
stops injecting the device. :func:`originals_exposed` puts the recorded
originals back on every slot that holds a torchlens wrapper for the duration
of the warm and reinstalls the same wrappers afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ... import _state
from ..._constants import get_orig_torch_funcs
from ..._utils._torch_compat import get_optional_torch_namespace
from ._transform_wrappers import (
    DIRECT_TRANSFORM_SITES,
    TRANSFORM_BUILDER_SITES,
    _setattr_ignoring_advisories,
)


def _wrapped_slot_names() -> Iterator[tuple[str, str]]:
    """Yield every ``(namespace, attribute)`` slot torchlens may have wrapped.

    Returns
    -------
    Iterator[tuple[str, str]]
        The torch-function pairs, then the transform-builder and direct
        transform sites, matching what ``unwrap_torch`` restores.
    """

    yield from get_orig_torch_funcs()
    for site in TRANSFORM_BUILDER_SITES:
        yield site[0], site[1]
    for site in DIRECT_TRANSFORM_SITES:
        yield site[0], site[1]


def _swap_in_originals() -> list[tuple[Any, str, Any]]:
    """Put the original back on each slot that currently holds a torchlens wrapper.

    Returns
    -------
    list[tuple[Any, str, Any]]
        ``(namespace, attribute, wrapper)`` for every slot swapped, so the
        caller can reinstall the same wrappers.
    """

    swapped: list[tuple[Any, str, Any]] = []
    for namespace_name, func_name in _wrapped_slot_names():
        namespace = get_optional_torch_namespace(namespace_name)
        if namespace is None or not hasattr(namespace, func_name):
            continue
        current = getattr(namespace, func_name)
        original = _state._decorated_to_orig.get(id(current))
        if original is None:
            continue
        try:
            _setattr_ignoring_advisories(namespace, func_name, original)
        except (AttributeError, TypeError):
            continue
        swapped.append((namespace, func_name, current))
    return swapped


@contextmanager
def originals_exposed() -> Iterator[None]:
    """Expose the recorded originals on wrapped slots for the body of the block.

    With no wrapper ever recorded (the first wrap) the namespace already holds
    originals and nothing is touched. After an ``unwrap_torch`` no slot holds
    a wrapper, so nothing is swapped either.

    Returns
    -------
    Iterator[None]
        A context in which every torchlens-wrapped slot shows its original;
        on exit the same wrapper objects are reinstalled.
    """

    if not _state._decorated_to_orig:
        yield
        return
    swapped = _swap_in_originals()
    try:
        yield
    finally:
        for namespace, func_name, wrapper in reversed(swapped):
            _setattr_ignoring_advisories(namespace, func_name, wrapper)
