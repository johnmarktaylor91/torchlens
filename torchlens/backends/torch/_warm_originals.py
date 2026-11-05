"""Show torch's original callables while a derived identity cache is warmed.

Torch memoizes some tables of callables resolved from the torch namespace at
first call (``torch.utils._device._device_constructors``, dynamo's rule
maps). ``wrappers._warm_derived_identity_caches`` warms them before wrappers
are installed so they stay keyed by the ORIGINALS for the whole wrapped epoch.
A decoration retry runs that warm while the slots a failed pass
already replaced still hold torchlens wrappers, so the rebuilt table holds
wrappers and ``DeviceContext.__torch_function__``, which receives originals,
stops injecting the device. :func:`originals_exposed` puts the recorded
originals back on every slot that holds a torchlens wrapper for the duration
of the warm and reinstalls the same wrappers afterwards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from ... import _state
from ..._constants import get_orig_torch_funcs
from ..._utils._torch_compat import get_optional_torch_namespace

SetAttr = Callable[[Any, str, Any], None]
"""The setattr the wrap/unwrap paths use for torch namespace slots."""


def _wrapped_slot_names(site_table: Iterable[Sequence[str]]) -> Iterator[tuple[str, str]]:
    """Yield every ``(namespace, attribute)`` slot torchlens may have wrapped.

    Parameters
    ----------
    site_table:
        The transform sites; each row starts with ``(namespace, attribute)``.

    Returns
    -------
    Iterator[tuple[str, str]]
        The torch-function pairs, then the transform sites, matching what
        ``unwrap_torch`` restores.
    """

    yield from get_orig_torch_funcs()
    for site in site_table:
        yield site[0], site[1]


def _swap_in_originals(
    set_slot: SetAttr, site_table: Iterable[Sequence[str]], decorated_to_orig: dict[int, Any]
) -> list[tuple[Any, str, Any]]:
    """Put the original back on each slot that currently holds a torchlens wrapper.

    Parameters
    ----------
    set_slot:
        The setattr used for torch namespace slots.
    site_table:
        The transform sites to visit after the torch-function pairs.
    decorated_to_orig:
        The read-only ``id(wrapper) -> original`` unwrap ledger.

    Returns
    -------
    list[tuple[Any, str, Any]]
        ``(namespace, attribute, wrapper)`` for every slot swapped, so the
        caller can reinstall the same wrappers.
    """

    swapped: list[tuple[Any, str, Any]] = []
    for namespace_name, func_name in _wrapped_slot_names(site_table):
        namespace = get_optional_torch_namespace(namespace_name)
        if namespace is None or not hasattr(namespace, func_name):
            continue
        current = getattr(namespace, func_name)
        original = decorated_to_orig.get(id(current))
        if original is None:
            continue
        try:
            set_slot(namespace, func_name, original)
        except (AttributeError, TypeError):
            continue
        swapped.append((namespace, func_name, current))
    return swapped


@contextmanager
def originals_exposed(set_slot: SetAttr, site_table: Iterable[Sequence[str]]) -> Iterator[None]:
    """Expose the recorded originals on wrapped slots for the body of the block.

    With no wrapper ever recorded (the first wrap) the namespace already holds
    originals and nothing is touched. After an ``unwrap_torch`` no slot holds
    a wrapper, so nothing is swapped either. The caller passes the setattr and
    transform sites so this module stays outside the wrapper import cycle.

    Parameters
    ----------
    set_slot:
        The setattr used for torch namespace slots.
    site_table:
        The transform sites (builder and direct) torchlens instruments.

    Returns
    -------
    Iterator[None]
        A context in which every torchlens-wrapped slot shows its original;
        on exit the same wrapper objects are reinstalled.
    """

    decorated_to_orig, _ = _state.wrap_epoch_ledgers()
    if not decorated_to_orig:
        yield
        return
    swapped = _swap_in_originals(set_slot, site_table, decorated_to_orig)
    try:
        yield
    finally:
        for namespace, func_name, wrapper in reversed(swapped):
            set_slot(namespace, func_name, wrapper)
