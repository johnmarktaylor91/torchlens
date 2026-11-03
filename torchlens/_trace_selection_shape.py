"""Shape queries over a public ``layers_to_save`` selection.

Each helper inspects the selection value alone (strings, integers, and
iterables of them) to say when it can be resolved: during the forward, or only
against the final graph numbering.
"""

from __future__ import annotations

import collections.abc
import re

# Numeric label components that depend on FINAL graph numbering only ever
# appear at the start of a request or after "_" (type index, layer ordinal).
# Digits embedded in stable text such as ``conv2d``, ``fc1``, or module
# addresses like ``features.3`` never sit in those positions, so they stay
# resolvable during the forward, as do ``:<pass>`` qualifiers (capture-time
# pass indexes are a live fact with a locked absorbed-path contract).
_FINAL_INDEX_STRING_PATTERN = re.compile(r"^\d|_\d")


def _selector_component_is_final_only(component: object) -> bool:
    """Return whether one selection component needs final graph numbering.

    Parameters
    ----------
    component
        One public ``layers_to_save`` component.

    Returns
    -------
    bool
        ``True`` when the component can only resolve after postprocess.

    Integer components are final layer ordinals, and label-shaped strings
    (``relu_1_2``, ``relu_1``) embed final type indexes or layer ordinals.
    Orphan removal renumbers both after capture, so matching them against
    capture-time raw indexes silently selects the wrong operation or nothing.
    """

    if isinstance(component, bool):
        return False
    if isinstance(component, int):
        return True
    if isinstance(component, str):
        return (
            component.startswith(("output", "identity"))
            or _FINAL_INDEX_STRING_PATTERN.search(component) is not None
        )
    return False


def _layers_to_save_needs_final_resolution(layers_to_save: object) -> bool:
    """Return whether any selection component needs final graph numbering.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when at least one component must resolve on the deferred
        post-postprocess path.
    """

    if isinstance(layers_to_save, (str, int)):
        return _selector_component_is_final_only(layers_to_save)
    if isinstance(layers_to_save, collections.abc.Iterable):
        return any(
            _layers_to_save_needs_final_resolution(component) for component in layers_to_save
        )
    return False


def _selector_requires_unwindowed_escrow(selector: object) -> bool:
    """Return whether a deferred selection forbids tail-window escrow eviction.

    Parameters
    ----------
    selector
        Public ``layers_to_save`` or gradient selection.

    Returns
    -------
    bool
        ``True`` when some component resolves against final graph numbering
        at an arbitrary graph position, so its escrowed payload must survive
        until post-postprocess resolution. Negative integers are windowable
        by construction (they ARE the tail window), ``output``/``identity``
        requests are served by the live output projection, and digit-free
        strings are covered by the live single-pass predicate. Any other
        component type (selector objects, callables) fails safe: retain.
    """

    if selector is None or isinstance(selector, bool):
        return False
    if isinstance(selector, int):
        return selector >= 0
    if isinstance(selector, str):
        return (
            not selector.startswith(("output", "identity"))
            and _FINAL_INDEX_STRING_PATTERN.search(selector) is not None
        )
    if isinstance(selector, (list, tuple, set, frozenset)):
        return any(_selector_requires_unwindowed_escrow(component) for component in selector)
    return True


def _layers_to_save_mentions_output(layers_to_save: object) -> bool:
    """Return whether a selection names a postprocess output wrapper.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when any token appears to target an output layer.
    """

    if isinstance(layers_to_save, str):
        values = (layers_to_save,)
    elif isinstance(layers_to_save, collections.abc.Iterable):
        values = tuple(layers_to_save)
    else:
        return False
    return any(str(value).startswith("output") for value in values)


def _layers_to_save_has_negative_index(layers_to_save: object) -> bool:
    """Return whether ``layers_to_save`` contains a negative op index.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when the selection needs final graph cardinality to resolve.
    """

    if isinstance(layers_to_save, bool):
        return False
    if isinstance(layers_to_save, int):
        return layers_to_save < 0
    if isinstance(layers_to_save, collections.abc.Iterable) and not isinstance(
        layers_to_save,
        str,
    ):
        return any(
            isinstance(value, int) and not isinstance(value, bool) and value < 0
            for value in layers_to_save
        )
    return False


def _layers_to_save_live_subset(layers_to_save: object) -> object | None:
    """Return selector components that can be resolved during the forward.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection containing a deferred component.

    Returns
    -------
    object | None
        Stable-text label components resolvable during the forward, or
        ``None`` when every component requires final graph structure.
        Integer ordinals and final-label-shaped strings are never live: they
        are defined against final layer numbering.
    """

    if isinstance(layers_to_save, bool):
        return None
    if isinstance(layers_to_save, (int, str)):
        if _selector_component_is_final_only(layers_to_save):
            return None
        return layers_to_save
    if isinstance(layers_to_save, collections.abc.Iterable):
        live_components = [
            component
            for component in layers_to_save
            if _layers_to_save_live_subset(component) is not None
        ]
        return live_components or None
    return None


def _layers_to_save_mentions_identity(layers_to_save: object) -> bool:
    """Return whether ``layers_to_save`` targets pass-through identity layers.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when matching needs the legacy module pass-through resolver.
    """

    if isinstance(layers_to_save, str):
        values = (layers_to_save,)
    elif isinstance(layers_to_save, collections.abc.Iterable):
        values = tuple(layers_to_save)
    else:
        return False
    return any(str(value).startswith("identity") for value in values)
