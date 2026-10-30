"""The call-site grouping contract (``callsite_v1``) and its in-memory evidence.

Grouping rule section 3 keeps the call-site key internal: it is never persisted
and never written into a public field, so ``equivalence_class`` keeps its
argument/parameter fingerprint for its other consumers. Step 7 groups by the
raw-graph workspace side table (``RawGraphWorkspace.call_site_classes``) and then
checks the contract against it; a weak per-trace registry keeps that table for
the life of the live Trace so re-validation of a fresh capture can re-check it.
A loaded trace has no table, and re-validation reports the contract as not
checked instead of passing it.

The contract, for every op whose call site the capture walk saw:

(i) every pass of one layer has one func_name and one call-site key;
(ii) two different layers never share a call-site key;
(iii) every recurrent group maps to exactly one call-site key.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = (
    "CallSiteEntry",
    "call_site_classes_for",
    "call_site_contract_violation",
    "remember_call_site_classes",
)

#: Live Trace -> raw op label -> call-site class. Weak, in memory only: the
#: entry dies with the Trace and never reaches a saved artifact.
_CALL_SITE_CLASSES: weakref.WeakKeyDictionary[Any, dict[str, str]] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class CallSiteEntry:
    """What the contract reads about one retained op.

    Attributes
    ----------
    raw_label:
        The op's raw capture label (``_label_raw``), the side table's key.
    layer:
        The layer the op belongs to (raw or final layer label).
    func_name:
        The function the op ran.
    group:
        Raw labels of the op's recurrent group (``recurrent_ops``).
    """

    raw_label: str
    layer: str
    func_name: Any
    group: tuple[str, ...]


def remember_call_site_classes(trace: Any, classes: dict[str, str]) -> None:
    """Keep a capture's call-site side table for re-validation of the live Trace."""

    try:
        _CALL_SITE_CLASSES[trace] = dict(classes)
    except TypeError:
        # An unhashable or non-weakrefable trace stand-in keeps no evidence;
        # re-validation then reports the contract as not checked.
        pass


def call_site_classes_for(trace: Any) -> dict[str, str] | None:
    """Return the live Trace's call-site side table, or None when it has none."""

    try:
        return _CALL_SITE_CLASSES.get(trace)
    except TypeError:
        return None


def call_site_contract_violation(
    entries: Iterable[CallSiteEntry], classes: dict[str, str]
) -> str | None:
    """Return the first breach of the call-site contract, or None when it holds.

    An op with no call-site class (inputs, outputs, buffers, positional
    fallback ops) carries no key; a layer or recurrent group that mixes ops with
    and without a key, or with two keys, breaches the contract.
    """

    entries = list(entries)
    layer_keys: dict[str, set[str | None]] = defaultdict(set)
    layer_funcs: dict[str, set[Any]] = defaultdict(set)
    key_layers: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        key = classes.get(entry.raw_label)
        layer_keys[entry.layer].add(key)
        if key is not None:
            layer_funcs[entry.layer].add(entry.func_name)
            key_layers[key].add(entry.layer)
    for layer, keys in layer_keys.items():
        if len(keys) > 1:
            return (
                f"(i) layer '{layer}' merges ops from {len(keys)} call sites: "
                f"{sorted(str(key) for key in keys)}"
            )
        if len(layer_funcs.get(layer, ())) > 1:
            return f"(i) layer '{layer}' runs {sorted(map(str, layer_funcs[layer]))}"
    for key, layers in key_layers.items():
        if len(layers) > 1:
            return f"(ii) call site '{key}' is split across layers {sorted(layers)}"
    return _group_contract_violation(entries, classes)


def _group_contract_violation(entries: list[CallSiteEntry], classes: dict[str, str]) -> str | None:
    """Contract (iii): the first recurrent group spanning two call sites, or None.

    Each group is checked once, at its first member; group symmetry (every
    member names the same group) is the loop-detection invariant's own check.
    """

    checked: set[str] = set()
    for entry in entries:
        if entry.raw_label in checked:
            continue
        checked.update(entry.group)
        group_keys = {classes.get(label) for label in entry.group}
        if len(group_keys) > 1:
            return (
                f"(iii) recurrent group of '{entry.raw_label}' spans call sites "
                f"{sorted(str(key) for key in group_keys)}"
            )
    return None
