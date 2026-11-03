"""Label-shape heuristics behind ``Bundle.at(label)``.

A string site names an entry in one of the Bundle accessor families (ops,
layers, modules, params, buffers, grad_fns, module_calls, grad_fn_calls).
These helpers guess the family from the label's shape first and fall back to
key presence, refusing an ambiguous label with the accessors that hold it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Bundle

_OP_LABEL_RE = re.compile(r"^.+_\d+_\d+$")
_BARE_LAYER_LABEL_RE = re.compile(r"^.+_\d+$")
_PASS_CALL_RE = re.compile(r"^(.+):(\d+)$")
_COMMON_PARAM_SUFFIXES = (".weight", ".bias")
_COMMON_BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


class AmbiguousLabelError(KeyError):
    """Raised when ``Bundle.at`` finds a label in multiple accessors."""


# Defined here so this leaf never imports the package at call time (layering:
# the import kept ``_labels`` inside the big cycle); pickles and the error
# registry keep the facade home.
AmbiguousLabelError.__module__ = "torchlens.bundle"
BUNDLE_ACCESSOR_NAMES = (
    "ops",
    "layers",
    "modules",
    "params",
    "buffers",
    "grad_fns",
    "module_calls",
    "grad_fn_calls",
)


def label_view(bundle: Bundle, site: str) -> Any:
    """Return the Super view for a label from any Bundle accessor family.

    Parameters
    ----------
    bundle:
        Bundle whose accessors are searched.
    site:
        Label from any Bundle accessor family.

    Returns
    -------
    Any
        Matching ``Super*`` view.

    Raises
    ------
    AmbiguousLabelError
        If an unrecognized-format label is present in multiple accessors.
    KeyError
        If the label is absent from every accessor.
    """

    preferred_names = preferred_accessor_names(site)
    for accessor_name in preferred_names:
        try:
            return getattr(bundle, accessor_name)[site]
        except KeyError:
            continue

    remaining_names = [name for name in BUNDLE_ACCESSOR_NAMES if name not in preferred_names]
    matches = matching_accessor_names(bundle, site, remaining_names)
    if len(matches) == 1:
        return getattr(bundle, matches[0])[site]
    if len(matches) > 1:
        raise AmbiguousLabelError(ambiguous_label_message(site, matches))
    raise KeyError(missing_label_message(bundle, site))


def preferred_accessor_names(label: str) -> list[str]:
    """Return format-preferred accessors for ``label`` in dispatch order.

    Parameters
    ----------
    label:
        Candidate label.

    Returns
    -------
    list[str]
        Accessor names to try before key-presence fallback.
    """

    call_base = pass_call_base(label)
    if call_base is not None:
        if looks_like_pass_qualified_op_label(call_base):
            return ["ops"]
        if looks_like_grad_fn_label(call_base):
            return ["grad_fn_calls"]
        return ["module_calls"]

    if looks_like_pass_qualified_op_label(label):
        return ["ops"]
    if looks_like_bare_layer_label(label):
        return ["layers"]
    if looks_like_module_address(label):
        return ["modules"]
    if looks_like_param_name(label):
        return ["params"]
    if looks_like_buffer_name(label):
        return ["buffers"]
    if looks_like_grad_fn_label(label):
        return ["grad_fns"]
    return []


def matching_accessor_names(bundle: Bundle, label: str, accessor_names: Sequence[str]) -> list[str]:
    """Return the accessors among ``accessor_names`` that contain ``label``."""

    return [name for name in accessor_names if label in getattr(bundle, name)]


def missing_label_message(bundle: Bundle, label: str) -> str:
    """Build a missing-label error message with accessor suggestions."""

    suggestions: list[str] = []
    seen: set[str] = set()
    for accessor_name in BUNDLE_ACCESSOR_NAMES:
        accessor = getattr(bundle, accessor_name)
        for suggestion in accessor._suggest(label):
            if suggestion not in seen:
                seen.add(suggestion)
                suggestions.append(suggestion)
    if suggestions:
        suggestion_str = ", ".join(repr(suggestion) for suggestion in suggestions)
        return f"Label {label!r} not found. Did you mean {suggestion_str}?"
    return f"Label {label!r} not found."


def ambiguous_label_message(label: str, matches: Sequence[str]) -> str:
    """Build the ambiguity error message for ``Bundle.at``."""

    match_str = " and ".join(f"bundle.{name}" for name in matches)
    disambiguators = " or ".join(f"bundle.{name}[{label!r}]" for name in matches)
    return f"Label {label!r} matches {match_str}; use {disambiguators} explicitly."


def pass_call_base(label: str) -> str | None:
    """Return the base label for ``base:N`` call labels, else ``None``."""

    match = _PASS_CALL_RE.match(label)
    return match.group(1) if match is not None else None


def looks_like_pass_qualified_op_label(label: str) -> bool:
    """Whether ``label`` has the ``op_type_group_pass`` shape."""

    return _OP_LABEL_RE.match(label) is not None


def looks_like_bare_layer_label(label: str) -> bool:
    """Whether ``label`` has the ``op_type_group`` shape but not an op label's."""

    return _BARE_LAYER_LABEL_RE.match(label) is not None and _OP_LABEL_RE.match(label) is None


def looks_like_module_address(label: str) -> bool:
    """Whether ``label`` has dotted or numeric module-address shape."""

    if ":" in label:
        return False
    if label.isdecimal():
        return True
    return "." in label and not (
        label.endswith(_COMMON_PARAM_SUFFIXES) or label.endswith(_COMMON_BUFFER_SUFFIXES)
    )


def looks_like_param_name(label: str) -> bool:
    """Whether ``label`` ends with a common parameter suffix."""

    return label.endswith(_COMMON_PARAM_SUFFIXES)


def looks_like_buffer_name(label: str) -> bool:
    """Whether ``label`` ends with a common buffer suffix."""

    return label.endswith(_COMMON_BUFFER_SUFFIXES)


def looks_like_grad_fn_label(label: str) -> bool:
    """Whether ``label`` has TorchLens backward grad-fn label shape."""

    return "_back_" in label
