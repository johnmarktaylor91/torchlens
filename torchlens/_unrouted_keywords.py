"""Remedies for ``trace()`` keywords it does not route.

A keyword that is a removed capture option names its replacement; a 1.x drawing
keyword (``vis_*``) names the ``Trace.draw`` keyword after capture; ``input_size``
names ``input_shape=``; one that names (or nearly names) a field of an options
dataclass names that class; the rest are model forward keywords that travel
through ``input_kwargs=``.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Mapping
from typing import Any

from ._data_classes._draw_keywords import LEGACY_DRAW_KEYWORDS, VIS_OPT_REMEDY
from ._errors import ArgumentTypeError
from ._options_removed import REMOVED_CAPTURE_MEMBERS

#: Keywords tl.trace does not take that are not model inputs either, with their remedy.
_MISPLACED_TRACE_KEYWORDS: dict[str, str] = {
    "input_size": "input_size is now input_shape=: tl.trace(model, input_shape=(1, 3, 224, 224))",
    "vis_opt": VIS_OPT_REMEDY,
    **{
        key: f"tl.trace does not draw; {key} is now Trace.draw({new}=...): "
        f"tl.trace(model, x).draw({new}=...)"
        for key, new in LEGACY_DRAW_KEYWORDS.items()
    },
}


def reject_unrouted_forward_kwargs(
    forward_kwargs: dict[str, Any], option_classes: Mapping[str, type] | None = None
) -> None:
    """Refuse unknown ``trace()`` keywords with a typed routing remedy.

    Typed recovery on the natural HF spelling (agent memo):
    ``tl.trace(model, input_ids=...)`` used to die with a bare CPython
    TypeError. Unknown keywords are NEVER silently absorbed as forward
    kwargs (a typo'd option must not become a model input); the refusal
    names both working spellings instead.

    Parameters
    ----------
    forward_kwargs:
        The unknown keyword arguments ``trace()`` collected. Empty means
        nothing to refuse and the function returns silently.
    option_classes:
        ``trace() parameter -> options class`` whose fields a keyword may name
        (``{"capture": CaptureOptions, ...}``); a match names its home.

    Raises
    ------
    ArgumentTypeError
        With code ``trace_forward_kwargs_unrouted`` when any unknown
        keyword is present.
    """

    if not forward_kwargs:
        return

    keys = sorted(forward_kwargs)
    # A removed capture option is never a model input: name its replacement first.
    removed = [
        key for key in keys if key in REMOVED_CAPTURE_MEMBERS or key in _MISPLACED_TRACE_KEYWORDS
    ]
    homes = option_homes([key for key in keys if key not in removed], option_classes or {})
    model_inputs = [key for key in keys if key not in removed and key not in homes]
    remedies = unrouted_remedies(removed, homes, model_inputs)
    if removed:
        unrouted = [key for key in keys if key not in removed]
        message = f"trace() got removed or misplaced keyword(s): {', '.join(removed)}" + (
            f" (and unrouted keyword(s): {', '.join(unrouted)})" if unrouted else ""
        )
    else:
        message = (
            f"trace() got keyword(s) it does not route: {', '.join(keys)}. Capture settings "
            "are grouped in option objects and forward keyword arguments travel through "
            "input_kwargs="
        )
    raise ArgumentTypeError(
        message,
        code="trace_forward_kwargs_unrouted",
        remedy="; ".join(remedies),
        unrouted_keywords=keys,
    )


def unrouted_remedies(
    removed: list[str], homes: Mapping[str, tuple[str, str, str]], model_inputs: list[str]
) -> list[str]:
    """One remedy per unrouted keyword class: removed option, option-class field, model input."""

    remedies = [
        _MISPLACED_TRACE_KEYWORDS.get(key) or f"{key}: {REMOVED_CAPTURE_MEMBERS[key]}"
        for key in removed
    ]
    remedies += [
        f"{key} is a {class_name} field: tl.trace(model, x, "
        f"{param}=tl.options.{class_name}({field}=...))"
        for key, (param, class_name, field) in homes.items()
    ]
    if model_inputs:
        remedies.append(
            "model forward keywords travel through input_kwargs=: tl.trace(model, (), "
            "input_kwargs={" + ", ".join(f"{key!r}: ..." for key in model_inputs) + "}); "
            "for HuggingFace text models, torchlens.bridge.transformers.trace_text(model, "
            "'prompt') builds the inputs for you"
        )
    return remedies


def option_homes(
    keys: list[str], option_classes: Mapping[str, type]
) -> dict[str, tuple[str, str, str]]:
    """Map each unknown keyword that names (or nearly names) an option field to its home.

    Parameters
    ----------
    keys:
        Unknown ``trace()`` keywords.
    option_classes:
        ``trace() parameter -> options dataclass`` to search.

    Returns
    -------
    dict[str, tuple[str, str, str]]
        ``keyword -> (trace parameter, options class name, field name)`` for the
        keywords that match an option field exactly or by a close spelling.
    """

    fields: dict[str, tuple[str, str]] = {}
    for param, option_class in option_classes.items():
        if not dataclasses.is_dataclass(option_class):
            continue
        for option_field in dataclasses.fields(option_class):
            if not option_field.name.startswith("_"):
                fields.setdefault(option_field.name, (param, option_class.__name__))
    homes: dict[str, tuple[str, str, str]] = {}
    for key in keys:
        match = key if key in fields else None
        if match is None:
            close = difflib.get_close_matches(key, list(fields), n=1, cutoff=0.8)
            match = close[0] if close else None
        if match is not None:
            homes[key] = (*fields[match], match)
    return homes
