"""Pass tokens: the ``:last`` and ``:all`` spellings a layer label may end in.

Passes are 1-based (``label:2`` is the second pass); ``label:last`` names the
final pass of a layer and ``label:all`` every pass. A leaf module: the accessor
base, the trace lookup and the selector universe all read it.
"""

from __future__ import annotations

from typing import Any

#: Pass tokens a label may end in besides a 1-based pass number: ``:last`` names
#: the final pass of a layer, ``:all`` every pass (a set spelling).
PASS_TOKEN_LAST = "last"
PASS_TOKEN_ALL = "all"
_PASS_TOKENS = (PASS_TOKEN_LAST, PASS_TOKEN_ALL)


def split_pass_token(label: str) -> tuple[str, str] | None:
    """Split ``layer_label:last`` / ``layer_label:all`` into ``(layer_label, token)``.

    Parameters
    ----------
    label:
        Label spelling.

    Returns
    -------
    tuple[str, str] | None
        The base label and token, or ``None`` when the spelling ends in neither token.
    """

    base, sep, token = label.rpartition(":")
    if not sep or token not in _PASS_TOKENS or not base:
        return None
    return base, token


def pass_token_labels(subject: Any) -> tuple[str, ...]:
    """Return the ``:all`` and ``:last`` spellings a finished op answers to.

    Every pass of a layer answers to ``layer_label:all``; the pass whose number
    equals the layer's pass count answers to ``layer_label:last`` as well. The
    short layer label (``linear_1``) takes both tokens the same way, since a
    label lookup accepts it wherever it accepts the full label. Subjects
    without a layer label (grad-fn nodes) answer to neither.
    """

    layer_label = getattr(subject, "layer_label", None)
    if not isinstance(layer_label, str):
        return ()
    short_label = getattr(subject, "short_layer_label", None)
    bases = [layer_label]
    if isinstance(short_label, str) and short_label and short_label != layer_label:
        bases.append(short_label)
    pass_number = getattr(subject, "pass_number", None) or 1
    num_passes = getattr(subject, "num_passes", None) or 1
    tokens = [PASS_TOKEN_ALL] + ([PASS_TOKEN_LAST] if pass_number == num_passes else [])
    return tuple(f"{base}:{token}" for base in bases for token in tokens)
