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
    equals the layer's pass count answers to ``layer_label:last`` as well.
    Subjects without a layer label (grad-fn nodes) answer to neither.
    """

    layer_label = getattr(subject, "layer_label", None)
    if not isinstance(layer_label, str):
        return ()
    spellings = [f"{layer_label}:{PASS_TOKEN_ALL}"]
    pass_number = getattr(subject, "pass_number", None) or 1
    num_passes = getattr(subject, "num_passes", None) or 1
    if pass_number == num_passes:
        spellings.append(f"{layer_label}:{PASS_TOKEN_LAST}")
    return tuple(spellings)
