"""Skip a known run of TorchLens frames in the per-op site-key walk.

The site-key walk (``_ops_call_site_fields._site_key``) reads every frame from
the op up to the innermost module wrapper or the call-site anchor. Most of those
frames are TorchLens's own: the emit machinery below the user's frame, and the
capture harness between the root ``forward`` and the anchor. The walk keeps
nothing from them; it reads each frame's name, file verdict and anchor match
only to learn that.

A run is a sequence of code objects met on consecutive frames, every one of
which the walk skips whatever frame runs it: TorchLens code (the verdict is a
property of the file), not named ``_call_impl`` and not named like a module
boundary wrapper. Whether the walk skips a frame depends only on its code, except
for the anchor check, and the anchor is never TorchLens code (it is a kept
frame). So frames whose codes are those of a run, in order, are skipped by the
walk, and a run is verified by code identity alone, one comparison per frame.
Runs are learned on first sight, keyed by the identity of their first code, and
hold their codes, so no id is reused while the table lives (one capture).
A changed set of boundary codes drops every run.

The site key names codes by ``id`` (a small int hashes in one step; a code
object hashes its bytecode, constants and names on every lookup), so every code
a stored key names is held by the table too (:func:`pin_codes`).
"""

from types import CodeType, FrameType
from typing import Any

from ..._utils._call_site import _BOUNDARY_CODES, _BOUNDARY_NAMES
from ._call_site_parts import is_internal

__all__ = ("RUN_BUDGET", "pin_codes", "skip_run")

#: Runs learned per capture at most; past it a run head is skipped frame by frame.
RUN_BUDGET = 32


def _skippable(code: CodeType) -> bool:
    """Whether the site-key walk skips every frame running ``code``."""

    name = code.co_name
    return name != "_call_impl" and name not in _BOUNDARY_NAMES and is_internal(code)


def _learn(memo: Any, frame: FrameType, code: CodeType) -> FrameType | None:
    """Record the run starting at ``frame`` (whose ``code`` the walk skips); the frame after it."""

    run = []
    above = frame.f_back
    while above is not None and _skippable(above.f_code):
        run.append(above.f_code)
        above = above.f_back
    memo.run_budget -= 1
    memo.runs.setdefault(id(code), []).append((code, tuple(run)))
    return above


def skip_run(memo: Any, frame: FrameType, code: CodeType) -> FrameType | None:
    """The first frame above ``frame`` the site-key walk must read.

    ``frame`` runs ``code``, TorchLens code the walk has already found it skips
    (not a boundary, not the anchor, not ``_call_impl``). Frames above it that
    match a known run starting at ``code`` are skipped by identity; otherwise the
    run is learned while the budget lasts, else only ``frame`` is skipped.
    """

    if len(_BOUNDARY_CODES) != memo.run_boundaries:
        memo.runs.clear()
        memo.run_boundaries = len(_BOUNDARY_CODES)
    for _head, run in memo.runs.get(id(code), ()):
        above = frame.f_back
        for expected in run:
            if above is None or above.f_code is not expected:
                break
            above = above.f_back
        else:
            return above
    if memo.run_budget > 0 and code.co_name not in _BOUNDARY_NAMES:
        return _learn(memo, frame, code)
    return frame.f_back


def pin_codes(memo: Any, frame: FrameType | None) -> None:
    """Hold every code from ``frame`` up to the first module wrapper, the anchor or the top.

    A superset of the codes the site key of an op at ``frame`` names, so no id
    in a stored key is reused while the table lives.
    """

    pinned = memo.pinned
    anchor = memo.anchor
    while frame is not None:
        code = frame.f_code
        pinned[id(code)] = code
        if code.co_name in _BOUNDARY_NAMES and code in _BOUNDARY_CODES:
            return
        if (
            anchor is not None
            and code is anchor[1]
            and frame.f_lasti == anchor[2]
            and frame.f_lineno == anchor[3]
            and id(frame) == anchor[0]
        ):  # the site-key walk's own anchor check, so this walk never stops sooner
            return
        frame = frame.f_back
