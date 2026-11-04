"""The lean recording contract: what ``lean=True`` skips, refuses and reports.

PROVISIONAL SPELLING: ``lean`` is the working name of the option until the
maintainer rules the public spelling; every reader goes through this module so
the rename is one edit.

A lean recording journals only the events its selection keeps. An op the save
predicate does not keep, and a module boundary no default retains, gets no
shell journal event, no frozen record and no replay clone; the predicate
context is still built (predicates, lookback history and module-boundary
retention read it) and the halt predicate still runs. Everything a default
recording derives from the full journal is therefore absent, and the recording
says so: :data:`LEAN_OMISSIONS` names the omitted content and the reads that
need it refuse with code ``recording_lean_omitted`` instead of answering from a
partial journal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._errors import InvalidArgumentError
from .exceptions import RecordingConfigError

if TYPE_CHECKING:
    from .options import _RecordingConfig


def validate_lean_value(value: object) -> None:
    """Refuse a non-bool ``lean=`` value with the recording option type error."""

    if not isinstance(value, bool):
        raise InvalidArgumentError(
            f"lean must be a bool; received {type(value).__name__}",
            code="recording_option_type_invalid",
            remedy="pass lean=True or lean=False",
            argument="lean",
        )


def validate_lean_config(options: _RecordingConfig, *, backward_requested: bool) -> None:
    """Refuse option combinations whose answers would depend on the dropped events.

    Parameters
    ----------
    options:
        The recorder's resolved configuration.
    backward_requested:
        Whether gradient hooks were asked for (``backward_ready``, ``save_grads``
        or ``default_grad``); gradient capture binds hooks to every op's journal
        event, which a lean journal does not hold.

    Raises
    ------
    RecordingConfigError
        ``record_lean_unsupported`` naming the conflicting option.
    """

    if not options.lean:
        return
    conflict: str | None = None
    if backward_requested:
        conflict = "gradient capture (backward_ready=True, save_grads= or default_grad=)"
    elif options.intervene is not None:
        conflict = "intervene="
    elif options.echo is not None:
        conflict = "narrate="
    elif options.on_forward_error != "raise":
        conflict = f"on_forward_error={options.on_forward_error!r}"
    if conflict is None:
        return
    raise RecordingConfigError(
        f"lean=True cannot be combined with {conflict}: that feature reads the "
        "journal events of ops the selection does not keep, which a lean "
        "recording does not hold. Remedy: drop lean=True or the conflicting option.",
        code="record_lean_unsupported",
    )
