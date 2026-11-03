"""Session-level suppression of mutate-in-place warnings.

The one instance is ``torchlens.options.suppress_mutate_warnings`` (re-exported as
``torchlens.io.suppress_mutate_warnings``).
"""

from __future__ import annotations


class _MutateWarningSuppression:
    """Session-level toggle and context manager for mutation warnings."""

    def __init__(self) -> None:
        """Initialize the suppression flag."""

        self._suppress = False
        # Stack of states to restore on ``__exit__``; supports nested ``with``.
        self._restore_stack: list[bool] = []
        # Pre-call state captured by ``__call__`` so ``with suppress(on):``
        # restores the state that existed *before* the call instead of leaking
        # the in-block value. ``None`` means no call-form entry is pending.
        self._pending_prior: bool | None = None

    def __call__(self, on: bool = True) -> _MutateWarningSuppression:
        """Set suppression state and return this context-capable object.

        Toggling immediately keeps the bare ``suppress_mutate_warnings(True)``
        session-level form working, while snapshotting the pre-call state lets a
        ``with suppress_mutate_warnings(on):`` restore it on exit rather than
        leaking the in-block value.

        Parameters
        ----------
        on:
            Whether mutate-in-place warnings should be suppressed.

        Returns
        -------
        _MutateWarningSuppression
            This suppression controller.
        """

        self._pending_prior = self._suppress
        self._suppress = bool(on)
        return self

    def __enter__(self) -> _MutateWarningSuppression:
        """Temporarily suppress mutate-in-place warnings.

        Returns
        -------
        _MutateWarningSuppression
            This suppression controller.
        """

        if self._pending_prior is not None:
            prior = self._pending_prior
            self._pending_prior = None
        else:
            prior = self._suppress
        self._restore_stack.append(prior)
        self._suppress = True
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the suppression state active before the context.

        Parameters
        ----------
        *exc:
            Exception triple supplied by the context manager protocol.
        """

        self._pending_prior = None
        self._suppress = self._restore_stack.pop() if self._restore_stack else False

    @property
    def is_suppressed(self) -> bool:
        """Whether mutate-in-place warnings are currently suppressed."""

        return self._suppress
