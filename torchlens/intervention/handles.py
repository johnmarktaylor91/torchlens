"""Scoped handles for removable TorchLens hook attachments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HookHandle:
    """Removable handle for one or more sticky hook specs.

    Parameters
    ----------
    owner:
        Object that owns the sticky hook specs.
    handle_ids:
        Opaque hook-spec identifiers to remove.
    """

    owner: Any
    handle_ids: tuple[str, ...]

    def remove(self) -> None:
        """Remove the hook specs owned by this handle.

        Returns
        -------
        None
            Matching hook specs are detached from the owning log.
        """

        self.owner._detach_handles(self.handle_ids)

    def __enter__(self) -> HookHandle:
        """Enter a scoped hook attachment.

        Returns
        -------
        HookHandle
            This handle.
        """

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Remove attached hooks when leaving a context manager.

        Parameters
        ----------
        exc_type:
            Exception type, if the context body raised.
        exc:
            Exception value, if the context body raised.
        traceback:
            Exception traceback, if the context body raised.

        Returns
        -------
        None
            The handle is removed.
        """

        self.remove()


__all__ = ["HookHandle"]
