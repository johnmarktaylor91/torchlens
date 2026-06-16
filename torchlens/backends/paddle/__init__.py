"""Paddle backend preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..registry import BackendUnsupportedError


@dataclass(frozen=True)
class GradOptions:
    """Paddle derived-gradient options placeholder for later backend phases."""


class PaddleBackend:
    """P1 Paddle backend shell with lazy runtime validation."""

    name = "paddle"
    supports_backward_capture = False

    def __init__(self) -> None:
        """Import Paddle lazily and require dygraph mode for backend use."""

        import paddle

        in_dynamic_mode = getattr(paddle, "in_dynamic_mode", None)
        if callable(in_dynamic_mode) and not in_dynamic_mode():
            raise BackendUnsupportedError("Paddle backend preview requires Paddle dygraph mode.")
        self._paddle = paddle

    def capture_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Raise the P1 capture placeholder error.

        Parameters
        ----------
        *args, **kwargs:
            Public trace arguments, unused until capture lands.

        Returns
        -------
        Any
            Never returns in P1.
        """

        del args, kwargs
        raise BackendUnsupportedError("Paddle backend capture is not yet implemented.")

    def validate_entry(self, *args: Any, **kwargs: Any) -> bool:
        """Raise the P1 model/input validation placeholder error.

        Parameters
        ----------
        *args, **kwargs:
            Public validation arguments, unused until validation lands.

        Returns
        -------
        bool
            Never returns in P1.
        """

        del args, kwargs
        raise BackendUnsupportedError("Paddle backend validation is not yet implemented.")

    def validate_trace(self, *args: Any, **kwargs: Any) -> bool:
        """Raise the P1 trace validation placeholder error.

        Parameters
        ----------
        *args, **kwargs:
            Trace validation arguments, unused until validation lands.

        Returns
        -------
        bool
            Never returns in P1.
        """

        del args, kwargs
        raise BackendUnsupportedError("Paddle backend trace validation is not yet implemented.")


__all__ = ["GradOptions", "PaddleBackend"]
