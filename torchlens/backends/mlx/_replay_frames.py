"""Module-call frames for the MLX AD replay observers."""

from __future__ import annotations


class _MLXReplayModuleFrames:
    """Mirror capture's module-call frames during an MLX AD replay.

    Capture stamps each op with the ``"address:call_index"`` frames of the
    discovered modules it ran inside (``op.modules``). The replay observers
    must build the SAME frames, or every op of a module-shaped model fails
    its 1:1 signature match and its intermediate gradient is silently absent.
    """

    def __init__(self) -> None:
        """Start with no frames and zeroed per-address call counts."""

        self.call_counts: dict[str, int] = {}
        self.stack: list[str] = []

    def push(self, address: str | None) -> bool:
        """Enter one module call; return whether a frame was pushed.

        Parameters
        ----------
        address
            Primary module address of the called module, or ``None`` for a
            call that is not a discovered module.

        Returns
        -------
        bool
            ``True`` when a frame was pushed (the caller must ``pop`` it).
        """

        if address is None:
            return False
        call_index = self.call_counts.get(address, 0) + 1
        self.call_counts[address] = call_index
        self.stack.append(f"{address}:{call_index}")
        return True

    def pop(self) -> None:
        """Leave the innermost module call."""

        self.stack.pop()

    def current(self) -> tuple[str, ...]:
        """Return the active frames, outermost first."""

        return tuple(self.stack)
