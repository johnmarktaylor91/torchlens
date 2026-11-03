"""Narration sinks: stderr, path, stream, or callable (snoop D2 sinks).

Default stderr (stdout stays clean). File sinks flush per line by default:
that durability is the entire reason live narration exists alongside the
post-hoc tail -- only bytes already on disk survive a segfault or OOM kill.
Colour and hyperlink capability resolve ONCE at sink-open from ``isatty()``;
the non-tty path is byte-clean (no ANSI/OSC-8 escapes, CI-asserted). The
callable sink is the sole seam to trackers/loggers; echo grows no dashboard
protocol. Distributed runs rank-prefix each line following the shipped
fastlog rank-prefix precedent.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

__tl_layer__ = "L5"


def _distributed_rank_prefix() -> str:
    """Return the ``[rank N] `` line prefix on initialized distributed runs.

    Returns
    -------
    str
        Rank prefix when ``torch.distributed`` is initialized, else ``""``.
    """

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return f"[rank {dist.get_rank()}] "
    except Exception:  # noqa: BLE001 - a prefix probe must never break the sink
        return ""
    return ""


class EchoSink:
    """One resolved narration destination with per-line delivery.

    Parameters
    ----------
    target:
        ``None`` (stderr), a path, an open text stream, or ``callable(str)``.
        The callable receives each line WITHOUT a trailing newline.

    Notes
    -----
    tty capability (``is_tty``) is resolved exactly once here, at open;
    formatters consult it instead of re-probing per line. Sink failures are
    the session's to handle (warn-once-disable) -- this class never swallows.
    """

    def __init__(self, target: str | Path | IO[str] | Callable[[str], Any] | None) -> None:
        """Resolve the sink target, its tty capability, and the rank prefix."""

        self._callable: Callable[[str], Any] | None = None
        self._stream: IO[str] | None = None
        self._owns_stream = False
        self._flush_per_line = False
        self.is_tty = False
        self.rank_prefix = _distributed_rank_prefix()
        if target is None:
            self._stream = sys.stderr
            self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        elif callable(target) and not isinstance(target, (str, Path)):
            self._callable = target
        elif isinstance(target, (str, Path)):
            self._stream = Path(target).expanduser().open("a", encoding="utf-8")
            self._owns_stream = True
            self._flush_per_line = True
        else:
            self._stream = target
            self.is_tty = bool(getattr(target, "isatty", lambda: False)())
            self._flush_per_line = True

    def write_line(self, line: str) -> None:
        """Deliver one narration line.

        Parameters
        ----------
        line:
            Rendered line without a trailing newline.
        """

        prefixed = f"{self.rank_prefix}{line}" if self.rank_prefix else line
        if self._callable is not None:
            self._callable(prefixed)
            return
        stream = self._stream
        if stream is None:
            return
        stream.write(prefixed + "\n")
        if self._flush_per_line:
            stream.flush()

    def flush(self) -> None:
        """Best-effort flush (used on interrupts and failure tails)."""

        stream = self._stream
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.flush()

    def close(self) -> None:
        """Close a sink-owned file stream; foreign streams stay open."""

        if self._owns_stream and self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.flush()
                self._stream.close()
            self._stream = None


__all__ = ["EchoSink"]
