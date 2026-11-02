"""The live model-return door of a Trace (``trace.output``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trace import Trace


class TraceOutputMixin:
    """Expose the forward's exact return object, held in memory only."""

    @property
    def output(self: Trace) -> Any:
        """Return the model's exact forward return value, held in memory only.

        This is the original object, independent of saved activations and
        output transforms. It is never written to a bundle or pickle. The trace
        keeps it alive for as long as the trace exists (for a Hugging Face
        model this includes ``past_key_values``); it is released when the
        trace is dropped.

        Returns
        -------
        Any
            The forward's return value, without copying or reconstruction.

        Raises
        ------
        PayloadUnavailableError
            ``output_not_stored`` when loaded, unpickled, or built without a
            live forward; ``output_capture_stopped`` when ``stop_after=``
            ended the capture before the model returned.
        """

        from ..recording.types import read_output

        return read_output(self._output_slot, owner="Trace")
