"""The model-return doors of a Trace (``trace.output``, ``trace.transformed_output``)."""

from __future__ import annotations

from typing import Any

from .._output_slot import read_output, read_transformed_output
from ._moe_views import _TraceRouting


class TraceOutputMixin(_TraceRouting):
    """Expose the forward's exact return object, held in memory only.

    ``self`` is typed ``Any``: importing ``Trace`` here, even for type checking,
    adds a module cycle the checker resolves in a different order.
    """

    @property
    def output(self: Any) -> Any:
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

        return read_output(self._output_slot, owner="Trace")

    @property
    def transformed_output(self: Any) -> Any:
        """Return the model output after ``output_transform``, or ``None`` without one.

        ``output_transform`` runs on the model's return value at capture, and
        again on each rerun. A replay recomputes ops from captured values and
        has no model return to re-apply it to: after a replay rewrites an
        output op's values, reading this refuses instead of serving the
        parent forward's result. A replay that leaves every output op's
        values unchanged keeps the inherited result. ``copy.deepcopy`` and
        pickle keep the refusal; a portable save (``tl.save``) of such a
        trace stores no transformed output, so the loaded trace reads ``None``.

        Returns
        -------
        Any
            The transform result, or ``None`` when no ``output_transform`` was
            given.

        Raises
        ------
        PayloadUnavailableError
            ``output_not_stored`` after a replay rewrote an output op; apply
            the edit on the rerun engine to recompute it.
        """

        try:
            value = self.__dict__["transformed_output"]
        except KeyError:
            raise AttributeError("transformed_output") from None
        return read_transformed_output(value, owner="Trace")

    @transformed_output.setter
    def transformed_output(self: Any, value: Any) -> None:
        """Store ``value`` under the plain ``transformed_output`` key."""

        self.__dict__["transformed_output"] = value

    @transformed_output.deleter
    def transformed_output(self: Any) -> None:
        """Drop the stored value (trace cleanup deletes fields by name)."""

        try:
            del self.__dict__["transformed_output"]
        except KeyError:
            raise AttributeError("transformed_output") from None
