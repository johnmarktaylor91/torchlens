"""Per-pass runtime data for autograd grad_fn_handle nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
import weakref

if TYPE_CHECKING:
    import pandas as pd

from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from ..constants import GRAD_FN_PASS_LOG_FIELD_ORDER
from ._tabular_export import TabularExportMixin


def _duration_str(duration: float) -> str:
    """Return a human-readable duration string.

    Parameters
    ----------
    duration:
        Duration in seconds.

    Returns
    -------
    str
        Duration formatted in milliseconds.
    """

    return f"{duration * 1000:.3f} ms"


@dataclass
class GradFnCall(TabularExportMixin):
    """Runtime data for one execution of an autograd ``grad_fn_handle`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "call_index": FieldPolicy.KEEP,
        "label": FieldPolicy.KEEP,
        "grad_inputs": FieldPolicy.BLOB_RECURSIVE,
        "grad_outputs": FieldPolicy.BLOB_RECURSIVE,
        "_time_started": FieldPolicy.KEEP,
        "_time_finished": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
    }

    call_index: int
    label: str = ""
    grad_inputs: Any = None
    grad_outputs: Any = None
    _time_started: float | None = None
    _time_finished: float | None = None
    _source_trace_ref: Any = None

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with an IO format marker."""

        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and fill fields added in newer versions."""

        read_tlspec_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "label": "",
                "grad_inputs": None,
                "grad_outputs": None,
                "_time_started": None,
                "_time_finished": None,
                "_source_trace_ref": None,
            },
        )
        if "duration" in state and "_time_started" not in state and "_time_finished" not in state:
            state["_time_started"] = 0.0
            state["_time_finished"] = float(state.pop("duration"))
        self.__dict__.update(state)

    @property
    def source_trace(self) -> Any:
        """Return the owning Trace if it is still alive."""

        ref = self._source_trace_ref
        return None if ref is None else ref()

    @source_trace.setter
    def source_trace(self, value: Any) -> None:
        """Set the owning Trace weakref."""

        self._source_trace_ref = weakref.ref(value) if value is not None else None

    @property
    def trace(self) -> Any:
        """Alias for the owning Trace."""

        return self.source_trace

    @property
    def ordinal_index(self) -> int:
        """Return this GradFnCall's 0-based position in ``trace.grad_fn_calls``."""

        trace = self.source_trace
        if trace is None:
            return -1
        return list(trace.grad_fn_calls).index(self)

    @property
    def call_label(self) -> str:
        """Return the pass-qualified GradFnCall label.

        Returns
        -------
        str
            GradFn label with the 1-based call index suffix.
        """

        return f"{self.label}:{self.call_index}" if self.label else str(self.call_index)

    @property
    def backward_duration(self) -> float:
        """Return the measured backward duration for this call.

        Returns
        -------
        float
            Seconds elapsed between ``_time_started`` and ``_time_finished``.
        """

        if self._time_started is None or self._time_finished is None:
            return 0.0
        return max(0.0, self._time_finished - self._time_started)

    @property
    def backward_duration_str(self) -> str:
        """Return backward duration in human-readable units.

        Returns
        -------
        str
            Human-readable duration.
        """

        return _duration_str(self.backward_duration)

    def to_pandas(self) -> "pd.DataFrame":
        """Export this pass as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``GRAD_FN_PASS_LOG_FIELD_ORDER``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        row = {field_name: getattr(self, field_name) for field_name in GRAD_FN_PASS_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=GRAD_FN_PASS_LOG_FIELD_ORDER)
