"""Per-pass runtime data for autograd grad_fn nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import pandas as pd

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..constants import GRAD_FN_PASS_LOG_FIELD_ORDER


@dataclass
class GradFnPassLog:
    """Runtime data for one execution of an autograd ``grad_fn`` node."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "pass_num": FieldPolicy.KEEP,
        "grad_inputs": FieldPolicy.BLOB_RECURSIVE,
        "grad_outputs": FieldPolicy.BLOB_RECURSIVE,
        "time_started": FieldPolicy.KEEP,
        "time_finished": FieldPolicy.KEEP,
    }

    pass_num: int
    grad_inputs: Any = None
    grad_outputs: Any = None
    time_started: float | None = None
    time_finished: float | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with an IO format marker."""

        state = self.__dict__.copy()
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and fill fields added in newer versions."""

        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults={
                "grad_inputs": None,
                "grad_outputs": None,
                "time_started": None,
                "time_finished": None,
            },
        )
        self.__dict__.update(state)

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
