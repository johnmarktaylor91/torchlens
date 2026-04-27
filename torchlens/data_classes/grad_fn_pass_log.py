"""Per-pass runtime data for autograd grad_fn nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..constants import GRAD_FN_PASS_LOG_FIELD_ORDER


@dataclass
class GradFnPassLog:
    """Runtime data for one execution of an autograd ``grad_fn`` node."""

    pass_num: int
    grad_inputs: Any = None
    grad_outputs: Any = None
    time_started: float | None = None
    time_finished: float | None = None

    def to_pandas(self) -> pd.DataFrame:
        """Export this pass as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``GRAD_FN_PASS_LOG_FIELD_ORDER``.
        """
        row = {field_name: getattr(self, field_name) for field_name in GRAD_FN_PASS_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=GRAD_FN_PASS_LOG_FIELD_ORDER)
