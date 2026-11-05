"""A missing text cell prints as ``None`` on a debug card under every pandas.

pandas 3 infers a text dtype for string columns and stores a missing cell as
NaN, so ``tl.debug.grad_health`` printed ``module=nan`` for a row with no
module where pandas 2 printed ``module=None``. Text-dtype columns (pandas 3's
inferred ``str`` and the explicit ``string`` dtype on any pandas) render a
missing cell as ``None``; numeric NaN is untouched.
"""

from __future__ import annotations

import math

import pandas as pd

from torchlens._debug_report import table_report


def test_missing_text_cells_print_none_and_numeric_nan_stays() -> None:
    """``string`` dtype NA prints ``None``; a float NaN still prints as NaN."""

    frame = pd.DataFrame(
        {
            "module": pd.Series([None, "fc"], dtype="string"),
            "norm": [math.nan, 1.5],
        }
    )
    report = table_report(
        helper="probe", frame=frame, subject="Net", finding_columns=("module", "norm")
    )
    findings = report._report_findings()

    assert findings[0].startswith("module=None, norm=")
    assert "None" not in findings[0].split("norm=")[1]
    assert findings[1].startswith("module=fc, norm=1.5")
