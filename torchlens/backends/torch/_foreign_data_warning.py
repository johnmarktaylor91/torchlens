"""Trace-time disclosure of a foreign data round trip TorchLens cannot re-run.

A tensor that leaves torch (``.numpy()``, ``.tolist()``, DLPack) and comes back through
a host constructor without matching any departure is recorded as an **unresolved**
``foreigndata`` op linked to the departed tensors (:mod:`._foreign_data`). Validation
refuses to replay it; this module says so at capture time, in the sentence validation
uses, so the gap is visible before anyone calls ``tl.validate``.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..._utils._package_frames import external_warning_stacklevel
from ..._utils._record_group_table import record_member
from ...errors._base import TorchLensWarning
from ._foreign_data import is_unresolved_foreign_data


def _roundtrip_sentence(op: Any) -> str:
    """Return validation's description of one unresolved round trip."""

    config = dict(record_member(op, "transform_config", None) or {})
    parents = tuple(getattr(op, "parents", ()) or ())
    return (
        f"foreign data round trip {op.label} ({config.get('entry')}) re-entered torch "
        f"after host computation TorchLens did not see (left through "
        f"{', '.join(config.get('exits', ()))} from {', '.join(parents)})"
    )


def warn_unresolved_foreign_data(trace: Any) -> None:
    """Warn once per capture when the trace holds an unresolved foreign data round trip.

    Parameters
    ----------
    trace:
        Completed capture.
    """

    unresolved = [
        op for op in getattr(trace, "ops", None) or () if is_unresolved_foreign_data(op.func)
    ]
    if not unresolved:
        return
    more = f" (and {len(unresolved) - 1} more)" if len(unresolved) > 1 else ""
    warnings.warn(
        TorchLensWarning(
            f"{_roundtrip_sentence(unresolved[0])}{more}; the recorded value is not "
            "replayed as a constant, so tl.validate fails this op. Remedy: wrap the "
            "foreign step in an autograd.Function or build the constant before the "
            "forward pass",
            code="foreign_data_roundtrip_unresolved",
            op_labels=[op.label for op in unresolved],
        ),
        stacklevel=external_warning_stacklevel(),
    )
