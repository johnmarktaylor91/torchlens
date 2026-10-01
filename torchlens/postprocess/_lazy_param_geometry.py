"""Step 15 support: finalize deferred geometry for lazy-at-prep parameters.

Extracted from finalization.py under the R43 size ratchet; called only by
`_finalize_param_logs` (step 15).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _finalize_lazy_param_geometry(self: "Trace") -> None:
    """Finalize deferred geometry for parameters that were lazy at prep.

    Lazy modules' UninitializedParameters materialized IN PLACE during the
    one captured forward; re-read shape/dtype/numel/bytes from the live
    reference so the inventory reports real counts (a never-materialized
    lazy param keeps its zero geometry and reads as never-executed).
    """

    from .._capture_state_helpers import _is_uninitialized_param
    from ..quantities import Bytes as _Bytes
    from ..utils.tensor_utils import get_memory_amount as _get_memory_amount

    for pl in self.param_logs:
        if not getattr(pl, "_lazy_at_prep", False):
            continue
        live_param = getattr(pl, "_param_ref", None)
        if live_param is None or _is_uninitialized_param(live_param):
            continue
        pl.shape = tuple(live_param.shape)
        pl.dtype = live_param.dtype
        pl.num_params = live_param.numel()
        pl.param_memory = _Bytes(_get_memory_amount(live_param))
        pl._lazy_at_prep = False  # type: ignore[attr-defined]
