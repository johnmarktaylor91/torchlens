"""Record display facade: one-line identities and bounded labelled cards.

Record values use words; explicit ``tl.stats.glance`` retains its sparkline.
Implementations are split by record family to keep display-only concerns small.
"""

from ._value_modules import (
    module_call_card as module_call_card,
    module_call_repr_line as module_call_repr_line,
    module_card as module_card,
    module_repr_line as module_repr_line,
)
from ._value_ops import (
    layer_card as layer_card,
    layer_repr_line as layer_repr_line,
    op_card as op_card,
    op_repr_line as op_repr_line,
)
from ._value_params import (
    buffer_card as buffer_card,
    buffer_repr_line as buffer_repr_line,
    param_card as param_card,
    param_repr_line as param_repr_line,
)
from ._value_state import (
    EDITED as EDITED,
    FREED as FREED,
    LAZY_DISK as LAZY_DISK,
    METADATA_ONLY as METADATA_ONLY,
    NOT_SAVED as NOT_SAVED,
    _payload_state as _payload_state,
    payload_core as payload_core,
)
