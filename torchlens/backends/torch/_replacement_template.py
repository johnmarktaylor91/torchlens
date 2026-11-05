"""Template field copies for a synthesized untagged-tensor boundary op."""

from __future__ import annotations

from typing import Any

from ..._constants import LAYER_PASS_LOG_FIELD_ORDER
from ..._utils._record_group_table import op_attr
from ._replacement_op_fields import _copy_field_value_for_replacement


def _replacement_template_fields(template_entry: Any | None) -> dict[str, Any]:
    """Return a copy of each pass-log field read from the template op.

    Parameters
    ----------
    template_entry
        The first parent op's live view, or ``None`` when the op has no parent;
        then every field copies ``None``.

    Returns
    -------
    dict[str, Any]
        Field name to its copied value, in ``LAYER_PASS_LOG_FIELD_ORDER``.
    """

    return {
        field_name: _copy_field_value_for_replacement(
            getattr(template_entry, op_attr(field_name), None)
            if template_entry is not None
            else None
        )
        for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }
