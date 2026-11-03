"""Compatibility path for the internal scalar-read guard (now ``torchlens._utils._internal_read``)."""

# layering: re-export kept for record importers; remove after the record-tier fold
from ..._utils._internal_read import (  # noqa: F401
    _internal_read_state as _internal_read_state,
    internal_scalar_read as internal_scalar_read,
)
