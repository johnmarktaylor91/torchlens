"""Compatibility re-export; the module lives in ``torchlens._distributed._dtensor``."""

# layering: re-export kept for record importers; remove after the record-tier fold
from .._distributed._dtensor import (  # noqa: F401
    RANK_LOCAL_SHARD as RANK_LOCAL_SHARD,
    _as_int_list as _as_int_list,
    dtensor_dual_geometry as dtensor_dual_geometry,
    shard_local_placements as shard_local_placements,
    value_marks_shard_local as value_marks_shard_local,
)
