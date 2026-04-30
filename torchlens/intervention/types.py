"""Shared dataclass ownership for future TorchLens intervention schemas."""

from dataclasses import dataclass
from typing import Any

from .errors import _not_implemented


@dataclass(frozen=True)
class TensorSliceSpec:
    """Placeholder for future tensor slicing metadata."""

    positions: Any | None = None
    heads: Any | None = None
    batch: Any | None = None
    output_index: int | None = None
    position_axis: int | None = None
    head_axis: int | None = None
    query_axis: int | None = None
    key_axis: int | None = None
    feature_axis: int | None = None

    def __post_init__(self) -> None:
        """Reject construction until tensor slicing is implemented.

        Raises
        ------
        NotImplementedError
            Always raised until Phase 1 implements shared intervention types.
        """

        _not_implemented("TensorSliceSpec", "Phase 1")


__all__ = ["TensorSliceSpec"]
