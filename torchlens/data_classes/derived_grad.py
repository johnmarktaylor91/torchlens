"""Backend-neutral derived gradient records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..ir.refs import DtypeRef
from ._accessor_base import Accessor


@dataclass(frozen=True)
class DerivedGradRecord:
    """Leaf-level gradient derived from a second functional backend run.

    Parameters
    ----------
    path
        Stable pytree leaf path for the differentiated value.
    source
        Differentiated source group, such as ``"params"`` or ``"inputs"``.
    argnum
        Backend positional argument index differentiated by the AD run.
    input_argnum
        Input-relative argument index for input gradients, or ``None`` for params.
    aval
        Backend abstract-value description for the gradient leaf.
    dtype_ref
        Backend-neutral dtype reference for the gradient payload.
    grad
        Gradient payload for the leaf.
    provenance
        Backend-owned validation and fingerprint metadata.
    """

    path: str
    source: str
    argnum: int
    input_argnum: int | None
    aval: str
    dtype_ref: DtypeRef | None
    grad: Any
    provenance: Mapping[str, Any]


class DerivedGradAccessor(Accessor[DerivedGradRecord]):
    """Dict-like accessor for leaf-level derived gradient records."""

    def __init__(self, records: Mapping[str, DerivedGradRecord] | None = None) -> None:
        """Initialize a derived-gradient accessor.

        Parameters
        ----------
        records
            Mapping from stable leaf paths to gradient records.
        """

        super().__init__({} if records is None else records)


@dataclass(frozen=True)
class IntermediateDerivedGradRecord:
    """Op-level gradient derived from a backend-owned auxiliary AD run.

    Parameters
    ----------
    op_label
        Stable pass-qualified TorchLens op label.
    layer_label
        Stable layer label for the owning op.
    aval
        Backend abstract-value description for the gradient payload.
    dtype_ref
        Backend-neutral dtype reference for the gradient payload.
    grad
        Gradient payload for the op output.
    provenance
        Backend-owned mechanism, loss, save predicate, and status metadata.
    """

    op_label: str
    layer_label: str
    aval: str
    dtype_ref: DtypeRef | None
    grad: Any
    provenance: Mapping[str, Any]


class IntermediateDerivedGradAccessor(Accessor[IntermediateDerivedGradRecord]):
    """Dict-like accessor for op-level derived gradient records."""

    def __init__(self, records: Mapping[str, IntermediateDerivedGradRecord] | None = None) -> None:
        """Initialize an intermediate-derived-gradient accessor.

        Parameters
        ----------
        records
            Mapping from stable op labels to gradient records.
        """

        exact_records = {
            key: record
            for key, record in ({} if records is None else records).items()
            if record.provenance.get("status") == "exact"
        }
        super().__init__(exact_records)
