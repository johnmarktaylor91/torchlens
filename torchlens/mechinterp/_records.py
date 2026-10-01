"""Typed result records + the shared labeled-coordinate protocol (mikit item 4).

Every kit function returns a TYPED record whose rows carry the same small
coordinate/provenance vocabulary: pass-qualified op labels, live site keys,
module-derived writer labels, capability/provenance strings, and validation
receipts. The records stay distinct classes (their invariants differ) but the
protocol is one, shared here so the library never grows two label systems --
``torchlens.attribution``'s per-layer tables are the intended second consumer.

TLens muscle-memory ergonomics: records with a component axis support
``stack, labels = record`` tuple unpacking, ``.to_pandas()``, and ``.top(k)``.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from ._errors import refuse

__all__ = [
    "ContributionScores",
    "ComponentRow",
    "ComponentStack",
    "Coordinate",
    "StackGrading",
]

#: Stack grading vocabulary (mikit D7). ``open`` never reaches a caller: an
#: OPEN stack refuses at construction, always.
StackGrading = Literal["complete", "closed_constant", "closed_unresolved"]

#: Row-kind vocabulary for spine writers and constants. ``expert`` is carried
#: from day one for future MoE per-expert rows (mikit plumbing ledger).
RowKind = Literal[
    "embedding",
    "attention",
    "mlp",
    "writer",
    "constant",
    "unresolved_remainder",
    "residual",
]


@dataclass(frozen=True)
class Coordinate:
    """One labeled graph coordinate (the shared protocol's address row).

    Parameters
    ----------
    label:
        Human-facing component label -- a module address in eval mode
        (``transformer.h.3.attn.c_proj``), never a raw op name (mikit D6).
    op_label:
        Pass-qualified captured op label the value rode (``add_10_101:2``).
    site_key:
        Live structural site key for the op (recapture-stable identity).
    pass_index:
        Pass qualifier of the op (within-forward reuse disambiguation).
    kind:
        Row-kind classification (embedding / attention / mlp / writer /
        constant / unresolved_remainder / residual).
    module_address:
        Innermost containing module address, when resolved.
    provenance:
        How the value was obtained (``"captured"``, ``"computed"``,
        ``"verified_constant"``, ``"injected"``), extension-open by design.
    expert:
        MoE expert index when the writer is a routed expert; ``None`` today.
    """

    label: str
    op_label: str | None = None
    site_key: str | None = None
    pass_index: int | None = None
    kind: str = "writer"
    module_address: str | None = None
    provenance: str = "captured"
    expert: int | None = None


@dataclass(frozen=True)
class ComponentRow:
    """One component row: a coordinate plus its tensor value."""

    coordinate: Coordinate
    value: torch.Tensor
    #: Scalar coefficient the spine add applied to this writer (1.0 for the
    #: plain ``a + b`` form; recorded, never silently folded).
    coefficient: float = 1.0


class ComponentStack:
    """An ordered, graded stack of component rows summing to a target.

    The record for residual decomposition/accumulation and every downstream
    stack-shaped consumer (norm folding, DLA). Invariants:

    - Rows are in EXECUTION order (the order whose accumulation replays the
      forward's own fp32 add sequence -- the bitwise identity's precondition).
    - ``grading`` is settled at construction (mikit D7): ``complete`` (every
      row a named writer), ``closed_constant`` (non-writer rows are VERIFIED
      constants), or ``closed_unresolved`` (a diagnostic ``strict=False``
      product, refused by DLA / gallery / launch claims). An OPEN stack never
      constructs -- the producer refuses instead.
    - ``target_value`` is the captured tensor the rows claim to sum to, and
      ``identity_receipt`` records how that claim was checked.

    Tuple unpacking (``stack_tensor, labels = record``) serves TLens
    muscle memory; iteration order is (values, labels).
    """

    def __init__(  # noqa: PLR0913 -- a graded stack's invariants are its fields
        self,
        rows: tuple[ComponentRow, ...],
        *,
        grading: StackGrading,
        target_coordinate: Coordinate,
        target_value: torch.Tensor | None,
        identity_receipt: dict[str, Any],
        diagnostic_only: bool = False,
    ) -> None:
        """Freeze a graded component stack.

        Parameters
        ----------
        rows:
            Component rows in execution order.
        grading:
            Settled D7 grading; ``open`` is not constructible.
        target_coordinate:
            Coordinate of the tensor the rows sum to.
        target_value:
            The captured target tensor (``None`` on structure-only products).
        identity_receipt:
            Validation receipt: check kind, result, and measured residual.
        diagnostic_only:
            Stamped by ``strict=False`` producers; DLA and gallery consumers
            refuse diagnostic stacks.
        """

        self._rows = rows
        self.grading: StackGrading = grading
        self.target_coordinate = target_coordinate
        self.target_value = target_value
        self.identity_receipt = dict(identity_receipt)
        self.diagnostic_only = diagnostic_only

    @property
    def rows(self) -> tuple[ComponentRow, ...]:
        """Return the component rows in execution order."""

        return self._rows

    @property
    def labels(self) -> tuple[str, ...]:
        """Return each row's component label, in row order."""

        return tuple(row.coordinate.label for row in self._rows)

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        """Return each row's coordinate, in row order."""

        return tuple(row.coordinate for row in self._rows)

    def __len__(self) -> int:
        """Return the number of rows."""

        return len(self._rows)

    def __getitem__(self, index: int) -> ComponentRow:
        """Return one row by position."""

        return self._rows[index]

    def stack(self) -> torch.Tensor:
        """Return the row values stacked on a new leading component axis.

        Broadcast-shaped rows (a position embedding captured at batch 1) are
        expanded to the target shape first so the axis is uniform.
        """

        shapes = {tuple(row.value.shape) for row in self._rows}
        if len(shapes) == 1:
            return torch.stack([row.value for row in self._rows])
        if self.target_value is None:
            refuse(
                code="mi_stack_shape_mixed",
                message="Rows have mixed shapes and no target value to broadcast against.",
                remedy="read rows individually via .rows, or produce the stack from a "
                "payload-bearing capture",
                shapes=sorted(str(shape) for shape in shapes),
            )
        target_shape = tuple(self.target_value.shape)
        return torch.stack([row.value.expand(target_shape) for row in self._rows])

    def sum(self) -> torch.Tensor:
        """Accumulate the rows in execution order with recorded coefficients.

        This replays the forward's own add sequence (left-to-right, one add
        per row), which is what makes the bitwise identity assertable.
        """

        if not self._rows:
            refuse(
                code="mi_stack_empty",
                message="The stack has no rows to accumulate.",
                remedy="produce the stack from a trace with at least one proven writer",
            )
        first = self._rows[0]
        total = first.value * first.coefficient if first.coefficient != 1.0 else first.value
        for row in self._rows[1:]:
            if row.coefficient != 1.0:
                total = torch.add(total, row.value, alpha=row.coefficient)
            else:
                total = total + row.value
        return total

    def __iter__(self) -> Iterator[Any]:
        """Yield ``(stacked_values, labels)`` -- TLens-style tuple unpacking."""

        yield self.stack()
        yield self.labels

    def top(self, k: int = 5, *, by: str = "norm") -> tuple[tuple[str, float], ...]:
        """Return the ``k`` largest rows as ``(label, score)`` pairs.

        Parameters
        ----------
        k:
            Number of rows to return (clamped to the row count).
        by:
            Scoring reduction: ``"norm"`` (L2 over the whole row value) is the
            only v1 option.
        """

        if by != "norm":
            refuse(
                code="mi_top_by_invalid",
                message=f"Unknown top() reduction {by!r}.",
                remedy='pass by="norm" (the only v1 reduction)',
            )
        scored = [
            (row.coordinate.label, float(torch.linalg.vector_norm(row.value.detach().float())))
            for row in self._rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return tuple(scored[: max(0, k)])

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame of coordinates + row norms.

        pandas is an optional dependency; absence refuses typed rather than
        ImportError-ing mid-analysis.
        """

        frame_rows = [_coordinate_frame_row(row) for row in self._rows]
        return _pandas_frame(frame_rows)

    def __repr__(self) -> str:
        """Return a compact, honest summary."""

        return (
            f"ComponentStack(rows={len(self._rows)}, grading={self.grading!r}, "
            f"target={self.target_coordinate.label!r}, "
            f"identity={self.identity_receipt.get('result', 'unchecked')!r})"
        )


@dataclass(frozen=True)
class _PandasRow:
    """Internal: one flattened frame row (kept dataclass for field order)."""

    label: str
    kind: str
    op_label: str | None
    site_key: str | None
    pass_index: int | None
    module_address: str | None
    provenance: str
    coefficient: float
    norm: float
    extras: dict[str, Any] = field(default_factory=dict)


def _coordinate_frame_row(row: ComponentRow) -> dict[str, Any]:
    """Flatten one component row into plain frame columns."""

    coord = row.coordinate
    return {
        "label": coord.label,
        "kind": coord.kind,
        "op_label": coord.op_label,
        "site_key": coord.site_key,
        "pass_index": coord.pass_index,
        "module_address": coord.module_address,
        "provenance": coord.provenance,
        "coefficient": row.coefficient,
        "norm": float(torch.linalg.vector_norm(row.value.detach().float())),
    }


def _pandas_frame(rows: list[dict[str, Any]]) -> Any:
    """Build a DataFrame, refusing typed when pandas is not installed."""

    try:
        import pandas
    except ImportError:
        refuse(
            code="mi_pandas_unavailable",
            message="to_pandas() needs the optional pandas dependency, which is not installed.",
            remedy="pip install pandas, or read .rows / .labels directly",
        )
    return pandas.DataFrame(rows)


class ContributionScores:
    """Per-component direct-logit contribution rows (mikit D9's record).

    Rows carry the shared coordinate protocol; ``values[i]`` is row ``i``'s
    contribution to each requested direction at each selected position
    (``[batch, n_positions, n_directions]``). ``constant`` is the ONE
    verified constant row (final-norm beta + unembedding bias, projected;
    nothing else -- writer-side biases already live inside the literal
    writers). ``native`` is the model's own captured logits (or logit
    diffs) at the same coordinates, and ``identity_receipt`` records the
    in-API check ``sum(rows) + constant == native``.
    """

    def __init__(  # noqa: PLR0913 -- a contribution table's invariants are its fields
        self,
        coordinates: tuple[Coordinate, ...],
        values: torch.Tensor,
        *,
        constant: torch.Tensor,
        native: torch.Tensor,
        answer_tokens: tuple[int, ...],
        vs_tokens: tuple[int, ...] | None,
        positions: tuple[int, ...],
        identity_receipt: dict[str, Any],
    ) -> None:
        """Freeze a contribution table (see class docstring for axes)."""

        self.coordinates = coordinates
        self.values = values
        self.constant = constant
        self.native = native
        self.answer_tokens = answer_tokens
        self.vs_tokens = vs_tokens
        self.positions = positions
        self.identity_receipt = dict(identity_receipt)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return each row's component label."""

        return tuple(coordinate.label for coordinate in self.coordinates)

    def __len__(self) -> int:
        """Return the number of component rows."""

        return len(self.coordinates)

    def __iter__(self) -> Iterator[Any]:
        """Yield ``(values, labels)`` -- TLens-style tuple unpacking."""

        yield self.values
        yield self.labels

    def top(
        self, k: int = 5, *, position: int = -1, direction: int = 0
    ) -> tuple[tuple[str, float], ...]:
        """Return the ``k`` largest contributors at one (position, direction).

        Parameters
        ----------
        k:
            Number of rows to return.
        position:
            Index into the SELECTED positions axis (default: the last).
        direction:
            Index into the requested directions axis.
        """

        scores = self.values[:, :, position, direction].detach().mean(dim=1)
        pairs = sorted(
            zip(self.labels, (float(v) for v in scores), strict=True),
            key=lambda p: p[1],
            reverse=True,
        )
        return tuple(pairs[: max(0, k)])

    def to_pandas(self) -> Any:
        """Return a DataFrame: one row per component, mean batch scores."""

        frame_rows = []
        for index, coordinate in enumerate(self.coordinates):
            row: dict[str, Any] = {
                "label": coordinate.label,
                "kind": coordinate.kind,
                "op_label": coordinate.op_label,
                "site_key": coordinate.site_key,
            }
            for p_index, position in enumerate(self.positions):
                for d_index in range(self.values.shape[-1]):
                    row[f"pos{position}_dir{d_index}"] = float(
                        self.values[index, :, p_index, d_index].mean()
                    )
            frame_rows.append(row)
        return _pandas_frame(frame_rows)

    def __repr__(self) -> str:
        """Return a compact, honest summary."""

        return (
            f"ContributionScores(rows={len(self.coordinates)}, "
            f"positions={self.positions}, answers={self.answer_tokens}, "
            f"identity={self.identity_receipt.get('result', 'unchecked')!r})"
        )
