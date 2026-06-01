"""Numeric quantity types with unit-aware display."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Self


class Quantity(ABC):
    """Marker base class for TorchLens numeric quantity wrappers.

    Quantity subclasses also subclass a concrete numeric type such as ``int`` or
    ``float``.  The marker base lets callers test for all unit-aware quantities
    with ``isinstance(value, Quantity)``.
    """


class _IntQuantity(int, Quantity):
    """Base class for integer-backed unit quantities."""

    _unit_name: ClassVar[str] = "quantity"

    def _new(self, value: int | float) -> Self:
        """Return ``value`` wrapped as this quantity type.

        Parameters
        ----------
        value:
            Numeric value to wrap.

        Returns
        -------
        Self
            Wrapped integer quantity.
        """

        return type(self)(int(value))

    def _check_other_quantity(self, other: Any) -> None:
        """Raise if ``other`` is an incompatible quantity.

        Parameters
        ----------
        other:
            Candidate right-hand operand.

        Raises
        ------
        TypeError
            If ``other`` is a different ``Quantity`` subtype.
        """

        if isinstance(other, Quantity) and not isinstance(other, type(self)):
            raise TypeError(f"Cannot mix {type(self).__name__} with {type(other).__name__}")

    def __add__(self, other: Any) -> Self:
        """Add a compatible scalar or same-unit quantity."""

        self._check_other_quantity(other)
        return self._new(int(self) + other)

    def __radd__(self, other: Any) -> Self:
        """Add this quantity to a compatible left operand."""

        self._check_other_quantity(other)
        return self._new(other + int(self))

    def __sub__(self, other: Any) -> Self:
        """Subtract a compatible scalar or same-unit quantity."""

        self._check_other_quantity(other)
        return self._new(int(self) - other)

    def __rsub__(self, other: Any) -> Self:
        """Subtract this quantity from a compatible left operand."""

        self._check_other_quantity(other)
        return self._new(other - int(self))

    def __mul__(self, other: Any) -> Self:
        """Scale this quantity by a scalar.

        Raises
        ------
        TypeError
            If multiplying by another quantity.
        """

        if isinstance(other, Quantity):
            raise TypeError(f"Cannot multiply {self._unit_name} quantities")
        return self._new(int(self) * other)

    def __rmul__(self, other: Any) -> Self:
        """Scale this quantity by a scalar left operand."""

        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Self | float:
        """Divide by a scalar, or return a ratio for same-unit division."""

        self._check_other_quantity(other)
        if isinstance(other, type(self)):
            return int(self) / int(other)
        return self._new(int(self) / other)

    def __floordiv__(self, other: Any) -> Self | int:
        """Floor-divide by a scalar, or return an integer same-unit ratio."""

        self._check_other_quantity(other)
        if isinstance(other, type(self)):
            return int(self) // int(other)
        return self._new(int(self) // other)

    def __neg__(self) -> Self:
        """Return the negated quantity."""

        return self._new(-int(self))

    def __abs__(self) -> Self:
        """Return the absolute quantity."""

        return self._new(abs(int(self)))

    def __format__(self, format_spec: str) -> str:
        """Format this quantity using the subclass unit grammar."""

        if format_spec == "raw":
            return str(int(self))
        return str(self)


class Bytes(_IntQuantity):
    """Memory quantity stored as bytes with human-readable display."""

    _unit_name: ClassVar[str] = "byte"
    _UNITS: ClassVar[tuple[tuple[str, float], ...]] = (
        ("B", 1.0),
        ("KB", 1024.0),
        ("MB", 1024.0**2),
        ("GB", 1024.0**3),
        ("TB", 1024.0**4),
        ("PB", 1024.0**5),
    )

    def __str__(self) -> str:
        """Return a human-readable byte string."""

        return self._format_bytes("")

    def __format__(self, format_spec: str) -> str:
        """Format bytes with optional precision and forced unit.

        Parameters
        ----------
        format_spec:
            ``"raw"`` for the raw byte count, a unit such as ``"MB"``, or a
            float precision plus unit such as ``".2f MB"``.

        Returns
        -------
        str
            Formatted byte quantity.
        """

        if format_spec == "raw":
            return str(int(self))
        return self._format_bytes(format_spec)

    def _format_bytes(self, format_spec: str) -> str:
        """Return a formatted byte string.

        Parameters
        ----------
        format_spec:
            Optional format spec containing precision and/or a forced unit.

        Returns
        -------
        str
            Human-readable byte string.
        """

        numeric_spec, unit = self._parse_format_spec(format_spec)
        value = float(int(self))
        sign = -1.0 if value < 0 else 1.0
        abs_value = abs(value)
        if unit is None:
            unit, divisor = self._best_unit(abs_value)
        else:
            divisor = self._unit_divisor(unit)
        scaled = sign * abs_value / divisor
        if unit == "B":
            return f"{int(scaled)} B"
        if numeric_spec:
            return f"{format(scaled, numeric_spec)} {unit}"
        return f"{scaled:.1f} {unit}"

    @classmethod
    def _parse_format_spec(cls, format_spec: str) -> tuple[str, str | None]:
        """Parse a byte format spec into numeric spec and unit.

        Parameters
        ----------
        format_spec:
            Format spec passed to ``__format__``.

        Returns
        -------
        tuple[str, str | None]
            Numeric format spec and optional unit.
        """

        stripped = format_spec.strip()
        if not stripped:
            return "", None
        parts = stripped.split()
        if len(parts) == 1 and parts[0] in {unit for unit, _ in cls._UNITS}:
            return "", parts[0]
        if len(parts) == 2 and parts[1] in {unit for unit, _ in cls._UNITS}:
            return parts[0], parts[1]
        return stripped, None

    @classmethod
    def _best_unit(cls, abs_value: float) -> tuple[str, float]:
        """Return the largest unit suitable for ``abs_value``.

        Parameters
        ----------
        abs_value:
            Absolute byte count.

        Returns
        -------
        tuple[str, float]
            Unit suffix and divisor.
        """

        chosen = cls._UNITS[0]
        for unit in cls._UNITS:
            if abs_value >= unit[1]:
                chosen = unit
        return chosen

    @classmethod
    def _unit_divisor(cls, unit_name: str) -> float:
        """Return the byte divisor for ``unit_name``.

        Parameters
        ----------
        unit_name:
            Unit suffix such as ``"MB"``.

        Returns
        -------
        float
            Unit divisor.

        Raises
        ------
        ValueError
            If ``unit_name`` is unsupported.
        """

        for unit, divisor in cls._UNITS:
            if unit == unit_name:
                return divisor
        raise ValueError(f"Unsupported byte unit: {unit_name}")


class Duration(float, Quantity):
    """Duration quantity stored in seconds."""


class Flops(_IntQuantity):
    """Floating-point operation count quantity."""


class Macs(_IntQuantity):
    """Multiply-accumulate operation count quantity."""


def as_bytes(value: int | float | None) -> Bytes | None:
    """Return ``value`` wrapped as ``Bytes`` unless it is ``None``.

    Parameters
    ----------
    value:
        Byte count or ``None``.

    Returns
    -------
    Bytes | None
        Wrapped byte count, preserving ``None``.
    """

    return None if value is None else Bytes(value)
