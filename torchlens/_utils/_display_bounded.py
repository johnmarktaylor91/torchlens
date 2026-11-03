"""Bounded printouts for collections and private records (house style v1).

:class:`RecordTuple` is the tuple TorchLens returns for owned sequences of
records (ATen rows, event buckets). It is a plain ``tuple`` in every respect
(iteration, indexing, equality with tuples, hashing, pickling) except its
repr and str, which print a count and at most five members instead of every
record. :func:`short_dataclass_repr` is the one family repr for private
dataclasses: scalar fields only, nested records as ``<Type>``, collections
as ``<N items>``, cut to one line.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any

from ._display_card import LIST_LIMIT, REPR_COLS, bounded_lines, inline_names
from .fail_open import fail_open

_SCALARS = (bool, int, float, str, bytes, type(None), enum.Enum)


class RecordTuple(tuple):
    """A tuple of records with a bounded repr: ``AtenOps: 2 calls (aten_825, aten_826)``.

    ``title`` names the collection and ``noun`` counts its members; both are
    display-only and do not take part in equality or hashing.
    """

    title: str
    noun: str
    singular: str

    def __new__(
        cls,
        items: Iterable[Any] = (),
        title: str = "Records",
        noun: str = "records",
        singular: str | None = None,
    ) -> RecordTuple:
        instance = super().__new__(cls, items)
        instance.title = title
        instance.noun = noun
        instance.singular = singular or noun.removesuffix("s")
        return instance

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (tuple(self), self.title, self.noun, self.singular))

    def _head(self) -> str:
        return f"{self.title}: {len(self)} {self.singular if len(self) == 1 else self.noun}"

    def __repr__(self) -> str:
        keys = [_member_key(item) for item in self[:LIST_LIMIT]]
        head = self._head()
        if not keys:
            return head
        line = f"{head} ({inline_names(keys, limit=2)})" if len(self) <= LIST_LIMIT else head
        return line if len(line) <= REPR_COLS else head

    def __str__(self) -> str:
        lines = [self._head()]
        lines.extend(bounded_lines(self, _member_line))
        return "\n".join(lines)


def short_dataclass_repr(obj: Any, *, width: int = REPR_COLS) -> str:
    """``ClassName(field=value, ...)`` with scalar fields only, at most ``width`` columns.

    Nested dataclasses print as ``<ClassName>``, sized collections as
    ``<N items>``, other objects as ``<TypeName>``. Never recurses and never
    calls a member's repr except for scalars.
    """

    name = type(obj).__name__
    if not dataclasses.is_dataclass(obj):
        return f"<{name}>"
    parts: list[str] = []
    for field in dataclasses.fields(obj):
        if not field.repr:
            continue
        value = _read_field(obj, field.name)
        parts.append(f"{field.name}={_short_value(value)}")
    text = f"{name}({', '.join(parts)})"
    if len(text) <= width:
        return text
    return text[: width - 4] + "...)"


def _short_value(value: Any) -> str:
    """Scalar repr, or a type or count placeholder for anything nested."""

    if isinstance(value, _SCALARS):
        text = repr(value)
        return text if len(text) <= 40 else text[:37] + "..."
    if dataclasses.is_dataclass(value):
        return f"<{type(value).__name__}>"
    if isinstance(value, (tuple, list, set, frozenset, dict)):
        return f"<{len(value)} items>"
    return f"<{type(value).__name__}>"


def _member_key(item: Any) -> str:
    """The identifying key of a member: its ``label`` when it has one."""

    label = fail_open(lambda: getattr(item, "label", None), lambda _error: None)
    return str(label) if isinstance(label, str) else type(item).__name__


def _member_line(item: Any) -> str:
    """One member's one-line repr, guarded so a bad member cannot break the list."""

    return fail_open(lambda: repr(item), lambda error: f"<{type(item).__name__}: display failed>")


def _read_field(obj: Any, name: str) -> Any:
    """Read one dataclass field for display; a failing property reads as ``None``."""

    def read() -> Any:
        return getattr(obj, name)

    def fallback(_error: Exception) -> Any:
        return None

    return fail_open(read, fallback)
