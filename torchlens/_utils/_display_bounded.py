"""Bounded printouts for collections and private records (house style v1).

:class:`RecordTuple` is the tuple TorchLens returns for owned sequences of
records (ATen rows, event buckets). It is a plain ``tuple`` in every respect
(iteration, indexing, equality with tuples, hashing, pickling) except its
repr and str: repr prints a count; str adds at most five members.
:func:`short_dataclass_repr` is the one family repr for private dataclasses: scalar fields first, nested objects omitted, collections
as ``<N items>``, cut to one line.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any

from ._display_card import REPR_COLS, bounded_lines, more_row, one_line
from ._display_keys import display_key
from ._display_numbers import plural
from .fail_open import fail_open

_SCALARS = (bool, int, float, str, bytes, type(None), enum.Enum)


class RecordTuple(tuple):
    """A tuple of records with a count-only repr: ``AtenOps: 2 calls``.

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
        return f"{self.title}: {plural(len(self), self.singular, self.noun)}"

    def __repr__(self) -> str:
        return self._head()

    def __str__(self) -> str:
        lines = [repr(self)]
        lines.extend(bounded_lines(self, _member_line))
        lines.append(more_row((".count()", ".index()")))
        return "\n".join(lines)


def short_dataclass_repr(obj: Any, *, width: int = REPR_COLS) -> str:
    """``ClassName(field=value, ...)`` with scalar fields only, at most ``width`` columns.

    Sized collections print as ``<N items>``; nested records are omitted.
    Never recurses and never calls a member's repr except for scalars.
    """

    name = type(obj).__name__
    if not dataclasses.is_dataclass(obj):
        return f"<{name}>"
    if name == "OpRecord":
        return one_line(
            f"OpRecord {display_key(obj)}",
            [f"kind {_read_field(obj, 'kind')}", f"sequence {_read_field(obj, 'seq')}"],
        )
    parts: list[str] = []
    identity = ("class_name", "label", "op_label", "label_raw", "name", "kind")
    ordered = sorted(
        dataclasses.fields(obj),
        key=lambda field: identity.index(field.name) if field.name in identity else len(identity),
    )
    for field in ordered:
        if not field.repr or field.name.endswith("object_id"):
            continue
        value = _read_field(obj, field.name)
        if not isinstance(value, (*_SCALARS, tuple, list, set, frozenset, dict)):
            continue
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
        return f"<{plural(len(value), 'item')}>"
    return f"<{type(value).__name__}>"


def _member_key(item: Any) -> str:
    """The identifying key of a member: its ``label`` when it has one."""

    return display_key(item)


def _member_line(item: Any) -> str:
    """One member's one-line repr, guarded so a bad member cannot break the list."""

    def render() -> str:
        line = repr(item).replace("\n", " ")
        key = _member_key(item)
        if key != "?":
            if type(item).__name__ == "OpRecord":
                line = f"kind {item.kind}, sequence {item.seq}"
            line = f"{key}: {line}"
        return line if len(line) <= REPR_COLS else line[: REPR_COLS - 3] + "..."

    return fail_open(render, lambda error: f"<{type(item).__name__}: display failed>")


def _read_field(obj: Any, name: str) -> Any:
    """Read one dataclass field for display; a failing property reads as ``None``."""

    def read() -> Any:
        return getattr(obj, name)

    def fallback(_error: Exception) -> Any:
        return None

    return fail_open(read, fallback)
