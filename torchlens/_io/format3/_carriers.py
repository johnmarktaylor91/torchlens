"""The opaque carrier values the value codec reads and writes.

A carrier stands in for a value the saved format cannot hold: ``UnencodableValue`` for a
value the writer could not encode (section 5), ``CallablePlaceholder`` for a callable an
intervention recipe could not persist (section 8). Both halves of the codec share these
classes: the decoder builds them, the encoder writes a decoded carrier back exactly (the
wire fields plus the private repr length), so ``encode(decode(x)) == x`` byte for byte.
The classes keep ``decode`` as their ``__module__``: that is their spelling in ``tl.types``
and in every artifact repr written before this module existed.
"""

from __future__ import annotations

import dataclasses

__all__ = ["CallablePlaceholder", "UnencodableValue"]

_CARRIER_MODULE = "torchlens._io.format3.decode"


@dataclasses.dataclass(frozen=True)
class UnencodableValue:
    """A value the writer could not encode: its type name, repr text and where it was.

    ``_repr_length`` (the unbounded repr's length) is carried for round trips, not public.

    Attributes
    ----------
    type_name:
        Dotted class path of the original value's type.
    repr_text:
        The original value's ``repr``, cut to the writer's length limit.
    at:
        Location of the value in the saved artifact: record kind, row index and
        column name, for example ``op[3].func_args``.
    """

    type_name: str
    repr_text: str
    at: str
    _repr_length: int = dataclasses.field(default=0, repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class CallablePlaceholder:
    """A callable the recipe could not persist: rule position and qualname (section 8)."""

    position: int | str
    qualname: str
    repr_length: int


UnencodableValue.__module__ = _CARRIER_MODULE
CallablePlaceholder.__module__ = _CARRIER_MODULE
