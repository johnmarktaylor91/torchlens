"""Typed refusals and capture options used by ``extract`` and ``pluck``."""

from __future__ import annotations

from typing import NoReturn

from ._errors import InvalidArgumentError
from .errors._base import TorchLensError
from .options import CaptureOptions

__all__ = [
    "CaptureOptions",
    "TorchLensError",
    "refuse_activation_not_saved",
    "refuse_lookup_not_found",
]

#: Remedy every extract-side lookup refusal carries.
LOOKUP_REMEDY = (
    "pass a layer label, module address or pass-qualified label the capture records; "
    "trace.find_layers(substring) lists the candidates"
)


def refuse_lookup_not_found(
    message: str, key: str, *, cause: BaseException | None = None
) -> NoReturn:
    """Raise the ``op_lookup_not_found`` refusal for a lookup that resolves to no layer."""

    raise InvalidArgumentError(
        message, code="op_lookup_not_found", remedy=LOOKUP_REMEDY, key=key
    ) from cause


def refuse_activation_not_saved(key: str) -> NoReturn:
    """Raise the ``activation_not_saved`` refusal for a layer that resolved but kept no out."""

    raise InvalidArgumentError(
        f"Layer {key!r} resolved but has no saved out.",
        code="activation_not_saved",
        remedy="request the layer through extract's layers argument so the capture saves it",
        key=key,
    )
