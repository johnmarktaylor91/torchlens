"""Typed refusals and the quiet capture used by ``extract`` and ``pluck``."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn

from ._errors import InvalidArgumentError
from .errors._base import TorchLensError

__all__ = [
    "TorchLensError",
    "capture_quiet",
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


@contextmanager
def capture_quiet() -> Iterator[None]:
    """Silence the capture-time ``layers_to_save`` zero-match warning during an extract capture.

    ``extract`` and ``pluck`` resolve every requested layer right after the capture and refuse
    typed (``op_lookup_not_found`` / ``activation_not_saved``) when nothing was saved, so the
    capture's zero-match disclosure would double-report ahead of that refusal.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"layers_to_save=.* matched zero layers", category=UserWarning
        )
        yield
