"""MLX model preparation helpers for technical-preview capture."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_named_modules(model: object) -> Iterator[tuple[str, object]]:
    """Yield an MLX module tree using best-effort public attributes.

    Parameters
    ----------
    model:
        Root MLX module.

    Yields
    ------
    tuple[str, object]
        Dotted module address and module object.
    """

    seen: set[int] = set()

    def _walk(module: object, address: str) -> Iterator[tuple[str, object]]:
        if id(module) in seen:
            return
        seen.add(id(module))
        yield address, module
        children = getattr(module, "_modules", None)
        if isinstance(children, dict):
            for child_name, child in children.items():
                child_address = f"{address}.{child_name}" if address else str(child_name)
                yield from _walk(child, child_address)
            return
        for child_name, child in vars(module).items():
            if child_name.startswith("_"):
                continue
            if child.__class__.__module__.startswith("mlx.nn"):
                child_address = f"{address}.{child_name}" if address else child_name
                yield from _walk(child, child_address)

    yield from _walk(model, "")


def prepare_model_once(model: object) -> object:
    """Apply one-time MLX preparation.

    Parameters
    ----------
    model:
        MLX model.

    Returns
    -------
    object
        The unchanged model.
    """

    return model


def prepare_model_session(session: object, model: object) -> object:
    """Apply per-session MLX preparation.

    Parameters
    ----------
    session:
        MLX backend session.
    model:
        MLX model.

    Returns
    -------
    object
        The unchanged model.
    """

    return model


def cleanup_model_session(session: object, prepared_model: object) -> None:
    """Clean up per-session MLX preparation.

    Parameters
    ----------
    session:
        MLX backend session.
    prepared_model:
        Prepared model object.
    """

    return None


__all__ = [
    "cleanup_model_session",
    "iter_named_modules",
    "prepare_model_once",
    "prepare_model_session",
]
