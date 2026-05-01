"""Experimental TorchLens APIs with unstable naming and behavior."""

from __future__ import annotations

import re
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from torch import nn

_ATTRIBUTE_PART_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<indexes>(?:\[[0-9]+\])*)")
_STOP_AFTER_SITE: Any | None = None


def attribute_walk(model: nn.Module, address: str) -> Any:
    """Resolve a dotted/indexed attribute-walk address on a module.

    Parameters
    ----------
    model:
        Root PyTorch module.
    address:
        Address such as ``"transformer.h[5].mlp.output"``.

    Returns
    -------
    Any
        Object reached by walking attributes and integer indexes.

    Raises
    ------
    AttributeError
        If any attribute component is missing.
    IndexError
        If any index component is invalid.
    ValueError
        If the address syntax is malformed.
    """

    current: Any = model
    if not address:
        return current
    for part in address.split("."):
        match = _ATTRIBUTE_PART_RE.fullmatch(part)
        if match is None:
            raise ValueError(f"Malformed attribute-walk segment: {part!r}")
        name = match.group("name")
        if not hasattr(current, name):
            raise AttributeError(f"{type(current).__name__!s} has no attribute {name!r}")
        current = getattr(current, name)
        indexes = re.findall(r"\[([0-9]+)\]", match.group("indexes"))
        for index_text in indexes:
            current = current[int(index_text)]
    return current


@contextmanager
def stop_after(site: Any) -> Iterator[None]:
    """Set the experimental stop-after site for ``torchlens.peek``.

    Parameters
    ----------
    site:
        Site where peek may stop early.

    Yields
    ------
    None
        Context body.
    """

    global _STOP_AFTER_SITE

    previous = _STOP_AFTER_SITE
    _STOP_AFTER_SITE = site
    try:
        yield
    finally:
        _STOP_AFTER_SITE = previous


@dataclass
class AutoCaptureSession:
    """State returned by ``auto_capture``."""

    model: nn.Module
    every: int
    logs: list[Any] = field(default_factory=list)
    calls: int = 0


@contextmanager
def auto_capture(model: nn.Module, every: int = 100) -> Iterator[AutoCaptureSession]:
    """Capture every Nth forward call in a block.

    Parameters
    ----------
    model:
        Model whose ``forward`` method is wrapped temporarily.
    every:
        Capture interval.

    Yields
    ------
    AutoCaptureSession
        Session containing captured logs.

    Raises
    ------
    RuntimeError
        If ``TORCHLENS_AUTO=1`` is set.
    """

    if os.environ.get("TORCHLENS_AUTO") == "1":
        raise RuntimeError("TORCHLENS_AUTO=1 is intentionally unsupported; use auto_capture().")
    if every <= 0:
        raise ValueError("every must be positive.")

    import torchlens

    session = AutoCaptureSession(model=model, every=every)
    original_forward = model.forward

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the original forward and capture every Nth call."""

        session.calls += 1
        if session.calls % every == 0:
            model.forward = original_forward  # type: ignore[method-assign]
            try:
                capture_input = list(args)
                log = torchlens.log_forward_pass(model, capture_input, kwargs or None)
                session.logs.append(log)
            finally:
                model.forward = wrapped_forward  # type: ignore[method-assign]
        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward  # type: ignore[method-assign]
    try:
        yield session
    finally:
        model.forward = original_forward  # type: ignore[method-assign]


def _active_stop_after_site() -> Any | None:
    """Return the active experimental stop-after site.

    Returns
    -------
    Any | None
        Active site, if any.
    """

    return _STOP_AFTER_SITE


__all__ = [
    "AutoCaptureSession",
    "attribute_walk",
    "auto_capture",
    "dagua",
    "node_styles",
    "stop_after",
]
