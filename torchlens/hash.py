"""Provisional public structural-hash helpers.

The names in this namespace are provisional and may become ``fingerprint``-style
names after public API review. The digest is address-free and describes the
captured operation graph; it intentionally does not encode parameter values,
tensor shapes, or dtypes.
"""

from __future__ import annotations

from typing import Any

from .options import CaptureOptions
from .utils.hashing import compute_graph_shape_hash

__all__ = ["StructuralHashMismatchError", "assert_unchanged", "model", "trace"]


class StructuralHashMismatchError(AssertionError):
    """Raised when a model's provisional structural hash differs from its pin."""


def trace(captured_trace: Any) -> str:
    """Return an address-free structural hash for an existing trace.

    This provisional name may change after public API review. The digest uses
    TorchLens' established graph-shape hasher with module addresses excluded.

    Parameters
    ----------
    captured_trace:
        Completed TorchLens trace to hash.

    Returns
    -------
    str
        SHA-256 structural hash for the trace.
    """

    return compute_graph_shape_hash(captured_trace, include_module_address=False)


def _model_hash(captured_model: Any, example_input: Any) -> str:
    """Capture a model with the canonical metadata-only hash options.

    Parameters
    ----------
    captured_model:
        Model or callable accepted by :func:`torchlens.trace`.
    example_input:
        Example positional input accepted by :func:`torchlens.trace`.

    Returns
    -------
    str
        SHA-256 structural hash for the captured trace.
    """

    from .user_funcs import trace as capture_trace

    captured_trace = capture_trace(
        captured_model,
        example_input,
        capture=CaptureOptions(layers_to_save=None, inference_only=True),
    )
    return trace(captured_trace)


def model(model: Any, example_input: Any) -> str:
    """Capture a model cheaply and return its address-free structural hash.

    This provisional name may change after public API review. The metadata-only
    capture deliberately uses ``layers_to_save=None`` so buffer events are
    represented identically to the corresponding metadata-only trace passed to
    :func:`trace`.

    Parameters
    ----------
    model:
        Model or callable accepted by :func:`torchlens.trace`.
    example_input:
        Example positional input accepted by :func:`torchlens.trace`.

    Returns
    -------
    str
        SHA-256 structural hash for the captured trace.
    """

    return _model_hash(model, example_input)


def assert_unchanged(model: Any, example_input: Any, expected: str | None) -> str:
    """Assert that a model still has a pinned address-free structural hash.

    This provisional name may change after public API review. Pass ``None`` to
    print and return a hash suitable for adding as a CI pin.

    Parameters
    ----------
    model:
        Model or callable accepted by :func:`torchlens.trace`.
    example_input:
        Example positional input accepted by :func:`torchlens.trace`.
    expected:
        Previously pinned structural hash, or ``None`` to bootstrap one.

    Returns
    -------
    str
        The actual structural hash.

    Raises
    ------
    StructuralHashMismatchError
        If ``expected`` differs from the actual hash.
    TypeError
        If ``expected`` is neither a string nor ``None``.
    """

    actual = _model_hash(model, example_input)
    if expected is None:
        print(f"TorchLens structural hash: {actual}")
        return actual
    if not isinstance(expected, str):
        raise TypeError("expected must be a structural hash string or None")
    if expected != actual:
        raise StructuralHashMismatchError(
            "TorchLens structural hash changed: "
            f"expected {expected}, got {actual}. "
            "Inspect the captured traces to find the structural divergence."
        )
    return actual
