"""Provisional public structural-hash helpers.

The names in this namespace are provisional and may become ``fingerprint``-style
names after public API review. The digest is address-free and describes the
captured operation graph; it intentionally does not encode parameter values,
tensor shapes, or dtypes.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence, Set
from hashlib import sha256
from typing import Any

import numpy as np
import torch

from ._errors import ArgumentTypeError
from ._transport import digest_byte_view
from ._utils.hashing import compute_graph_shape_hash
from .errors._base import TorchLensWarning, ValidationError
from .options import CaptureOptions

__all__ = ["StructuralHashMismatchError", "assert_unchanged", "content", "model", "trace"]


class StructuralHashMismatchError(ValidationError, AssertionError):
    """Raised when a model's provisional structural hash differs from its pin.

    Part of the typed taxonomy (catchable as ``tl.errors.ValidationError``)
    while keeping ``AssertionError`` in the MRO so historical
    ``except AssertionError`` handlers keep working. Structured context on
    ``fields``: ``code`` is always ``"structural_hash_mismatch"``, ``remedy``
    names the fix, and ``expected`` / ``actual`` carry both hashes.
    """


def content(value: Any) -> str:
    """Return a deterministic SHA-256 digest of nested tensor content.

    Tensor dtype, shape, and exact logical bytes are included. Lists, tuples,
    and mappings are framed recursively so distinct input structures cannot
    collide merely because their tensor bytes concatenate to the same stream.

    Parameters
    ----------
    value:
        Tensor or nested tensor-containing value to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = sha256()
    _update_content_digest(digest, value)
    return digest.hexdigest()


def _update_content_digest(digest: Any, value: Any) -> None:
    """Add one nested value to a framed content digest.

    Parameters
    ----------
    digest:
        Hashlib-compatible mutable digest.
    value:
        Value to frame and hash.
    """

    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            digest.update(b"tensor\0meta\0")
            digest.update(str(value.dtype).encode("utf-8") + b"\0")
            digest.update(repr(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(str(value.layout).encode("utf-8") + b"\0")
            return
        tensor = value.detach().cpu()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8") + b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.layout).encode("utf-8") + b"\0")
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        if tensor.is_quantized:
            _update_content_digest(digest, tensor.int_repr())
            digest.update(repr(tensor.qscheme()).encode("ascii") + b"\0")
            return
        # The shared transport byte view flattens to 1-D before the uint8
        # reinterpret (``view(torch.uint8)`` refuses a 0-dim scalar tensor,
        # which otherwise crashed ``content()`` and, via a bare-except in the
        # runnable-bundle path, silently dropped the manifest ``input_hash``
        # for scalar inputs), hashes through the buffer protocol (no
        # whole-payload ``bytes`` copy; digest bytes identical to the prior
        # ``.tobytes()`` spelling), and resolves lazy conj/neg bits first
        # (r8 R35: the hand-rolled view here crashed on ``x.conj()`` inputs).
        byte_view = digest_byte_view(tensor)
        digest.update(byte_view.nbytes.to_bytes(8, "big"))
        digest.update(byte_view)
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        framed_items = []
        for key, item_value in value.items():
            key_digest = sha256()
            value_digest = sha256()
            _update_content_digest(key_digest, key)
            _update_content_digest(value_digest, item_value)
            framed_items.append((key_digest.digest(), value_digest.digest()))
        framed_items.sort()
        digest.update(len(framed_items).to_bytes(8, "big"))
        for key_frame, value_frame in framed_items:
            digest.update(key_frame)
            digest.update(value_frame)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_content_digest(digest, item)
        return
    if isinstance(value, bytes | bytearray):
        raw_bytes = bytes(value)
        digest.update(b"bytes\0" + len(raw_bytes).to_bytes(8, "big") + raw_bytes)
        return
    if isinstance(value, Set):
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        item_frames = []
        for item in value:
            item_digest = sha256()
            _update_content_digest(item_digest, item)
            item_frames.append(item_digest.digest())
        item_frames.sort()
        digest.update(len(item_frames).to_bytes(8, "big"))
        for item_frame in item_frames:
            digest.update(item_frame)
        return
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        raw = contiguous.tobytes(order="C")
        digest.update(b"numpy.ndarray\0")
        digest.update(str(contiguous.dtype).encode("utf-8") + b"\0")
        digest.update(repr(tuple(contiguous.shape)).encode("ascii") + b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, np.generic):
        _update_content_digest(digest, value.item())
        return
    if value is None or isinstance(value, str | int | float | bool | complex):
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        digest.update(repr(value).encode("utf-8") + b"\0")
        return
    if isinstance(value, torch.dtype | torch.device):
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        digest.update(str(value).encode("utf-8") + b"\0")
        return
    digest.update(type(value).__module__.encode("utf-8") + b".")
    digest.update(type(value).__qualname__.encode("utf-8") + b"\0")
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        _update_content_digest(digest, attributes)
        return
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    raise ArgumentTypeError(
        f"torchlens.fingerprint.content cannot deterministically encode {value_type}",
        code="hash_content_type_unsupported",
        remedy=(
            "pass tensors, numpy arrays, builtin scalars/containers, or objects "
            "whose state is an inspectable __dict__"
        ),
        value_type=value_type,
    )


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
    torchlens.errors.ArgumentTypeError
        If ``expected`` is neither a string nor ``None`` (a ``TypeError``).
    """

    actual = _model_hash(model, example_input)
    if expected is None:
        # Bootstrap surface: disclose the fresh pin through the warning
        # machinery instead of an unconditional stdout write, so callers and
        # test harnesses can capture or silence it like any other diagnostic.
        warnings.warn(
            f"TorchLens structural hash: {actual}",
            TorchLensWarning,
            stacklevel=2,
        )
        return actual
    if not isinstance(expected, str):
        raise ArgumentTypeError(
            "expected must be a structural hash string or None",
            code="hash_expected_type_invalid",
            remedy="pass the pinned hash string, or None to bootstrap a pin",
        )
    if expected != actual:
        raise StructuralHashMismatchError(
            "TorchLens structural hash changed: "
            f"expected {expected}, got {actual}. "
            "Inspect the captured traces to find the structural divergence.",
            code="structural_hash_mismatch",
            remedy=(
                "inspect the captured traces to find the structural divergence, "
                "or re-pin the hash if the change is intentional"
            ),
            expected=expected,
            actual=actual,
        )
    return actual
