"""Bounded portable previews of raw inputs and outputs under ``save_raw_input="small"``.

The save scrub (``scrub.py``) calls ``_small_raw_value``; the load side
(``rehydrate.py``) reads the image sentinel and caps. This module imports
nothing from the package.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import torch

_RAW_INPUT_TEXT_LIMIT = 10_000
_RAW_INPUT_TENSOR_BYTES_LIMIT = 1_000_000
_RAW_OUTPUT_TEXT_LIMIT = _RAW_INPUT_TEXT_LIMIT
_RAW_OUTPUT_TENSOR_BYTES_LIMIT = _RAW_INPUT_TENSOR_BYTES_LIMIT
_RAW_CONTAINER_ITEM_LIMIT = 20
_RAW_INPUT_IMAGE_MAX_EDGE = 256
_RAW_INPUT_IMAGE_BYTES_LIMIT = 256_000
_RAW_IMAGE_SENTINEL = "__torchlens_small_raw_image__"
# Records keep the save scrub's logger name, as before the move.
_LOGGER = logging.getLogger("torchlens._io.scrub")


def _small_raw_value(value: Any, *, field_name: str) -> Any:
    """Return a bounded portable representation of a raw value.

    Parameters
    ----------
    value:
        Raw user input or output metadata.
    field_name:
        Raw-value field being serialized.

    Returns
    -------
    Any
        Truncated string, small tensor, bounded image bytes, recursively
        bounded container, or ``None`` when the value is too large or
        unsupported. PIL images under ``save_raw_input="small"`` are copied,
        downsampled to at most 256 px on the longest edge, encoded as PNG or
        JPEG bytes, and dropped if the encoded payload exceeds 256 KB.
    """

    if value is None:
        return None
    is_output = field_name == "transformed_output"
    if isinstance(value, str):
        return value[: _RAW_OUTPUT_TEXT_LIMIT if is_output else _RAW_INPUT_TEXT_LIMIT]
    if isinstance(value, torch.Tensor):
        tensor_bytes = value.nelement() * value.element_size()
        tensor_limit = (
            _RAW_OUTPUT_TENSOR_BYTES_LIMIT if is_output else _RAW_INPUT_TENSOR_BYTES_LIMIT
        )
        if tensor_bytes <= tensor_limit:
            return value
        _LOGGER.debug(
            "Dropping %s tensor over small-policy cap: %s bytes", field_name, tensor_bytes
        )
        return None
    if field_name == "input":
        small_image = _small_raw_image_value(value)
        if small_image is not None:
            return small_image
    if isinstance(value, (list, tuple, dict)):
        return _small_raw_container(value, field_name=field_name)
    _LOGGER.debug(
        "Dropping unsupported %s type under small policy: %s",
        field_name,
        type(value).__name__,
    )
    return None


def _small_raw_container(
    value: list[Any] | tuple[Any, ...] | dict[Any, Any], *, field_name: str
) -> Any:
    """Keep a raw list, tuple or dict's first items, each bounded by ``_small_raw_value``.

    Parameters
    ----------
    value:
        Raw container value.
    field_name:
        Raw-value field being serialized.

    Returns
    -------
    Any
        A container of the same kind holding at most ``_RAW_CONTAINER_ITEM_LIMIT`` items.
    """

    if isinstance(value, list):
        return [
            _small_raw_value(item, field_name=field_name)
            for item in value[:_RAW_CONTAINER_ITEM_LIMIT]
        ]
    if isinstance(value, tuple):
        return tuple(
            _small_raw_value(item, field_name=field_name)
            for item in value[:_RAW_CONTAINER_ITEM_LIMIT]
        )
    return {
        key: _small_raw_value(item, field_name=field_name)
        for key, item in list(value.items())[:_RAW_CONTAINER_ITEM_LIMIT]
    }


def _small_raw_image_value(value: Any) -> dict[str, Any] | None:
    """Return bounded bytes for a PIL image under the small raw-input policy.

    Parameters
    ----------
    value:
        Candidate raw input value.

    Returns
    -------
    dict[str, Any] | None
        Sentinel dictionary containing encoded image bytes and metadata, or
        ``None`` when ``value`` is not a PIL image or cannot fit the cap.
    """

    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return None
    if not isinstance(value, PILImage):
        return None

    image = value.copy()
    original_size = tuple(image.size)
    image.thumbnail((_RAW_INPUT_IMAGE_MAX_EDGE, _RAW_INPUT_IMAGE_MAX_EDGE))
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    encoded = _encode_small_raw_image(image, fmt="PNG")
    image_format = "PNG"
    if encoded is None and image.mode != "RGBA":
        encoded = _encode_small_raw_image(image.convert("RGB"), fmt="JPEG")
        image_format = "JPEG"
    if encoded is None:
        _LOGGER.debug("Dropping raw_input PIL image over small-policy cap.")
        return None
    return {
        _RAW_IMAGE_SENTINEL: True,
        "format": image_format,
        "mode": image.mode,
        "size": tuple(image.size),
        "original_size": original_size,
        "max_edge": _RAW_INPUT_IMAGE_MAX_EDGE,
        "bytes_limit": _RAW_INPUT_IMAGE_BYTES_LIMIT,
        "data": encoded,
    }


def _encode_small_raw_image(image: Any, *, fmt: str) -> bytes | None:
    """Encode a PIL image and enforce the small-policy byte cap.

    Parameters
    ----------
    image:
        PIL image to encode.
    fmt:
        Pillow format name.

    Returns
    -------
    bytes | None
        Encoded bytes when they fit the cap, otherwise ``None``.
    """

    buffer = BytesIO()
    save_kwargs: dict[str, Any] = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs.update({"quality": 85, "optimize": True})
    image.save(buffer, **save_kwargs)
    data = buffer.getvalue()
    if len(data) > _RAW_INPUT_IMAGE_BYTES_LIMIT:
        return None
    return data
