"""Canonical modality-aware standard inputs with honest random fallback."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np

from menagerie.crawler.identity import hash_bytes


ASSET_ROOT = Path(__file__).with_name("assets") / "standard"


class StandardInputError(ValueError):
    """Raised when an input specification cannot produce any valid tensor."""


@dataclass(frozen=True)
class InputSpec:
    """Concrete single-tensor input specification.

    Parameters
    ----------
    shape:
        Positive concrete tensor dimensions.
    dtype:
        Framework-neutral dtype token.
    """

    shape: tuple[int, ...]
    dtype: str = "float32"

    @classmethod
    def from_value(cls, value: Union["InputSpec", Mapping[str, Any]]) -> "InputSpec":
        """Normalize a typed input specification.

        Parameters
        ----------
        value:
            Existing spec or mapping with shape and dtype.

        Returns
        -------
        InputSpec
            Concrete normalized specification.
        """

        if isinstance(value, InputSpec):
            return value
        shape_value = value.get("shape")
        if not isinstance(shape_value, (list, tuple)) or not shape_value:
            raise StandardInputError("input spec shape must be a non-empty integer sequence")
        try:
            shape = tuple(int(dimension) for dimension in shape_value)
        except (TypeError, ValueError) as exc:
            raise StandardInputError("input spec shape must contain concrete integers") from exc
        if any(dimension <= 0 for dimension in shape):
            raise StandardInputError("input spec dimensions must be positive")
        dtype = value.get("dtype", "float32")
        if not isinstance(dtype, str) or not dtype:
            raise StandardInputError("input spec dtype must be a non-empty string")
        return cls(shape=shape, dtype=dtype.lower().replace("torch.", ""))


@dataclass(frozen=True)
class MaterializedInput:
    """Tensor plus mechanically observed input provenance.

    Parameters
    ----------
    value:
        Framework-native tensor.
    input_kind:
        Canonical asset kind or ``random-fallback``.
    input_asset:
        Content-addressed asset identifier, or ``None`` for random fallback.
    input_note:
        Brief mechanical shaping/fallback note.
    spec:
        Concrete fulfilled input specification.
    """

    value: object
    input_kind: str
    input_asset: str | None
    input_note: str
    spec: InputSpec

    @property
    def args(self) -> tuple[object, ...]:
        """Return this single tensor as positional forward arguments.

        Returns
        -------
        tuple[object, ...]
            One positional input.
        """

        return (self.value,)

    @property
    def kwargs(self) -> dict[str, object]:
        """Return empty keyword arguments for the standard call.

        Returns
        -------
        dict[str, object]
            Empty keyword mapping.
        """

        return {}


def _asset_id(path: Path) -> str:
    """Return a content-addressed standard asset identifier.

    Parameters
    ----------
    path:
        Bundled asset path.

    Returns
    -------
    str
        Asset name and SHA-256 digest.
    """

    return f"standard:{path.name}:{hash_bytes(path.read_bytes())}"


def _load_ppm(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Load the bundled ASCII PPM image without an imaging dependency.

    Parameters
    ----------
    path:
        P3 PPM asset.

    Returns
    -------
    numpy.ndarray
        Height-by-width-by-three uint8 image.
    """

    tokens: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        content = line.partition("#")[0]
        tokens.extend(content.split())
    if len(tokens) < 4 or tokens[0] != "P3":
        raise StandardInputError(f"invalid standard PPM asset: {path}")
    width, height, maximum = (int(tokens[index]) for index in (1, 2, 3))
    if maximum != 255:
        raise StandardInputError("standard PPM must use an 8-bit maximum")
    pixels = np.asarray([int(token) for token in tokens[4:]], dtype=np.uint8)
    if pixels.size != width * height * 3:
        raise StandardInputError("standard PPM pixel count is inconsistent")
    return pixels.reshape(height, width, 3)


def _resize_image(image: np.ndarray[Any, Any], height: int, width: int) -> np.ndarray[Any, Any]:
    """Resize an image deterministically with nearest-neighbor sampling.

    Parameters
    ----------
    image:
        Source HWC image.
    height, width:
        Requested dimensions.

    Returns
    -------
    numpy.ndarray
        Resized HWC image.
    """

    rows = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    columns = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[rows][:, columns]


def _image_array(spec: InputSpec, *, video: bool = False) -> np.ndarray[Any, Any]:
    """Shape the canonical image for an NCHW/NHWC vision or video spec.

    Parameters
    ----------
    spec:
        Requested tensor specification.
    video:
        Whether a five-dimensional video layout is allowed.

    Returns
    -------
    numpy.ndarray
        Standard image/video array.
    """

    shape = spec.shape
    if video and len(shape) == 5:
        if shape[1] in {1, 3, 4}:
            batch, channels, frames, height, width = shape
            frame_spec = InputSpec((batch, channels, height, width), spec.dtype)
            frame = _image_array(frame_spec)
            return np.repeat(frame[:, :, None, :, :], frames, axis=2)
        if shape[-1] in {1, 3, 4}:
            batch, frames, height, width, channels = shape
            frame_spec = InputSpec((batch, height, width, channels), spec.dtype)
            frame = _image_array(frame_spec)
            return np.repeat(frame[:, None, :, :, :], frames, axis=1)
        raise StandardInputError("video shape has no recognizable channel axis")
    if len(shape) == 4 and shape[1] in {1, 3, 4}:
        batch, channels, height, width = shape
        channel_first = True
    elif len(shape) == 4 and shape[-1] in {1, 3, 4}:
        batch, height, width, channels = shape
        channel_first = False
    elif len(shape) == 3 and shape[0] in {1, 3, 4}:
        channels, height, width = shape
        batch = 0
        channel_first = True
    elif len(shape) == 3 and shape[-1] in {1, 3, 4}:
        height, width, channels = shape
        batch = 0
        channel_first = False
    else:
        raise StandardInputError("vision shape has no recognizable image layout")
    image = _resize_image(_load_ppm(ASSET_ROOT / "image.ppm"), height, width)
    if channels == 1:
        image = image.astype(np.float32).mean(axis=2, keepdims=True)
    elif channels == 4:
        alpha = np.full((height, width, 1), 255, dtype=image.dtype)
        image = np.concatenate((image, alpha), axis=2)
    if channel_first:
        image = np.transpose(image, (2, 0, 1))
    if batch:
        image = np.repeat(image[None, ...], batch, axis=0)
    return image


def _text_array(spec: InputSpec) -> np.ndarray[Any, Any]:
    """Tokenize the fixed standard text into the requested shape.

    Parameters
    ----------
    spec:
        Requested tensor specification.

    Returns
    -------
    numpy.ndarray
        Deterministic UTF-8-byte token array.
    """

    text = (ASSET_ROOT / "text.txt").read_text(encoding="utf-8").strip()
    tokens = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64) + 1
    return np.resize(tokens, spec.shape)


def _csv_array(path: Path, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Resize numeric CSV asset values to a concrete tensor shape.

    Parameters
    ----------
    path:
        Bundled numeric CSV.
    shape:
        Requested tensor shape.

    Returns
    -------
    numpy.ndarray
        Resized floating array.
    """

    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(value) for row in csv.reader(handle) for value in row]
    return np.resize(np.asarray(values, dtype=np.float32), shape)


def _random_array(spec: InputSpec, seed: int) -> np.ndarray[Any, Any]:
    """Generate a deterministic random fallback array.

    Parameters
    ----------
    spec:
        Requested tensor specification.
    seed:
        RNG seed.

    Returns
    -------
    numpy.ndarray
        Random array before framework conversion.
    """

    rng = np.random.default_rng(seed)
    dtype = spec.dtype
    if dtype in {"bool"}:
        return rng.integers(0, 2, size=spec.shape, dtype=np.int8).astype(np.bool_)
    if dtype.startswith("int") or dtype in {"long", "uint8"}:
        return rng.integers(0, 16, size=spec.shape, dtype=np.int64)
    return rng.standard_normal(spec.shape).astype(np.float32)


def _convert_array(
    array: np.ndarray[Any, Any], spec: InputSpec, framework: str, device: str
) -> object:
    """Convert a NumPy array into the requested native framework tensor.

    Parameters
    ----------
    array:
        Source array.
    spec:
        Requested dtype.
    framework:
        Native execution framework.
    device:
        Native device string.

    Returns
    -------
    object
        Framework-native tensor.
    """

    normalized = framework.lower()
    if spec.dtype.startswith("float") or spec.dtype in {"bfloat16", "half", "double"}:
        array = array.astype(np.float32) / (255.0 if array.max(initial=0) > 1.0 else 1.0)
    if normalized in {"torch", "pytorch"}:
        import torch

        dtype_map = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "double": torch.float64,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "long": torch.int64,
            "int32": torch.int32,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        try:
            dtype = dtype_map[spec.dtype]
        except KeyError as exc:
            raise StandardInputError(f"unsupported PyTorch dtype: {spec.dtype!r}") from exc
        return torch.as_tensor(array, dtype=dtype, device=device)
    if normalized in {"numpy", "np"}:
        return array
    if normalized in {"tensorflow", "tf"}:
        import tensorflow as tf

        return tf.convert_to_tensor(array, dtype=getattr(tf, spec.dtype))
    if normalized in {"jax", "flax"}:
        import jax.numpy as jnp

        return jnp.asarray(array, dtype=getattr(jnp, spec.dtype))
    if normalized in {"paddle", "paddlepaddle"}:
        import paddle

        return paddle.to_tensor(array, dtype=spec.dtype, place=device)
    raise StandardInputError(f"unsupported input framework: {framework!r}")


def _normalized_modalities(modality: Union[str, Sequence[str], None]) -> tuple[str, ...]:
    """Normalize source-gated modality values for matching.

    Parameters
    ----------
    modality:
        One modality, several modalities, or null.

    Returns
    -------
    tuple[str, ...]
        Lowercase modality tokens.
    """

    if modality is None:
        return ()
    values = (modality,) if isinstance(modality, str) else tuple(modality)
    return tuple(str(value).strip().lower() for value in values)


def materialize_standard_input(
    modality: Union[str, Sequence[str], None],
    input_spec: Union[InputSpec, Mapping[str, Any]],
    *,
    framework: str = "pytorch",
    device: str = "cpu",
    seed: int = 0,
) -> MaterializedInput:
    """Build a canonical modality input or an honest random fallback.

    Parameters
    ----------
    modality:
        Source-gated ``external_metadata.modality`` value.
    input_spec:
        Concrete shape and dtype contract.
    framework, device:
        Native tensor target.
    seed:
        Deterministic fallback seed.

    Returns
    -------
    MaterializedInput
        Tensor plus ``input_kind``, asset hash, and shaping/fallback note.
    """

    spec = InputSpec.from_value(input_spec)
    modalities = _normalized_modalities(modality)
    selected: tuple[str, Path] | None = None
    try:
        if any(value in {"vision", "image", "computer-vision"} for value in modalities):
            asset = ASSET_ROOT / "image.ppm"
            array = _image_array(spec)
            selected = ("standard-image", asset)
        elif any(value in {"language", "text", "nlp"} for value in modalities):
            asset = ASSET_ROOT / "text.txt"
            array = _text_array(spec)
            selected = ("standard-text", asset)
        elif any(value in {"audio", "speech"} for value in modalities):
            asset = ASSET_ROOT / "audio.csv"
            array = _csv_array(asset, spec.shape)
            selected = ("standard-audio", asset)
        elif "video" in modalities:
            asset = ASSET_ROOT / "image.ppm"
            array = _image_array(spec, video=True)
            selected = ("standard-video", asset)
        elif any(value in {"tabular", "recsys"} for value in modalities):
            asset = ASSET_ROOT / "tabular.csv"
            array = _csv_array(asset, spec.shape)
            selected = ("standard-tabular", asset)
        else:
            raise StandardInputError("no canonical asset for the declared modality")
        if array.shape != spec.shape:
            raise StandardInputError(
                f"standard asset produced {array.shape}, expected {spec.shape}"
            )
        value = _convert_array(array, spec, framework, device)
        kind, asset = selected
        return MaterializedInput(
            value=value,
            input_kind=kind,
            input_asset=_asset_id(asset),
            input_note=f"canonical {asset.name} shaped to {list(spec.shape)}",
            spec=spec,
        )
    except StandardInputError as exc:
        fallback = _convert_array(_random_array(spec, seed), spec, framework, device)
        return MaterializedInput(
            value=fallback,
            input_kind="random-fallback",
            input_asset=None,
            input_note=f"random fallback: {exc}",
            spec=spec,
        )
