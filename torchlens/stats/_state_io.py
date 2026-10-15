"""Save and reload streaming-accumulator state (DOCUMENTED-UNSTABLE).

``tl.stats.save_fitted(stat, path)`` writes an accumulator's whole state
(its configuration, its folded tensors and its private random stream) to a
``tl_stats_state_v1`` safetensors file, and ``tl.stats.load_fitted(path)``
rebuilds an object that keeps folding as if it had never stopped: the same
later batches give the same values, example ids, positions and quantiles as
one uninterrupted run.

File layout: the tensors are safetensors entries; the metadata holds the
schema id, a JSON description of every attribute (``state``) and a
``sha256:`` digest over both. Loading recomputes the digest and refuses a
mismatch. Nothing is unpickled or imported from the file: only the
accumulator classes in ``_CLASSES`` can be rebuilt, and an attribute of any
other type refuses at save time.

An ``encoder=`` callable is not saved (it is code, often holding an SAE's
weights); the file records where one was set, and ``load_fitted`` needs the
same callable passed back as ``encoder=``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from .._io._json import loads_bounded
from ._fitted import FittedArtifactError

__tl_layer__ = "L5"

#: Schema id of the accumulator-state file (self-owned artifact, not a tlspec field family).
STATE_SCHEMA = "tl_stats_state_v1"
_SCHEMA_PREFIX = "tl_stats_state_v"
_DOCS = "docs/reference/stats.md, Save, reload and cache"

#: Tensor dtypes a state file may name (looked up, never ``getattr`` on ``torch``).
_DTYPES: dict[str, torch.dtype] = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}

#: Decoder method per stored node kind (scalars and objects are handled inline).
_READERS = {
    "tensor": "_read_tensor",
    "dtype": "_read_dtype",
    "device": "_read_device",
    "generator": "_read_generator",
    "pyrandom": "_read_pyrandom",
    "floats": "_read_floats",
    "list": "_read_sequence",
    "tuple": "_read_sequence",
    "dict": "_read_dict",
    "encoder": "_read_encoder",
}

# Attribute values longer than this that are all floats are stored as one
# float64 tensor (reservoirs, pooled top-k heaps) instead of JSON numbers.
_FLOAT_LIST_TENSOR_MIN = 16


def _classes() -> dict[str, type]:
    """Return the allow-list of classes a state file may rebuild, by name."""

    from ..observability._kernels import Histogram, HistogramDescriptor, Spine
    from ._aggregate import Aggregator
    from ._per_unit import Quantile, TopK, _UnitLayout
    from ._streaming import CKA, PCA, Covariance, CrossCovariance, Mean, Norm

    allowed: tuple[type, ...] = (
        Aggregator,
        CKA,
        Covariance,
        CrossCovariance,
        Histogram,
        HistogramDescriptor,
        Mean,
        Norm,
        PCA,
        Quantile,
        Spine,
        TopK,
        _UnitLayout,
    )
    return {cls.__name__: cls for cls in allowed}


def is_saveable_state(value: Any) -> bool:
    """Return whether ``value`` is an accumulator whose state this module saves."""

    return type(value) in _classes().values() and not dataclasses.is_dataclass(value)


class _Encoder:
    """Write side: turns attribute values into JSON nodes plus named tensors."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.encoder_paths: list[str] = []
        self._classes = _classes()

    def _tensor(self, tensor: torch.Tensor) -> str:
        """Store a tensor in the side table and return its key."""
        key = f"t{len(self.tensors)}"
        self.tensors[key] = tensor.detach().to("cpu").contiguous().clone()
        return key

    def node(self, value: Any, path: str) -> Any:
        """Encode one value found at attribute ``path``."""

        leaf = self._leaf(value)
        if leaf is not None:
            return leaf
        if isinstance(value, (list, tuple)):
            return self._sequence(value, path)
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return {
                "t": "dict",
                "items": [[key, self.node(item, f"{path}.{key}")] for key, item in value.items()],
            }
        return self._object(value, path)

    def _leaf(self, value: Any) -> dict[str, Any] | None:
        """Encode a scalar, tensor, dtype or device; ``None`` for anything else."""

        if value is None or isinstance(value, (bool, int, str)):
            return {"t": "v", "v": value}
        if isinstance(value, float):
            return {"t": "f", "v": value}
        if isinstance(value, torch.Tensor):
            return {"t": "tensor", "key": self._tensor(value), "device": str(value.device)}
        if isinstance(value, torch.dtype):
            return {"t": "dtype", "v": str(value).removeprefix("torch.")}
        if isinstance(value, torch.device):
            return {"t": "device", "v": str(value)}
        return self._stream(value)

    def _stream(self, value: Any) -> dict[str, Any] | None:
        """Encode a random stream (torch or Python); ``None`` for anything else."""

        if isinstance(value, torch.Generator):
            return {
                "t": "generator",
                "key": self._tensor(value.get_state()),
                "device": str(value.device),
            }
        if isinstance(value, random.Random):
            version, internal, gauss = value.getstate()
            key = self._tensor(torch.tensor(internal, dtype=torch.int64))
            return {"t": "pyrandom", "version": version, "key": key, "gauss": gauss}
        return None

    def _sequence(self, value: list[Any] | tuple[Any, ...], path: str) -> Any:
        """Encode a list or tuple; a long run of floats is stored as one tensor."""
        kind = "tuple" if isinstance(value, tuple) else "list"
        if len(value) >= _FLOAT_LIST_TENSOR_MIN and all(type(item) is float for item in value):
            key = self._tensor(torch.tensor(value, dtype=torch.float64))
            return {"t": "floats", "key": key, "kind": kind}
        return {
            "t": kind,
            "items": [self.node(item, f"{path}[{index}]") for index, item in enumerate(value)],
        }

    def _object(self, value: Any, path: str) -> Any:
        """Encode an allow-listed accumulator object attribute by attribute."""
        cls = type(value)
        if self._classes.get(cls.__name__) is not cls:
            raise FittedArtifactError(
                f"The accumulator state at {path!r} holds a {cls.__module__}.{cls.__qualname__}, "
                "which a tl_stats_state_v1 file cannot store (it keeps tensors, numbers, "
                "strings, random streams and tl.stats accumulators only). "
                f"See {_DOCS}.",
                code="stats_state_unsupported",
                remedy=(
                    "save a tl.stats accumulator as tl.stats.save_fitted(stat, path) "
                    "without extra attributes set on it"
                ),
                path=path,
                value_type=f"{cls.__module__}.{cls.__qualname__}",
            )
        if dataclasses.is_dataclass(value):
            fields = [
                [field.name, self.node(getattr(value, field.name), f"{path}.{field.name}")]
                for field in dataclasses.fields(value)
            ]
            return {"t": "dataclass", "cls": cls.__name__, "fields": fields}
        attrs: list[list[Any]] = []
        for name, item in vars(value).items():
            if name == "encoder" and item is not None and callable(item):
                self.encoder_paths.append(f"{path}.{name}")
                attrs.append([name, {"t": "encoder"}])
                continue
            attrs.append([name, self.node(item, f"{path}.{name}")])
        return {"t": "obj", "cls": cls.__name__, "attrs": attrs}


class _Decoder:
    """Read side: rebuilds values from JSON nodes and the file's tensors."""

    def __init__(
        self, tensors: Mapping[str, torch.Tensor], encoder: Callable[[Any], Any] | None
    ) -> None:
        self.tensors = tensors
        self.encoder = encoder
        self._classes = _classes()

    def _tensor(self, key: str) -> torch.Tensor:
        """Return the side-table tensor stored under ``key``."""
        return self.tensors[key]

    def node(self, node: Mapping[str, Any]) -> Any:
        """Decode one JSON node (an unknown kind raises ``KeyError``)."""

        kind = str(node["t"])
        if kind in {"v", "f"}:
            return node["v"]
        if kind in {"obj", "dataclass"}:
            return self._object(node)
        return getattr(self, _READERS[kind])(node)

    def _read_tensor(self, node: Mapping[str, Any]) -> torch.Tensor:
        """Rebuild a tensor on its saved device."""
        return self._tensor(node["key"]).to(node["device"])

    def _read_dtype(self, node: Mapping[str, Any]) -> torch.dtype:
        """Rebuild a dtype from its saved name."""
        return _DTYPES[str(node["v"])]

    def _read_device(self, node: Mapping[str, Any]) -> torch.device:
        """Rebuild a device from its saved name."""
        return torch.device(node["v"])

    def _read_generator(self, node: Mapping[str, Any]) -> torch.Generator:
        """Rebuild a torch random generator with its saved state."""
        generator = torch.Generator(device=node["device"])
        generator.set_state(self._tensor(node["key"]))
        return generator

    def _read_pyrandom(self, node: Mapping[str, Any]) -> random.Random:
        """Rebuild a Python random stream with its saved state."""
        stream = random.Random()
        internal = tuple(int(item) for item in self._tensor(node["key"]).tolist())
        stream.setstate((int(node["version"]), internal, node["gauss"]))
        return stream

    def _read_floats(self, node: Mapping[str, Any]) -> list[float] | tuple[float, ...]:
        """Rebuild a list or tuple of floats from its stored tensor."""
        items = [float(item) for item in self._tensor(node["key"]).tolist()]
        return tuple(items) if node["kind"] == "tuple" else items

    def _read_sequence(self, node: Mapping[str, Any]) -> list[Any] | tuple[Any, ...]:
        """Rebuild a list or tuple item by item."""
        items = [self.node(item) for item in node["items"]]
        return tuple(items) if node["t"] == "tuple" else items

    def _read_dict(self, node: Mapping[str, Any]) -> dict[str, Any]:
        """Rebuild a dict with string keys."""
        return {str(key): self.node(item) for key, item in node["items"]}

    def _read_encoder(self, node: Mapping[str, Any]) -> Callable[[Any], Any] | None:
        """Return the caller's encoder in place of a saved encoder placeholder."""
        return self.encoder

    def _object(self, node: Mapping[str, Any]) -> Any:
        """Rebuild an allow-listed accumulator object without calling its constructor."""
        cls = self._classes[str(node["cls"])]
        if node["t"] == "dataclass":
            return cls(**{name: self.node(item) for name, item in node["fields"]})
        attrs = {name: self.node(item) for name, item in node["attrs"]}
        obj: Any = object.__new__(cls)
        obj.__dict__.update(attrs)
        return obj


def state_digest(state_json: str, tensors: Mapping[str, torch.Tensor]) -> str:
    """Digest the state description and every tensor's dtype, shape and bytes."""

    hasher = hashlib.sha256()
    hasher.update(STATE_SCHEMA.encode("ascii"))
    hasher.update(state_json.encode("utf-8"))
    for key in sorted(tensors):
        tensor = tensors[key].detach().to("cpu").contiguous()
        hasher.update(repr((key, str(tensor.dtype), tuple(tensor.shape))).encode("utf-8"))
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + hasher.hexdigest()


def encode_state(root: Any) -> tuple[str, dict[str, torch.Tensor], list[str]]:
    """Encode one accumulator, or a ``{name: accumulator}`` bundle, to JSON plus tensors.

    Returns
    -------
    tuple[str, dict[str, torch.Tensor], list[str]]
        The canonical state JSON, the named tensors, and the attribute paths
        where an ``encoder=`` callable was set.
    """

    writer = _Encoder()
    if isinstance(root, Mapping):
        node: Any = {
            "t": "bundle",
            "items": [[str(name), writer.node(stat, str(name))] for name, stat in root.items()],
        }
    else:
        node = writer.node(root, type(root).__name__)
    node["encoders"] = writer.encoder_paths
    state_json = json.dumps(node, sort_keys=True, separators=(",", ":"))
    return state_json, writer.tensors, writer.encoder_paths


def write_state(root: Any, path: str | Path) -> Path:
    """Write ``root`` (an accumulator or a bundle) as a ``tl_stats_state_v1`` file.

    The file is written beside its destination and renamed into place, so a
    reader never sees a half-written file.
    """

    from safetensors.torch import save_file

    import torchlens

    state_json, tensors, _paths = encode_state(root)
    metadata = {
        "schema": STATE_SCHEMA,
        "state": state_json,
        "digest": state_digest(state_json, tensors),
        "torchlens": str(torchlens.__version__),
    }
    destination = Path(path)
    handle, scratch = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(handle)
    try:
        save_file(tensors, scratch, metadata=metadata)
        os.replace(scratch, destination)
    finally:
        if os.path.exists(scratch):
            os.unlink(scratch)
    return destination


def read_state(
    source: Path,
    metadata: Mapping[str, str],
    tensors: Mapping[str, torch.Tensor],
    *,
    encoder: Callable[[Any], Any] | None,
    require_encoder_match: bool = True,
) -> Any:
    """Rebuild the accumulator (or ``{name: accumulator}`` bundle) a state file holds.

    Raises
    ------
    FittedArtifactError
        ``stats_state_version_unknown`` for a newer or unknown format version,
        ``stats_state_digest_mismatch`` when the contents do not hash to the
        recorded digest, ``stats_state_encoder_mismatch`` when ``encoder=``
        is missing for a file saved with one (or given for a file saved
        without), and ``stats_state_invalid`` for a malformed file.
    """

    schema = str(metadata.get("schema", ""))
    if schema != STATE_SCHEMA:
        raise FittedArtifactError(
            f"The accumulator state file at {source} has format {schema!r}; this "
            f"torchlens reads {STATE_SCHEMA!r} only, so loading it could misread "
            f"the state. See {_DOCS}.",
            code="stats_state_version_unknown",
            remedy=(
                "load it with the torchlens version that wrote it "
                f"(it records {metadata.get('torchlens', 'unknown')!r}), or refold and "
                "re-save with tl.stats.save_fitted(stat, path)"
            ),
            path=str(source),
            schema=schema,
        )
    state_json = str(metadata.get("state", ""))
    recorded = str(metadata.get("digest", ""))
    recomputed = state_digest(state_json, tensors)
    if recomputed != recorded:
        raise FittedArtifactError(
            f"The accumulator state file at {source} records digest {recorded!r} but "
            f"its contents hash to {recomputed!r}; the file changed after it was "
            f"written, and resuming from it would fold onto unknown state. See {_DOCS}.",
            code="stats_state_digest_mismatch",
            remedy="re-save the state from the original accumulator with tl.stats.save_fitted",
            path=str(source),
            recorded=recorded,
            recomputed=recomputed,
        )
    try:
        node = loads_bounded(state_json)
        saved_encoders = [str(item) for item in node.get("encoders", [])]
        if require_encoder_match and bool(saved_encoders) != (encoder is not None):
            refuse_encoder_mismatch(source, saved_encoders, encoder)
        reader = _Decoder(tensors, encoder)
        if node.get("t") == "bundle":
            return {name: reader.node(item) for name, item in node["items"]}
        return reader.node(node)
    except FittedArtifactError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise FittedArtifactError(
            f"The accumulator state file at {source} has a valid digest but a state "
            f"description this torchlens cannot rebuild ({type(exc).__name__}: {exc}). "
            f"See {_DOCS}.",
            code="stats_state_invalid",
            remedy="re-save the state with tl.stats.save_fitted(stat, path) from this torchlens",
            path=str(source),
        ) from exc


def refuse_encoder_mismatch(
    source: Path, saved_encoders: list[str], encoder: Callable[[Any], Any] | None
) -> None:
    """Refuse a load whose ``encoder=`` does not match how the state was saved."""

    if saved_encoders:
        problem = (
            f"The accumulator state at {source} was saved with an encoder= callable "
            f"(at {', '.join(saved_encoders)}); callables are not stored, and folding "
            "on without it would mix raw activations into encoded state."
        )
        remedy = "pass the same callable back: tl.stats.load_fitted(path, encoder=sae.encode)"
    else:
        problem = (
            f"The accumulator state at {source} was saved without an encoder=, but "
            "load_fitted got one; folding encoded batches onto raw state mixes the two."
        )
        remedy = "drop encoder=: tl.stats.load_fitted(path)"
    raise FittedArtifactError(
        f"{problem} See {_DOCS}.",
        code="stats_state_encoder_mismatch",
        remedy=remedy,
        path=str(source),
        saved_encoders=saved_encoders,
    )


def can_merge_state(target: Any, loaded: Any) -> bool:
    """Return whether ``loaded`` has ``target``'s class and child structure."""

    from ._aggregate import Aggregator

    if type(target) is not type(loaded):
        return False
    if isinstance(target, Aggregator):
        return len(target.stats) == len(loaded.stats) and all(
            can_merge_state(child, other)
            for child, other in zip(target.stats, loaded.stats, strict=True)
        )
    return True


def merge_state_into(target: Any, loaded: Any) -> None:
    """Copy ``loaded``'s state onto ``target`` in place, keeping ``target``'s encoders.

    Callers check :func:`can_merge_state` first.
    """

    from ._aggregate import Aggregator

    if isinstance(target, Aggregator):
        for child, loaded_child in zip(target.stats, loaded.stats, strict=True):
            merge_state_into(child, loaded_child)
        target.name = loaded.name
        return
    for name, value in vars(loaded).items():
        if name != "encoder":
            setattr(target, name, value)
