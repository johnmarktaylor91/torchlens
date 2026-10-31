"""Safetensors export of a Trace (spellings DOCUMENTED-UNSTABLE).

``tl.export.safetensors(trace, path)`` writes ONE ``.safetensors`` file: every
tensor-valued op payload (``out``, ``transformed_out``, ``grad``,
``transformed_grad``) as a tensor entry, and the metadata as one JSON
document in the safetensors header. ``tl.io.read_safetensors(path)`` reads it
back. The file holds no pickle: safetensors stores raw tensor bytes plus a
string header, so reading it never executes code.

Source metadata passes through ``Trace.save``'s own source-privacy policy (at
its default: paths reduced to basenames), so no host path reaches the file.
Metadata follows the writer contract of record (``writer_contract()``): for
the ``trace`` record and every ``op`` record, each KEEP-policy field whose
value has an exact JSON encoding is written, and every KEEP field without
one is listed in the ``omitted`` disclosure with its value type, never
silently dropped. The encoding is tagged so values decode to the same
Python values (tuples stay tuples, ``torch.dtype`` stays a dtype). Complex
tensors are stored as their real view and restored bit-for-bit.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import torch

from .._errors import InvalidArgumentError
from .._utils._torch_symbols import torch_attr

__tl_layer__ = "L5"

__all__ = [
    "TRACE_SAFETENSORS_SCHEMA",
    "SafetensorsTrace",
    "read_trace_safetensors",
    "write_trace_safetensors",
]

#: Schema id of the JSON document in the safetensors header.
TRACE_SAFETENSORS_SCHEMA = "tl_trace_safetensors_v1"

#: The safetensors ``__metadata__`` key holding that document.
_METADATA_KEY = "torchlens"

#: Op record fields whose tensor values are stored as tensor entries.
_PAYLOAD_FIELDS = ("out", "transformed_out", "grad", "transformed_grad")

#: Nesting bound of the tagged encoding (deeper values are disclosed omitted).
_MAX_DEPTH = 32

_UNENCODABLE = object()


@dataclasses.dataclass(frozen=True)
class SafetensorsTrace:
    """A Trace read back from its safetensors export (read-only record).

    Attributes
    ----------
    header:
        Schema id, tlspec version, writer-contract digest, TorchLens version,
        and the capture-honesty facts at export.
    trace_fields:
        Exported ``trace`` record fields, decoded.
    op_fields:
        ``op label -> {field: value}`` for every op, decoded.
    tensors:
        ``op label -> {payload field: tensor}`` for every stored payload.
    omitted:
        ``{"trace": {field: type}, "op": {label: {field: type}}}``: KEEP
        fields whose values have no exact JSON encoding.
    """

    header: dict[str, Any]
    trace_fields: dict[str, Any]
    op_fields: dict[str, dict[str, Any]]
    tensors: dict[str, dict[str, torch.Tensor]]
    omitted: dict[str, Any]


def _encode(value: Any, depth: int = 0) -> Any:
    """Encode one value as tagged JSON, or return the unencodable sentinel.

    Dispatch is on the EXACT type: a subclass (an enum over ``str``, a
    namedtuple) is not silently flattened to its base, it is disclosed.

    Parameters
    ----------
    value:
        Field value.
    depth:
        Current nesting depth.

    Returns
    -------
    Any
        JSON-native tagged form, or ``_UNENCODABLE``.
    """

    encoder = _ENCODERS.get(type(value))
    if encoder is None or depth > _MAX_DEPTH:
        return _UNENCODABLE
    return encoder(value, depth)


def _encode_float(value: float, depth: int) -> Any:
    """Encode a float; non-finite values take the ``$float`` tag."""

    return value if math.isfinite(value) else {"$float": repr(value)}


def _encode_items(values: Any, depth: int) -> list[Any] | None:
    """Encode a sequence's items, or ``None`` when any item is unencodable."""

    items = [_encode(item, depth + 1) for item in values]
    return None if any(item is _UNENCODABLE for item in items) else items


def _encode_sequence(value: Any, depth: int) -> Any:
    """Encode a list natively and a tuple / set / frozenset under its tag."""

    items = _encode_items(value, depth)
    if items is None:
        return _UNENCODABLE
    if type(value) is list:
        return items
    if type(value) is tuple:
        return {"$tuple": items}
    ordered = sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    return {"$set" if type(value) is set else "$frozenset": ordered}


def _encode_dict(value: dict[Any, Any], depth: int) -> Any:
    """Encode a dict: plain object for safe string keys, else a pair list.

    Parameters
    ----------
    value:
        The dict.
    depth:
        Current nesting depth.

    Returns
    -------
    Any
        JSON-native tagged form, or ``_UNENCODABLE``.
    """

    plain = all(isinstance(key, str) and not key.startswith("$") for key in value)
    pairs = [(_encode(key, depth + 1), _encode(item, depth + 1)) for key, item in value.items()]
    if any(key is _UNENCODABLE or item is _UNENCODABLE for key, item in pairs):
        return _UNENCODABLE
    if plain:
        return dict(pairs)
    return {"$map": [[key, item] for key, item in pairs]}


#: Exact-type encoders of the tagged JSON form.
_ENCODERS: Mapping[type, Any] = MappingProxyType(
    {
        type(None): lambda value, depth: value,
        bool: lambda value, depth: value,
        int: lambda value, depth: value,
        str: lambda value, depth: value,
        float: _encode_float,
        torch.dtype: lambda value, depth: {"$dtype": str(value).removeprefix("torch.")},
        torch.Size: lambda value, depth: {"$size": [int(dim) for dim in value]},
        torch.device: lambda value, depth: {"$device": str(value)},
        list: _encode_sequence,
        tuple: _encode_sequence,
        set: _encode_sequence,
        frozenset: _encode_sequence,
        dict: _encode_dict,
    }
)


def _refuse_invalid(problem: str, path: Path, cause: BaseException | None = None) -> NoReturn:
    """Refuse a file that is not a valid TorchLens safetensors Trace export.

    Parameters
    ----------
    problem:
        What is wrong.
    path:
        The file read.
    cause:
        The underlying parse error, chained when given.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``trace_safetensors_invalid``.
    """

    raise InvalidArgumentError(
        f"{str(path)!r} is not a valid TorchLens safetensors Trace export: {problem}.",
        code="trace_safetensors_invalid",
        remedy="read files written by tl.export.safetensors(trace, path)",
        path=str(path),
    ) from cause


def _decode_tag(tag: str, payload: Any, path: Path) -> Any:
    """Decode one tagged value.

    Parameters
    ----------
    tag:
        The ``$`` tag.
    payload:
        Its JSON payload.
    path:
        The file read (refusal text).

    Returns
    -------
    Any
        The decoded Python value.
    """

    if tag == "$float" and payload in ("nan", "inf", "-inf"):
        return float(payload)
    if tag == "$dtype":
        dtype = torch_attr(str(payload))
        if isinstance(dtype, torch.dtype):
            return dtype
    if tag == "$size" and isinstance(payload, list):
        return torch.Size(int(dim) for dim in payload)
    if tag == "$device" and isinstance(payload, str):
        return torch.device(payload)
    if tag in ("$tuple", "$set", "$frozenset") and isinstance(payload, list):
        items = [_decode(item, path) for item in payload]
        return {"$tuple": tuple, "$set": set, "$frozenset": frozenset}[tag](items)
    if tag == "$map" and isinstance(payload, list):
        return {_decode(key, path): _decode(item, path) for key, item in payload}
    _refuse_invalid(f"unknown or malformed value tag {tag!r}", path)


def _decode(value: Any, path: Path) -> Any:
    """Decode one tagged-JSON value.

    Parameters
    ----------
    value:
        JSON-native tagged form.
    path:
        The file read (refusal text).

    Returns
    -------
    Any
        The decoded Python value.
    """

    if isinstance(value, list):
        return [_decode(item, path) for item in value]
    if isinstance(value, dict):
        if len(value) == 1:
            ((key, payload),) = value.items()
            if key.startswith("$"):
                return _decode_tag(key, payload, path)
        return {key: _decode(item, path) for key, item in value.items()}
    return value


def _keep_fields(contract: dict[str, Any], record: str) -> list[str]:
    """Return one record's KEEP-policy field names from the writer contract.

    Parameters
    ----------
    contract:
        The live writer contract.
    record:
        Record type (``"trace"`` / ``"op"``).

    Returns
    -------
    list[str]
        Sorted KEEP field names.
    """

    policies = contract["record_contracts"][record]["field_policies"]
    return sorted(name for name, policy in policies.items() if policy == "keep")


def _encode_record(
    state: dict[str, Any], fields: list[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Encode one record's KEEP fields; disclose the unencodable ones.

    Parameters
    ----------
    state:
        The record's live state fields.
    fields:
        KEEP field names.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        Encoded fields, and ``field -> value type`` for omitted ones.
    """

    encoded: dict[str, Any] = {}
    omitted: dict[str, str] = {}
    for name in fields:
        if name not in state:
            continue
        value = _encode(state[name])
        if value is _UNENCODABLE:
            omitted[name] = type(state[name]).__name__
        else:
            encoded[name] = value
    return encoded, omitted


def _payload_entry(tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Return the storable form of one payload tensor and its encoding.

    Parameters
    ----------
    tensor:
        The payload.

    Returns
    -------
    tuple[torch.Tensor, str]
        A dense CPU contiguous copy (complex as its real view) and
        ``"raw"`` / ``"view_as_real"``.
    """

    copy = tensor.detach().to("cpu").contiguous().clone()
    if copy.is_complex():
        return torch.view_as_real(copy).contiguous(), "view_as_real"
    return copy, "raw"


def _storable(tensor: Any) -> bool:
    """Return whether a payload is a dense, materialized tensor.

    Parameters
    ----------
    tensor:
        Candidate payload.

    Returns
    -------
    bool
        ``True`` for strided, non-meta, non-quantized tensors.
    """

    return (
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and not tensor.is_meta
        and not tensor.is_quantized
    )


def _portable_trace_state(trace: Any) -> dict[str, Any]:
    """Return the Trace's state under ``Trace.save``'s source-privacy policy.

    The same policy bodies a portable save applies (at its default,
    ``include_source=True``): source-file paths reduce to bare basenames,
    signature defaults lose absolute-path literals, capture advisories lose
    their absolute locations, and the source-code blob is re-made
    path-free. The live Trace is never mutated (``annotations`` is copied
    before the advisory policy rebuilds its rows).

    Parameters
    ----------
    trace:
        The Trace being exported.

    Returns
    -------
    dict[str, Any]
        A shallow state copy with the privacy policy applied.
    """

    from .._data_classes._state_adapter import state_items
    from .._io._source_privacy import (
        _apply_advisory_source_policy,
        _apply_source_metadata_policy,
        _apply_trace_blob_policy,
    )
    from .._io.scrub import _ScrubOptions

    state = dict(state_items(trace))
    if isinstance(state.get("annotations"), dict):
        state["annotations"] = dict(state["annotations"])
    options = _ScrubOptions(
        include_outs=False, include_grads=False, include_saved_args=False, include_rng_states=False
    )
    _apply_advisory_source_policy(state, options)
    _apply_source_metadata_policy(state, options)
    _apply_trace_blob_policy(state, options)
    return state


def write_trace_safetensors(trace: Any, path: str | Path) -> Path:
    """Export one Trace to a single safetensors file (no pickle).

    Parameters
    ----------
    trace:
        A :class:`torchlens.Trace`.
    path:
        Destination file (written atomically).

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``trace_safetensors_trace_required`` for a non-Trace argument.
    """

    from safetensors.torch import save_file

    from .. import __version__
    from .._capture_honesty import capture_honesty_facts
    from .._data_classes._state_adapter import state_items
    from .._data_classes.trace import Trace
    from .._io import TLSPEC_VERSION
    from .._io.writer_contract import writer_contract, writer_contract_digest

    if not isinstance(trace, Trace):
        raise InvalidArgumentError(
            f"tl.export.safetensors exports one Trace; got {type(trace).__name__}.",
            code="trace_safetensors_trace_required",
            remedy="pass a Trace (export each Bundle member separately)",
            value_type=type(trace).__name__,
        )
    contract = writer_contract()
    trace_fields, trace_omitted = _encode_record(
        _portable_trace_state(trace), _keep_fields(contract, "trace")
    )
    op_keep = _keep_fields(contract, "op")
    ops: dict[str, Any] = {}
    op_omitted: dict[str, dict[str, str]] = {}
    tensors: dict[str, torch.Tensor] = {}
    entries: dict[str, dict[str, Any]] = {}
    for label in trace.op_labels:
        state = dict(state_items(trace.ops[label]))
        ops[label], omitted = _encode_record(state, op_keep)
        for slot in _PAYLOAD_FIELDS:
            value = state.get(slot)
            if value is None:
                continue
            if not _storable(value):
                omitted[slot] = type(value).__name__
                continue
            name = f"t{len(entries)}"
            tensors[name], encoding = _payload_entry(value)
            entries[name] = {"op": label, "field": slot, "encoding": encoding}
        if omitted:
            op_omitted[label] = omitted
    document = {
        "schema": TRACE_SAFETENSORS_SCHEMA,
        "tlspec_version": TLSPEC_VERSION,
        "writer_contract_digest": writer_contract_digest(contract),
        "torchlens_version": __version__,
        "records": ["trace", "op"],
        "source_policy": "Trace.save default: source text kept, source paths reduced to basenames",
        "capture_honesty": capture_honesty_facts(trace),
        "op_labels": list(trace.op_labels),
        "trace": trace_fields,
        "ops": ops,
        "tensors": entries,
        "omitted": {"trace": trace_omitted, "op": op_omitted},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        save_file(
            tensors,
            str(temporary),
            metadata={_METADATA_KEY: json.dumps(document, sort_keys=True, separators=(",", ":"))},
        )
        os.replace(temporary, target)
    finally:
        # A failed write leaves no partial file behind (the replace already
        # consumed the temporary on success).
        temporary.unlink(missing_ok=True)
    return target


def _require_honourable(document: dict[str, Any], path: Path) -> None:
    """Refuse a document this reader's contract cannot honour.

    The reader decodes every field generically, so any writer contract whose
    ``tlspec_version`` lies in this build's readable window is honoured (the
    field names it carries are that contract's KEEP fields; the header says
    whether the digest matches this build's contract). A newer version, or a
    digest that is not a digest, refuses.

    Parameters
    ----------
    document:
        The parsed header document.
    path:
        The file read (refusal text).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``trace_safetensors_version_unsupported`` outside the readable
        window; ``trace_safetensors_invalid`` for a malformed digest.
    """

    from .._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION

    version = document.get("tlspec_version")
    if type(version) is not int or not MIN_TLSPEC_VERSION <= version <= TLSPEC_VERSION:
        raise InvalidArgumentError(
            f"{str(path)!r} was written under tlspec version {version!r}; this "
            f"build reads versions {MIN_TLSPEC_VERSION} to {TLSPEC_VERSION}.",
            code="trace_safetensors_version_unsupported",
            remedy="read it with the TorchLens release that wrote it, or a newer one",
            path=str(path),
            tlspec_version=repr(version),
        )
    digest = document.get("writer_contract_digest")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    ):
        _refuse_invalid("its writer_contract_digest is not a SHA-256 hex digest", path)


def read_trace_safetensors(path: str | Path) -> SafetensorsTrace:
    """Read a Trace safetensors export back (no code execution).

    Parameters
    ----------
    path:
        File written by :func:`write_trace_safetensors`.

    Returns
    -------
    SafetensorsTrace
        Decoded header, fields, payload tensors, and the omitted disclosure.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``trace_safetensors_invalid`` when the file is not a valid export.
    """

    from safetensors import safe_open

    from .._io._json import loads_bounded
    from .._io.writer_contract import writer_contract_digest

    source = Path(path)
    with safe_open(str(source), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if _METADATA_KEY not in metadata:
            _refuse_invalid("its header carries no TorchLens document", source)
        try:
            document = loads_bounded(metadata[_METADATA_KEY])
        except json.JSONDecodeError as exc:
            _refuse_invalid(f"its header document is not valid JSON ({exc.msg})", source, exc)
        if not isinstance(document, dict) or document.get("schema") != TRACE_SAFETENSORS_SCHEMA:
            _refuse_invalid(f"its header schema is not {TRACE_SAFETENSORS_SCHEMA!r}", source)
        _require_honourable(document, source)
        tensors: dict[str, dict[str, torch.Tensor]] = {}
        for name, entry in (document.get("tensors") or {}).items():
            tensor = handle.get_tensor(name)
            if entry.get("encoding") == "view_as_real":
                tensor = torch.view_as_complex(tensor)
            tensors.setdefault(str(entry["op"]), {})[str(entry["field"])] = tensor
    header_keys = ("schema", "tlspec_version", "writer_contract_digest", "torchlens_version")
    header = {key: document.get(key) for key in header_keys}
    header["capture_honesty"] = document.get("capture_honesty")
    header["writer_contract_current"] = document["writer_contract_digest"] == (
        writer_contract_digest()
    )
    return SafetensorsTrace(
        header=header,
        trace_fields=_decode(document.get("trace") or {}, source),
        op_fields={
            str(label): _decode(fields, source)
            for label, fields in (document.get("ops") or {}).items()
        },
        tensors=tensors,
        omitted=document.get("omitted") or {},
    )
