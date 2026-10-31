"""The embedded-tensor route (``torch.storage._load_from_bytes``) is refused by name.

Torch's default tensor reduction carries its storage as
``_load_from_bytes(<torch.save bytes>)``: a nested ``torch.load`` over bytes taken from
the file. The restricted metadata unpickler refuses that global before anything resolves,
so no byte of an untrusted file reaches torch's unpickler, on any torch version. Plain
metadata tensors (annotation values, small raw inputs/outputs) travel through the
TorchLens-owned inline reduction instead (``torchlens/_io/_inline_tensor.py``), which
validates every argument and allocates exactly the bytes present in the stream into a
non-resizable storage.

Three hostile shapes are driven here: the bare reduction at the stream root, the same
reduction nested inside a container the loader would otherwise accept, and the
reduction planted in the metadata of an otherwise valid bundle (manifest and blobs
genuine). Each must be refused with the loader's typed error naming the construct, and
``torch.load`` must never be entered.
"""

from __future__ import annotations

import io
import pickle
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io._canonical_pickle import dump_canonical_metadata
from torchlens._io._inline_tensor import INLINE_TENSOR_GLOBAL, rebuild_inline_tensor
from torchlens._io._safe_unpickle import SafeBundleUnpickler
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.options import CaptureOptions

_UNPICKLERS = [SafeBundleUnpickler, _RenameAwareUnpickler]
_REFUSAL = r"torch\.storage\._load_from_bytes"


@pytest.fixture
def torch_load_tripwire(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fail the test if anything calls ``torch.load`` while a payload is being refused."""

    calls: list[Any] = []

    def _trip(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("torch.load was entered with file bytes")

    monkeypatch.setattr(torch, "load", _trip)
    return calls


def _glob(module: str, name: str) -> bytes:
    return pickle.GLOBAL + module.encode() + b"\n" + name.encode() + b"\n"


def _binbytes(data: bytes) -> bytes:
    return pickle.BINBYTES + struct.pack("<I", len(data)) + data


def _short_str(text: str) -> bytes:
    raw = text.encode()
    return pickle.SHORT_BINUNICODE + bytes([len(raw)]) + raw


def _embedded_blob() -> bytes:
    """The bytes torch's own reduction would hand to ``_load_from_bytes``."""

    buffer = io.BytesIO()
    torch.save(torch.arange(3, dtype=torch.float32), buffer)
    return buffer.getvalue()


def _load_from_bytes_reduce(blob: bytes) -> bytes:
    """``_load_from_bytes(blob)`` as a REDUCE."""

    return (
        _glob("torch.storage", "_load_from_bytes") + _binbytes(blob) + pickle.TUPLE1 + pickle.REDUCE
    )


def _root_payload() -> bytes:
    return pickle.PROTO + bytes([2]) + _load_from_bytes_reduce(_embedded_blob()) + pickle.STOP


def _nested_payload() -> bytes:
    """``{"annotations": {"probe": [<reduction>]}}``: a shape a genuine root would carry."""

    return (
        pickle.PROTO
        + bytes([2])
        + pickle.EMPTY_DICT
        + _short_str("annotations")
        + pickle.EMPTY_DICT
        + _short_str("probe")
        + pickle.EMPTY_LIST
        + _load_from_bytes_reduce(_embedded_blob())
        + pickle.APPEND
        + pickle.SETITEM
        + pickle.SETITEM
        + pickle.STOP
    )


def _rebuild_chain_payload() -> bytes:
    """The exact shape an earlier writer emitted: ``_rebuild_tensor_v2(_load_from_bytes(..), ...)``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _glob("torch._utils", "_rebuild_tensor_v2")
        + pickle.MARK
        + _load_from_bytes_reduce(_embedded_blob())
        + pickle.BININT1
        + bytes([0])
        + pickle.BININT1
        + bytes([3])
        + pickle.TUPLE1
        + pickle.BININT1
        + bytes([1])
        + pickle.TUPLE1
        + pickle.NEWFALSE
        + pickle.NONE
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(value))


# --------------------------------------------------------------------------- #
# 1. The route is refused by name, before torch.load, in every position.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke_cells(
    "test_load_from_bytes_refused_before_torch_load[rebuild_chain-_RenameAwareUnpickler]"
)
@pytest.mark.parametrize("unpickler", _UNPICKLERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "payload",
    [_root_payload, _nested_payload, _rebuild_chain_payload],
    ids=["root", "nested", "rebuild_chain"],
)
def test_load_from_bytes_refused_before_torch_load(
    unpickler: type[Any], payload: Any, torch_load_tripwire: list[Any]
) -> None:
    with pytest.raises(pickle.UnpicklingError, match=_REFUSAL):
        unpickler(io.BytesIO(payload())).load()
    assert torch_load_tripwire == []


def test_load_from_bytes_refused_inside_a_valid_bundle(
    tmp_path: Path, torch_load_tripwire: list[Any]
) -> None:
    """A genuine envelope (manifest, blobs) around planted metadata refuses typed at tl.load."""

    path = tmp_path / "planted.tlspec"
    tl.trace(_Model(), torch.ones(2, 3), capture=CaptureOptions(cache=False)).save(path)
    (path / "metadata.pkl").write_bytes(_nested_payload())

    with pytest.raises(TorchLensIOError) as caught:
        tl.load(path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert isinstance(caught.value.__cause__, pickle.UnpicklingError)
    assert "torch.storage._load_from_bytes" in str(caught.value.__cause__)
    assert "re-saved" in str(caught.value.__cause__)
    assert torch_load_tripwire == []


# --------------------------------------------------------------------------- #
# 2. The inline reduction the writer emits instead: exact round trips.
# --------------------------------------------------------------------------- #


def _writer_round_trip(value: Any) -> tuple[bytes, Any]:
    buffer = io.BytesIO()
    dump_canonical_metadata({"value": value}, buffer)
    data = buffer.getvalue()
    return data, SafeBundleUnpickler(io.BytesIO(data)).load()["value"]


def _same_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Exact element equality for every dtype (float8 and bool have no ``equal`` kernel everywhere)."""

    left_flat = left.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
    right_flat = right.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
    if left_flat.numel() == 0:
        return right_flat.numel() == 0
    return torch.equal(left_flat.view(torch.uint8), right_flat.view(torch.uint8))


def _dtype_cases() -> list[Any]:
    names = [
        "float32",
        "float64",
        "float16",
        "bfloat16",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
        "complex64",
        "complex128",
        "float8_e4m3fn",
        "float8_e5m2",
    ]
    return [pytest.param(getattr(torch, name), id=name) for name in names if hasattr(torch, name)]


@pytest.mark.parametrize("dtype", _dtype_cases())
def test_inline_tensor_round_trips_every_dtype(dtype: torch.dtype) -> None:
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor = (source > 2) if dtype is torch.bool else source.to(dtype)
    data, loaded = _writer_round_trip(tensor)

    assert b"_load_from_bytes" not in data
    assert INLINE_TENSOR_GLOBAL[1].encode() in data
    assert type(loaded) is torch.Tensor
    assert loaded.dtype == dtype
    assert loaded.shape == tensor.shape
    assert _same_bytes(loaded, tensor)


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(torch.tensor(3.5), id="scalar"),
        pytest.param(torch.empty(0, 3), id="zero_size"),
        pytest.param(torch.arange(12.0).reshape(3, 4)[:, 1:3], id="strided_view"),
        pytest.param(torch.arange(6.0).reshape(2, 3).t(), id="transposed"),
        pytest.param(torch.arange(6.0).reshape(2, 3)[1], id="offset_view"),
        pytest.param(torch.view_as_complex(torch.arange(8.0).reshape(4, 2)).conj(), id="conj_bit"),
        pytest.param(torch._neg_view(torch.arange(4.0)), id="neg_bit"),
    ],
)
def test_inline_tensor_round_trips_shapes_and_views(tensor: torch.Tensor) -> None:
    data, loaded = _writer_round_trip(tensor)

    assert b"_load_from_bytes" not in data
    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype
    assert loaded.is_contiguous()
    assert not loaded.is_conj() and not loaded.is_neg()
    assert _same_bytes(loaded, tensor)


def test_inline_tensor_keeps_requires_grad_and_parameter_identity() -> None:
    plain = torch.ones(2, requires_grad=True)
    _, loaded_plain = _writer_round_trip(plain)
    assert type(loaded_plain) is torch.Tensor
    assert loaded_plain.requires_grad and loaded_plain.is_leaf

    for flag in (True, False):
        data, loaded = _writer_round_trip(nn.Parameter(torch.arange(3.0), requires_grad=flag))
        assert b"_load_from_bytes" not in data
        assert isinstance(loaded, nn.Parameter)
        assert loaded.requires_grad is flag
        assert _same_bytes(loaded, torch.arange(3.0))


def test_bundle_with_metadata_tensors_round_trips(tmp_path: Path) -> None:
    """An annotation tensor saves without the route and loads back exactly through tl.load."""

    probe = torch.arange(6.0).reshape(2, 3)
    x = torch.ones(2, 3)
    trace = tl.trace(_Model(), x, capture=CaptureOptions(cache=False))
    trace.annotations["probe"] = probe
    path = tmp_path / "genuine.tlspec"
    trace.save(path)

    metadata = (path / "metadata.pkl").read_bytes()
    assert b"_load_from_bytes" not in metadata
    assert INLINE_TENSOR_GLOBAL[1].encode() in metadata

    loaded = tl.load(path)
    assert torch.equal(loaded.annotations["probe"], probe)


# --------------------------------------------------------------------------- #
# 3. The reconstructor itself fails closed on malformed or oversized arguments.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("args", "detail"),
    [
        pytest.param((b"\x00" * 4, "torch.float32", (100_000_000,)), "byte length", id="bomb"),
        pytest.param((b"\x00" * 4, "torch.float32", (2,)), "byte length", id="short"),
        pytest.param((bytearray(4), "torch.float32", (1,)), "must be bytes", id="not_bytes"),
        pytest.param((b"\x00" * 4, "float32", (1,)), "unknown dtype", id="bare_dtype_name"),
        pytest.param((b"\x00" * 4, "torch.qint8", (4,)), "unknown dtype", id="quantized"),
        pytest.param((b"\x00" * 4, "torch.float32", [1]), "shape must be", id="list_shape"),
        pytest.param((b"\x00" * 4, "torch.float32", (-1,)), "shape must be", id="negative_dim"),
        pytest.param((b"\x00" * 4, "torch.float32", (1,), 1), "requires_grad", id="int_flag"),
        pytest.param(
            (b"\x00" * 4, "torch.int32", (1,), True), "cannot require grad", id="int_grad"
        ),
        pytest.param(
            (b"\x00" * 4, "torch.float32", (1,), False, "cpu; import os"),
            "device must be",
            id="device_junk",
        ),
        pytest.param(
            (b"\x00" * 4, "torch.float32", (1,), False, "nosuchdevice:0"),
            "unavailable here",
            id="device_unknown",
        ),
    ],
)
def test_inline_tensor_rejects_malformed_arguments(args: tuple[Any, ...], detail: str) -> None:
    with pytest.raises(pickle.UnpicklingError, match="malformed inline tensor") as caught:
        rebuild_inline_tensor(*args)
    assert detail in str(caught.value)


def test_inline_tensor_storage_cannot_be_grown() -> None:
    """The buffer-backed storage is not resizable, so ``_rebuild_tensor_v2`` cannot amplify it."""

    tensor = rebuild_inline_tensor(bytes(range(4)), "torch.uint8", (4,))
    assert not tensor.untyped_storage().resizable()
    # ``_rebuild_tensor_v2`` takes a typed storage; wrapping shares the same (frozen) bytes.
    storage = torch.storage.TypedStorage(
        wrap_storage=tensor.untyped_storage(), dtype=torch.uint8, _internal=True
    )
    with pytest.raises(RuntimeError, match="(?i)resiz"):
        torch._utils._rebuild_tensor_v2(storage, 0, (2_000_000,), (1,), False, None)
    exact_fit = torch._utils._rebuild_tensor_v2(storage, 0, (4,), (1,), False, None)
    assert exact_fit.tolist() == [0, 1, 2, 3]
