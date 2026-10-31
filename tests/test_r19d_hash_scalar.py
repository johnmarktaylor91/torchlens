"""Regression tests for ``torchlens.hash.content`` on 0-dim (scalar) tensors.

Finding r19d / F1-hash. ``hash._update_content_digest`` crashed on a 0-dim
tensor because ``torch.Tensor.view(torch.uint8)`` refuses a scalar
(``self.dim() cannot be 0 to view Float as Byte``). Two impacts:

1. Public ``tl.hash.content(scalar_tensor)`` raised ``RuntimeError``.
2. Worse and SILENT: the runnable-bundle provenance builder in
   ``torchlens/_io/bundle.py`` calls ``hash.content([...inputs...])`` inside a
   bare ``except Exception`` and, on the crash, wrote ``input_hash=None`` into
   the runnable manifest -- so a scalar input silently lost its attestation
   hash while a 1-D input recorded a real one.

The fix flattens with ``reshape(-1)`` before the ``uint8`` view. It is
byte-identical to the prior expression for every contiguous >=1-D tensor, so
existing pinned digests are unchanged.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

# 0-dim tensors of every replayed numeric dtype (mix of 1/2/4/8-byte elements).
SCALAR_DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
]


def _is_sha256_hex(value: object) -> bool:
    """Return True for a lowercase 64-char hexadecimal SHA-256 digest."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def test_content_scalar_float_hashes() -> None:
    """A 0-dim float scalar hashes (fail-before: RuntimeError on view)."""

    assert _is_sha256_hex(tl.hash.content(torch.tensor(1.5)))


@pytest.mark.parametrize("dtype", SCALAR_DTYPES)
def test_content_scalar_all_dtypes(dtype: torch.dtype) -> None:
    """Every 0-dim numeric dtype hashes without crashing."""

    assert _is_sha256_hex(tl.hash.content(torch.tensor(1, dtype=dtype)))


def test_content_scalar_bool() -> None:
    """0-dim bool scalars hash and distinguish their value."""

    assert _is_sha256_hex(tl.hash.content(torch.tensor(True)))
    assert tl.hash.content(torch.tensor(True)) != tl.hash.content(torch.tensor(False))


def test_content_scalar_distinguishes_values() -> None:
    """Different scalar values produce different digests."""

    assert tl.hash.content(torch.tensor(1.5)) != tl.hash.content(torch.tensor(2.5))


def test_content_scalar_differs_from_1d_sibling() -> None:
    """Shape is framed in: a 0-dim scalar must not collide with a (1,) tensor."""

    assert tl.hash.content(torch.tensor(1.5)) != tl.hash.content(torch.tensor([1.5]))


def test_content_scalar_deterministic() -> None:
    """The scalar digest is stable across repeated calls."""

    assert tl.hash.content(torch.tensor(3.25)) == tl.hash.content(torch.tensor(3.25))


def test_content_scalar_in_list_hashes() -> None:
    """Lists carrying a scalar hash -- exactly the shape ``_io/bundle.py`` passes."""

    assert _is_sha256_hex(tl.hash.content([torch.tensor(1.5)]))
    assert _is_sha256_hex(tl.hash.content([torch.tensor(1.5), torch.arange(4)]))


def test_content_scalar_in_mapping_hashes() -> None:
    """Mappings carrying a scalar value hash."""

    assert _is_sha256_hex(tl.hash.content({"a": torch.tensor(1.5)}))


def test_reshape_is_byte_identical_for_ndim_tensors() -> None:
    """Portable stability proof: the flattened view keeps >=1-D bytes identical.

    ``reshape(-1)`` on a contiguous tensor is a contiguous view, so the raw
    bytes fed to the digest are unchanged for every >=1-D tensor. This is the
    platform-independent guarantee that no pinned digest can drift from the fix.
    """

    tensors = [
        torch.tensor([1.5, 2.5, -3.0]),
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(24, dtype=torch.int64).reshape(2, 3, 4),
        torch.tensor([True, False, True]),
    ]
    for tensor in tensors:
        contiguous = tensor.contiguous()
        old = contiguous.view(torch.uint8).numpy().tobytes()
        new = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        assert old == new


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            torch.tensor([1.5, 2.5, -3.0]),
            "25545ac439b24b12e7a62d8b65fe5c37eec4e5e16452a5b451a62a5bc82c1b74",  # pragma: allowlist secret
        ),
        (
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "9f6847a8224a325adafb1604b00f920df15f77a70881f5fce6b8d09360aa9bb2",  # pragma: allowlist secret
        ),
        (
            torch.arange(24, dtype=torch.int64).reshape(2, 3, 4),
            "cb6889685c6d2535452a95e9ae001c06ed812d744557f1330eb6b394ccf0c5b5",  # pragma: allowlist secret
        ),
    ],
)
def test_ndim_content_hash_is_pinned(value: torch.Tensor, expected: str) -> None:
    """Absolute regression pins (little-endian x86 CI) for >=1-D digests.

    Captured on base 60416438 before the fix; the fix must leave them unchanged.
    """

    assert tl.hash.content(value) == expected


class _ScalarInputNet(nn.Module):
    """Model whose sole forward input is a 0-dim scalar tensor."""

    def __init__(self) -> None:
        """Initialize a small linear head."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        """Broadcast the scalar into a vector and run the linear head.

        Parameters
        ----------
        scalar:
            0-dim input scalar.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.lin(scalar.reshape(1) * torch.ones(4))


def test_scalar_input_runnable_manifest_records_input_hash(tmp_path) -> None:
    """Attestation gap closed: a scalar input records a real manifest input_hash.

    Before the fix the scalar input crashed ``hash.content`` inside the bundle
    provenance builder's bare-except, silently writing ``input_hash=null``.
    """

    log = tl.trace(
        _ScalarInputNet(),
        torch.tensor(1.5),
        capture=CaptureOptions(intervention_ready=True),
    )
    dest = tmp_path / "run.tlspec"
    tl.save(log, str(dest), level="runnable")
    manifest = json.loads((dest / "manifest.json").read_text())
    input_hash = manifest.get("provenance", {}).get("input_hash")
    assert _is_sha256_hex(input_hash)
