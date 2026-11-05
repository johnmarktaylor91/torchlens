"""Saved format 3 stores byte-identical payloads once and hashes its files in blocks.

A trace often holds the same bytes several times: a clone, an eval-mode dropout's
copy, a view saved as a payload of its own. The writer stores one entry per
payload identity (kind, dtype, shape, strides, device, ``requires_grad``, codec and
the SHA-256 of the bytes) and every cell names that entry; the reader already
checks that repeated references agree and loads each cell as a tensor of its own.
The counters pinned here (payload references, stored entries, stored bytes) are
deterministic for a fixed model and input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torchvision
from safetensors import safe_open
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
from torchlens._io.format3 import writer
from torchlens._io.format3.payloads import PayloadStore


class _Twins(nn.Module):
    """A linear layer whose output is cloned twice, so three op outputs hold equal bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sum of two clones of the linear output."""

        y = self.lin(x)
        return y.clone() + y.clone()


def _tensor_cells(value: Any) -> list[dict[str, Any]]:
    """Return every stored tensor reference cell in a decoded record file."""

    if isinstance(value, dict):
        if value.get("$tl") == "tensor" and "key" in value:
            return [value]
        return [cell for item in value.values() for cell in _tensor_cells(item)]
    if isinstance(value, list):
        return [cell for item in value for cell in _tensor_cells(item)]
    return []


def payload_counters(root: Path) -> dict[str, int]:
    """Return the references, stored entries and stored bytes of a saved trace."""

    cells = [
        cell
        for table in sorted((root / "records").glob("*.json"))
        for cell in _tensor_cells(json.loads(table.read_text(encoding="utf-8")))
    ]
    entries = 0
    stored = 0
    for chunk in sorted((root / "blobs").glob("*.safetensors")):
        with safe_open(str(chunk), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open handles are not iterable
                tensor = handle.get_tensor(key)
                entries += 1
                stored += tensor.numel() * tensor.element_size()
    return {"references": len(cells), "entries": entries, "stored_bytes": stored}


def _gpt2() -> tuple[nn.Module, torch.Tensor]:
    """A two-block GPT-2 with a small vocabulary, randomly initialised."""

    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
    return GPT2LMHeadModel(config).eval(), torch.arange(8).reshape(1, 8)


def _resnet() -> tuple[nn.Module, torch.Tensor]:
    """torchvision's ResNet-18, randomly initialised, on one 32x32 image."""

    return torchvision.models.resnet18(weights=None).eval(), torch.randn(1, 3, 32, 32)


def _twins() -> tuple[nn.Module, torch.Tensor]:
    """The cloning model above on a 2x4 input."""

    return _Twins().eval(), torch.randn(2, 4)


_MODELS = {"gpt2": _gpt2, "resnet18": _resnet, "twins": _twins}

#: Counters measured at the writer that stores each payload identity once.
_PINNED = {
    "gpt2": {"references": 0, "entries": 0, "stored_bytes": 0},
    "resnet18": {"references": 0, "entries": 0, "stored_bytes": 0},
    "twins": {"references": 0, "entries": 0, "stored_bytes": 0},
}


def save_model(name: str, root: Path) -> tl.Trace:
    """Trace one of the pinned models with a fixed seed and save it under ``root``."""

    torch.manual_seed(0)
    model, x = _MODELS[name]()
    trace = tl.trace(model, x)
    trace.save(root, level="portable")
    return trace


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_pinned_payload_counters(tmp_path: Path, name: str) -> None:
    """References, stored entries and stored bytes match their pins, and the trace reloads."""

    root = tmp_path / f"{name}.tlspec"
    trace = save_model(name, root)
    assert payload_counters(root) == _PINNED[name]
    loaded = tl.load(root)
    for live, back in zip(trace.ops, loaded.ops, strict=True):
        if isinstance(live.out, torch.Tensor):
            assert torch.equal(back.out, live.out), live.label


def test_shared_entry_loads_as_independent_tensors(tmp_path: Path) -> None:
    """Three equal outputs name one entry; each loads as its own tensor, so writes stay local."""

    root = tmp_path / "twins.tlspec"
    trace = save_model("twins", root)
    op_cells = json.loads((root / "records/op.json").read_text(encoding="utf-8"))["columns"]["out"]
    ops = list(trace.ops)
    equal = [index for index, op in enumerate(ops) if op.func_name in {"linear", "clone"}]
    assert len(equal) == 3
    assert len({(op_cells[index]["blob"], op_cells[index]["key"]) for index in equal}) == 1
    loaded_ops = list(tl.load(root).ops)
    first, second, third = (loaded_ops[index].out for index in equal)
    assert len({tensor.untyped_storage().data_ptr() for tensor in (first, second, third)}) == 3
    expected = ops[equal[2]].out.detach().clone()
    with torch.no_grad():
        first.add_(1.0)
        second.mul_(0.0)
    assert torch.equal(third, expected)


def test_payload_identity_separates_every_declared_field(tmp_path: Path) -> None:
    """Equal bytes share an entry only when kind, strides and grad flag agree too."""

    store = PayloadStore(tmp_path)
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    strided = torch.empty(3, 2).t().copy_(base)
    assert strided.stride() != base.stride() and torch.equal(strided, base)
    keys = [
        store.submit_payload(base, kind="out")[:2],
        store.submit_payload(base.clone(), kind="out")[:2],
        store.submit_payload(base.clone(), kind="grad")[:2],
        store.submit_payload(base.clone().requires_grad_(), kind="out")[:2],
        store.submit_payload(strided, kind="out")[:2],
        store.unshared().submit_payload(base.clone(), kind="out")[:2],
        store.submit_payload(base.clone(), kind="out")[:2],
    ]
    assert keys[1] == keys[0] == keys[6]
    assert len(set(keys)) == 5
    assert store.payload_counts == {"submitted": 7, "stored": 5}


def test_file_digests_hash_in_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The envelope digests never read a whole file into memory."""

    data = bytes(range(256)) * 4099
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs/chunk.safetensors").write_bytes(data)
    whole_reads: list[Path] = []
    real_read_bytes = Path.read_bytes

    def counting_read_bytes(self: Path) -> bytes:
        """Record a whole-file read, then perform it."""

        whole_reads.append(self)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    digests = writer.file_digests(tmp_path)
    assert whole_reads == []
    assert digests == {
        "blobs/chunk.safetensors": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    }
