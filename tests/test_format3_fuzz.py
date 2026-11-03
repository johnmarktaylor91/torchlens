"""Seeded fuzz of the 3.0 typed-value codec and of whole saves through the reload check.

Random nested Python values go through ``values.ValueEncoder``, a strict JSON round
trip (``allow_nan=False``) and ``values.decode_value`` and must come back identical
(same types, floats by exact repr). Random small models are traced, saved and put
through ``writer.reload_check``. Every case is seeded, so a failure reproduces.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import ArtifactError
from torchlens._io.format3 import writer as format_writer
from torchlens._io.format3.contract import TYPED_VALUE_KEY, TYPED_VALUE_TAGS
from torchlens._io.format3.values import ValueEncoder, decode_value, enum_table
from torchlens.io import UnencodableValue

_VALUE_SEEDS = range(40)
_MODEL_SEEDS = range(6)
_DTYPES = (torch.float32, torch.float64, torch.int64, torch.bool, torch.bfloat16, torch.uint8)
_DEVICES = (torch.device("cpu"), torch.device("meta"), torch.device("cuda", 1))
_SPECIAL_FLOATS = (math.nan, math.inf, -math.inf, -0.0, 1e308, 5e-324)
_SPECIAL_INTS = (0, -1, 2**53, 2**53 + 1, -(2**63), 2**70)
_SPECIAL_STRINGS = ("", "$tl", "aé中\U0001f600", "line\nbreak", '"quoted"')


def _scalar(rng: random.Random) -> Any:
    choice = rng.randrange(9)
    if choice == 0:
        return None
    if choice == 1:
        return rng.random() < 0.5
    if choice == 2:
        return rng.choice((rng.randint(-1000, 1000), rng.choice(_SPECIAL_INTS)))
    if choice == 3:
        return rng.choice((rng.uniform(-1e6, 1e6), rng.choice(_SPECIAL_FLOATS)))
    if choice == 4:
        return rng.choice(_SPECIAL_STRINGS + ("".join(rng.choices("abcxyz_ ", k=6)),))
    if choice == 5:
        return bytes(rng.randrange(256) for _ in range(rng.randrange(6)))
    if choice == 6:
        return complex(rng.uniform(-5, 5), rng.uniform(-5, 5))
    if choice == 7:
        return rng.choice(_DTYPES)
    return rng.choice(_DEVICES)


def _hashable(rng: random.Random) -> Any:
    value = _scalar(rng)
    if isinstance(value, float) and math.isnan(value):
        return 1.5
    return value


def _value(rng: random.Random, depth: int = 0) -> Any:
    if depth >= 4 or rng.random() < 0.35:
        return _scalar(rng)
    choice = rng.randrange(7)
    width = rng.randrange(4)
    if choice == 0:
        return [_value(rng, depth + 1) for _ in range(width)]
    if choice == 1:
        return tuple(_value(rng, depth + 1) for _ in range(width))
    if choice == 2:
        return {_hashable(rng) for _ in range(width)}
    if choice == 3:
        return frozenset(_hashable(rng) for _ in range(width))
    if choice == 4:
        return {f"k{i}": _value(rng, depth + 1) for i in range(width)}
    if choice == 5:
        keys = [_hashable(rng) for _ in range(width)] + [TYPED_VALUE_KEY]
        return {key: _value(rng, depth + 1) for key in keys}
    return (tuple(_hashable(rng) for _ in range(width)), _value(rng, depth + 1))


def _assert_same(decoded: Any, original: Any, at: str = "value") -> None:
    """Exact equality: same type, floats by repr, containers member by member."""

    assert type(decoded) is type(original), (at, decoded, original)
    if isinstance(original, float):
        assert repr(decoded) == repr(original), at
    elif isinstance(original, complex):
        _assert_same(decoded.real, original.real, f"{at}.real")
        _assert_same(decoded.imag, original.imag, f"{at}.imag")
    elif isinstance(original, (list, tuple)):
        assert len(decoded) == len(original), at
        for index, (left, right) in enumerate(zip(decoded, original, strict=True)):
            _assert_same(left, right, f"{at}[{index}]")
    elif isinstance(original, dict):
        assert list(decoded) == list(original), at
        for key in original:
            _assert_same(decoded[key], original[key], f"{at}[{key!r}]")
    else:
        assert decoded == original, at


def _round_trip(value: Any, encoder: ValueEncoder | None = None) -> tuple[Any, Any]:
    encoder = encoder or ValueEncoder()
    encoded = encoder.encode(value, "fuzz")
    text = json.dumps(encoded, allow_nan=False, ensure_ascii=False)
    parsed = json.loads(text)
    return parsed, decode_value(parsed, at="fuzz")


def _tags(data: Any) -> set[str]:
    if isinstance(data, list):
        return set().union(*(_tags(item) for item in data)) if data else set()
    if isinstance(data, dict):
        found = {data[TYPED_VALUE_KEY]} if TYPED_VALUE_KEY in data else set()
        return found.union(*(_tags(item) for item in data.values())) if data else found
    return set()


@pytest.mark.parametrize("seed", _VALUE_SEEDS)
def test_random_values_round_trip_exactly(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(25):
        value = _value(rng)
        parsed, decoded = _round_trip(value)
        assert _tags(parsed) <= set(TYPED_VALUE_TAGS)
        _assert_same(decoded, value)


def test_complex_with_non_finite_parts_round_trips() -> None:
    value = complex(1.0, math.inf)
    _, decoded = _round_trip(value)
    _assert_same(decoded, value)


def test_torchlens_enums_round_trip() -> None:
    members = [member for cls in enum_table().values() for member in cls]
    assert members
    rng = random.Random(7)
    for member in rng.sample(members, min(40, len(members))):
        _, decoded = _round_trip(member)
        assert decoded is member


def test_unencodable_values_are_disclosed_never_pickled() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "O" * 400

    encoder = ValueEncoder()
    parsed, decoded = _round_trip({"a": [1, Opaque()], "b": Opaque()}, encoder)
    assert [item["at"] for item in encoder.disclosures] == ["fuzz['a'][1]", "fuzz['b']"]
    assert isinstance(decoded["b"], UnencodableValue)
    assert decoded["b"].repr_text == "O" * 256
    assert decoded["b"].at == "fuzz['b']"
    assert parsed["b"]["repr_length"] == 400
    assert _tags(parsed) == {"unencodable"}


def test_tensor_in_payload_position_becomes_a_reference() -> None:
    stored: list[str] = []

    def tensor_ref(value: Any, at: str) -> dict[str, Any]:
        stored.append(at)
        return {"blob": "blobs/torch_safetensors_v1-0000.safetensors", "key": f"t{len(stored) - 1}"}

    rng = random.Random(3)
    tensors = [torch.randn(rng.randrange(1, 4), 2) for _ in range(3)]
    encoder = ValueEncoder(tensor_ref=tensor_ref)
    encoded = encoder.encode({"x": tensors}, "op[0].out", payload=True)
    assert stored == [f"op[0].out['x'][{i}]" for i in range(3)]
    assert [cell["key"] for cell in encoded["x"]] == ["t0", "t1", "t2"]
    assert all(cell[TYPED_VALUE_KEY] == "tensor" for cell in encoded["x"])
    assert not encoder.disclosures


def test_tensor_in_metadata_position_is_disclosed() -> None:
    encoder = ValueEncoder(tensor_ref=lambda value, at: {"blob": "b", "key": "t0"})
    encoded = encoder.encode(torch.zeros(2), "op[0].func_config", payload=False)
    assert encoded[TYPED_VALUE_KEY] == "unencodable"
    assert [item["at"] for item in encoder.disclosures] == ["op[0].func_config"]


@pytest.mark.parametrize("seed", range(10))
def test_decoder_refuses_unknown_tags(seed: int) -> None:
    rng = random.Random(seed)
    bogus = "".join(rng.choices("abcdefgh", k=8))
    with pytest.raises(ArtifactError, match="unknown tag") as caught:
        decode_value({"x": [{TYPED_VALUE_KEY: bogus}]}, at="fuzz")
    assert caught.value.fields["code"] == "record_schema_invalid"


class _RandomNet(nn.Module):
    """A small random feed-forward net (linear and conv branches, random activations)."""

    def __init__(self, rng: random.Random) -> None:
        super().__init__()
        self.use_conv = rng.random() < 0.5
        width = rng.randrange(2, 6)
        self.conv = nn.Conv2d(1, width, 3) if self.use_conv else None
        features = width * 4 if self.use_conv else 16
        depth = rng.randrange(1, 4)
        sizes = [features] + [rng.randrange(2, 8) for _ in range(depth)]
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(depth))
        self.activations = [rng.choice((torch.relu, torch.tanh, torch.sigmoid)) for _ in sizes]
        self.residual = rng.random() < 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = nn.functional.adaptive_avg_pool2d(self.conv(x), 2)
        x = x.flatten(1)
        for layer, activation in zip(self.layers, self.activations, strict=False):
            y = activation(layer(x))
            x = y + y.mean() if self.residual else y
        return x


@pytest.mark.parametrize("seed", _MODEL_SEEDS)
def test_random_model_saves_pass_the_reload_check(seed: int, tmp_path: Path) -> None:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = _RandomNet(rng).eval()
    x = torch.randn(2, 1, 4, 4)
    trace = tl.trace(model, x)
    level = rng.choice(("portable", "audit"))
    trace.save(tmp_path / "fuzz", level=level)
    format_writer.reload_check(tmp_path / "fuzz")
    manifest = json.loads((tmp_path / "fuzz" / "manifest.json").read_text())
    assert manifest["save_level"] == level
    assert manifest["records"]["op"]["rows"] == len(trace._layer_list)
