"""The reader side of the ``tuple`` and ``dict`` subtype carrier (saved format 3).

A tuple subtype selects a constructor from the code-owned table only; a defaultdict
cell rebuilds with a builtin factory or reloads as a plain dict when its factory was
disclosed; anything else is a typed ``record_schema_invalid`` refusal. No artifact
string is imported or looked up by name and no factory executes while decoding.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from torchlens._io import ArtifactError
from torchlens._io.format3.decode import TensorCellFields, UnencodableValue, decode_value
from torchlens._io.format3.values import ValueEncoder, tuple_subtypes

AT = "op[0].func_config"


def _round_trip(value: object) -> object:
    return decode_value(ValueEncoder().encode(value, AT), at=AT)


def _refuses(cell: dict) -> ArtifactError:
    with pytest.raises(ArtifactError) as caught:
        decode_value(cell, at=AT)
    assert caught.value.fields["code"] == "record_schema_invalid"
    assert AT in str(caught.value)
    return caught.value


def test_torch_size_reloads_as_torch_size() -> None:
    back = _round_trip(torch.Size([2, 3]))
    assert type(back) is torch.Size
    assert back == (2, 3)


def test_structseq_reloads_through_its_known_constructor() -> None:
    # Plain members: a tensor member needs the record-table payload sink, not this unit.
    maximum = torch.return_types.max((3, 1))
    back = _round_trip(maximum)
    assert type(back) is type(maximum)
    assert back.values == 3 and back.indices == 1


def test_tuple_subtype_table_is_closed_and_code_owned() -> None:
    table = tuple_subtypes()
    assert table["torch.Size"] is torch.Size
    assert all(issubclass(cls, tuple) for cls in table.values())
    assert all(name == "torch.Size" or name.startswith("torch.return_types.") for name in table)


def test_plain_tuple_keeps_its_old_meaning() -> None:
    back = decode_value({"$tl": "tuple", "items": [1, 2]}, at=AT)
    assert type(back) is tuple and back == (1, 2)


@pytest.mark.parametrize(
    "subtype", ["os.system", "builtins.tuple", "torch.return_types.__loader__", 7, ""]
)
def test_unknown_tuple_subtype_refuses_typed(subtype: object) -> None:
    error = _refuses({"$tl": "tuple", "items": [1, 2], "subtype": subtype})
    assert "tuple subtype" in str(error)


def test_tuple_subtype_that_rejects_the_items_refuses_typed() -> None:
    error = _refuses({"$tl": "tuple", "items": ["a"], "subtype": "torch.Size"})
    assert "does not accept" in str(error)


@pytest.mark.parametrize("factory", [list, dict, set, int, float, str, tuple])
def test_builtin_defaultdict_factory_rebuilds(factory: type) -> None:
    back = _round_trip(defaultdict(factory, {"seen": 3}))
    assert isinstance(back, defaultdict)
    assert back.default_factory is factory
    assert back["seen"] == 3
    assert back["unseen"] == factory()


def test_disclosed_foreign_factory_reloads_as_a_plain_dict() -> None:
    calls: list[bool] = []

    def factory() -> list:
        calls.append(True)
        return []

    encoder = ValueEncoder()
    cell = encoder.encode(defaultdict(factory, {"seen": 3}), AT)
    assert cell["default_factory"]["$tl"] == "unencodable"
    back = decode_value(cell, at=AT)
    assert type(back) is dict
    assert back == {"seen": 3}
    assert not calls
    disclosure = decode_value(cell["default_factory"], at=f"{AT}.default_factory")
    assert isinstance(disclosure, UnencodableValue)


@pytest.mark.parametrize(
    "cell",
    [
        {
            "$tl": "dict",
            "items": [],
            "subtype": "collections.defaultdict",
            "default_factory": "eval",
        },
        {"$tl": "dict", "items": [], "subtype": "collections.defaultdict", "default_factory": 3},
        {"$tl": "dict", "items": [], "subtype": "collections.defaultdict"},
        {"$tl": "dict", "items": [], "subtype": "collections.OrderedDict"},
        {"$tl": "dict", "items": [], "default_factory": "list"},
        {
            "$tl": "dict",
            "items": [],
            "subtype": "collections.defaultdict",
            "default_factory": {"$tl": "tuple", "items": []},
        },
    ],
)
def test_malformed_defaultdict_cells_refuse_typed(cell: dict) -> None:
    _refuses(cell)


def test_plain_dict_cell_keeps_its_old_meaning() -> None:
    back = decode_value({"$tl": "dict", "items": [[1, "a"]]}, at=AT)
    assert type(back) is dict and back == {1: "a"}


def test_frozenset_with_a_tensor_member_decodes_without_a_payload() -> None:
    tensor_cell = {
        "$tl": "tensor",
        "dtype": "torch.int64",
        "shape": [],
        "device": "cpu",
        "requires_grad": False,
        "codec": "torch_safetensors_v1",
        "blob": "blobs/torch_safetensors_v1-0000.safetensors",
        "key": "t0",
    }
    cell = {"frozen": {"$tl": "frozenset", "items": [tensor_cell]}}
    back = decode_value(cell, at=AT)
    assert isinstance(back["frozen"], frozenset)
    (member,) = back["frozen"]
    assert isinstance(member, TensorCellFields)
    assert member["dtype"] == "torch.int64"
