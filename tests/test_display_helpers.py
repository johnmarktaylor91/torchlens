"""House style v1 display helpers: numbers, tensor text, cards, keys, bounded printouts."""

import dataclasses
import pickle

import pytest

from torchlens._utils._display_bounded import RecordTuple, short_dataclass_repr
from torchlens._utils._display_card import (
    REPR_COLS,
    bounded_lines,
    inline_names,
    one_line,
    render_card,
)
from torchlens._utils._display_keys import candidates_text, display_key
from torchlens._utils._display_numbers import (
    format_count,
    format_duration,
    format_exact_count,
    format_iec_bytes,
    format_sig,
    plural,
)
from torchlens._utils._display_text import tensor_stand_in, tensor_text


def test_numbers_follow_the_house_spellings() -> None:
    assert format_sig(-1.0) == "-1.000"
    assert format_sig(0.99637) == "0.9964"
    assert format_sig(3.27e-5) == "3.270e-05"
    assert format_count(15360) == "15.4k"
    assert format_exact_count(124439808) == "124,439,808"
    assert format_iec_bytes(61440) == "60.0 KiB"
    assert format_iec_bytes(12) == "12 B"
    assert format_duration(0.00402) == "4.02 ms"
    assert format_duration(151e-6) == "151 us"
    assert format_duration(3.52) == "3.52 s"
    assert plural(1, "call") == "1 call"
    assert plural(3, "pass", "passes") == "3 passes"


@pytest.mark.parametrize(
    "count, expected",
    [
        (999, "999"),
        (1000, "1.0k"),
        (999949, "999.9k"),
        (999950, "1.0M"),
        (999950000, "1.0G"),
        (124439808, "124.4M"),
        (999950000000, "1.0T"),
        (3400000000000, "3.4T"),
    ],
)
def test_compact_counts_keep_one_decimal_and_roll_over(count: int, expected: str) -> None:
    assert format_count(count) == expected


@pytest.mark.parametrize("unit, scale", [("KiB", 1024), ("MiB", 1024**2), ("GiB", 1024**3)])
@pytest.mark.parametrize("value, expected", [(999.5, "0.976"), (1000, "0.977"), (1023.9, "1.00")])
def test_iec_rounding_promotes_without_scientific_notation(
    unit: str, scale: int, value: float, expected: str
) -> None:
    next_unit = {"KiB": "MiB", "MiB": "GiB", "GiB": "TiB"}[unit]
    assert format_iec_bytes(round(value * scale)) == f"{expected} {next_unit}"
    assert format_iec_bytes(15 * scale) == f"15.0 {unit}"


@pytest.mark.parametrize("scale, next_unit", [(1e-9, "us"), (1e-6, "ms"), (1e-3, "s")])
@pytest.mark.parametrize("value, expected", [(999.5, "1.00"), (1000, "1.00"), (1023.9, "1.02")])
def test_duration_rounding_promotes_without_scientific_notation(
    scale: float, next_unit: str, value: float, expected: str
) -> None:
    assert format_duration(value * scale) == f"{expected} {next_unit}"
    assert format_duration(-value * scale) == f"-{expected} {next_unit}"


def test_duration_keeps_significant_zeros() -> None:
    assert format_duration(1.5) == "1.50 s"


def test_tensor_text_reads_metadata_only() -> None:
    assert tensor_text("torch.float32", (1, 5, 3072), "cpu") == "float32 [1, 5, 3072] on cpu"
    assert tensor_text("torch.int64", ()) == "int64 []"
    assert tensor_text(None, None) == "dtype unknown shape unknown"

    class Meta:
        dtype = "torch.float32"
        shape = (4,)
        device = "cpu"

    assert tensor_stand_in(Meta()) == "<tensor float32 [4] on cpu>"


def test_card_layout_and_cut() -> None:
    card = render_card(
        "Op tanh_6: float32 [1, 5, 3072] on cpu",
        [("position", "operation 270 of 545"), ("values", "")],
        [".out", ".parents"],
    )
    assert card.splitlines() == [
        "Op tanh_6: float32 [1, 5, 3072] on cpu",
        "  position  operation 270 of 545",
        "  more      .out  .parents",
    ]
    many = render_card("X", [(f"r{i}", "v") for i in range(20)], [".a"], max_rows=4)
    assert many.splitlines()[-2].strip() == "... and 18 more rows"
    assert len(many.splitlines()) == 5


def test_one_line_drops_optional_facts_to_fit() -> None:
    line = one_line("Op k", ["float32 [1] on cpu"], ["x" * 200])
    assert line == "Op k: float32 [1] on cpu"
    assert len(one_line("Op k", ["a"], ["b" * 50, "c" * 50])) <= REPR_COLS


def test_lists_cut_before_rendering() -> None:
    rendered: list[int] = []

    def render(item: int) -> str:
        rendered.append(item)
        return str(item)

    lines = bounded_lines(list(range(500)), render)
    assert lines[-1] == "  ... and 495 more"
    assert rendered == [0, 1, 2, 3, 4]
    assert inline_names(["a", "b", "c", "d"]) == "a, b, c, and 1 more"


def test_display_key_uses_stored_short_label() -> None:
    class Single:
        short_layer_label = "tanh_6"
        num_passes = 1
        pass_number = 1
        label = "tanh_6_270:1"

    class Multi(Single):
        short_layer_label = "tanh_1"
        num_passes = 3
        pass_number = 2

    assert display_key(Single()) == "tanh_6"
    assert display_key(Multi()) == "tanh_1:2"
    assert candidates_text(list("abcdefg")) == "'a', 'b', 'c', 'd', 'e'; and 2 more"


@dataclasses.dataclass
class _Row:
    label: str
    count: int
    items: tuple
    nested: object


def test_record_tuple_is_a_tuple_with_a_bounded_repr() -> None:
    rows = RecordTuple(
        [_Row(f"aten_{i}", i, (), None) for i in range(500)], title="AtenOps", noun="calls"
    )
    assert repr(rows) == "AtenOps: 500 calls"
    assert str(rows).splitlines()[-2] == "  ... and 495 more"
    assert str(rows).splitlines()[-1] == "  more      .count()  .index()"
    assert rows == tuple(rows)
    assert isinstance(rows, tuple)
    assert hash(RecordTuple((1, 2))) == hash((1, 2))
    clone = pickle.loads(pickle.dumps(rows))
    assert clone == rows and repr(clone) == repr(rows)
    small = RecordTuple([_Row("aten_825", 1, (), None)], title="AtenOps", noun="calls")
    assert repr(small) == "AtenOps: 1 call"


def test_short_dataclass_repr_never_recurses() -> None:
    row = _Row("aten_1", 1, (1, 2), _Row("n", 0, (), None))
    assert short_dataclass_repr(row) == "_Row(label='aten_1', count=1, items=<2 items>)"
    long_row = _Row("x" * 300, 1, (), None)
    assert len(short_dataclass_repr(long_row)) <= REPR_COLS
