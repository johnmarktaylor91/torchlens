"""F12 pins: the display filter (N9) -- tokens, polarity, disclosure,
bridged edges, and the teaching boundary raise.

Composition-test rows covered: 8 (filter-token yield disclosure), 9 (the
raw-predicate boundary raise names the real attributes and the token
remedy), 14 (bridged-edge label spelling: midpoint label=, dashed, never a
new xlabel/headlabel family).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses.audit import CORPUS

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)


def _member(name: str) -> Any:
    """Return one corpus member by name."""

    return next(member for member in CORPUS if member.name == name)


@pytest.fixture(scope="module")
def glue_log() -> Any:
    """The reshape-heavy filtered-chain toy."""

    model, x = _member("filtered_chain").build()
    log = tl.trace(model, x)
    yield log
    log.cleanup()


def test_tokens_are_the_closed_v1_vocabulary() -> None:
    """reshapes / constants / non_module_ops; nothing else."""

    assert lenses.FILTER_TOKENS == ("constants", "non_module_ops", "reshapes")


def test_unknown_token_refuses_typed(glue_log: Any) -> None:
    """A token outside the closed vocabulary refuses with the roster."""

    with pytest.raises(Exception) as excinfo:
        lenses.compile_display_filter(glue_log, lenses.DisplayFilter(exclude="noise"))
    assert excinfo.value.fields["code"] == "display_filter_token_unknown"


def test_polarity_is_exactly_one(glue_log: Any) -> None:
    """Both or neither polarity refuses typed."""

    with pytest.raises(Exception) as excinfo:
        lenses.compile_display_filter(
            glue_log, lenses.DisplayFilter(exclude="reshapes", include="constants")
        )
    assert excinfo.value.fields["code"] == "display_filter_polarity_conflict"
    with pytest.raises(Exception) as excinfo:
        lenses.compile_display_filter(glue_log, lenses.DisplayFilter())
    assert excinfo.value.fields["code"] == "display_filter_polarity_conflict"


def test_reshapes_token_counts_and_caption(glue_log: Any) -> None:
    """The caption reports actual counts (composition row 8)."""

    compiled = lenses.compile_display_filter(glue_log, lenses.DisplayFilter(exclude="reshapes"))
    assert compiled.filtered == 3  # reshape, permute, contiguous
    assert f"displayed {compiled.eligible - 3} of {compiled.eligible} eligible ops" in (
        compiled.caption
    )
    assert "reshapes" in compiled.caption
    assert "reachability through omitted operations" in compiled.legend_line


def test_boundaries_auto_exempt_on_tokens(glue_log: Any) -> None:
    """Token spellings never raise the boundary refusal."""

    compiled = lenses.compile_display_filter(glue_log, lenses.DisplayFilter(exclude="reshapes"))
    for op in glue_log.ops:
        if op.is_input or op.is_final_output:
            assert compiled.skip_fn(op) is False


def test_include_polarity_inverts(glue_log: Any) -> None:
    """include= hides everything NOT matched."""

    compiled = lenses.compile_display_filter(glue_log, lenses.DisplayFilter(include="reshapes"))
    assert compiled.eligible - compiled.filtered == 3


def test_skip_fn_conflict_refuses(glue_log: Any) -> None:
    """display_filter plus explicit skip_fn is a typed conflict."""

    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(
            glue_log,
            "blueprint",
            {"skip_fn": lambda layer: False},
            display_filter=lenses.DisplayFilter(exclude="reshapes"),
        )
    assert excinfo.value.fields["code"] == "display_filter_skip_fn_conflict"


def test_bridged_edges_are_dashed_with_midpoint_labels(glue_log: Any, tmp_path: Any) -> None:
    """Composition row 14: dashed style, midpoint label= 'via N hidden',
    the rendered caption, and no new xlabel family."""

    resolution = lenses.resolve_lens(
        glue_log, "blueprint", display_filter=lenses.DisplayFilter(exclude="reshapes")
    )
    graph = glue_log.draw(
        **resolution.draw_kwargs,
        vis_outpath=str(tmp_path / "filtered"),
        vis_fileformat="svg",
        vis_save_only=True,
        return_graph=True,
    )
    source = graph.source
    assert "style=dashed" in source
    labels = re.findall(r'label="via (\d+) hidden"', source)
    assert labels and int(labels[0]) >= 1
    assert "filtered (exclude: reshapes)" in source
    assert "xlabel=" not in source


def test_predicate_boundary_raise_teaches(glue_log: Any, tmp_path: Any) -> None:
    """Composition row 9: the strict predicate boundary raise names the REAL
    attributes, the multi-output case, and the token remedy."""

    with pytest.raises(Exception) as excinfo:
        glue_log.draw(
            skip_fn=lambda layer: True,
            vis_outpath=str(tmp_path / "boundary"),
            vis_save_only=True,
        )
    message = str(excinfo.value)
    assert excinfo.value.fields["code"] == "skip_fn_boundary_invalid"
    assert "is_input" in message
    assert "is_final_output" in message
    assert "DisplayFilter" in message


def test_non_module_ops_is_migration_parity(glue_log: Any) -> None:
    """The torchview-parity token compiles and counts module-less ops."""

    compiled = lenses.compile_display_filter(
        glue_log, lenses.DisplayFilter(exclude="non_module_ops")
    )
    expected = sum(
        1
        for op in glue_log.ops
        if not (op.is_input or op.is_final_output) and not tuple(op.modules or ())
    )
    assert compiled.filtered == expected


def test_operand_shape_refuses_typed(glue_log: Any) -> None:
    """A dict operand is not a token, list, Selection, or predicate."""

    with pytest.raises(Exception) as excinfo:
        lenses.compile_display_filter(glue_log, lenses.DisplayFilter(exclude={"a": 1}))
    assert excinfo.value.fields["code"] == "display_filter_operand_invalid"


def test_selection_operand_hides_the_selected_site(glue_log: Any) -> None:
    """A same-trace Selection operand hides the selected family (never a
    silent no-op: the resolved site labels must reach the match fn)."""

    target = next(op for op in glue_log.ops if not (op.is_input or op.is_final_output))
    resolved = target.__selection__().resolve(glue_log)
    compiled = lenses.compile_display_filter(glue_log, lenses.DisplayFilter(exclude=resolved))
    assert compiled.filtered >= 1


def test_foreign_resolved_selection_refuses_typed(glue_log: Any) -> None:
    """A ResolvedSelection bound to another trace refuses; align_to is the door."""

    model, x = _member("filtered_chain").build()
    other = tl.trace(model, x)
    try:
        foreign = other.ops[0].__selection__().resolve(other)
        with pytest.raises(Exception) as excinfo:
            lenses.compile_display_filter(glue_log, lenses.DisplayFilter(exclude=foreign))
        assert excinfo.value.fields["code"] == "display_filter_selection_foreign"
        assert "align_to" in str(excinfo.value)
    finally:
        other.cleanup()
