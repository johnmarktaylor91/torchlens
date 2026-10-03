"""F11 visual-token gates (collapse memo item 10 / D10, machinery wave).

The ONE ``collapse_tokens`` table in ``themes.py`` holds every kind's marks
and panel-fixed wording (the naming sprint renames in one place; a compact
mode is a table change), the K4/K3 theming bug is fixed (hardcoded grays
ignored dark themes), and the kind-word default flip stays behind the
FORK-5 flag until the legibility protocol ratifies it.
"""

from __future__ import annotations

import graphviz

from torchlens.visualization.collapse_plan import SegmentDescriptor
from torchlens.visualization.themes import (
    COLLAPSE_KIND_TOKENS_DEFAULT,
    THEME_PRESETS,
    CollapseTokens,
    collapse_tokens,
)


def test_tokens_table_wording_is_panel_fixed() -> None:
    """The kind wording carries the D10 content: kind + honest counts.

    "different" for segments (never a sameness claim), "1 of N shown" for
    folds (separate instances, never "distinct weights"), reuse counts for
    K1, and the K5 grammar for pattern chips.
    """

    tokens = collapse_tokens(THEME_PRESETS["torchlens"])
    assert tokens.word_reused.format(n=8) == "reused x8"
    assert tokens.word_box.format(n=12) == "12 ops inside"
    assert tokens.word_fold.format(n=20) == "1 of 20 shown"
    assert tokens.word_segment.format(n=20) == "20 different ops"
    assert tokens.word_pattern.format(name="ConvBnRelu", n=3) == "PATTERN 'ConvBnRelu' -- 3 ops"
    assert dict(tokens.label_rows).keys() == {"reused", "box", "fold", "segment", "pattern"}


def test_flag_default_off_until_protocol_ratifies() -> None:
    """FORK-5: the kind-word default flip waits for the evaluator battery."""

    assert COLLAPSE_KIND_TOKENS_DEFAULT is False


def test_tokens_derive_from_theme() -> None:
    """Dark themes get dark capsule/chip colors (the fixed hardcode bug)."""

    light = collapse_tokens(THEME_PRESETS["torchlens"])
    dark = collapse_tokens(THEME_PRESETS["dark"])
    assert isinstance(light, CollapseTokens) and isinstance(dark, CollapseTokens)
    assert light.segment_fill != dark.segment_fill
    assert dark.segment_fill == "#1F2937"
    assert dark.ellipsis_border == "#9CA3AF"
    assert light.segment_fill == "#f7f7f7"  # the historical light value, now themed


def test_segment_capsule_uses_theme_tokens() -> None:
    """The K4 emitter paints from the active theme, not hardcodes."""

    from torchlens.visualization._render_nodes import _queue_segment_node

    descriptor = SegmentDescriptor(
        name="seg_test",
        kind="op",
        label="a ... b -- 3 ops",
        ops=("a:1", "b:1", "c:1"),
        owner=None,
        num_ops=3,
    )
    for theme_name, expected_fill in (("torchlens", "#f7f7f7"), ("dark", "#1F2937")):
        graph = graphviz.Digraph()
        _queue_segment_node(graph, {}, set(), descriptor, ("unrolled", THEME_PRESETS[theme_name]))
        assert expected_fill in graph.source, (theme_name, graph.source)
