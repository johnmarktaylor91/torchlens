"""F11 visual-token gates (collapse memo item 10 / D10, machinery wave).

The ONE ``collapse_tokens`` table in ``themes.py`` holds every kind's marks
and panel-fixed wording (the naming sprint renames in one place; a compact
mode is a table change), the K4/K3 theming bug is fixed (hardcoded grays
ignored dark themes), and the kind-word default flip stays behind the
FORK-5 flag until the legibility protocol ratifies it.
"""

from __future__ import annotations

import graphviz

from torchlens._visualization.collapse_plan import SegmentDescriptor
from torchlens._visualization.themes import (
    THEME_PRESETS,
)


def test_segment_capsule_uses_theme_tokens() -> None:
    """The K4 emitter paints from the active theme, not hardcodes."""

    from torchlens._visualization._render_nodes import _queue_segment_node

    descriptor = SegmentDescriptor(
        name="seg_test",
        kind="op",
        label="a ... b -- 3 ops",
        ops=("a:1", "b:1", "c:1"),
        owner=None,
        num_ops=3,
    )
    for theme_name, expected_fill in (("torchlens", "#f7f7f7"), ("dark", "#262626")):
        graph = graphviz.Digraph()
        _queue_segment_node(graph, {}, set(), descriptor, ("unrolled", THEME_PRESETS[theme_name]))
        assert expected_fill in graph.source, (theme_name, graph.source)
