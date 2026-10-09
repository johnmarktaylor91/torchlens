"""Coverage gate of the visual language deck (``scripts/visual_language``).

The deck must teach everything the renderer can draw. ``coverage.check_all`` derives six
universes from the renderer (draw parameters, closed vocabularies, legend rows, Graphviz
tokens and hex colours, label templates, emission sites) and runs eight checkers against
the slide table and the committed coverage tables; a new renderer item fails with its name,
its source and the remedy.

Red-capability: each checker is a plain function over its two sides, and
``TestMechanismIsRedCapable`` plants one drift into each and proves it is reported. A gate
nobody has proved can fail is not a gate.

Only the completeness test is smoke tier (no rendering; the AST scans reuse the shared
source corpus). The red-capability tests re-derive universes and stay unmarked.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from _source_corpus import package_files

from scripts.visual_language import coverage, slides
from scripts.visual_language.coverage_scan import (
    DrawParam,
    LegendText,
    SourceItem,
    VocabValue,
    emission_site_universe,
    scanned_files,
    visual_token_universe,
)


@pytest.mark.smoke
def test_deck_covers_every_renderer_item() -> None:
    """Every draw option, vocabulary value, legend row, token, template and site is taught."""

    assert set(scanned_files()) <= set(package_files())
    start = time.perf_counter()
    failures = coverage.check_all()
    elapsed = time.perf_counter() - start
    for line in failures:
        print(line)
    print(f"visual language coverage: {len(failures)} failure(s) in {elapsed:.2f}s")
    assert not failures, f"{len(failures)} coverage failure(s):\n" + "\n".join(failures)


def _reported(failures: list[str], *needles: str) -> bool:
    return any(all(needle in line for needle in needles) for line in failures)


class TestMechanismIsRedCapable:
    """Plant one drift per checker and prove the checker reports it."""

    def test_new_draw_parameter_is_reported(self) -> None:
        """A draw option no slide, convention or cheat list teaches fails."""

        planted = (*coverage.draw_surface_universe(), DrawParam("Trace.draw", "edge_wobble"))
        failures = coverage.check_draw_surface(planted, coverage.draw_lessons())
        assert _reported(failures, "`edge_wobble`", "Trace.draw", "no lesson")
        assert _reported(failures, "`edge_wobble`", "no cheat group")

    def test_new_visualization_option_field_is_reported(self) -> None:
        """A grouped option field with no draw-kwarg alias fails."""

        planted = (
            *coverage.draw_surface_universe(),
            DrawParam("VisualizationOptions", "wobble"),
        )
        failures = coverage.check_draw_surface(planted, coverage.draw_lessons())
        assert _reported(failures, "`wobble`", "VIS_OPTION_ALIASES")

    def test_new_literal_value_is_reported(self) -> None:
        """A closed-vocabulary value with no witness fails."""

        planted = (
            *coverage.vocabulary_universe(),
            VocabValue("_literals.VisDirectionLiteral", "rightleft_spiral", "planted"),
        )
        failures = coverage.check_vocabularies(
            planted,
            coverage.filled_deck_text(),
            coverage.panel_kwarg_values(),
            coverage.row_variants(),
        )
        assert _reported(failures, "`rightleft_spiral`", "VisDirectionLiteral")

    def test_new_theme_legend_row_is_reported(self) -> None:
        """A legend row text no slide shows fails, and so does an unmapped section."""

        planted = (
            *coverage.legend_universe(),
            LegendText("TorchLens legend", "sparkle node", "theme_role_sections(fake)"),
            LegendText("TorchLens sparkles", "sparkle", "fake_sections()"),
        )
        failures = coverage.check_legend_rows(planted, coverage.filled_deck_text())
        assert _reported(failures, "`sparkle node`", "theme_role_sections(fake)")
        assert _reported(failures, "`TorchLens sparkles`", "LEGEND_SECTION_ROWS")

    def test_new_hex_token_is_reported(self, tmp_path: Path) -> None:
        """The scanner finds a new colour and shape, and the checker reports both."""

        source = tmp_path / "fake_render.py"
        source.write_text(
            'def emit(graph):\n    graph.node("n", shape="hexagon", fillcolor="#123456")\n',
            encoding="utf-8",
        )
        tokens = visual_token_universe([source])
        assert {item.key for item in tokens} == {"#123456", "shape=hexagon"}
        planted = (*coverage.visual_token_universe(), *tokens)
        failures = coverage.check_visual_tokens(planted)
        assert _reported(failures, "`#123456`", "fake_render.py:2", "unclassified")
        assert _reported(failures, "`shape=hexagon`", "unclassified")

    def test_stale_and_misclassified_tokens_are_reported(self) -> None:
        """A table entry for a vanished token, or one naming no row, fails."""

        table = {**coverage.VISUAL_TOKENS, "#abcdef": "VN01", "#98fb98": "VZ99"}
        failures = coverage.check_visual_tokens(coverage.visual_token_universe(), table)
        assert _reported(failures, "Stale", "`#abcdef`")
        assert _reported(failures, "`#98fb98`", "unknown row `VZ99`")

    def test_new_label_template_is_reported(self) -> None:
        """A label template with no classification fails."""

        planted = (
            *coverage.label_template_universe(),
            SourceItem("wobble factor:", "torchlens/visualization/_label_format.py:999"),
        )
        failures = coverage.check_label_templates(planted)
        assert _reported(failures, "`wobble factor:`", "_label_format.py:999")

    def test_unclassified_emission_site_is_reported(self, tmp_path: Path) -> None:
        """A new emitting function needs classification; it is never auto-accepted."""

        source = tmp_path / "fake_emit.py"
        source.write_text(
            'def draw_extra(graph):\n    graph.edge("a", "b", style="dashed")\n',
            encoding="utf-8",
        )
        sites = emission_site_universe([source])
        assert [site.site.rsplit(":", 1)[-1] for site in sites] == ["draw_extra"]
        failures = coverage.check_emission_sites((*coverage.emission_site_universe(), *sites))
        assert _reported(failures, "draw_extra", "needs classification")

    def test_changed_emission_fingerprint_is_reported(self, tmp_path: Path) -> None:
        """A site whose emitted vocabulary changed fails as needing classification."""

        source = tmp_path / "fake_emit.py"
        source.write_text('def emit(g):\n    g.node("a", shape="box")\n', encoding="utf-8")
        before = emission_site_universe([source])[0]
        changed = tmp_path / "fake_emit2.py"
        changed.write_text('def emit(g):\n    g.node("a", shape="box3d")\n', encoding="utf-8")
        after = emission_site_universe([changed])[0]
        assert before.fingerprint != after.fingerprint
        table = {before.site: (before.fingerprint, "VN10")}
        renamed = dataclasses.replace(after, site=before.site)
        failures = coverage.check_emission_sites((renamed,), table)
        assert _reported(failures, before.site, "changed its emitted vocabulary")

    def test_flipped_gating_flag_is_reported(self) -> None:
        """An inactive row whose gating flag flipped fails."""

        namespace = {
            **coverage.gate_namespace(),
            "themes": SimpleNamespace(COLLAPSE_KIND_TOKENS_DEFAULT=True),
        }
        failures = coverage.check_inactive_rows(coverage.ROWS, namespace)
        assert _reported(failures, "VT04", "no longer holds")
        assert not coverage.check_inactive_rows(coverage.ROWS, coverage.gate_namespace())

    def test_removed_slide_id_is_reported(self) -> None:
        """A row pointing at a slide the deck dropped fails, as does a stale locator."""

        deck = tuple(slide for slide in slides.SLIDES if slide.id != "buffers")
        failures = coverage.check_map_integrity(coverage.ROWS, deck)
        assert _reported(failures, "VN07", "`buffers`")
        rows = {
            **coverage.ROWS,
            "VN01": dataclasses.replace(coverage.ROWS["VN01"], locator="_render_nodes:gone"),
        }
        failures = coverage.check_map_integrity(rows, slides.SLIDES)
        assert _reported(failures, "VN01", "does not resolve")
