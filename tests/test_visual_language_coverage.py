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

from scripts.visual_language import coverage, slides


def _reported(failures: list[str], *needles: str) -> bool:
    return any(all(needle in line for needle in needles) for line in failures)


class TestMechanismIsRedCapable:
    """Plant one drift per checker and prove the checker reports it."""

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
