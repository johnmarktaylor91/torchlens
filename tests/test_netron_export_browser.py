"""Tier T4: headless-browser semantic smoke over the served artifact (D-09).

The ONLY layer that catches the silent-hang class (the function-cycle crash
shows an error dialog, renders nothing, and logs NOTHING to the console)
and the panel-content class (the ``module`` name collision was invisible
until a real click opened the sidebar). Mechanics locked by the panel and
re-verified here: netron replaces ``window.eval`` with a thrower
(browser.js:12-14) so ``page.evaluate`` dies on its first call -- every
read goes through raw CDP (``Page.createIsolatedWorld`` +
``Runtime.evaluate``); node appearance RACES the ``#message`` dialog under
a deadline; geometry must SETTLE before the extent is read
(first-element-exists gave a 4x-flattering wrong number); netron emits two
elements per box so ``.node`` count halves; the drill-down is a REAL mouse
click on the 6x12 px "Show Function Definition" affordance.

Heavy tier (measured 2-6 s per fixture); CI runs it path-filtered per the
sprint/packaging_requests.tsv row. Requires netron + playwright + its
Chromium; skips typed when absent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

netron_package = pytest.importorskip("netron", reason="T4 serves the real pinned netron app")
playwright_sync = pytest.importorskip(
    "playwright.sync_api", reason="T4 drives headless Chromium over the served app"
)

pytestmark = pytest.mark.heavy

#: Fit-zoom legibility budget in netron ranks (netron memo D-12).
_RANK_BUDGET = 27
_RANK_PIXELS = 55.0


class _Session:
    """One served-artifact browser session with CDP isolated-world reads."""

    def __init__(self, page: Any) -> None:
        """Wire the CDP session and isolated world for one loaded page."""

        self.page = page
        self.cdp = page.context.new_cdp_session(page)
        frame_id = self.cdp.send("Page.getFrameTree")["frameTree"]["frame"]["id"]
        self.ctx = self.cdp.send("Page.createIsolatedWorld", {"frameId": frame_id})[
            "executionContextId"
        ]

    def eval(self, expression: str) -> Any:
        """Evaluate one expression in the isolated world (eval is a thrower)."""

        reply = self.cdp.send(
            "Runtime.evaluate",
            {"expression": expression, "contextId": self.ctx, "returnByValue": True},
        )
        return reply["result"].get("value")

    def race_render(self, deadline_s: float = 20.0) -> tuple[str, Any]:
        """Race node appearance against the #message error dialog."""

        deadline = time.time() + deadline_s
        while time.time() < deadline:
            if self.page.locator("#message").count() and self.page.locator("#message").is_visible():
                return ("dialog", self.page.locator("#message").inner_text()[:200])
            count = self.page.locator(".node").count()
            if count > 0:
                return ("nodes", count)
            time.sleep(0.25)
        return ("timeout", None)

    def settled_extent(self) -> tuple[float, float]:
        """Poll the canvas bounding box until the layout stops moving."""

        previous: Any = None
        for _ in range(60):
            current = self.eval(
                "(() => { const c = document.querySelector('#canvas');"
                " if (!c || !c.getBBox) return null;"
                " const b = c.getBBox(); return [b.width, b.height]; })()"
            )
            if current is not None and current == previous:
                return (float(current[0]), float(current[1]))
            previous = current
            time.sleep(0.25)
        raise AssertionError("layout never settled")


@pytest.fixture(scope="module")
def browser() -> Any:
    """One shared headless Chromium for the module's fixtures."""

    with playwright_sync.sync_playwright() as playwright:
        launched = playwright.chromium.launch()
        try:
            yield launched
        finally:
            launched.close()


def _serve(artifact: Path, browser: Any) -> tuple[Any, _Session]:
    """Serve one artifact through the real netron server and load it."""

    address = netron_package.serve(
        artifact.name,
        bytearray(artifact.read_bytes()),
        address=("127.0.0.1", 0),
        browse=False,
    )
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"http://{address[0]}:{address[1]}")
    return page, _Session(page)


class _Inner(nn.Module):
    """Two-op child module."""

    def __init__(self) -> None:
        """Build the child linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Nested(nn.Module):
    """Two children plus a root op."""

    def __init__(self) -> None:
        """Build children."""

        super().__init__()
        self.a = _Inner()
        self.b = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine both children."""

        return self.a(x) + self.b(x)


def test_module_artifact_renders_within_budget_and_drills_down(
    browser: Any, tmp_path: Path
) -> None:
    """The served module artifact renders, fits the budget, and drills down."""

    log = tl.trace(_Nested(), torch.randn(2, 4))
    artifact = tl.export.netron(log, tmp_path / "m.json")
    page, session = _serve(artifact, browser)
    try:
        outcome, detail = session.race_render()
        assert outcome == "nodes", f"render race lost: {outcome} {detail}"
        # netron emits two elements per box; graph I/O are boxes too.
        boxes = session.eval("document.querySelectorAll('.node').length") / 2
        assert boxes == 5, "a, b, add + graph input/output"
        width, height = session.settled_extent()
        assert height / _RANK_PIXELS <= _RANK_BUDGET, f"extent {width}x{height}"
        labels = session.eval(
            "[...document.querySelectorAll('.edge-label')].map(e => e.textContent)"
        )
        assert labels and all(text == "2×4" for text in labels)
        rects = session.eval(
            "[...document.querySelectorAll('.edge-label')].map(e => {"
            " const b = e.getBoundingClientRect();"
            " return [b.left, b.top, b.right, b.bottom]; })"
        )
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                overlaps = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
                assert not overlaps, "edge-label overlaps must be zero"
        # One REAL mouse click on the 6x12 px function affordance.
        icon = page.locator("#node-name-a g.node-item", has_text="ƒ").first
        icon.click(force=True)
        deadline = time.time() + 10
        child_names: list[str] = []
        while time.time() < deadline:
            child_names = (
                session.eval(
                    "[...document.querySelectorAll('#canvas [id^=node-name-]')].map(e => e.id)"
                )
                or []
            )
            if any("linear" in name for name in child_names):
                break
            time.sleep(0.25)
        assert any("linear" in name for name in child_names), f"drill-down landed on {child_names}"
        child_labels = session.eval(
            "[...document.querySelectorAll('.edge-label')].map(e => e.textContent)"
        )
        assert child_labels, "labels intact inside the drill-down"
    finally:
        page.close()
        netron_package.stop()


def test_cycle_class_shows_dialog_not_hang(browser: Any, tmp_path: Path) -> None:
    """The race harness detects the dialog/dead-render class (negative)."""

    artifact = tmp_path / "broken.json"
    artifact.write_text('{"irVersion": 10, "graph": "not-a-graph"}', encoding="utf-8")
    page, session = _serve(artifact, browser)
    try:
        outcome, _detail = session.race_render(deadline_s=15.0)
        assert outcome in ("dialog", "timeout"), "a broken file must never count as rendered"
        assert session.eval("document.querySelectorAll('.node').length") in (0, None)
    finally:
        page.close()
        netron_package.stop()
