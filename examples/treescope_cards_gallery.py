"""Render the treescope-cards browser packet (F16 B7 gallery).

Writes card fragments and offline reports for the reachable fixtures into
one output directory with an ``index.html``, for the D03 human /
naive-evaluator browser signoff. Zero-network by default: config-built
fixtures always render; real-checkpoint fixtures (torchvision resnet18,
gpt2) render only when their weights are ALREADY cached locally -- the
gallery never downloads, and every skip is disclosed on the index page.

Usage::

    python examples/treescope_cards_gallery.py [out_dir]

Default output: ``/tmp/treescope_cards_gallery``.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchlens as tl  # noqa: E402 - path bootstrap above, examples idiom

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/treescope_cards_gallery")

#: Byte-stable reports for the gallery (regeneration-friendly).
_DETERMINISTIC = tl.export.ReportOptions(deterministic=True)


class _Recurrent(nn.Module):
    """Tiny weight-reused model: exercises pass-aware Layer cards."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three shared-weight steps then a head."""
        for _ in range(3):
            x = torch.relu(self.fc(x))
        return self.head(x)


class _Poisoned(nn.Module):
    """Mid-graph NaN injection: exercises the frontier report."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with a deliberate 0/0 between the two layers."""
        h = torch.relu(self.a(x))
        h = h / 0.0 * 0.0
        return self.b(h)


class _MidwayFailure(nn.Module):
    """Deliberate mid-forward failure: exercises the PartialTrace forms."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward that dies after one committed op."""
        torch.relu(self.a(x))
        raise RuntimeError("deliberate mid-forward failure (gallery fixture)")


def _write(name: str, fragment: str, entries: list[tuple[str, str]]) -> None:
    """Write one gallery artifact and record it on the index."""

    path = OUT_DIR / name
    path.write_text(fragment, encoding="utf-8")
    entries.append((name, name))


def _config_built_entries(entries: list[tuple[str, str]]) -> None:
    """Render the always-available config-built fixtures."""

    log = tl.trace(_Recurrent(), torch.randn(2, 8))
    _write("trace_card.html", log._repr_html_(), entries)
    _write("layer_card_multipass.html", log.layer_logs["relu_1_2"]._repr_html_(), entries)
    _write("op_card.html", log["relu_1_2:2"]._repr_html_(), entries)
    tl.export.html(log, OUT_DIR / "report_clean.html", options=_DETERMINISTIC)
    entries.append(("report_clean.html", "report_clean.html"))

    dirty = tl.trace(_Poisoned(), torch.randn(2, 8))
    tl.export.html(dirty, OUT_DIR / "report_frontier.html", options=_DETERMINISTIC)
    entries.append(("report_frontier.html", "report_frontier.html"))
    tl.export.html(
        dirty,
        OUT_DIR / "report_share_safe.html",
        options=tl.export.ReportOptions(share_safe=True, deterministic=True),
    )
    entries.append(("report_share_safe.html", "report_share_safe.html"))

    try:
        tl.trace(_MidwayFailure(), torch.randn(2, 8))
    except Exception as error:  # noqa: BLE001 - the failure IS the fixture
        partial = tl.partial.from_failed_capture(error)
        if partial is not None:
            _write("partial_card.html", partial._repr_html_(), entries)
            tl.export.html(partial, OUT_DIR / "report_partial.html", options=_DETERMINISTIC)
            entries.append(("report_partial.html", "report_partial.html"))


def _cached_checkpoint_entries(entries: list[tuple[str, str]], skips: list[str]) -> None:
    """Render real-checkpoint fixtures ONLY from a warm local cache."""

    try:
        import torchvision  # noqa: F401
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        log = tl.trace(model.eval(), torch.randn(1, 3, 224, 224))
        tl.export.html(log, OUT_DIR / "report_resnet18.html", options=_DETERMINISTIC)
        entries.append(("report_resnet18.html", "report_resnet18.html"))
    except Exception as error:  # noqa: BLE001 - cold cache / absent extra: skip, disclosed
        skips.append(f"resnet18: {type(error).__name__}: {error}")


def main() -> None:
    """Render the packet and write the index page."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str]] = []
    skips: list[str] = []
    _config_built_entries(entries)
    _cached_checkpoint_entries(entries, skips)

    items = "".join(
        f'<li><a href="{escape(href, quote=True)}">{escape(title)}</a></li>'
        for title, href in entries
    )
    skipped = "".join(f"<li>{escape(line)}</li>" for line in skips)
    index = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>TorchLens treescope-cards gallery</title></head><body>"
        "<h1>TorchLens treescope-cards gallery (F16)</h1>"
        f"<ul>{items}</ul>"
        + (f"<h2>skipped (no local cache; never downloads)</h2><ul>{skipped}</ul>" if skips else "")
        + "</body></html>"
    )
    (OUT_DIR / "index.html").write_text(index, encoding="utf-8")
    print(f"gallery written: {OUT_DIR} ({len(entries)} artifacts, {len(skips)} skips)")


if __name__ == "__main__":
    main()
