"""Repeatable facet-coverage audit producing recipe PROPOSAL inputs.

Runs a roster of models through TorchLens, computes
``torchlens._semantic.facet_coverage`` for each, and writes ONE review document
aggregating classified/unclassified modules and typed facet absences. The
output is evidence for the maintenance sweep prompt
(``tools/facet_maintenance/DISCOVER_FACETS.md``); this script NEVER registers,
edits, or merges facet recipes, and its output directory must be treated as
proposals for human review.

Usage::

    python tools/facet_maintenance/run_facet_audit.py --out /tmp/facet-proposals
    python tools/facet_maintenance/run_facet_audit.py --spec my_models.py

A ``--spec`` module exposes ``iter_models()`` yielding
``(name, model, input_args)`` tuples; the built-in roster is a small synthetic
demonstration set so the pipeline stays runnable offline.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BANNER = (
    "# Facet coverage audit -- PROPOSALS FOR REVIEW ONLY\n\n"
    "This document is machine-generated evidence for the facet-maintenance sweep\n"
    "(`tools/facet_maintenance/DISCOVER_FACETS.md`). Nothing in it has been\n"
    "applied: facet recipes are NEVER auto-merged, because a wrong facet label is\n"
    "a confidently mislabelled part of someone's model. Review each candidate\n"
    "against the real module source before writing a recipe.\n"
)


def _builtin_roster() -> Iterator[tuple[str, Any, Any]]:
    """Yield the offline demonstration roster.

    Yields
    ------
    tuple[str, Any, Any]
        ``(name, model, input_args)`` entries.
    """

    import torch
    from torch import nn

    class Block(nn.Module):
        """Tiny pre-LN transformer block."""

        def __init__(self, d: int = 16) -> None:
            """Initialize children."""

            super().__init__()
            self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run pre-norm attention and MLP residual updates."""

            normed = self.ln1(x)
            attended, _weights = self.attn(normed, normed, normed)
            x = x + attended
            return x + self.mlp(self.ln2(x))

    class TinyLM(nn.Module):
        """Tiny GPT-shaped LM exercising the lm_head recipe."""

        def __init__(self) -> None:
            """Initialize the stack and head."""

            super().__init__()
            self.transformer = nn.ModuleDict(
                {
                    "h": nn.ModuleList([Block(), Block()]),
                    "ln_f": nn.LayerNorm(16),
                }
            )
            self.lm_head = nn.Linear(16, 32, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return logits."""

            for block in self.transformer["h"]:
                x = block(x)
            return self.lm_head(self.transformer["ln_f"](x))

    class UnclassifiedMixer(nn.Module):
        """Deliberately recipe-less module family, exercising candidate reporting."""

        def __init__(self, d: int = 16) -> None:
            """Initialize the gate projection."""

            super().__init__()
            self.gate = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Mix tokens with a data-dependent gate."""

            return x * torch.sigmoid(self.gate(x.flip(-2)))

    class MixerNet(nn.Module):
        """Stack of unclassified mixers."""

        def __init__(self) -> None:
            """Initialize mixers."""

            super().__init__()
            self.mixers = nn.ModuleList([UnclassifiedMixer(), UnclassifiedMixer()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the mixers."""

            for mixer in self.mixers:
                x = mixer(x)
            return x

    torch.manual_seed(0)
    yield "tiny_lm", TinyLM().eval(), torch.randn(2, 4, 16)
    yield "mixer_net", MixerNet().eval(), torch.randn(2, 4, 16)


def _load_spec_roster(spec_path: Path) -> Iterable[tuple[str, Any, Any]]:
    """Load ``iter_models()`` from a user spec module.

    Parameters
    ----------
    spec_path:
        Path to a Python module exposing ``iter_models()``.

    Returns
    -------
    Iterable[tuple[str, Any, Any]]
        The spec's roster.
    """

    module_spec = importlib.util.spec_from_file_location("facet_audit_spec", spec_path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"Cannot import spec module {spec_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    iter_models = getattr(module, "iter_models", None)
    if not callable(iter_models):
        raise SystemExit(f"Spec module {spec_path} does not define iter_models()")
    return iter_models()


def run_audit(roster: Iterable[tuple[str, Any, Any]], out_dir: Path) -> Path:
    """Trace every roster entry and write the aggregated proposal document.

    Parameters
    ----------
    roster:
        ``(name, model, input_args)`` entries.
    out_dir:
        Output directory for the proposal document.

    Returns
    -------
    Path
        Path of the written document.
    """

    import torchlens as tl
    from torchlens._semantic import facet_coverage

    sections: list[str] = [_BANNER]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(f"Generated: {stamp}\n")
    for name, model, input_args in roster:
        sections.append(f"\n## {name}\n")
        try:
            log = tl.trace(
                model,
                input_args,
                capture=tl.options.CaptureOptions(layers_to_save="all"),
            )
            report = facet_coverage(log)
            sections.append(report.to_markdown())
            log.cleanup()
        except Exception:  # noqa: BLE001 - an unaudited model must not sink the sweep.
            sections.append(
                "AUDIT FAILED for this entry (disclosed, not skipped silently):\n\n"
                "```\n" + traceback.format_exc() + "```"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "facet_coverage_audit.md"
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("facet-proposals"),
        help="Output directory for the proposal document (default: ./facet-proposals)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Python module exposing iter_models() -> (name, model, input_args)",
    )
    args = parser.parse_args(argv)
    roster = _load_spec_roster(args.spec) if args.spec else _builtin_roster()
    out_path = run_audit(roster, args.out)
    print(f"Wrote {out_path}")
    print("PROPOSALS FOR REVIEW ONLY -- facet recipes are never auto-merged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
