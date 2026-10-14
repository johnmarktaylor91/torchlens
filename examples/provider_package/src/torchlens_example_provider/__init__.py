"""Example TorchLens provider: one export target behind an entry point.

The ``torchlens.export_targets`` entry point in ``pyproject.toml`` names
:func:`export_target`, a zero-arg factory. TorchLens never imports this
module during discovery; ``torchlens.ecosystem.plugins.activate`` imports it,
calls the factory once and registers the returned row through the export
door. The module needs nothing from TorchLens at import time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Name the export target registers under.
TARGET_NAME = "example_provider_summary"

#: Format label written into every summary.
SUMMARY_FORMAT = "torchlens-example-provider/summary/1"


def summarize(trace: Any, path: str | Path | None = None) -> dict[str, Any]:
    """Summarize a trace as plain JSON-ready data, optionally writing it.

    Parameters
    ----------
    trace:
        A ``torchlens`` Trace.
    path:
        Optional file to write the summary to as JSON.

    Returns
    -------
    dict[str, Any]
        The format label, the number of layers and their labels in order.
    """

    labels = [str(label) for label in trace.layer_labels]
    summary = {"format": SUMMARY_FORMAT, "num_layers": len(labels), "layer_labels": labels}
    if path is not None:
        Path(path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def export_target() -> dict[str, Any]:
    """Zero-arg factory returning the frozen export-target row.

    Returns
    -------
    dict[str, Any]
        ``name``, ``fn`` and ``tier``: the keys the export door requires.
    """

    return {"name": TARGET_NAME, "fn": summarize, "tier": "present"}
