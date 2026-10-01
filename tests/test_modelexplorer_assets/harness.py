"""Shared helpers for the Model Explorer contract harness tests (F15).

Runs the pinned real ``dist/worker.js`` graph processor under Node and
returns its per-graph stats. The worker is the panel's stand-in for the
browser: three of the four data-loss bugs the design round found are
invisible to a dataclass parse and visible only here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ASSETS_DIR = Path(__file__).resolve().parent
WORKER_JS = ASSETS_DIR / "worker.js"
RUN_WORKER = ASSETS_DIR / "run_worker.cjs"


def node_available() -> bool:
    """Return whether a Node runtime is on PATH for the worker oracle."""

    return shutil.which("node") is not None


def run_worker_oracle(
    payload: dict[str, Any], *, keep_single_child: bool = False, timeout: float = 60.0
) -> list[dict[str, Any]]:
    """Run every graph of one collection payload through the real worker.

    Parameters
    ----------
    payload:
        A ``{label, graphs}`` collection dict.
    keep_single_child:
        Forward ``keepLayersWithASingleChild=true`` (the ``faithful_layers``
        viewer profile) instead of the vendor default (pruned).
    timeout:
        Subprocess timeout in seconds.

    Returns
    -------
    list[dict[str, Any]]
        One row per graph: ``{collection, graphId, ms, err, stats}``.
    """

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="tl-me-oracle-", delete=False
    ) as handle:
        json.dump(payload, handle)
        payload_path = handle.name
    command = ["node", str(RUN_WORKER), str(WORKER_JS), payload_path]
    if keep_single_child:
        command.append("--keep-single-child")
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"worker oracle runner failed rc={completed.returncode}: {completed.stderr[:2000]}"
        )
    return json.loads(completed.stdout)


def assert_no_silent_node_loss(payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Assert the worker processed EVERY declared node and edge of each graph.

    Model Explorer's worker silently drops duplicate-id nodes (with their
    edges, namespace, and overlay binding) and silently drops dangling
    edges; equality against the declared counts is the only thing that
    catches it.
    """

    declared = {
        str(graph.get("id", "")): (
            len(graph.get("nodes") or []),
            sum(len(node.get("incomingEdges") or []) for node in graph.get("nodes") or []),
        )
        for graph in payload.get("graphs") or []
    }
    for row in rows:
        assert row["err"] is None, f"worker error on {row['graphId']}: {row['err']}"
        stats = row["stats"]
        assert stats is not None, f"worker produced no modelGraph for {row['graphId']}"
        assert stats["layoutGraphError"] is None
        expected_nodes, expected_edges = declared[row["graphId"]]
        assert stats["opNodes"] == expected_nodes, (
            f"SILENT NODE LOSS on {row['graphId']}: declared {expected_nodes}, "
            f"processed {stats['opNodes']}"
        )
        assert stats["totalIncomingEdges"] == expected_edges, (
            f"SILENT EDGE LOSS on {row['graphId']}: declared {expected_edges}, "
            f"processed {stats['totalIncomingEdges']}"
        )
