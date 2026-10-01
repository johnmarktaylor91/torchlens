"""Visualization identity oracle (separate from the surface byte oracle).

Forward and rolled DOT sources are byte-stable across processes (verified by
the two-process control below, run BEFORE the goldens were frozen), so they
are compared against full-source goldens. Backward DOT embeds ``id()``-derived
grad_fn node names, so per the locked cross-process lesson it is checked
IN-PROCESS ONLY: double-render equality plus structural invariants.
"""

from __future__ import annotations

import difflib
import functools
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_UPDATE_ENV = "TORCHLENS_UPDATE_GODOBJECT_VIZ_ORACLE"
_SEED = 20260812

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_SUBPROCESS_PROBE = """
import hashlib, tempfile, torch, warnings
from torch import nn
import sys
sys.path.insert(0, {repo_root!r})
warnings.filterwarnings("ignore")
import torchlens as tl

class VizCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = nn.Linear(2, 3)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))

torch.manual_seed({seed})
trace = tl.trace(VizCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))
with tempfile.TemporaryDirectory() as tmp:
    source = trace.draw(
        vis_save_only=True, vis_fileformat="svg", vis_outpath=tmp + "/g"
    )
print(hashlib.sha256(source.encode()).hexdigest())
"""


class VizCNN(nn.Module):
    """Deterministic conv model for the viz oracle."""

    def __init__(self) -> None:
        """Initialize conv and head layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv-relu-pool-linear."""

        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))


class VizRecurrent(nn.Module):
    """Three-step recurrent model exercising rolled-graph folding."""

    def __init__(self) -> None:
        """Initialize the shared cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared cell three times."""

        state = x
        for _ in range(3):
            state = torch.tanh(self.cell(state))
        return state


def _capture(model_key: str) -> tl.Trace:
    """Capture one deterministic trace for a viz model key."""

    torch.manual_seed(_SEED)
    if model_key == "viz_cnn":
        return tl.trace(VizCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))
    return tl.trace(VizRecurrent(), torch.linspace(-1.0, 1.0, 4).reshape(1, 4))


def _dot(trace: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render a trace and return its DOT source."""

    return trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


#: The ``graphviz`` python package is the direct DOT emitter for these
#: goldens (quoting included), so the family fingerprint extends with its
#: version (b10 R78 round-3); ``ENV-graphviz`` in the goldens dir records the
#: canonical emitter version.
_EMITTER_PACKAGES = ("graphviz",)


def _assert_matches_golden(actual: str, golden_name: str) -> None:
    """Compare DOT text to a committed golden, with the update escape hatch."""

    from _oracle_env import (
        flag_armed,
        require_env_golden,
        require_update_reason,
        resolve_env_golden,
        write_provenance,
    )

    actual = actual.rstrip("\n")
    # No wrap-state guard here: generation runs in an isolated subprocess
    # with a clean interpreter (SF-53 is closed structurally for this family).
    if flag_armed(os.environ, _UPDATE_ENV):
        # Reason BEFORE bytes (b10 R78 round-4): the write used to precede
        # require_update_reason, so a reasonless update run FAILED but had
        # already rebaselined the committed goldens in the working tree.
        reason = require_update_reason(_UPDATE_ENV)
        golden_path, _ = resolve_env_golden(_GOLDEN_DIR, golden_name, _EMITTER_PACKAGES)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        write_provenance(golden_path.parent, "tests/godobject_oracle viz", _UPDATE_ENV, reason)
        pytest.skip(f"updated golden {golden_name}; re-run without {_UPDATE_ENV} to verify")
    golden_path = require_env_golden(
        _GOLDEN_DIR, golden_name, _UPDATE_ENV, extra_packages=_EMITTER_PACKAGES
    )
    if not golden_path.exists():
        reason = require_update_reason(_UPDATE_ENV)
        golden_path.write_text(actual + "\n")
        write_provenance(golden_path.parent, "tests/godobject_oracle viz", _UPDATE_ENV, reason)
        pytest.skip(f"recorded first-run viz golden for this environment: {golden_path}")
    expected = golden_path.read_text().rstrip("\n")
    if actual != expected:
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="golden",
                    tofile="actual",
                    lineterm="",
                )
            )[:60]
        )
        raise AssertionError(f"DOT diverged from {golden_name}:\n{diff}")


@functools.lru_cache(maxsize=1)
def _worker_renders() -> dict[str, str]:
    """Generate every forward-DOT golden in one isolated subprocess.

    The worker constructs both models before its first capture, so no model
    ctor runs on wrapped torch and the goldens cannot silently freeze
    session-dependent wrap-state artifacts (b10 R78-1).
    """

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    python_paths = (str(Path(_REPO_ROOT) / "tests"), _REPO_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, *((existing_pythonpath,) if existing_pythonpath else ()))
    )
    completed = subprocess.run(
        [sys.executable, "-m", "godobject_oracle._worker"],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("viz worker produced no JSON")
    renders = json.loads(lines[-1])
    assert isinstance(renders, dict)
    return renders


@pytest.mark.smoke
@pytest.mark.parametrize("model_key", ("viz_cnn", "viz_recurrent"))
@pytest.mark.parametrize("vis_mode", ("unrolled", "rolled"))
def test_forward_dot_matches_golden(model_key: str, vis_mode: str) -> None:
    """Forward/rolled DOT source is byte-identical to the frozen golden."""

    source = _worker_renders()[f"viz_{model_key}_{vis_mode}.gv"]
    _assert_matches_golden(source, f"viz_{model_key}_{vis_mode}.gv")


@pytest.mark.heavy
def test_forward_dot_two_process_control() -> None:
    """Baseline-vs-baseline: two FRESH processes emit the same DOT hash.

    This is the control that legitimizes the cross-process goldens above. If
    it fails, the DOT stream has a process-dependent leak and the goldens
    must not be trusted (fix the normalization, do not refresh goldens).
    """

    probe = _SUBPROCESS_PROBE.format(repo_root=_REPO_ROOT, seed=_SEED)
    hashes = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        hashes.append(result.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1]


@pytest.mark.smoke
def test_backward_dot_in_process_stable(tmp_path: Path) -> None:
    """Backward DOT: double-render byte equality + structural invariants.

    NO cross-process golden here: grad_fn node names embed ``id()`` values
    (locked lesson: cross-process backward-DOT comparison is INVALID).
    """

    torch.manual_seed(_SEED)
    x = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4).requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            VizCNN(),
            x,
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
    loss = trace[trace.ops.keys()[-1]].out.sum()
    trace.log_backward(loss)

    first = trace.draw_backward(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "b1"),
    )
    second = trace.draw_backward(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "b2"),
    )
    assert first == second

    node_count = sum(1 for line in first.splitlines() if " [" in line and "->" not in line)
    edge_count = sum(1 for line in first.splitlines() if "->" in line)
    # 17 nodes: 16 graph nodes + the one in-frame backward-key table node
    # (vizmech item 17; AUTO renders the key, one row per painted style).
    # (17, 12) -> (16, 11) with the W051-CAPT3 redundant-output-copy removal:
    # the capture-side clone of the model output contributed one grad_fn node
    # and its edge; the retained op payload is now the single copy.
    assert (node_count, edge_count) == (16, 11), (
        f"backward graph structure changed: {node_count} nodes, {edge_count} edges"
    )
