"""Compile all public Python fences and execute the corrected flagship workflows."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

PYTHON_FENCE_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


class _RunnableDocsModel(nn.Module):
    """Small model used to execute the documented runnable workflow."""

    def __init__(self) -> None:
        """Initialize one deterministic linear layer."""

        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a ReLU-transformed projection.

        Parameters
        ----------
        value:
            Input batch.

        Returns
        -------
        torch.Tensor
            Projected batch.
        """

        return torch.relu(self.linear(value))


def _repo_root() -> Path:
    """Return the repository root.

    Returns
    -------
    Path
        Absolute repository root.
    """

    return Path(__file__).resolve().parents[1]


def _public_python_fences() -> list[tuple[Path, int, str]]:
    """Collect Python fences from every public documentation page.

    Returns
    -------
    list[tuple[Path, int, str]]
        Page, one-based fence index, and source text.
    """

    root = _repo_root()
    pages = (
        root / "README.md",
        root / "CLAUDE.md",
        root / "AGENTS.md",
        *sorted((root / "docs").rglob("*.md")),
    )
    fences: list[tuple[Path, int, str]] = []
    for page in pages:
        for index, match in enumerate(
            PYTHON_FENCE_RE.finditer(page.read_text(encoding="utf-8")), start=1
        ):
            fences.append((page, index, match.group("code")))
    return fences


def _torchlens_imports(tree: ast.AST) -> list[tuple[str, str | None]]:
    """Extract TorchLens modules and imported attributes from one syntax tree.

    Parameters
    ----------
    tree:
        Parsed documentation snippet.

    Returns
    -------
    list[tuple[str, str | None]]
        Module and optional imported attribute pairs.
    """

    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, None) for alias in node.names if alias.name.startswith("torchlens")
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith("torchlens"):
                imports.extend((node.module, alias.name) for alias in node.names)
    return imports


@pytest.mark.smoke
def test_every_public_python_fence_compiles_and_torchlens_imports_resolve() -> None:
    """Cover every public Python fence without swallowing misspelled TorchLens imports."""

    root = _repo_root()
    for page, index, code in _public_python_fences():
        source_name = f"{page.relative_to(root)}:python-{index}"
        tree = ast.parse(code, filename=source_name)
        compile(tree, source_name, "exec")
        for module_name, attribute in _torchlens_imports(tree):
            module = importlib.import_module(module_name)
            if attribute is not None:
                assert hasattr(module, attribute), f"{source_name}: {module_name}.{attribute}"


@pytest.mark.smoke_cells(
    'test_performance_python_fence_runs[3-import torch\\nfrom torch import nn\\nimport torchlens as tl\\nfrom torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch\\n\\n\\nmodel = nn.ReLU()\\nx = torch.randn(4)\\nwrap_torch(escape_detector="shadow")\\ntrace = tl.trace(model, x)\\nprint(trace.escape_detector_event_count, trace.escape_detector_callback_ns)\\nunwrap_torch()]',
    'test_performance_python_fence_runs[8-import torch\\nfrom torch import nn\\nimport torchlens as tl\\n\\n\\nmodel = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()\\nx = torch.randn(2, 4)\\n\\npatched = tl.trace(\\n    model,\\n    x,\\n    save=tl.func("relu"),\\n    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),\\n)\\n\\nassert torch.count_nonzero(patched.find_sites(tl.func("relu")).first().out) == 0]',
)
@pytest.mark.parametrize(
    ("block_index", "code"),
    [
        (index, match.group("code"))
        for index, match in enumerate(
            PYTHON_FENCE_RE.finditer(
                (_repo_root() / "docs/performance.md").read_text(encoding="utf-8")
            ),
            start=1,
        )
    ],
)
def test_performance_python_fence_runs(block_index: int, code: str, tmp_path: Path) -> None:
    """Execute every performance-guide example without broad import skips.

    Parameters
    ----------
    block_index:
        One-based fence index used in the synthetic filename.
    code:
        Python fence source.
    tmp_path:
        Temporary output directory for disk-backed examples.
    """

    namespace: dict[str, Any] = {
        "__file__": f"docs/performance.md:python-{block_index}",
        "__name__": f"docs_performance_{block_index}",
        "DOCS_TMPDIR": str(tmp_path),
    }
    exec(compile(code, namespace["__file__"], "exec"), namespace)


def test_root_agent_runnable_example_runs(tmp_path: Path) -> None:
    """Execute the documented capture-save-load-run ordering and prerequisite."""

    torch.manual_seed(7)
    model = _RunnableDocsModel().eval()
    inputs = torch.randn(2, 4)
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            layers_to_save="all",
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    artifact = tmp_path / "architecture.tlspec"
    tl.save(trace, artifact, level="runnable", include_weights=True)
    loaded = tl.load(artifact)
    result = loaded.run(inputs=inputs, seed=42, on_divergence="raise")

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_root_agent_receptive_gradient_and_overlay_run() -> None:
    """Execute gradient and overlay calls on the documented armed Trace."""

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(1.0)
    inputs = torch.ones(1, 1, 5, 5, requires_grad=True)
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    target = trace.find_sites(tl.func("conv2d")).first()
    unit = target.receptive_field.center_unit(batch_index=0)
    gradient = target.receptive_field.gradient(unit, retain_graph=True)
    overlay = target.receptive_field.show(unit, gradient=True)

    assert any(result.support_mask.any() for result in gradient.values())
    assert overlay.size == (5, 5)
