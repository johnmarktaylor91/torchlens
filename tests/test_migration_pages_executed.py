"""Execute every Python example on the coming-from and migration pages listed here.

Each fence on these pages starts with ``# migration-test: tool=<tool> expected=<literal>``
and assigns ``RESULT``; the test runs the fence in a fresh namespace and compares
``RESULT`` with the declared literal. A ``torchlens`` fence never skips: it needs only
the core test environment (torch, torchvision, transformers). A fence for another tool
skips when that tool is not installed, so the third-party side of a page runs wherever
the tool is present. A fence without a header fails the lint below, so no example on
these pages can go unexecuted.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import linecache
import re
from pathlib import Path
from typing import Any

import pytest
import torch

#: Page (under ``docs/``) -> tools its fences may name.
PAGES: dict[str, frozenset[str]] = {
    "migration/from_torchview.md": frozenset({"torchview", "torchlens"}),
    "migration/from_torchviz_hiddenlayer.md": frozenset({"torchviz", "torchlens"}),
    "migration/from_bertviz_ecco.md": frozenset({"torchlens"}),
    "migration/from_transformers_interpret.md": frozenset({"transformers_interpret", "torchlens"}),
    "migration/from_torchvision_feature_extraction.md": frozenset({"torchvision", "torchlens"}),
    "migration/from_intermediate_layer_getter.md": frozenset({"torchvision", "torchlens"}),
    "migration/from_timm_features_only.md": frozenset({"timm", "timm_torchlens"}),
    "migration/from_surgeon_pytorch.md": frozenset({"surgeon_pytorch", "torchlens"}),
    "migration/multi_gpu_extraction.md": frozenset({"torchlens"}),
    "migration/from_hf_kwargs.md": frozenset({"torchlens"}),
    "migration/from_model_zoos.md": frozenset({"torchlens"}),
    "migration/penzai.md": frozenset({"torchlens"}),
    "migration/pyvene.md": frozenset({"torchlens"}),
}

#: Tool key -> modules that must import for the fence to run.
TOOL_MODULES: dict[str, tuple[str, ...]] = {
    "torchlens": (),
    "torchvision": ("torchvision",),
    "torchview": ("torchview",),
    "torchviz": ("torchviz",),
    "transformers_interpret": ("transformers_interpret",),
    "timm": ("timm",),
    "timm_torchlens": ("timm",),
    "surgeon_pytorch": ("surgeon_pytorch",),
}

HEADER_RE = re.compile(r"# migration-test: tool=(?P<tool>[a-z0-9_]+) expected=(?P<expected>.+)")
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)
DOCS = Path(__file__).resolve().parents[1] / "docs"


def _fences(page: str) -> list[str]:
    """Return the Python fence bodies of one page, in order.

    Parameters
    ----------
    page:
        Page path relative to ``docs``.

    Returns
    -------
    list[str]
        Fence bodies.
    """

    return [m.group("code") for m in BLOCK_RE.finditer((DOCS / page).read_text("utf-8"))]


def _examples() -> list[Any]:
    """Collect one pytest param per headed fence.

    Returns
    -------
    list[Any]
        ``pytest.param(page, index, tool, expected, code)`` entries.
    """

    params: list[Any] = []
    for page in PAGES:
        for index, code in enumerate(_fences(page), start=1):
            header = HEADER_RE.match(code.splitlines()[0])
            if header is None:
                continue
            expected = ast.literal_eval(header.group("expected"))
            params.append(
                pytest.param(
                    page, index, header.group("tool"), expected, code, id=f"{page}:{index}"
                )
            )
    return params


@pytest.mark.parametrize(("page", "index", "tool", "expected", "code"), _examples())
def test_migration_page_example_runs(
    page: str, index: int, tool: str, expected: Any, code: str
) -> None:
    """Run one fence and compare its ``RESULT`` with the declared value."""

    for module in TOOL_MODULES[tool]:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{page} fence {index} needs {module!r}")
    filename = f"{page}:python-block-{index}"
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    namespace: dict[str, Any] = {"__file__": filename, "__name__": f"migration_page_{index}"}
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        exec(compile(code, filename, "exec"), namespace)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    assert namespace["RESULT"] == expected


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_fence_is_headed_with_a_known_tool(page: str) -> None:
    """Every fence on a listed page is executed: it carries a header naming a page tool."""

    fences = _fences(page)
    assert fences, f"{page} has no executed example"
    for index, code in enumerate(fences, start=1):
        header = HEADER_RE.match(code.splitlines()[0])
        assert header is not None, f"{page} fence {index} has no migration-test header"
        assert header.group("tool") in PAGES[page], (page, index, header.group("tool"))
    tools = {HEADER_RE.match(code.splitlines()[0]).group("tool") for code in fences}  # type: ignore[union-attr]
    assert tools & {"torchlens", "timm_torchlens"}, f"{page} has no TorchLens example"


def test_saelens_page_names_the_splice_result_latents_fields() -> None:
    """The SAELens page's read-latents row names SpliceResult fields that exist."""

    from torchlens.bridge.sae import SpliceResult

    fields = {field.name for field in dataclasses.fields(SpliceResult)}
    row = next(
        line
        for line in (DOCS / "migration/saelens.md").read_text("utf-8").splitlines()
        if line.startswith("| Read feature activations")
    )
    named = set(re.findall(r"`SpliceResult\.([a-z_]+)`", row))
    assert named == {"latents", "edited_latents"}, named
    assert named <= fields, named - fields
    assert "kept by default" in row and "keep_latents=False" in row


def test_gaussian_rdm_migration_note_matches_the_formula() -> None:
    """MIGRATIONS.md's gaussian example values are what rdm(metric='gaussian') returns."""

    import numpy as np

    from torchlens.repgeom import rdm

    text = (DOCS.parent / "MIGRATIONS.md").read_text("utf-8")
    assert "0.148, 0.765 and 0.474 and are now 0.275, 0.945 and\n0.724" in text
    features = np.array([[0.0], [1.0], [3.0]])
    new = np.asarray(rdm(features, metric="gaussian"))
    squared = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 4.0], [9.0, 4.0, 0.0]])
    old = 1.0 - np.exp(-squared / (2.0 * squared.mean()))
    np.testing.assert_allclose(new, 1.0 - np.exp(-squared / squared.mean()), rtol=0, atol=1e-12)
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert [round(float(new[i, j]), 3) for i, j in pairs] == [0.275, 0.945, 0.724]
    assert [round(float(old[i, j]), 3) for i, j in pairs] == [0.148, 0.765, 0.474]


def test_v216_erratum_names_the_live_floor_refusal() -> None:
    """The 2.16 erratum retracts 'permanently readable' and names the real refusal."""

    from torchlens._io import format_contract

    text = (DOCS.parent / "MIGRATIONS.md").read_text("utf-8")
    compat = text.split("## Visualization Collapse Engine", 1)[0]
    assert compat.count("permanently readable") == 1  # only inside the quoted retraction
    assert '"are permanently readable". That sentence is retracted.' in compat
    assert f"`tlspec_version`\n{format_contract.MIN_TLSPEC_VERSION}" in compat
    assert "artifact_version_below_floor" in compat
    assert "torchlens==2.17.0" in compat


def test_lovely_sd_divergence_note_is_true() -> None:
    """The stats-line page's population-versus-sample sd numbers hold on arange(8)."""

    text = (DOCS / "migration/lovely_stats_line.md").read_text("utf-8")
    assert "**2.291 (TorchLens) vs 2.449 (lovely)**" in text
    values = torch.arange(8, dtype=torch.float64)
    assert f"{values.std(unbiased=False).item():.3f}" == "2.291"
    assert f"{values.std(unbiased=True).item():.3f}" == "2.449"
