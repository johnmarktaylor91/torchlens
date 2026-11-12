"""Backend tutorial fences execute: each backend page's Tutorial runs top to bottom.

Every registered backend's ``docs/backends/<name>.md`` carries a ``## Tutorial`` section;
its python fences run in one namespace under the backend's framework marker, when the
backend's runtime is available here. The docs-versus-code lockstep this file used to hold
(the pages' sections, the backend table in ``docs/backends.md`` and the can and cannot
tables against the shipped scorecards) reads no runtime and moved to
``scripts/checks/backend_docs_lockstep.py``, run by ``scripts/checks/run_all.py``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

import torchlens.backends as tl_backends

_REPO = Path(__file__).resolve().parents[1]
_PAGES = _REPO / "docs" / "backends"

_TUTORIAL = "## Tutorial"

#: The pytest marker each backend's framework-dependent tests carry (``pyproject.toml``).
_FRAMEWORK_MARKERS: dict[str, str] = {
    "tf": "tf_backend",
    "jax": "backend_jax",
    "mlx": "backend_mlx",
    "paddle": "backend_paddle",
    "tinygrad": "backend_tinygrad",
}

_BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _registered() -> tuple[str, ...]:
    """Names of the registered backends that carry a protocol, sorted."""

    return tuple(sorted(spec.name for spec in tl_backends.specs() if spec.protocol is not None))


_BACKENDS = _registered()


def _section(text: str, heading: str) -> str:
    """The body of one ``## `` section, up to the next ``## `` heading.

    Parameters
    ----------
    text:
        Markdown page text.
    heading:
        The exact heading line, such as ``"## Tutorial"``.

    Returns
    -------
    str
        The section body; empty when the heading is absent.
    """

    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


def _page_text(backend: str) -> str:
    """The backend page's text; fails when the page is missing."""

    page = _PAGES / f"{backend}.md"
    assert page.is_file(), f"docs/backends/{backend}.md is missing"
    return page.read_text(encoding="utf-8")


def _tutorial_params() -> list[object]:
    """One parameter per backend, carrying its framework marker when it has one."""

    params: list[object] = []
    for backend in _BACKENDS:
        marker = _FRAMEWORK_MARKERS.get(backend)
        marks = [getattr(pytest.mark, marker)] if marker else []
        params.append(pytest.param(backend, marks=marks, id=backend))
    return params


def _skip_reason(backend: str) -> str | None:
    """Why the tutorial cannot run here, or ``None`` when it can."""

    spec = tl_backends.spec(backend)
    assert spec.protocol is not None
    code = spec.protocol().runtime_available()
    if code is not None:
        return f"{backend} runtime unavailable here ({code})"
    if (
        backend == "mlx"
        and sys.platform != "darwin"
        and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1"
    ):
        return "MLX off macOS runs only with TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1"
    return None


@pytest.mark.parametrize("backend", _tutorial_params())
def test_tutorial_fences_execute(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Tutorial section's fences run top to bottom in one namespace."""

    reason = _skip_reason(backend)
    if reason is not None:
        pytest.skip(reason)
    page = _PAGES / f"{backend}.md"
    blocks = [m.group("code") for m in _BLOCK_RE.finditer(_section(_page_text(backend), _TUTORIAL))]
    assert blocks, f"docs/backends/{backend}.md: no tutorial fence"
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": f"__{backend}_backend_tutorial__"}
    for block in blocks:
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102 - documentation code
