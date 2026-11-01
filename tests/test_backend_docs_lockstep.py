"""Backend docs lockstep: the backends guide and the per-backend pages match the code.

Graduation item G5 and acceptance criterion A26 hold the docs to measured facts:

- every registered backend has ``docs/backends/<name>.md`` with a ``## Tutorial`` and a
  ``## Can and cannot`` section;
- the maturity table in ``docs/backends.md`` lists exactly the registered backends, each with
  its registered ``maturity``, its protocol's ``materialization`` and a link to its page;
- each page's can and cannot table equals the capability summary of the backend's shipped
  scorecard record: every ``BackendCapabilities`` flag the summary reads appears in exactly one
  row, spelled exactly in that row's first cell, and the row's state cell reads ``supported``,
  ``unsupported`` or ``not yet proven`` (``untested`` is accepted as the summary's own word).
  Rows that name no flag (extra refusals, known limits) are free text and are not compared;
- each page's tutorial fences execute, under the backend's framework marker, when the
  backend's runtime is available here.

The shipped scorecard is the measured source (regenerated and checked by the backend's pinned
CI leg); a page whose table disagrees with it is stale, and so is a backend that ships no record.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

import torchlens.backends as tl_backends
from torchlens.conformance._scorecard import shipped_scorecard

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_GUIDE = _DOCS / "backends.md"
_PAGES = _DOCS / "backends"

_GUIDE_SECTION = "## Backend protocol and maturity"
_TUTORIAL = "## Tutorial"
_CAN_AND_CANNOT = "## Can and cannot"

#: The page's word for each capability-summary state.
_STATE_WORDS: dict[str, str] = {
    "supported": "supported",
    "unsupported": "unsupported",
    "not yet proven": "untested",
    "untested": "untested",
}

#: The pytest marker each backend's framework-dependent tests carry (``pyproject.toml``).
_FRAMEWORK_MARKERS: dict[str, str] = {
    "tf": "tf_backend",
    "jax": "backend_jax",
    "mlx": "backend_mlx",
    "paddle": "backend_paddle",
    "tinygrad": "backend_tinygrad",
}

_BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)
_ROW_RE = re.compile(r"^\|(?P<cells>.*)\|\s*$")
_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)]+)\)")


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


def _table_rows(section: str) -> list[list[str]]:
    """Data rows of every pipe table in ``section``, header and rule rows dropped.

    Parameters
    ----------
    section:
        Markdown text holding one or more tables.

    Returns
    -------
    list[list[str]]
        Each row's stripped cells.
    """

    rows: list[list[str]] = []
    previous_was_row = False
    for line in section.splitlines():
        match = _ROW_RE.match(line.strip())
        if match is None:
            previous_was_row = False
            continue
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        is_rule = all(set(cell) <= {"-", ":", " "} and cell for cell in cells)
        if not previous_was_row or is_rule:
            previous_was_row = True
            continue  # the header row, or the rule under it
        rows.append(cells)
    return rows


def _first_table_rows(section: str) -> list[list[str]]:
    """Data rows of the first pipe table in ``section`` only.

    Parameters
    ----------
    section:
        Markdown text that may hold several tables.

    Returns
    -------
    list[list[str]]
        The first table's data rows; empty when the section has no table.
    """

    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if _ROW_RE.match(line.strip())), None)
    if start is None:
        return []
    end = start
    while end < len(lines) and _ROW_RE.match(lines[end].strip()):
        end += 1
    return _table_rows("\n".join(lines[start:end]))


def _plain(cell: str) -> str:
    """A table cell with Markdown emphasis and code ticks removed, lower-cased."""

    return re.sub(r"[`*_]", " ", cell).strip().lower()


def _page_text(backend: str) -> str:
    """The backend page's text; fails when the page is missing."""

    page = _PAGES / f"{backend}.md"
    assert page.is_file(), f"docs/backends/{backend}.md is missing"
    return page.read_text(encoding="utf-8")


def test_registry_has_the_six_protocol_backends() -> None:
    """The parametrized backends are the six the protocol covers, torch included."""

    assert set(_BACKENDS) >= {"torch", "tf", "jax", "mlx", "paddle", "tinygrad"}, _BACKENDS


@pytest.mark.parametrize("backend", _BACKENDS)
def test_page_has_tutorial_and_can_and_cannot_sections(backend: str) -> None:
    """Every registered backend's page carries both sections G5 reads, with a runnable fence."""

    text = _page_text(backend)
    lines = text.splitlines()
    for heading in (_TUTORIAL, _CAN_AND_CANNOT):
        assert heading in lines, f"docs/backends/{backend}.md lacks {heading!r}"
    assert _BLOCK_RE.search(_section(text, _TUTORIAL)), (
        f"docs/backends/{backend}.md: the Tutorial section has no python fence"
    )
    assert _table_rows(_section(text, _CAN_AND_CANNOT)), (
        f"docs/backends/{backend}.md: the Can and cannot section has no table"
    )


def test_guide_maturity_table_matches_the_registry() -> None:
    """``docs/backends.md`` lists every registered backend with its maturity and page link."""

    rows = _first_table_rows(_section(_GUIDE.read_text(encoding="utf-8"), _GUIDE_SECTION))
    assert rows, f"docs/backends.md lacks the {_GUIDE_SECTION!r} table"
    documented: dict[str, tuple[str, str, str]] = {}
    for cells in rows:
        assert len(cells) == 4, f"maturity table row needs four cells: {cells}"
        name = cells[0].strip("` ")
        assert name not in documented, f"docs/backends.md lists {name} twice"
        link = _LINK_RE.search(cells[3])
        assert link is not None, f"docs/backends.md: {name} row has no page link"
        documented[name] = (_plain(cells[1]), _plain(cells[2]), link.group("target"))
    assert sorted(documented) == list(_BACKENDS), (
        f"docs/backends.md lists {sorted(documented)}; the registry has {list(_BACKENDS)}"
    )
    for name, (maturity, materialization, target) in documented.items():
        spec = tl_backends.spec(name)
        protocol = spec.protocol() if spec.protocol is not None else None
        assert maturity == spec.maturity, (
            f"docs/backends.md says {name} is {maturity!r}; the registry says {spec.maturity!r}"
        )
        assert protocol is not None
        assert materialization == protocol.materialization, (
            f"docs/backends.md says {name} is {materialization!r}; "
            f"its protocol says {protocol.materialization!r}"
        )
        assert target == f"backends/{name}.md", f"{name} links {target!r}"
        assert (_DOCS / target).is_file(), f"docs/backends.md links a missing page {target}"


def _documented_states(backend: str, flags: tuple[str, ...]) -> dict[str, list[str]]:
    """Each capability flag's state words as the page's can and cannot table spells them.

    Parameters
    ----------
    backend:
        Backend name.
    flags:
        Capability flags to look for, spelled exactly.

    Returns
    -------
    dict[str, list[str]]
        Flag to the plain state cell of every row whose first cell names it.
    """

    rows = _table_rows(_section(_page_text(backend), _CAN_AND_CANNOT))
    found: dict[str, list[str]] = {flag: [] for flag in flags}
    for cells in rows:
        if len(cells) < 2:
            continue
        for flag in flags:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(flag)}(?![A-Za-z0-9_])", cells[0]):
                found[flag].append(_plain(cells[1]))
    return found


@pytest.mark.parametrize("backend", _BACKENDS)
def test_can_and_cannot_table_equals_the_shipped_capability_summary(backend: str) -> None:
    """The page's table reads the same state per flag as the backend's shipped scorecard."""

    card = shipped_scorecard(backend)
    assert card is not None, (
        f"{backend} ships no scorecard record, so its can and cannot table has no measured "
        f"basis; write it with python -m torchlens.conformance.scorecard --backend {backend} "
        "--write on the pinned leg"
    )
    summary = dict(card.capabilities)
    assert summary, f"{backend}'s shipped scorecard has an empty capability summary"
    documented = _documented_states(backend, tuple(sorted(summary)))
    problems: list[str] = []
    for flag, expected in sorted(summary.items()):
        cells = documented[flag]
        if len(cells) != 1:
            problems.append(f"{flag}: {len(cells)} rows name it (need exactly 1)")
            continue
        state = _STATE_WORDS.get(cells[0])
        if state != expected:
            problems.append(f"{flag}: page reads {cells[0]!r}, the scorecard reads {expected!r}")
    assert not problems, f"docs/backends/{backend}.md can and cannot table: " + "; ".join(problems)


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
