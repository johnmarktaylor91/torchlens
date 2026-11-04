"""The stale ``kw=`` check tells a TorchLens call from a third-party one by its receiver.

A removed TorchLens keyword (``format=``, which only ``export_extraction`` took) is
still spelled by a call that hands it to TorchLens, directly, through an alias, through
a TorchLens object or through a local wrapper; it is not spelled by matplotlib's
``fig.savefig(svg, format="svg")``. Each shape is tested both ways in notebooks (code
cells, with a stored SVG output beside them) and in Python files. Synthetic
inventories and repositories in ``tmp_path``; no torch.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import check  # noqa: E402
from name_lockstep.diff import diff_inventories  # noqa: E402

TRACE = {"id": "root:tl.trace", "kind": "root", "name": "trace", "path": "tl.trace"}
FORMAT = {
    "id": "param:format",
    "kind": "param",
    "name": "format",
    "path": "format=",
    "uses": ["tl.dataset_extraction.export_extraction"],
}
SETUP = [
    "import io\n",
    "import matplotlib.pyplot as plt\n",
    "import torchlens as tl\n",
    "from torchlens.dataset_extraction import export_extraction as ee\n",
    "from my_helpers import export_extraction as mine\n",
]
SVG_OUTPUT = {
    "data": {"image/svg+xml": ['<svg><text x="1" format="svg">loss</text></svg>\n']},
    "metadata": {},
    "output_type": "display_data",
}


def _rows(*rows: dict) -> dict[str, dict]:
    """Key rows on their ids, with a fingerprint each."""

    return {row["id"]: {**row, "fingerprint": f"f-{row['name']}"} for row in rows}


def _notebook(*cells: list[str]) -> str:
    """Return nbformat JSON: one code cell per source list, each with an SVG output."""

    body = [
        {
            "cell_type": "code",
            "execution_count": index,
            "id": f"c{index}",
            "metadata": {},
            "outputs": [SVG_OUTPUT],
            "source": source,
        }
        for index, source in enumerate(cells, start=1)
    ]
    notebook = {"cells": body, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(notebook, indent=1, sort_keys=True) + "\n"


def _stale(root: Path, relative: str, text: str) -> tuple[set[int], list[str]]:
    """Check ``text`` written at ``relative`` against the removal of ``format=``.

    Returns the stale line numbers in that file and the file's lines.
    """

    files = {
        "docs/reference/glossary.md": "Nothing.\n",
        "docs/internal/agent-reference/surface.md": "Use `tl.trace(model, x)`.\n",
        "examples/demo.py": "y = 1\n",
        relative: text,
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    corpus = check.Corpus.read(root, [Path("docs/reference/glossary.md")], snippets=True)
    base, head = _rows(TRACE, FORMAT), _rows(TRACE)
    findings = check.check_lockstep(diff_inventories(base, head), corpus, head)
    prefix = f"{relative}:"
    lines = {
        int(f.location[len(prefix) :])
        for f in findings
        if f.problem == "stale" and f.location.startswith(prefix)
    }
    return lines, text.splitlines()


def _line(lines: list[str], fragment: str) -> int:
    """Return the 1-based line holding ``fragment`` (exactly one must)."""

    hits = [number for number, line in enumerate(lines, start=1) if fragment in line]
    assert len(hits) == 1, (fragment, hits)
    return hits[0]


def test_savefig_format_on_a_matplotlib_figure_is_not_the_gone_keyword(tmp_path: Path) -> None:
    """``fig.savefig(svg, format="svg")`` passes matplotlib's keyword: no finding."""

    plot = [
        "%matplotlib inline\n",
        "fig, ax = plt.subplots(figsize=(7, 3.4))\n",
        "svg = io.BytesIO()\n",
        'with plt.rc_context({"svg.hashsalt": "x"}):\n',
        '    fig.savefig(svg, format="svg", metadata={"Date": None})\n',
        "fig.savefig(\n",
        "    svg,\n",
        '    format="png",\n',
        ")\n",
        "plt.close(fig)\n",
    ]
    found, lines = _stale(tmp_path, "examples/plot.ipynb", _notebook(SETUP, plot))
    assert any('format=\\"svg\\", metadata' in line for line in lines)
    assert found == set()


def test_the_gone_keyword_handed_to_torchlens_still_reports(tmp_path: Path) -> None:
    """Direct, aliased, TorchLens-object, wrapper and commented spellings all report."""

    calls = [
        "trace = tl.trace(model, x)\n",
        'tl.dataset_extraction.export_extraction(trace, format="npy")\n',
        'ee(trace, format="hdf5")\n',
        "save = tl.dataset_extraction.export_extraction\n",
        'save(trace, format="mat")\n',
        'tl.record(model, x, format="npy")\n',
        'trace.export(format="npy")\n',
        'mine(trace, format="npy")\n',
        'bundle, sizes = tl.Bundle(), {}\nbundle.dump(format="npy")\n',
        "def write(t, **kw):\n",
        "    return tl.dataset_extraction.export_extraction(t, **kw)\n",
        'write(trace, format="npy")\n',
        "fig, ax = plt.subplots()\n",
        'fig.savefig(svg, format="svg")  # was export_extraction(format="svg")\n',
        'fig.savefig(svg, format="svg")\n',
    ]
    found, lines = _stale(tmp_path, "examples/save.ipynb", _notebook(SETUP, calls))
    expected = {
        _line(lines, 'export_extraction(trace, format=\\"npy\\")'),
        _line(lines, 'ee(trace, format=\\"hdf5\\")'),
        _line(lines, 'save(trace, format=\\"mat\\")'),
        _line(lines, 'tl.record(model, x, format=\\"npy\\")'),
        _line(lines, 'trace.export(format=\\"npy\\")'),
        _line(lines, 'mine(trace, format=\\"npy\\")'),
        _line(lines, 'bundle.dump(format=\\"npy\\")'),
        _line(lines, 'write(trace, format=\\"npy\\")'),
        _line(lines, "# was export_extraction"),
    }
    assert found == expected


def test_a_local_function_that_declares_the_keyword_owns_it(tmp_path: Path) -> None:
    """A local function or ``__init__`` that declares ``format=`` owns it; ``**kw`` may not."""

    cells = [
        'def plot(data, format="png"):\n',
        "    return data\n",
        'plot(1, format="svg")\n',
        "def forward(data, **kw):\n",
        "    return data\n",
        'forward(1, format="svg")\n',
        "class Writer:\n",
        '    def __init__(self, format="png"):\n',
        "        self.format = format\n",
        'Writer(format="svg")\n',
        "class Opaque(tl.Bundle):\n",
        "    pass\n",
        'Opaque(format="svg")\n',
    ]
    found, lines = _stale(tmp_path, "examples/local.ipynb", _notebook(SETUP, cells))
    owned = {_line(lines, 'plot(1, format=\\"svg\\")'), _line(lines, 'Writer(format=\\"svg\\")')}
    assert found.isdisjoint(owned)
    assert {
        _line(lines, 'forward(1, format=\\"svg\\")'),
        _line(lines, 'Opaque(format=\\"svg\\")'),
    } <= found


def test_python_files_resolve_receivers_the_same_way(tmp_path: Path) -> None:
    """In a ``.py`` example the same rules hold: matplotlib's keyword is skipped."""

    source = textwrap.dedent(
        """
        import io
        import matplotlib.pyplot as plt
        import torchlens as tl
        from torchlens.dataset_extraction import export_extraction as ee

        fig, ax = plt.subplots()
        fig.savefig(io.BytesIO(), format="svg")
        tl.record(model, x, format="npy")
        ee(trace, format="hdf5")
        """
    ).lstrip("\n")
    found, lines = _stale(tmp_path, "examples/script.py", source)
    assert found == {
        _line(lines, 'tl.record(model, x, format="npy")'),
        _line(lines, 'ee(trace, format="hdf5")'),
    }
