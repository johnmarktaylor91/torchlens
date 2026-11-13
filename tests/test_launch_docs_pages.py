"""Execute and check the launch docs pages: Field guide, doors, gallery framing, relation, benchmark.

The Field guide task pages run every Python fence in order in one namespace (a page is one
running session). The any-architecture strip runs each fence alone and skips a fence whose
declared module is absent. The structural tests hold each page to its contract: Field guide
pages open with a canonical running-example fence, the doors are equal on the docs landing,
the basis grid has the six entries and six verbs, the comparison table keeps the jobs another
tool covers, and every relative link on these pages resolves.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BLOCK_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)
LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")
STRIP_HEADER_RE = re.compile(r"# strip: family=(?P<family>[a-z-]+) requires=(?P<module>[a-z_]+)")

#: Field guide task pages, executed top to bottom; each opens with a canonical fence.
FIELD_GUIDE_TASK_PAGES = (
    "field-guide/see_and_inspect.md",
    "field-guide/extract_features.md",
    "field-guide/intervene.md",
)
#: Pages whose fences run in sequence in one namespace.
SEQUENTIAL_PAGES = ("field-guide/running_examples.md", *FIELD_GUIDE_TASK_PAGES)
#: Pages whose single fence runs alone and fast.
LIGHT_PAGES = ("migration/comparison.md",)
#: Every page this module owns; links on these must resolve.
LAUNCH_PAGES = (
    "README.md",
    *SEQUENTIAL_PAGES,
    "doors/mechinterp.md",
    "doors/neuroai.md",
    "doors/vision.md",
    "doors/agents.md",
    "gallery/basis_grid.md",
    "gallery/any_architecture.md",
    "migration/index.md",
    "migration/comparison.md",
    "migration/interop.md",
    "migration/where_to_go_instead.md",
    "benchmarks/capture.md",
)
DOORS = ("mechinterp", "neuroai", "vision", "agents")
VERBS = ("capture", "select", "read", "do", "compare", "validate")
COMPARISON_TOOLS = (
    "TorchLens",
    "Plain hooks",
    "TransformerLens",
    "nnsight",
    "pyvene",
    "Captum",
    "thingsvision",
    "torchvision feature extraction",
)
HOOKS_HEADING = "## When you need TorchLens, and when five lines of hooks are enough"


def _text(page: str) -> str:
    """Return one docs page's text."""

    return (DOCS / page).read_text(encoding="utf-8")


def _fences(page: str) -> list[str]:
    """Return the Python fence bodies of one page, in order."""

    return [match.group("code") for match in BLOCK_RE.finditer(_text(page))]


def _uses_draw(code: str) -> bool:
    """Whether a fence renders a picture (which needs the Graphviz binary)."""

    return ".draw(" in code


def _skip_without_dot(codes: list[str]) -> None:
    """Skip when a fence draws and the Graphviz ``dot`` binary is absent."""

    if any(_uses_draw(code) for code in codes) and shutil.which("dot") is None:
        pytest.skip("Graphviz dot binary not installed; the page draws")


def _run(codes: list[str], page: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Execute fences in order in one namespace, inside a scratch directory."""

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": f"docs_{page.replace('/', '_')}"}
    for index, code in enumerate(codes, start=1):
        exec(compile(code, f"docs/{page}:python-{index}", "exec"), namespace)  # noqa: S102


@pytest.mark.heavy
@pytest.mark.parametrize("page", SEQUENTIAL_PAGES)
def test_field_guide_page_runs(page: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every fence on a Field guide page runs, in order, as one session."""

    codes = _fences(page)
    assert codes, page
    _skip_without_dot(codes)
    _run(codes, page, monkeypatch, tmp_path)


@pytest.mark.parametrize("page", LIGHT_PAGES)
def test_light_page_runs(page: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The comparison page's hooks-versus-capture fence runs and its asserts hold."""

    codes = _fences(page)
    assert len(codes) == 1, page
    _run(codes, page, monkeypatch, tmp_path)


def _strip_fences() -> list[tuple[str, str, str]]:
    """Return (family, module, code) for each fence of the any-architecture strip."""

    out = []
    for code in _fences("gallery/any_architecture.md"):
        match = STRIP_HEADER_RE.match(code)
        assert match is not None, code.splitlines()[0]
        out.append((match["family"], match["module"], code))
    return out


@pytest.mark.heavy
@pytest.mark.parametrize(("family", "module", "code"), _strip_fences(), ids=lambda v: str(v)[:24])
def test_strip_fence_runs(
    family: str, module: str, code: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each architecture runs the same three lines; an absent optional module skips."""

    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed for the {family} fence")
    _skip_without_dot([code])
    _run([code], f"gallery/any_architecture.md#{family}", monkeypatch, tmp_path)


def test_strip_has_six_families_and_one_preview() -> None:
    """Six Menagerie families plus one preview backend, all with the same closing lines."""

    fences = _strip_fences()
    families = [family for family, _module, _code in fences]
    assert families == [
        "decoder",
        "cnn",
        "vision-transformer",
        "recurrent",
        "diffusion-unet",
        "graph-net",
        "preview-jax",
    ]
    closing = [
        "trace = tl.trace(model, x)",
        "trace.draw()",
        "edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)",
    ]
    for family, _module, code in fences[:6]:
        assert code.strip().splitlines()[-3:] == closing, family
    page = _text("gallery/any_architecture.md")
    assert "https://modelmenagerie.ai" in page
    assert "thousands of models across all major architecture families" in page
    assert "## Another framework: JAX" in page


def _canonical_fences() -> list[str]:
    """The two running-example fences: the first two Python fences of their page."""

    fences = _fences("field-guide/running_examples.md")[:2]
    for fence in fences:
        assert fence.rstrip().endswith("trace = tl.trace(model, x)"), fence
    return fences


@pytest.mark.parametrize("page", FIELD_GUIDE_TASK_PAGES)
def test_field_guide_page_opens_with_a_running_example(page: str) -> None:
    """Each task page's first fence is one running example, byte for byte."""

    assert _fences(page)[0] in _canonical_fences(), page
    text = _text(page)
    for heading in (
        "## Minimal example on the running model",
        "## Output",
        "## A check",
        "## Vary it",
        "## Performance note",
        "## Limits",
        "## Next task",
    ):
        assert heading in text, (page, heading)
    assert text.index("**Result in one line:**") < text.index("## Minimal example")


def test_doors_are_equal_on_the_docs_landing() -> None:
    """Four door rows of the same shape, each door page with the same sections."""

    landing = _text("README.md")
    rows = [line for line in landing.splitlines() if line.startswith("| [")]
    door_rows = [row for row in rows if "](doors/" in row]
    assert [re.search(r"\(doors/(\w+)\.md\)", row)[1] for row in door_rows] == list(DOORS)
    assert len({row.count("|") for row in door_rows}) == 1
    sections = None
    for door in DOORS:
        text = _text(f"doors/{door}.md")
        assert "<!--" not in text, f"{door} carries a placeholder comment"
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        sections = sections or headings
        assert headings == sections, door


def _table_rows(text: str, header_start: str) -> tuple[list[str], list[list[str]]]:
    """Return (header cells, body rows) of the first table whose header starts so."""

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    body = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        body.append([cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip("|"))])
    return header, body


def test_basis_grid_has_six_entries_and_six_verbs() -> None:
    """Rows are the six Gallery entries; pending rows say pending in both closing cells."""

    header, body = _table_rows(_text("gallery/basis_grid.md"), "| Entry |")
    assert tuple(header[1:7]) == VERBS
    assert header[7].startswith("Load-bearing lines") and header[8] == "Rerun on"
    assert [row[0].split(".")[0] for row in body] == ["1", "2", "3", "4", "5", "6"]
    for row in body:
        assert len(row) == len(header), row[0]
        if "pending" in row[7:]:
            assert row[7:] == ["pending", "pending"], row[0]


def test_comparison_table_keeps_the_jobs_torchlens_loses() -> None:
    """At least 15 jobs by 8 tools, dated, with the hooks section above the table."""

    text = _text("migration/comparison.md")
    assert "Checked 2026-11-03" in text
    assert text.index(HOOKS_HEADING) < text.index("## The table")
    header, body = _table_rows(text, "| Job |")
    tools = header[1:-1]
    assert len(tools) == len(COMPARISON_TOOLS)
    for tool, expected in zip(tools, COMPARISON_TOOLS, strict=True):
        assert tool.startswith(expected), (tool, expected)
    assert len(body) >= 15
    assert all(len(row) == len(header) for row in body)
    lost = [row[0] for row in body if row[-1]]
    assert len(lost) >= 5, lost
    for job in ("Serving-scale", "Sparse autoencoder", "Attribution", "Trainable", "One module"):
        assert any(row.startswith(job) for row in lost), job


def test_coming_from_index_lists_every_coming_from_page() -> None:
    """Every migration page titled "Coming from" is linked from the relation index."""

    index = _text("migration/index.md")
    for page in sorted((DOCS / "migration").glob("*.md")):
        first = page.read_text(encoding="utf-8").splitlines()[0]
        if first.startswith("# Coming from"):
            assert f"]({page.name})" in index, page.name


def _slug(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""

    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


@pytest.mark.parametrize("page", LAUNCH_PAGES)
def test_relative_links_resolve(page: str) -> None:
    """Every relative link on a launch page names a file and, if given, a heading in it."""

    source = DOCS / page
    for match in LINK_RE.finditer(source.read_text(encoding="utf-8")):
        target = match["target"]
        if re.match(r"^[a-z]+:", target):
            continue
        path_part, _, anchor = target.partition("#")
        resolved = (source.parent / path_part).resolve() if path_part else source
        assert resolved.exists(), (page, target)
        if anchor and resolved.suffix == ".md":
            headings = [
                _slug(line.lstrip("#"))
                for line in resolved.read_text(encoding="utf-8").splitlines()
                if line.startswith("#")
            ]
            assert anchor in headings, (page, target)


def _benchmark_data() -> list[dict[str, Any]]:
    """Every committed capture benchmark result the page quotes, one per run."""

    paths = sorted((DOCS / "benchmarks" / "data").glob("capture_*.json"))
    assert paths, "no capture benchmark result file"
    return [json.loads(path.read_text()) for path in paths]


def test_benchmark_page_quotes_its_result_file() -> None:
    """Both headings, each run's host and date, and every quoted number from its result file."""

    text = _text("benchmarks/capture.md")
    assert text.index("## Live edits at hook speed") < text.index("## Where TorchLens is slower")
    for data in _benchmark_data():
        assert data["host"]["name"] in text
        assert data["measured_on"] in text
        assert data["page_rows"], data["host"]["name"]
        for row in data["page_rows"]:
            assert row["text"] in text, row["text"]
