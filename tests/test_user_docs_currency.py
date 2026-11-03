"""User-docs currency gates: named replacements resolve, galleries list every notebook.

Three defect families found by the pre-launch docs audit, each pinned so it
cannot silently reopen:

* the removed-spellings ledger (``docs/reference/deprecations.md``) tells users
  that its "New name" column "is the spelling to use", so every dotted public
  path in that column must import today (a later rename left
  ``torchlens.errors.PostTraceParamUnavailable`` there after the class became
  ``ParamReleasedError``);
* each example gallery's README is the reading order a new user follows, so it
  must link every notebook in its folder (``extract_dataset.ipynb`` went
  unlisted);
* the 2.0 field rename (activation to ``out``) also rewrote English prose
  ("Pluck your first out", "Training on saved outs"); the noun in prose is
  "activation", and the code field stays ``.out``.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPRECATIONS = REPO_ROOT / "docs" / "reference" / "deprecations.md"
GALLERIES = ("examples/5min", "examples/50min", "examples/recipes")

#: A ledger row whose "New name" cell is one backticked dotted public path.
_LEDGER_ROW_RE = re.compile(r"^\| `[^`]+` \| `(?P<new>(?:torchlens|tl)\.[A-Za-z0-9_.]+)` \|", re.M)


def _resolve(dotted: str) -> object:
    """Import the longest module prefix of ``dotted`` and walk the rest as attributes."""

    parts = dotted.replace("tl.", "torchlens.", 1).split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


def _ledger_new_names() -> list[str]:
    return sorted(set(_LEDGER_ROW_RE.findall(DEPRECATIONS.read_text(encoding="utf-8"))))


def test_ledger_has_dotted_replacements() -> None:
    """The row pattern still matches the ledger (guards the resolver test below)."""

    assert len(_ledger_new_names()) >= 20


@pytest.mark.parametrize("dotted", _ledger_new_names())
def test_deprecations_ledger_new_name_resolves(dotted: str) -> None:
    """Every "New name" the removed-spellings ledger teaches imports today."""

    try:
        _resolve(dotted)
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"deprecations.md teaches {dotted!r}, which does not resolve: {exc}")


@pytest.mark.parametrize("gallery", GALLERIES)
def test_gallery_readme_links_every_notebook(gallery: str) -> None:
    """A gallery README links every notebook in its folder, and only existing ones."""

    folder = REPO_ROOT / gallery
    readme = (folder / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^)#]+\.ipynb)\)", readme))
    present = {path.name for path in folder.glob("*.ipynb")}
    assert present - linked == set(), (
        f"{gallery}/README.md does not list {sorted(present - linked)}"
    )
    assert linked - present == set(), (
        f"{gallery}/README.md links missing {sorted(linked - present)}"
    )


#: Prose that only the activation-to-out field codemod writes.
_CODEMOD_PROSE_RE = re.compile(
    r"\b(?:first|saved|clean|stimulus|ReLU|positive|negative) outs?\b|\ban out\b(?!-)|\bout-space\b|"
    r"\bcopy of an out\b|\bcompare outs\b|\bout (?:patching|direction|intervention)\b"
)
_FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)


def _user_prose() -> list[tuple[str, str]]:
    """(label, prose) for README, user docs pages, gallery READMEs and notebook text."""

    pages = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "examples").rglob("*.md"))]
    pages += [
        page
        for page in sorted((REPO_ROOT / "docs").rglob("*.md"))
        if "agent-reference" not in page.parts and page.name != "glossary.md"
    ]
    out = [
        (
            page.relative_to(REPO_ROOT).as_posix(),
            _FENCE_RE.sub("", page.read_text(encoding="utf-8")),
        )
        for page in pages
    ]
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    notebooks += sorted((REPO_ROOT / "examples").rglob("*.ipynb"))
    for nb_path in notebooks:
        cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
        for index, cell in enumerate(cells):
            text = "".join(cell.get("source", []))
            if cell["cell_type"] == "code":
                # Comments and docstrings only; identifiers named ``out``/``outs`` are fine.
                text = "\n".join(line for line in text.splitlines() if "#" in line or '"""' in line)
            rel = nb_path.relative_to(REPO_ROOT).as_posix()
            out.append((f"{rel}::cell{index}", text))
    return out


def test_user_prose_says_activation_not_out() -> None:
    """User-facing prose never carries the field codemod's "an out" / "saved outs"."""

    offenders = [
        f"{label}: {match.group(0)!r}"
        for label, text in _user_prose()
        for match in _CODEMOD_PROSE_RE.finditer(text)
    ]
    assert not offenders, "codemod prose (write 'activation'):\n" + "\n".join(offenders)


def test_intervention_examples_readme_names_every_script() -> None:
    """The worked-examples README names each numbered script by its real file name."""

    folder = REPO_ROOT / "examples" / "intervention"
    readme = (folder / "README.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`(\d\d_[a-z0-9_]+\.py)`", readme))
    present = {path.name for path in folder.glob("[0-9][0-9]_*.py")}
    assert named == present, (
        f"missing from README: {sorted(present - named)}; "
        f"README names absent files: {sorted(named - present)}"
    )
