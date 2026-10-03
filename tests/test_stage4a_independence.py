"""L6 stage-4a exit gate: GREP-VERIFIED independence from the attribution kit.

The design memo (sec 2.4, r3 fix for sol MAJOR-5) split stage 4 in two
merges precisely so the SHEDDABLE attribution-kit capstone (4b) is
demonstrably detachable from the non-shed 4a content: Selection resolution,
cross-run patching, and every gallery acceptance test must have NO import
of, reference to, or test dependency on ``torchlens/attribution``. This
module is that check, shipped as a test rather than a claim.

Two scanners over the enumerated 4a territory:

1. AST import scan — no ``import``/``from`` statement may resolve into
   ``torchlens.attribution`` (absolute or relative).
2. Raw token scan — the token ``attribution`` may not appear AT ALL
   (imports, attribute access, strings, comments; a docstring mention is
   already a coupling a shed would have to edit).

Plus a red-capability check proving both scanners can actually fail.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The stage-4a territory (memo 2.4): selection resolution + cross-run
#: alignment modules, and every 4a acceptance/algebra/canary test module.
#: Dropping 4b (torchlens/attribution + its tests) must leave every one of
#: these byte-identical and green.
STAGE_4A_FILES: tuple[str, ...] = (
    "torchlens/selection.py",
    "torchlens/_selection_align.py",
    "tests/test_selection_algebra.py",
    "tests/test_selection_do.py",
    "tests/test_selection_align.py",
    "tests/test_selection_gallery.py",
    "tests/test_dna_canary.py",
)

_FORBIDDEN_TOKEN = re.compile(r"\battribution\b")


def _imported_module_names(source: str, module_path: Path) -> list[str]:
    """Return every module name an import statement in ``source`` resolves to.

    Relative imports are resolved against the module's package location so a
    hypothetical ``from .attribution import x`` cannot hide from the scan.
    """

    tree = ast.parse(source, filename=str(module_path))
    try:
        package_parts = module_path.relative_to(_REPO_ROOT).parts[:-1]
    except ValueError:
        package_parts = ()  # outside the repo (red-capability fixtures)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package_parts[: len(package_parts) - node.level + 1])
            else:
                base = []
            module = node.module or ""
            resolved = ".".join(part for part in (*base, module) if part)
            names.append(resolved)
            names.extend(f"{resolved}.{alias.name}" for alias in node.names)
    return names


def test_stage_4a_files_exist() -> None:
    """Anti-vacuity: the enumerated territory is real (a rename goes red)."""

    missing = [relative for relative in STAGE_4A_FILES if not (_REPO_ROOT / relative).is_file()]
    assert not missing, f"stage-4a territory rows point at missing files: {missing}"


@pytest.mark.parametrize("relative", STAGE_4A_FILES)
def test_no_attribution_import_in_stage_4a(relative: str) -> None:
    """No 4a module imports anything under ``torchlens.attribution``."""

    path = _REPO_ROOT / relative
    offenders = [
        name
        for name in _imported_module_names(path.read_text(encoding="utf-8"), path)
        if name == "torchlens.attribution" or name.startswith("torchlens.attribution.")
    ]
    assert not offenders, f"{relative} imports the 4b attribution kit: {offenders}"


@pytest.mark.parametrize("relative", STAGE_4A_FILES)
def test_no_attribution_token_in_stage_4a(relative: str) -> None:
    """The token ``attribution`` appears nowhere in the 4a territory."""

    source = (_REPO_ROOT / relative).read_text(encoding="utf-8")
    hits = [
        f"line {line_number}: {line.strip()}"
        for line_number, line in enumerate(source.splitlines(), start=1)
        if _FORBIDDEN_TOKEN.search(line)
    ]
    assert not hits, f"{relative} references the 4b attribution kit:\n" + "\n".join(hits)


def test_independence_scanners_are_red_capable(tmp_path: Path) -> None:
    """Both scanners demonstrably fail on real couplings."""

    coupled = tmp_path / "coupled.py"
    coupled.write_text("from torchlens.attribution import saliency\n", encoding="utf-8")
    names = _imported_module_names(coupled.read_text(encoding="utf-8"), coupled)
    assert any(name.startswith("torchlens.attribution") for name in names)
    assert _FORBIDDEN_TOKEN.search(coupled.read_text(encoding="utf-8"))
    # The relative spelling is caught too (resolved against the package).
    relative_form = _REPO_ROOT / "torchlens" / "_fake_lane_module.py"
    names = _imported_module_names("from .attribution import _core\n", relative_form)
    assert "torchlens.attribution" in names
