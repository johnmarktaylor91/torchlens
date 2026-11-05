"""The render-identity golden must hold on every host.

``tests/golden/viz_render_identity_oracle.json`` freezes raw DOT bytes.  The
key draws its swatches as PNG files under the system temp folder and the DOT
``<IMG SRC>`` names them by absolute path, so a regeneration that skipped the
record builder's swatch-path normalization would bake the generating host's
directory (a CI runner's temp folder, a worker's job folder) into the golden
and the byte oracle would go red on every other host.  These scans fail such
a golden before it is committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GOLDEN = Path(__file__).parent / "golden" / "viz_render_identity_oracle.json"

#: Two or more path components after a leading slash, or a Windows drive root.
_ABSOLUTE_PATH = re.compile(r"(?<![\w<])(?:/[\w.-]+){2,}|[A-Za-z]:\\")
_IMG_SRC = re.compile(r'<IMG SRC="([^"]*)"')
#: Substrings only a fleet or CI job folder ever carries.
_JOB_FOLDER_MARKERS = ("run-on-job-", ".orca-run", "pytest-of-", "/tmp/", "/home/", "/Users/")


def _golden_strings() -> list[str]:
    """Return every string leaf of the golden, JSON-decoded."""

    leaves: list[str] = []

    def walk(value: Any) -> None:
        """Collect string leaves depth-first."""
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            leaves.append(value)

    walk(json.loads(_GOLDEN.read_text(encoding="utf-8")))
    return leaves


def test_render_identity_golden_names_no_job_folder() -> None:
    """No worker, CI or pytest temp folder marker survives in the golden."""

    text = _GOLDEN.read_text(encoding="utf-8")
    found = [marker for marker in _JOB_FOLDER_MARKERS if marker in text]
    assert found == [], f"host folder markers in {_GOLDEN.name}: {found}"


def test_render_identity_golden_holds_no_absolute_path() -> None:
    """No decoded string in the golden contains an absolute file path."""

    hits = sorted(
        {match.group(0) for leaf in _golden_strings() for match in _ABSOLUTE_PATH.finditer(leaf)}
    )
    assert hits == [], f"absolute paths in {_GOLDEN.name}: {hits}"


def test_render_identity_golden_swatch_sources_are_bare_file_names() -> None:
    """Every key swatch ``<IMG SRC>`` is the content-addressed file name alone."""

    sources = [src for leaf in _golden_strings() for src in _IMG_SRC.findall(leaf)]
    assert sources, "the golden draws no key swatch; the themed CNN case is expected to"
    offenders = [src for src in sources if "/" in src or "\\" in src or not src.startswith("key-")]
    assert offenders == [], f"swatch sources that are not bare file names: {offenders}"
