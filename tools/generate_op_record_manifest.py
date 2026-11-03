"""Generate ``torchlens/_ir/op_record_manifest.py`` from the source spec.

The manifest is the FROZEN ``CellSourceManifest`` v1: one row per op store
cell naming its step-0 ingest source class. Regenerate-and-diff is CI-gated
(``tests/producer_parity/test_p1_record_types.py``); editing the generated
file by hand is a build failure.

Run: ``python -m tools.generate_op_record_manifest [--check]``
"""

from __future__ import annotations

import sys
from pathlib import Path

# Match the sibling generate_record_schema.py: the plain-script spelling
# (``python tools/generate_op_record_manifest.py``) puts tools/ -- not the repo
# root -- at sys.path[0], so without this the torchlens import below fails
# with ModuleNotFoundError (grind b7 R53-A).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

HEADER = '''"""GENERATED FILE — do not edit. CellSourceManifest v{version}.

Regenerate with ``python -m tools.generate_op_record_manifest`` after editing
the source spec in ``torchlens/_ir/op_record_scatter.py``. The joint-freeze
contract (producer DoR v4 section 5.2 / ppdag v3 section 9) versions this
table; post-freeze changes bump the version, never mutate v{version}.
"""

from __future__ import annotations

CELL_SOURCE_MANIFEST_VERSION = {version}

CELL_SOURCE_MANIFEST: dict[str, str] = {{
'''


def generate() -> str:
    from torchlens._ir.op_record_scatter import CELL_SOURCE_MANIFEST_VERSION, CELL_SOURCES

    lines = [HEADER.format(version=CELL_SOURCE_MANIFEST_VERSION)]
    for field_name in sorted(CELL_SOURCES):
        lines.append(f"    {field_name!r}: {CELL_SOURCES[field_name]!r},\n")
    lines.append("}\n")
    return "".join(lines)


def main() -> int:
    """Generate (or with --check, verify) the manifest module."""

    target = _REPO_ROOT / "torchlens" / "_ir" / "op_record_manifest.py"
    rendered = generate()
    if "--check" in sys.argv:
        current = target.read_text() if target.exists() else ""
        if current != rendered:
            print("stale: op_record_manifest.py differs from a fresh generation")
            return 1
        print("ok: op_record_manifest.py is current")
        return 0
    target.write_text(rendered)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
