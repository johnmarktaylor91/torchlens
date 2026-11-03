"""API map tripwire: the generated page covers the whole root surface, or fails loudly.

``tools/generate_api_map.py`` renders the interactive API map from a static
parse of the package. The page is never committed, so the only way it can
drift from the code is by silently dropping a name; this suite pins the two
guarantees that prevent that: every row of the frozen root SURFACE table
(the same table ``test_arch_spine_surface.py`` holds in lockstep with the
live facade) appears in the generated map, and a root name the static parse
cannot resolve aborts generation with ``ApiMapError`` instead of producing a
page with a hole.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "tools" / "generate_api_map.py"


def _load_generator():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("generate_api_map", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("generate_api_map", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():  # type: ignore[no-untyped-def]
    return _load_generator()


@pytest.fixture(scope="module")
def model(generator):  # type: ignore[no-untyped-def]
    """The root-scope model: every surface row resolved, namespaces shallow (seconds, not tens)."""

    return generator.build_model(REPO_ROOT, scope="root")


def test_generated_page_names_every_root_surface_name(generator, model) -> None:  # type: ignore[no-untyped-def]
    html = generator.render_html(model)
    assert html.count("<script") == html.count("</script"), (
        "unbalanced script tags: the JSON payload leaked a terminator"
    )
    for row in generator.read_surface_rows(REPO_ROOT):
        name = row["canonical_path"].split(".")[-1]
        assert f'"n":"{name}"' in html, f"{row['canonical_path']} is not in the rendered page"
    assert str(model["commit"]) in html
    assert "tools/generate_api_map.py" in html
