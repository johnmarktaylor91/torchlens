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


@pytest.mark.smoke
def test_every_root_surface_name_is_in_the_map(generator, model) -> None:  # type: ignore[no-untyped-def]
    rows = generator.read_surface_rows(REPO_ROOT)
    assert rows, "the frozen root surface table is empty"
    nodes = model["nodes"]
    root = nodes[model["roots"]["tl"]]
    mapped = {nodes[child]["n"] for child in root["c"]}
    expected = {row["canonical_path"].split(".")[-1] for row in rows}
    assert expected <= mapped, f"root names missing from the map: {sorted(expected - mapped)}"
    assert model["root_count"] == len(rows)
    for child in root["c"]:
        alias = nodes[child]
        target = nodes[alias["A"]]
        assert target["k"] in {"function", "class", "module", "package", "variable", "constant"}, (
            alias
        )
        assert "f" in target, f"root name {alias['n']} has no defining file"


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_unresolvable_root_name_fails_loudly(generator, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real_rows = generator.read_surface_rows(REPO_ROOT)
    fake = dict(real_rows[0], canonical_path="torchlens.name_that_does_not_exist_anywhere")
    monkeypatch.setattr(generator, "read_surface_rows", lambda repo: real_rows + [fake])
    with pytest.raises(generator.ApiMapError, match="name_that_does_not_exist_anywhere"):
        generator.build_model(REPO_ROOT)
    assert generator.main([str(Path("/nonexistent") / "out.html"), "--repo", str(REPO_ROOT)]) == 2


@pytest.mark.heavy
def test_cli_writes_the_full_page(generator, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The full build (every module parsed): heavy, not smoke (about 10 s under load)."""

    out = tmp_path / "map.html"
    assert generator.main([str(out), "--repo", str(REPO_ROOT)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    for row in generator.read_surface_rows(REPO_ROOT):
        assert f'"n":"{row["canonical_path"].split(".")[-1]}"' in text
    assert '"scope":"full"' in text
    assert "http://" not in text.split('<script id="api-map-data"')[0].replace(
        "http://www.w3.org", ""
    )
    assert "<link" not in text, "the page must stay self-contained (no external stylesheet)"
