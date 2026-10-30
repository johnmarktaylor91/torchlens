"""First-page allowlist: the README's first screen and the landing page name only core calls.

The allowed names and the pages live in ``tools/docs/first_page_allowlist.json``.
A page whose ``enforced`` switch is off must say why beside the switch; its check
still runs here (its violations are computed and must be well formed), it just does
not fail the suite. A missing landing page has nothing to check yet.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "docs" / "first_page.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_first_page_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


first_page = _load_module()
ALLOW = first_page.load_allowlist()
PAGES = {page["path"]: page for page in ALLOW["pages"]}

CORE_PAGE = """# Title

```python
import torchlens as tl
trace = tl.trace(model, x)
trace["conv2d_1_1"].out.shape
trace.draw()
feats = tl.extract(model, x, ["layer4"])
trace.do(site, edit)  # a comment naming trace.summary() is ignored
```

Use `tl.trace(model, x)` first.

## More

```python
trace.summary()
```
"""


def test_core_page_passes() -> None:
    text = first_page.first_screen(CORE_PAGE)
    assert "## More" not in text
    assert first_page.violations(text, "fixture.md", ALLOW) == []


@pytest.mark.parametrize(
    ("snippet", "kind", "name"),
    [
        ("```python\nlog = tl.trace(m, x)\nlog.summary()\n```", "trace_method", "summary"),
        ("```python\nrun = tl.trace(m, x)\nrun.fork()\n```", "trace_method", "fork"),
        (
            "```python\ntrace = tl.trace(m, x)\ntrace[7].func_name\n```",
            "layer_attribute",
            "func_name",
        ),
        ("Call `tl.validate(model, x)` too.", "tl", "validate"),
        ("```python\ntorchlens.bundle({})\n```", "tl", "bundle"),
    ],
)
def test_non_core_names_are_flagged(snippet: str, kind: str, name: str) -> None:
    found = first_page.violations(f"# T\n\n{snippet}\n", "fixture.md", ALLOW)
    assert [(v.kind, v.name) for v in found] == [(kind, name)]


def test_heading_inside_a_fence_does_not_end_the_first_screen() -> None:
    text = "# T\n\n```python\n## not a heading\ntl.validate(m, x)\n```\n\n## Next\n"
    assert "tl.validate" in first_page.first_screen(text)


def test_allowlist_data_is_well_formed(tmp_path: Path) -> None:
    assert set(PAGES) >= {"README.md", "docs/index.md"}
    for page in ALLOW["pages"]:
        assert isinstance(page["enforced"], bool)
    bad = json.loads(first_page.ALLOWLIST.read_text(encoding="utf-8"))
    bad["pages"][0].update(enforced=False, reason=" ")
    path = tmp_path / "allow.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="must state its reason"):
        first_page.load_allowlist(path)


@pytest.mark.parametrize("path", sorted(PAGES))
def test_first_pages_name_only_core_calls(path: str) -> None:
    page = PAGES[path]
    found = first_page.page_violations(page, ALLOW)
    assert all(v.path == path and v.name for v in found)
    if page["enforced"]:
        assert found == [], "non-core names on the first page:\n" + "\n".join(
            f"  {v.kind} {v.name!r}: {v.snippet}" for v in found
        )
    else:
        # Switch off on purpose; the reason is written beside the switch in the data file.
        assert page["reason"].strip()
