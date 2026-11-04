"""Wiring of the docs-site deploy workflow, ``.github/workflows/api-map.yml``.

The workflow publishes the documentation site, with the interactive API map at
``api-map/``, to GitHub Pages. It may start only after a successful Release run
of a version-tag push, or by manual dispatch; only its deploy job holds the
Pages scopes; and its build runs every check of ``docs.yml`` (the pull-request
docs check) so the published site is the one pull requests prove. Static reads
of the YAML only: no network, well under a second.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: The build job's condition, whitespace-normalised.
_RELEASE_GATE = (
    "github.event_name == 'workflow_dispatch' || "
    "(github.event.workflow_run.conclusion == 'success' && "
    "github.event.workflow_run.event == 'push' && "
    "startsWith(github.event.workflow_run.head_branch, 'v'))"
)

#: Build commands in the order they must run: the strict build writes ``site/``
#: from scratch, so the API map is generated after it and the links are checked
#: over the finished site.
_BUILD_ORDER = (
    "uv pip install --system --require-hashes -r requirements-docs.lock",
    "python tools/docs/build_agent_index.py --check",
    "python tools/docs/site_nav.py --check",
    "zensical build --strict --clean",
    "python tools/generate_api_map.py site/api-map/index.html",
    "python tools/docs/check_site_links.py site --base-path /torchlens/",
)


def _load(name: str) -> dict[Any, Any]:
    """Return the parsed workflow file ``name``.

    PyYAML reads the bare key ``on`` as ``True``; it is stored back under
    ``"on"`` so the tests read like the file.
    """
    data = yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))
    if True in data:
        data["on"] = data.pop(True)
    return data


def _run_steps(job: dict[str, Any]) -> list[str]:
    """Return the ``run`` text of each step of ``job`` that has one."""
    return [step["run"] for step in job["steps"] if "run" in step]


def _position(steps: list[str], command: str) -> int:
    """Return the index of the one step that runs ``command``."""
    hits = [i for i, text in enumerate(steps) if command in text]
    assert len(hits) == 1, f"expected one step running {command!r}, found {len(hits)}"
    return hits[0]


def test_deploy_starts_only_after_a_release_run_or_by_dispatch() -> None:
    """Only a completed Release run or a manual dispatch starts the workflow."""
    workflow = _load("api-map.yml")
    triggers = workflow["on"]
    assert set(triggers) == {"workflow_run", "workflow_dispatch"}
    assert triggers["workflow_run"] == {"workflows": ["Release"], "types": ["completed"]}
    assert _load("release.yml")["name"] == "Release", "the trigger names release.yml's workflow"


def test_build_refuses_failed_runs_dry_runs_and_non_tag_pushes() -> None:
    """The build job runs only for a successful Release run of a ``v*`` push."""
    condition = _load("api-map.yml")["jobs"]["build"]["if"]
    assert " ".join(condition.split()) == _RELEASE_GATE


def test_only_the_deploy_job_holds_the_pages_scopes() -> None:
    """Read-only token by default; Pages write and OIDC only on deploy."""
    workflow = _load("api-map.yml")
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert set(jobs) == {"build", "deploy"}
    assert "permissions" not in jobs["build"]
    deploy = jobs["deploy"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["needs"] == "build"
    assert deploy["environment"]["name"] == "github-pages"
    assert any("actions/deploy-pages@" in step.get("uses", "") for step in deploy["steps"])


def test_build_makes_the_whole_site_then_the_api_map_in_order() -> None:
    """Strict site build, then the API map, then links over ``site/``."""
    build = _load("api-map.yml")["jobs"]["build"]
    steps = _run_steps(build)
    positions = [_position(steps, command) for command in _BUILD_ORDER]
    assert positions == sorted(positions), "build steps run out of order"
    assert not any("site/index.html <<" in text or "http-equiv" in text for text in steps), (
        "the site has its own index page; no redirect may overwrite it"
    )
    python = [s for s in build["steps"] if "actions/setup-python@" in s.get("uses", "")]
    assert [s["with"]["python-version"] for s in python] == ["3.12"]
    upload = [s for s in build["steps"] if "actions/upload-pages-artifact@" in s.get("uses", "")]
    assert [s["with"]["path"] for s in upload] == ["site/"]


def test_deploy_build_runs_every_check_of_the_pull_request_docs_build() -> None:
    """Each ``run`` step of ``docs.yml`` also runs before a deploy."""
    deploy_steps = _run_steps(_load("api-map.yml")["jobs"]["build"])
    for command in _run_steps(_load("docs.yml")["jobs"]["build"]):
        assert command.strip() in "\n".join(deploy_steps), (
            f"docs.yml runs {command.strip()!r} but the deploy build does not"
        )
