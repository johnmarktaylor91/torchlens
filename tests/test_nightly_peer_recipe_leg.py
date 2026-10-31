"""The nightly peer-recipe leg runs every recipe page with its peer installed.

``tests/test_recipe_pages_executed.py`` skips a third-party recipe page when the
page's tool is not installed, so on an ordinary leg those pages cover nothing.
The nightly ``peer-recipes`` job is the one place they run: it installs every
peer at the version its page pins, sets ``TORCHLENS_RECIPE_PEERS=1`` (a missing
tool then fails instead of skipping) and fails the leg on any skip. These checks
keep that job present, complete and in step with the pins the pages print.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from tests.test_recipe_pages_executed import PAGES

_REPO = Path(__file__).resolve().parents[1]
_NIGHTLY = _REPO / ".github" / "workflows" / "nightly.yml"
_JOB = "peer-recipes"
_SHA_PINNED_USE = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")
#: A pinned peer on a page's install line: ``name==version`` or a ``git+URL@sha``.
_PAGE_PIN = re.compile(r'([A-Za-z0-9_.-]+==[0-9][^\s"]*)|(git\+[^\s"]+@[0-9a-f]{40})')


def _job() -> dict[str, Any]:
    """Return the nightly ``peer-recipes`` job."""

    jobs = yaml.safe_load(_NIGHTLY.read_text(encoding="utf-8"))["jobs"]
    assert _JOB in jobs, "nightly.yml lost the peer-recipes job"
    return jobs[_JOB]


def _run_text(job: dict[str, Any]) -> str:
    """Return every ``run:`` script of the job, joined."""

    return "\n".join(step.get("run", "") for step in job["steps"])


def test_peer_recipe_job_follows_the_workflow_conventions() -> None:
    """SHA-pinned actions, no persisted checkout credentials, a bounded runtime."""

    job = _job()
    assert isinstance(job.get("timeout-minutes"), int)
    uses = [step["uses"] for step in job["steps"] if "uses" in step]
    assert uses and all(_SHA_PINNED_USE.match(use) for use in uses), uses
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/"))
    assert checkout["with"]["persist-credentials"] is False


def test_peer_recipe_job_requires_peers_and_fails_on_any_skip() -> None:
    """The recipe run sets the peer flag and a zero-skip check follows it."""

    job = _job()
    runs = [
        step
        for step in job["steps"]
        if "tests/test_recipe_pages_executed.py" in step.get("run", "")
    ]
    assert len(runs) == 1, "exactly one step runs the recipe pages"
    assert runs[0].get("env", {}).get("TORCHLENS_RECIPE_PEERS") == "1"
    assert "-rA" in runs[0]["run"] and "--junitxml=" in runs[0]["run"]
    # No marker expression: heavy recipe pages (HEAVY_PAGES) must run here too.
    pytest_args = shlex.split(runs[0]["run"].split("-m pytest", 1)[1].replace("\\\n", " "))
    assert not any(arg == "-m" or arg.startswith("-m=") for arg in pytest_args), pytest_args
    check = re.search(
        r'echo "peer-recipes\.junit\.xml (\d+) (\S+)"',
        _run_text(job),
    )
    assert check, "the leg must attest its junit with the executed-test check"
    floor, max_skipped = int(check.group(1)), float(check.group(2))
    assert max_skipped == 0.0, "a skipped recipe page counts as a failure on this leg"
    assert floor >= len(PAGES), (floor, len(PAGES))


def test_peer_recipe_job_installs_every_page_pin_and_holds_the_base() -> None:
    """Each pin a recipe page's install line prints is installed verbatim, and the
    base install's torch, transformers and torchvision are held."""

    text = _run_text(_job())
    page_pins = {
        pin
        for page in PAGES
        for line in (_REPO / page).read_text(encoding="utf-8").splitlines()
        if "pip install" in line
        for match in _PAGE_PIN.findall(line.split("pip install", 1)[1])
        for pin in match
        if pin
    }
    assert page_pins, "the recipe pages print no pinned install line"
    missing = sorted(pin for pin in page_pins if pin not in text)
    assert not missing, f"peer pins on recipe pages but not in the nightly leg: {missing}"
    for held in ('"torch==$TV"', '"transformers==$TF"', '"torchvision==$TVV"'):
        assert held in text, held
    assert "uv pip freeze" in text and "peer-freeze.txt" in text


def test_peer_recipe_job_uploads_the_freeze() -> None:
    """The resolved peer set ships as an artifact even when the leg fails."""

    uploads = [
        step
        for step in _job()["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    assert uploads[0].get("if") == "always()"
    assert "peer-freeze.txt" in uploads[0]["with"]["path"]
