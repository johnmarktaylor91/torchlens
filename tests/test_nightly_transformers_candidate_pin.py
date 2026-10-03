"""Nightly jobs that run a broad selection must pin the R0 candidate leg.

``tests/real_model/r0/test_realism_randinit_sweep.py`` and
``test_realism_randinit_axes.py`` assert the resolved-config fingerprint
(``tests/real_model/registry.py::resolved_config_fingerprint``) and the
op-count/behavior goldens it gates against a value RECORDED AGAINST ONE
EXACT transformers release (``tests/real_model/r0/expectations.py``
docstring: "generated against transformers 5.14.1 ... regenerate
deliberately"). The declared support band (``pyproject.toml``,
``tests/support/proofnet/support_policy.tsv``) is the open interval
``>=4.45,<6``: an install that does not pin transformers resolves to
whatever is newest inside that band, which silently drifts off the
recorded golden the moment a new transformers release ships (measured: a
routine 5.14.1 -> 5.18.0 move fails every row in the deep sweep with
"resolved-config fingerprint drifted", plus a T5/SDPA upstream-behavior
assertion, 24 failures total -- a mix of pure version-string noise
``config["transformers_version"]`` and genuine op-count changes for
llama/mamba/t5).

``tests.yml`` already pins every row's install to
``transformers==5.14.1`` and ``nightly.yml``'s ``hf-band-legs`` job pins
through the named ``hf_5_candidate`` constraints leg. The four jobs below
run a smoke-tier-or-wider pytest selection with NO deselect of the R0 deep
sweep, so they must pin the SAME exact version those legs use; letting the
resolver pick a transformers release ad hoc inside the open band is
exactly the drift class the named-leg system (``support_policy.tsv``,
``test_proofnet_version_policy.py``) exists to prevent. The fix is a
workflow edit (not owned by this test file): pin
``transformers==<candidate pin>`` (or ``-c
tests/support/proofnet/constraints/hf_5_candidate.txt``) in each job's
install step. ``latest-canary.yml`` is deliberately excluded: it is the
one leg designed to run transformers fully unpinned and surface this exact
drift non-blockingly (PASSED-count attested, never a PR gate).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NIGHTLY = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
CANDIDATE_CONSTRAINTS = (
    REPO_ROOT / "tests" / "support" / "proofnet" / "constraints" / "hf_5_candidate.txt"
)

#: Nightly jobs that run a pytest selection wide enough to include
#: ``tests/real_model/r0``'s smoke-marked deep sweep and axes files
#: (``-m smoke`` or ``-m "not slow and not rare"``) with no per-test
#: deselect of the band-keyed rows. ``hf-band-legs`` is excluded: its
#: non-candidate legs explicitly ``--deselect`` the sweep file by name
#: (nightly.yml, band-keyed-rows comment), and its candidate leg already
#: pins through ``-c .../hf_5_candidate.txt``.
JOBS_RUNNING_R0_BROAD_SELECTION = ("fast-tier", "canonical-tier", "coverage", "shuffle-stress")

_JOB_BLOCK_RE_TEMPLATE = r"\n  {job}:\n(.*?)(?=\n  [A-Za-z][A-Za-z0-9_-]*:\n|\Z)"


def _candidate_transformers_pin() -> str:
    for line in CANDIDATE_CONSTRAINTS.read_text().splitlines():
        line = line.strip()
        if line.startswith("transformers=="):
            return line.split("==", 1)[1]
    raise AssertionError(f"{CANDIDATE_CONSTRAINTS}: no transformers== pin found")


def _job_block(workflow_text: str, job_id: str) -> str:
    pattern = re.compile(_JOB_BLOCK_RE_TEMPLATE.format(job=re.escape(job_id)), re.DOTALL)
    match = pattern.search(workflow_text)
    assert match, f"job {job_id!r} not found in {NIGHTLY}; this job was renamed or removed"
    return match.group(1)


@pytest.mark.smoke_cells(
    "test_broad_selection_nightly_jobs_pin_transformers_to_the_candidate_leg[fast-tier]",
    "test_broad_selection_nightly_jobs_pin_transformers_to_the_candidate_leg[shuffle-stress]",
)
@pytest.mark.parametrize("job_id", JOBS_RUNNING_R0_BROAD_SELECTION)
def test_broad_selection_nightly_jobs_pin_transformers_to_the_candidate_leg(job_id: str) -> None:
    text = NIGHTLY.read_text()
    block = _job_block(text, job_id)
    assert "test_realism_randinit_sweep" not in block, (
        f"nightly.yml job {job_id!r} now deselects the R0 sweep by name --"
        " if that is the intended fix instead of pinning, update this test"
        " (and the README support-policy table) to match; do not leave both"
        " unreconciled"
    )
    pin = _candidate_transformers_pin()
    assert f"transformers=={pin}" in block or f'transformers: "{pin}"' in block, (
        f"nightly.yml job {job_id!r} installs `.[test]` with transformers"
        f" unpinned inside the open >=4.45,<6 band -- it must pin"
        f" transformers=={pin} (the exact version"
        " tests/real_model/r0/expectations_r0.json's fingerprints were"
        " recorded against, tracked in"
        " tests/support/proofnet/constraints/hf_5_candidate.txt), else it"
        " intermittently fails the whole R0 deep sweep on every routine"
        " transformers release (resolved-config fingerprint drift)."
    )
