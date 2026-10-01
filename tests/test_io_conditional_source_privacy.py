"""B8-18: a captured conditional must not leak an absolute source path on save.

``Trace.conditional_records`` holds ``ConditionalEvent`` objects whose
``source_file`` is the ABSOLUTE path of the user's ``forward``-defining module.
Those dataclasses had no ``PORTABLE_STATE_SPEC``, so the scrubber returned them
verbatim and the source-path relativizer never ran -- the absolute path (with its
``$HOME``, OS username, and filesystem layout) persisted into ``metadata.pkl`` at
EVERY save level, INCLUDING ``include_source=False``. That is a direct regression
of the source-privacy guarantee the scrubber documents for every other
source-bearing record (``Trace``/``Module``/``FuncCallLocation``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _BranchingModel(nn.Module):
    """A forward with an if-chain over a captured tensor, so a conditional records."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        if y.sum() > 0:
            return y.relu()
        return y


def _conditional_source_files(trace: tl.Trace) -> list[str]:
    """Return the ``source_file`` of every captured conditional record."""

    return [
        getattr(record, "source_file", "")
        for record in trace.conditional_records
        if getattr(record, "source_file", None)
    ]


def _capture() -> tl.Trace:
    trace = tl.trace(_BranchingModel().eval(), torch.randn(2, 4))
    assert _conditional_source_files(trace), "test model did not capture a conditional"
    # The live capture legitimately holds the absolute path (that is not saved yet).
    assert any(os.path.isabs(path) for path in _conditional_source_files(trace))
    return trace


def _assert_no_host_path(paths: list[str]) -> None:
    home = os.path.expanduser("~")
    for path in paths:
        assert not os.path.isabs(path), f"absolute conditional source path persisted: {path!r}"
        assert home not in path, f"$HOME leaked in a conditional source path: {path!r}"
        assert os.sep not in path.replace("\\", "/").rstrip(), (
            f"a path separator survived in a saved conditional source path: {path!r}"
        )


def test_conditional_source_path_relativized_with_source_kept(tmp_path: Path) -> None:
    """At ``include_source=True`` the path is reduced to a bare basename."""

    trace = _capture()
    spec = tmp_path / "kept.tlspec"
    tl.save(trace, str(spec), include_source=True)
    loaded = tl.load(str(spec))
    saved_paths = _conditional_source_files(loaded)
    assert saved_paths, "conditional records vanished on round-trip"
    _assert_no_host_path(saved_paths)


def test_conditional_source_path_dropped_with_source_excluded(tmp_path: Path) -> None:
    """At ``include_source=False`` no conditional source path reaches the bundle.

    Fail-before (B8-18): the absolute forward-module path persisted verbatim in
    ``metadata.pkl`` even though the caller asked for no embedded source.
    """

    trace = _capture()
    spec = tmp_path / "dropped.tlspec"
    tl.save(trace, str(spec), include_source=False)

    # Nothing in the on-disk artifact may carry the absolute path.
    absolute_here = os.path.abspath(__file__)
    for artifact in spec.rglob("*"):
        if artifact.is_file():
            blob = artifact.read_bytes()
            assert absolute_here.encode() not in blob, (
                f"absolute source path leaked into {artifact.name}"
            )

    loaded = tl.load(str(spec))
    _assert_no_host_path(_conditional_source_files(loaded))


def test_capture_advisory_location_honors_source_privacy(tmp_path: Path) -> None:
    """The persisted scalar-escape advisory never carries an absolute path.

    Fail-before (SF5 leak): ``annotations["capture_advisories"]`` rows stored
    the user's escape site as an ABSOLUTE ``"<file>:<line>"`` string, which
    persisted verbatim into ``metadata.pkl`` even at ``include_source=False``.
    The advisory FACT (kind, count, message) must survive both save modes; the
    location is a source-file reference and follows the source policy: bare
    basename with source kept, dropped entirely with source excluded. The live
    trace keeps its session-time absolute location either way.
    """

    trace = _capture()
    live_rows = trace.annotations.get("capture_advisories")
    assert live_rows and live_rows[0]["kind"] == "scalar_escape"
    assert os.path.basename(__file__) in (live_rows[0]["first_location"] or "")

    kept = tmp_path / "advisory_kept.tlspec"
    tl.save(trace, str(kept), include_source=True)
    kept_rows = tl.load(str(kept)).annotations["capture_advisories"]
    assert kept_rows[0]["count"] >= 1
    kept_location = kept_rows[0]["first_location"]
    assert kept_location and kept_location.startswith(os.path.basename(__file__))
    _assert_no_host_path([kept_location.rsplit(":", 1)[0]])

    dropped = tmp_path / "advisory_dropped.tlspec"
    tl.save(trace, str(dropped), include_source=False)
    dropped_rows = tl.load(str(dropped)).annotations["capture_advisories"]
    assert dropped_rows and dropped_rows[0]["kind"] == "scalar_escape"
    assert dropped_rows[0]["first_location"] is None
    # The live trace's session-time annotations were never mutated by either save.
    assert os.path.basename(__file__) in (live_rows[0]["first_location"] or "")
    assert os.path.isabs(live_rows[0]["first_location"].rsplit(":", 1)[0])
