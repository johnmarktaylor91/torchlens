"""Family grammar refuses forged cells before record construction, with cell paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _format3_envelope import copy_artifact, save_tiny, set_cell

import torchlens as tl
from torchlens._io.format3.decode import Decoder
from torchlens._io.format3.values import ValueEncoder
from torchlens._io.format_errors import ArtifactError


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a real small trace for cell surgery.

    Parameters
    ----------
    tmp_path_factory:
        Isolated test directory factory.
    """

    root = tmp_path_factory.mktemp("family-schemas") / "trace"
    trace = save_tiny(root)
    trace.cleanup()
    return root


def _forged_cell(saved: Path, target: Path, column: str, value: Any) -> Path:
    """Copy an artifact, replace one cell and restamp its manifest digests.

    Parameters
    ----------
    saved, target:
        Original artifact and mutation destination.
    column, value:
        Trace column and decoded replacement value.
    """

    root = copy_artifact(saved, target)
    cell = ValueEncoder().encode(value, at=f"trace[0].{column}")
    set_cell(root, "trace", column, 0, cell)
    return root


@pytest.mark.parametrize(
    "column,value,code,path",
    [
        ("grad_fn_timing_provenance", "invented_clock", "grad_fn_timing_provenance", ""),
        ("_provenance_distributed_scope", "global", "distributed_scope", ""),
        ("checkpoint_invocation_witness", {}, "checkpoint_witness", ""),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": True,
                "tokens": {},
                "degrade_flags": [],
                "verdict": "no_checkpoint_invocation_observed",
            },
            "checkpoint_witness",
            ".token_count",
        ),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": 0,
                "tokens": {},
                "degrade_flags": [],
                "verdict": "evidence_incomplete",
            },
            "checkpoint_witness",
            ".verdict",
        ),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": 1,
                "tokens": {
                    1: {
                        "pack_count": -1,
                        "unpack_evidence_count": 0,
                        "site_key_candidates": [],
                        "window_evidence": [],
                    }
                },
                "degrade_flags": [],
                "verdict": "checkpoint_invocations_observed",
            },
            "checkpoint_witness",
            ".tokens.1.pack_count",
        ),
        (
            "annotations",
            {
                "capture_advisories": [
                    {
                        "kind": "escape",
                        "count": True,
                        "first_location": None,
                        "message": "m",
                    }
                ]
            },
            "capture_advisories",
            ".capture_advisories[0].count",
        ),
        (
            "annotations",
            {
                "sidecar": {
                    "example.facts": {
                        "schema_id": "facts_v1",
                        "version": False,
                        "owner": "example",
                        "payload": {},
                    }
                }
            },
            "sidecar",
            ".sidecar.example.facts.version",
        ),
        (
            "annotations",
            {
                "_kernel_telemetry": {
                    "_available": True,
                    "_launches": [
                        {
                            "attribution_status": "guessed",
                            "launch_name": None,
                            "device": None,
                            "duration": None,
                            "stream": None,
                            "runtime_correlation": None,
                        }
                    ],
                    "_relations": [],
                }
            },
            "kernel_telemetry",
            "._kernel_telemetry._launches[0].attribution_status",
        ),
        (
            "annotations",
            {
                "_kernel_telemetry": {
                    "_available": False,
                    "_launches": [],
                    "_relations": [],
                }
            },
            "kernel_telemetry",
            "._kernel_telemetry._launches",
        ),
        (
            "_source_snapshots",
            [{"path": "model.py", "digest": "bad", "text": None}],
            "source_snapshots",
            "[0].digest",
        ),
    ],
)
def test_forged_family_cell_refuses_with_code_and_path(
    saved: Path,
    tmp_path: Path,
    column: str,
    value: Any,
    code: str,
    path: str,
) -> None:
    """A digest-valid forgery reaches its family's grammar refusal.

    Parameters
    ----------
    saved, tmp_path:
        Real artifact and isolated mutation directory.
    column, value, code, path:
        Cell mutation, family code stem and expected nested diagnostic path.
    """

    loaded = tl.load(saved)
    loaded.cleanup()
    root = _forged_cell(saved, tmp_path / "forged", column, value)
    with pytest.raises(ArtifactError) as caught:
        tl.load(root)
    assert caught.value.fields["code"] == f"artifact_{code}_invalid"
    assert caught.value.fields["at"] == f"trace[0].{column}{path}"
    assert f"trace[0].{column}{path}" in str(caught.value)


@pytest.mark.parametrize(
    "column,value",
    [
        ("grad_fn_timing_provenance", "unmeasured"),
        ("_provenance_distributed_scope", "rank_local_shard"),
        (
            "checkpoint_invocation_witness",
            {
                "token_count": 0,
                "tokens": {},
                "degrade_flags": (),
                "verdict": "no_checkpoint_invocation_observed",
            },
        ),
        (
            "annotations",
            {
                "capture_advisories": (
                    {
                        "kind": "escape",
                        "count": 1,
                        "first_location": None,
                        "message": "",
                    },
                )
            },
        ),
        (
            "annotations",
            {
                "sidecar": {
                    "example.facts": {
                        "schema_id": "facts_v1",
                        "version": 1,
                        "owner": "example",
                        "payload": {"any": 3},
                    }
                }
            },
        ),
        (
            "annotations",
            {
                "_kernel_telemetry": {
                    "_available": False,
                    "_relations": [],
                    "_launches": [
                        {
                            "attribution_status": "unavailable",
                            "device": None,
                            "launch_name": None,
                            "duration": None,
                            "stream": None,
                            "runtime_correlation": None,
                        }
                    ],
                }
            },
        ),
        ("_source_snapshots", [{"path": "model.py", "digest": "a" * 64, "text": None}]),
    ],
)
def test_valid_family_cells_remain_loadable(
    saved: Path,
    tmp_path: Path,
    column: str,
    value: Any,
) -> None:
    """Admitted tuple containers, optional values and open payloads still load.

    Parameters
    ----------
    saved, tmp_path:
        Baseline artifact and isolated mutation directory.
    column, value:
        Valid replacement cell.
    """

    root = _forged_cell(saved, tmp_path / "valid", column, value)
    loaded = tl.load(root)
    loaded.cleanup()


def test_family_node_checks_do_not_require_record_construction() -> None:
    """Decoding alone rejects grammar before trace state hooks can run.

    Returns
    -------
    None
        Assertions verify the decoder's independent grammar boundary.
    """

    with pytest.raises(ArtifactError) as caught:
        Decoder(defer_records=True).decode("invented", at="trace[2].grad_fn_timing_provenance")
    assert caught.value.fields["code"] == "artifact_grad_fn_timing_provenance_invalid"
    assert caught.value.fields["at"] == "trace[2].grad_fn_timing_provenance"
