"""tlspec v9 cross-version compatibility proofs (C07 coordinated schema write).

Barrier-S gate rows: the schema field-name diff between the v8 and v9
contracts of record is pinned EXACTLY (reviewable KEEP-policy diff, zero
removals), the harvested v8 golden still loads at its recorded schema under
the v9 runtime, and the two contract digests are distinct -- the committed
per-version contracts are the compatibility ledger's digest inputs (ecosystem
MEMO 3.1: the ledger row carries the digest; F32 builds the ledger over these
committed rows).
"""

from __future__ import annotations

import json
import tarfile
import warnings
from pathlib import Path

import pytest

import torchlens as tl
from torchlens._io import TLSPEC_VERSION

_HERE = Path(__file__).parent
V8_CONTRACT = _HERE / "release_goldens" / "writer_contract_v8.json"
V9_CONTRACT = _HERE.parent / "torchlens" / "schemas" / "writer_contract_v9.json"
CORPUS = _HERE / "release_goldens" / "genuine_release_artifacts.tar.gz"

#: The EXACT persisted-field-name additions of the v9 write, per record
#: contract. A change here is a schema-territory change (C07 fence) and must
#: be a reviewed diff of this pin. The C07X amendment (the ONE coordinated
#: v9 amendment; TLSPEC_VERSION stays 9) adds its entry-dark slots to the
#: same window: Op.episode_step + Op.tl_authored_root and
#: Trace.root_entry_point (written unconditionally at capture; None only on
#: legacy artifacts).
V9_FIELD_ADDITIONS: dict[str, frozenset[str]] = {
    "op": frozenset({"injection_provenance", "episode_step", "tl_authored_root"}),
    "trace": frozenset({"source_snapshots", "structure_evidence", "root_entry_point"}),
}


def _record_contracts(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["contract"]["record_contracts"]


@pytest.mark.smoke
def test_v8_to_v9_field_name_diff_is_exactly_the_pinned_additions() -> None:
    old = _record_contracts(V8_CONTRACT)
    new = _record_contracts(V9_CONTRACT)
    assert set(old) == set(new), "no record contract may appear or disappear at v9"
    for record_key in sorted(new):
        added = set(new[record_key]["persisted_field_names"]) - set(
            old[record_key]["persisted_field_names"]
        )
        removed = set(old[record_key]["persisted_field_names"]) - set(
            new[record_key]["persisted_field_names"]
        )
        assert removed == set(), (
            f"{record_key}: the v9 write removes persisted field(s) {sorted(removed)}; "
            "v9 is add-only (removals need alias rows and a reviewed pin change)"
        )
        assert added == set(V9_FIELD_ADDITIONS.get(record_key, frozenset())), (
            f"{record_key}: v9 additions {sorted(added)} differ from the pinned "
            f"diff {sorted(V9_FIELD_ADDITIONS.get(record_key, frozenset()))}"
        )


def test_contract_versions_and_digests_are_ledger_distinct() -> None:
    old = json.loads(V8_CONTRACT.read_text(encoding="utf-8"))
    new = json.loads(V9_CONTRACT.read_text(encoding="utf-8"))
    assert old["contract"]["tlspec_version"] == 8
    assert new["contract"]["tlspec_version"] == TLSPEC_VERSION == 9
    assert old["writer_contract_digest"] != new["writer_contract_digest"], (
        "the two contracts of record must be digest-distinct (the ledger keys "
        "on the digest, never the stamp alone)"
    )


@pytest.mark.heavy
def test_v8_golden_loads_at_its_recorded_schema_under_v9(tmp_path: Path) -> None:
    """The harvested tlspec-8 main golden loads green under the v9 runtime."""

    with tarfile.open(CORPUS, "r:gz") as tar:
        tar.extractall(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.load(tmp_path / "art_main_portable")
    assert trace.tlspec_version == 8, "a loaded artifact keeps its recorded schema"
    assert trace.outcome.status.name == "COMPLETE"
    # The v9 entry-dark slots default-fill tolerantly on the older artifact.
    assert trace.structure_evidence is None
    assert trace.source_snapshots == []
    assert all(op.injection_provenance is None for op in trace.ops)
