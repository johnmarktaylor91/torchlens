"""Artifact-registry contract tests (testing MEMO build rows A1/A2).

The registry is the ONE authority for real-model artifacts: closed row
schema, pinned revisions and per-blob digests, never-count enforcement for
R0 rows, the exact-key cache digest, the byte-budget cap formula, and the
consumes-never-constructs rule enforced as an AST lint over this tree.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json

import pytest

from tests.real_model.registry import (
    NATURAL_INPUTS_DIR,
    PERMISSIVE_LICENSES,
    REGISTRY_DIR,
    REGISTRY_PATH,
    VENDORED_DIR,
    CheckpointClaimError,
    RegistryError,
    resolved_config_fingerprint,
)

pytestmark = pytest.mark.real_model

# Files licensed to call pretrained-artifact loaders, and only with a
# registry-pinned revision= (the acquisition rule, memo 4.3).
LOADER_ALLOWLIST = ("r1/conftest.py",)
CONSTRUCTOR_CALLS = frozenset(
    {"from_pretrained", "hf_hub_download", "snapshot_download", "load_state_dict_from_url"}
)


def test_registry_loads_and_ids_are_unique(artifact_registry):
    ids = [row.artifact_id for row in artifact_registry.rows]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 30  # 14 families + fixtures + the R1/R2 slate


def test_registry_digest_is_the_exact_cache_key_material(artifact_registry):
    digest = artifact_registry.digest()
    assert digest == hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    key = artifact_registry.cache_key()
    assert key.startswith("torchlens-artifacts-v1-")
    assert digest[:32] in key
    # Any row edit moves the key: no prefix fallback can serve stale blobs.
    tampered = REGISTRY_PATH.read_bytes().replace(b"r0-gpt2", b"r0-gpt2x", 1)
    assert hashlib.sha256(tampered).hexdigest() != digest


def test_randinit_rows_never_satisfy_checkpoint_claims(artifact_registry):
    for row in artifact_registry.rows:
        if row.band == "R0":
            assert not row.satisfies_checkpoint_claim
            with pytest.raises(CheckpointClaimError):
                artifact_registry.checkpoint_evidence(row.artifact_id)
    # And the R1 core rows DO serve as checkpoint evidence.
    assert artifact_registry.checkpoint_evidence("r1-distilgpt2").band == "R1"


def test_unknown_artifact_refuses_with_teaching_message(artifact_registry):
    with pytest.raises(RegistryError, match="consumes, never constructs"):
        artifact_registry.get("r9-not-a-row")


def test_hub_rows_are_fully_pinned(artifact_registry):
    for row in artifact_registry.rows:
        if row.kind != "hf_hub":
            continue
        assert row.revision, f"{row.artifact_id}: hub row without a pinned revision"
        assert row.blobs, f"{row.artifact_id}: hub row without blobs"
        for blob in row.blobs:
            assert blob.sha256, f"{row.artifact_id}/{blob.filename}: missing sha256"
            assert blob.size_bytes, f"{row.artifact_id}/{blob.filename}: missing size"


def test_redistributable_requires_permissive_license(artifact_registry):
    for row in artifact_registry.rows:
        if row.redistributable:
            assert row.license in PERMISSIVE_LICENSES, (
                f"{row.artifact_id}: redistributable without a permissive license;"
                " only permissive rows may enter the Release asset"
            )


def test_vendored_files_match_their_ledgered_digests():
    ledger = [
        json.loads(line)
        for line in (VENDORED_DIR / "PROVENANCE.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert ledger, "empty vendored provenance ledger"
    for row in ledger:
        path = VENDORED_DIR / row["family"] / row["filename"]
        assert path.exists(), f"vendored file missing: {path}"
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], (
            f"vendored file drifted from its provenance digest: {path}"
        )
        if row.get("stored_encoding") == "gzip":
            decompressed = gzip.decompress(data)
            assert hashlib.sha256(decompressed).hexdigest() == row["decompressed_sha256"]


def test_registry_randinit_blobs_are_the_vendored_files(artifact_registry):
    for row in artifact_registry.rows:
        if row.kind != "local_class" or not row.blobs:
            continue
        for blob, path in zip(row.blobs, row.vendored_paths(), strict=True):
            assert path.exists(), f"{row.artifact_id}: vendored blob missing {path}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == blob.sha256, (
                f"{row.artifact_id}: vendored blob digest drifted for {blob.filename}"
            )


def test_natural_input_assets_exist_and_are_licensed(artifact_registry):
    prompts = {
        json.loads(line)["id"]
        for line in (NATURAL_INPUTS_DIR / "prompts.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert {"clean-factual-1", "left-padded-batch-1", "clip-captions-1"} <= prompts
    image = NATURAL_INPUTS_DIR / "pd_astronaut_256.jpg"
    assert image.exists()
    licenses = (NATURAL_INPUTS_DIR / "LICENSES.md").read_text()
    assert "ni-image-pd-1" in licenses and "Public domain" in licenses
    referenced = {
        natural_id for row in artifact_registry.rows for natural_id in row.natural_input_ids
    }
    for natural_id in referenced:
        assert natural_id in {"ni-prompts-v1", "ni-image-pd-1"}, (
            f"registry references unledgered natural input {natural_id!r}"
        )


def test_nightly_cache_cap_formula(artifact_registry):
    nightly_rows = artifact_registry.rows_for_venue("nightly")
    assert nightly_rows, "empty nightly venue"
    cap = artifact_registry.nightly_cap_bytes()
    total = sum(
        row.measured_bytes or sum(blob.size_bytes or 0 for blob in row.blobs)
        for row in nightly_rows
    )
    assert cap == int(total * 1.2)
    assert total < cap < 10 * 1024**3, (
        "the set+20% cap must leave real headroom and stay under the free"
        " 10 GB Actions ceiling (memo 4.3)"
    )


def test_resolved_config_fingerprint_is_canonical():
    fp_a = resolved_config_fingerprint(config={"b": 1, "a": 2})
    fp_b = resolved_config_fingerprint(config={"a": 2, "b": 1})
    assert fp_a == fp_b
    assert fp_a != resolved_config_fingerprint(config={"a": 2, "b": 2})
    with pytest.raises(RegistryError):
        resolved_config_fingerprint()


def test_consumes_never_constructs_ast_lint():
    """No test under tests/real_model/ constructs an artifact ad hoc.

    Pretrained-loader calls (``from_pretrained``/``hf_hub_download``/...)
    are licensed ONLY in the r1 loader conftest, and there every call must
    carry an explicit ``revision=`` keyword (fed from a registry row).
    """

    offenders: list[str] = []
    for path in sorted(REGISTRY_DIR.rglob("*.py")):
        rel = str(path.relative_to(REGISTRY_DIR))
        tree = ast.parse(path.read_text(), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in CONSTRUCTOR_CALLS:
                continue
            if rel not in LOADER_ALLOWLIST:
                offenders.append(f"{rel}:{node.lineno} calls {name} outside the loader")
                continue
            if name == "from_pretrained" and not any(kw.arg == "revision" for kw in node.keywords):
                offenders.append(f"{rel}:{node.lineno} {name} without revision=")
    assert not offenders, (
        "consumes-never-constructs violations (memo 4.3: tests consume registry"
        f" rows; nothing constructs artifacts ad hoc): {offenders}"
    )


def test_vendored_gpt2_tokenizer_loads_offline(tmp_path):
    from tests.real_model.r0.families import load_vendored_gpt2_tokenizer

    tokenizer = load_vendored_gpt2_tokenizer(str(tmp_path))
    prompt = json.loads((NATURAL_INPUTS_DIR / "prompts.jsonl").read_text().splitlines()[0])["clean"]
    ids = tokenizer(prompt)["input_ids"]
    assert len(ids) >= 5
    assert tokenizer.decode(ids) == prompt
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(["short", prompt], padding=True, return_tensors="pt")
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"][0, 0].item() == 0  # left padding really applied
