"""Identity determinism and exact staleness tests."""

from __future__ import annotations

from typing import Any

from menagerie.crawler.identity import (
    assign_stable_id,
    compute_env_generation,
    compute_evidence_identity,
    compute_execution_identity,
    compute_recipe_revision,
    compute_source_identity,
    stable_hash,
    stale_dependencies,
)


def test_hashes_are_deterministic_and_key_order_independent() -> None:
    """Canonical hashing is deterministic across mapping insertion order."""

    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    stable_id, full_digest = assign_stable_id("menagerie", "ExampleNet/base")
    replay_id, replay_digest = assign_stable_id("menagerie", "ExampleNet/base")
    assert (stable_id, full_digest) == (replay_id, replay_digest)
    assert stable_id.startswith("m_")
    assert len(stable_id) == 22


def test_identity_inputs_change_the_correct_products() -> None:
    """Each byte-level dependency changes its own dependent identity."""

    source_a = compute_source_identity(
        [
            {
                "url": "HTTPS://EXAMPLE.COM/model#readme",
                "revision": "1",
                "locator": "a",
                "content_sha256": "x",
            }
        ],
        {"queries": ["one"]},
    )
    source_b = compute_source_identity(
        [
            {
                "url": "https://example.com/model",
                "revision": "2",
                "locator": "a",
                "content_sha256": "x",
            }
        ],
        {"queries": ["one"]},
    )
    evidence_a = compute_evidence_identity([{"text": "literal", "supports": ["field"]}])
    evidence_b = compute_evidence_identity([{"text": "changed", "supports": ["field"]}])
    assert source_a != source_b
    assert evidence_a != evidence_b
    assert compute_recipe_revision({"symbol": "A"}, source_a) != compute_recipe_revision(
        {"symbol": "A"}, source_b
    )
    assert compute_env_generation({}, "lock-a", "export", {}, []) != compute_env_generation(
        {}, "lock-b", "export", {}, []
    )


def test_staleness_propagates_exactly_to_dependent_facts() -> None:
    """Lock and metadata changes stale disjoint dependent products."""

    lock_change = stale_dependencies(["lock"])
    assert lock_change.stale == frozenset({"environment", "execution", "attempts", "run_status"})
    metadata_change = stale_dependencies(["authored_metadata"])
    assert metadata_change.stale == frozenset({"vet", "accuracy_gate"})
    runner_change = stale_dependencies(["runner"])
    assert runner_change.stale == frozenset({"execution", "attempts", "run_status"})


def test_execution_identity_changes_with_environment() -> None:
    """An environment-generation change invalidates execution identity."""

    common: dict[str, Any] = {
        "stable_id": "m_example",
        "recipe_revision": "recipe",
        "runner_version": "runner",
        "target": "test",
        "machine_class": "cpu",
        "seed_policy": {"seed": 0},
        "framework_adapter": {"framework": "pytorch"},
        "device": "cpu",
    }
    first = compute_execution_identity(env_generation="env-a", **common)
    second = compute_execution_identity(env_generation="env-b", **common)
    assert first != second
