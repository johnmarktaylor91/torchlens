"""Digest primitives + model identity contract tests (extract memo items 1 + 6).

Covers the pinned threaded-Merkle crypto digest (thread-count invariance is
THE thing a Merkle fold can silently get wrong — T-IDENTITY-COST's invariance
half), the order-sensitive value reduction (the collision battery that killed
the sampled-fingerprint proposal — the exact two-word algebraic collision is
provoked and must NOT collide here), and the D6 model-identity record
(T-IDENTITY-BUFFERS: an integer 0-dim buffer mutation must change the digest).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens._data_substrate import (
    MERKLE_ALGORITHM_ID,
    MODEL_IDENTITY_LEVELS,
    MODEL_STATE_DIGEST_ID,
    compare_model_identity,
    compute_model_identity,
    merkle_digest,
    value_reduction,
)
from torchlens._errors import InvalidArgumentError

pytestmark = pytest.mark.smoke


def _entries(n: int = 4, seed: int = 0) -> list[tuple[str, torch.Tensor]]:
    """Build deterministic named-tensor entries.

    Parameters
    ----------
    n:
        Number of entries.
    seed:
        Torch seed.

    Returns
    -------
    list[tuple[str, torch.Tensor]]
        Ordered ``(name, tensor)`` pairs of mixed shapes and dtypes.
    """

    torch.manual_seed(seed)
    return [
        ("w", torch.randn(8, 3)),
        ("b", torch.randn(8)),
        ("count", torch.tensor(7, dtype=torch.int64)),
        ("flag", torch.tensor(True)),
    ][:n]


# --- threaded Merkle digest ---------------------------------------------------


def test_merkle_digest_is_thread_count_invariant() -> None:
    """Identical digest at 1, 4, and 32 workers (the Merkle silent-wrong case)."""

    entries = _entries()
    digests = {merkle_digest(entries, max_workers=workers).digest for workers in (1, 4, 32, None)}
    assert len(digests) == 1


def test_merkle_digest_is_order_and_content_sensitive() -> None:
    """Reordering entries, renaming, or flipping one byte changes the digest."""

    base = merkle_digest(_entries()).digest
    assert merkle_digest(list(reversed(_entries()))).digest != base
    renamed = _entries()
    renamed[0] = ("w2", renamed[0][1])
    assert merkle_digest(renamed).digest != base
    flipped = _entries()
    flipped[0][1].view(-1)[0] += 1.0
    assert merkle_digest(flipped).digest != base


def test_merkle_digest_folds_shape_and_dtype_not_just_bytes() -> None:
    """Two byte-identical payloads with different geometry digest differently."""

    flat = torch.arange(6, dtype=torch.float32)
    assert (
        merkle_digest([("t", flat.reshape(2, 3))]).digest
        != merkle_digest([("t", flat.reshape(3, 2))]).digest
    )
    assert (
        merkle_digest([("t", flat)]).digest != merkle_digest([("t", flat.view(torch.int32))]).digest
    )


def test_merkle_digest_record_and_hash_selection() -> None:
    """The record carries the pinned algorithm id/version; sha256 selectable."""

    record = merkle_digest(_entries(), hash_name="sha256").record()
    assert record["algorithm_id"] == MERKLE_ALGORITHM_ID
    assert record["algorithm_version"] == 1
    assert record["hash"] == "sha256"
    assert str(record["digest"]).startswith("sha256:")
    assert record["n_leaves"] == 4
    empty = merkle_digest([])
    assert empty.n_leaves == 0 and empty.digest.startswith("blake2b:")


def test_merkle_digest_unknown_hash_refuses_typed() -> None:
    """The hash name is a recomputation contract: closed vocabulary."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        merkle_digest(_entries(), hash_name="md5")
    assert excinfo.value.fields["code"] == "digest_hash_invalid"


# --- order-sensitive value reduction ------------------------------------------


def test_value_reduction_collision_battery() -> None:
    """The battery that killed the sampled fingerprint: none of these collide.

    The two-word algebraic collision (fp([0, 0]) == fp([-G, 1]) for the
    rejected sum-based fingerprint, constructed without search) plus swaps,
    rolls, and reversals — each must produce a DIFFERENT reduction, because
    the Horner fold weights every byte by position.
    """

    base_tensor = torch.arange(16384, dtype=torch.float32)
    base = value_reduction("t", base_tensor)

    def variant(mutate) -> str:
        t = base_tensor.clone()
        mutate(t)
        return value_reduction("t", t)

    def swap(t: torch.Tensor, i: int, j: int) -> None:
        t[i], t[j] = t[j].item(), t[i].item()

    assert variant(lambda t: swap(t, 0, 1)) != base, "adjacent swap"
    assert variant(lambda t: swap(t, 0, 4096)) != base, "distant swap"
    assert value_reduction("t", torch.flip(base_tensor, dims=(0,))) != base, "full reverse"
    assert value_reduction("t", torch.roll(base_tensor, 1)) != base, "roll by 1"
    shuffled = torch.roll(base_tensor, 7)
    assert value_reduction("t", shuffled.msort()) != value_reduction("t", shuffled), "sort"
    # The exact two-word algebraic collision class: zero-sum rearrangements.
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([-1.0, 1.0])
    assert value_reduction("t", a) != value_reduction("t", b)


def test_value_reduction_folds_name_shape_and_dtype() -> None:
    """The per-tensor header fold distinguishes name, geometry, and dtype."""

    t = torch.arange(6, dtype=torch.float32)
    assert value_reduction("a", t) != value_reduction("b", t)
    assert value_reduction("a", t.reshape(2, 3)) != value_reduction("a", t.reshape(3, 2))


def test_value_reduction_block_boundaries_and_zero_dim() -> None:
    """Sizes straddling the 65536-byte block slice stay well-defined and distinct."""

    for n in (65535, 65536, 65537):
        t = torch.zeros(n, dtype=torch.uint8).view(torch.uint8)
        t2 = t.clone()
        t2[n - 1] = 1
        assert value_reduction("t", t) != value_reduction("t", t2), n
    scalar = value_reduction("s", torch.tensor(3.0))
    assert scalar.startswith("0x") and len(scalar) == 18
    assert value_reduction("s", torch.tensor([], dtype=torch.float32)).startswith("0x")


def test_value_reduction_is_deterministic_across_calls() -> None:
    """Same name + tensor -> same hex, every time (a recomputation contract)."""

    t = torch.arange(100, dtype=torch.float16)
    assert value_reduction("x", t) == value_reduction("x", t.clone())


# --- model identity (D6) --------------------------------------------------------


class _BnModel(nn.Module):
    """Tiny model with an integer 0-dim persistent buffer (num_batches_tracked)."""

    def __init__(self) -> None:
        """Build a conv + batchnorm stack deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv then batchnorm.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Normalized features.
        """

        return self.bn(self.conv(x))


def test_measured_identity_covers_every_state_entry() -> None:
    """The digest covers params AND persistent buffers, int/bool/0-dim included."""

    model = _BnModel()
    record = compute_model_identity(model)
    assert record["level"] == "measured"
    assert record["algorithm_id"] == MODEL_STATE_DIGEST_ID
    assert record["n_state_entries"] == len(model.state_dict())
    assert record["digest"].startswith("blake2b:")


def test_resume_identity_integer_buffer_mutation_changes_the_digest() -> None:
    """T-IDENTITY-BUFFERS: mutate num_batches_tracked -> identity changes."""

    model = _BnModel()
    before = compute_model_identity(model)["digest"]
    model.bn.num_batches_tracked += 1
    after = compute_model_identity(model)["digest"]
    assert before != after


def test_resume_identity_weight_edit_changes_the_digest() -> None:
    """An in-place weight edit (LoRA-merge / lesion class) changes the identity."""

    model = _BnModel()
    before = compute_model_identity(model)["digest"]
    with torch.no_grad():
        model.conv.weight[0, 0, 0, 0] += 1.0
    assert compute_model_identity(model)["digest"] != before


def test_identity_levels_and_refusals() -> None:
    """The closed level vocabulary; 'sampled' is not offered."""

    model = _BnModel()
    assert MODEL_IDENTITY_LEVELS == ("measured", "asserted", "none")
    none_record = compute_model_identity(model, level="none")
    assert none_record["level"] == "none" and "digest" not in none_record
    asserted = compute_model_identity(
        model, level="asserted", assertion={"checkpoint": "org/m", "revision": "abc"}
    )
    assert asserted["level"] == "asserted"
    assert asserted["assertion"] == {"checkpoint": "org/m", "revision": "abc"}
    with pytest.raises(InvalidArgumentError) as excinfo:
        compute_model_identity(model, level="sampled")
    assert excinfo.value.fields["code"] == "extraction_model_identity_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        compute_model_identity(model, level="asserted")
    assert excinfo.value.fields["code"] == "extraction_model_identity_invalid"


def test_meta_device_state_degrades_to_unavailable() -> None:
    """Unmeasurable state records 'unavailable' (resume refuses on it) unless asserted."""

    with torch.device("meta"):
        meta_model = nn.Linear(4, 2)
    record = compute_model_identity(meta_model)
    assert record["level"] == "unavailable"
    assert "meta" in record["measurement_unavailable_reason"]
    asserted = compute_model_identity(meta_model, assertion={"checkpoint": "org/m"})
    assert asserted["level"] == "asserted"
    assert asserted["measurement_unavailable_reason"]


def test_hub_identity_is_provenance_never_the_check() -> None:
    """config._commit_hash is recorded defensively, independent of the level."""

    model = _BnModel()

    class _Config:
        _commit_hash = "deadbeef"

    model.config = _Config()  # type: ignore[assignment]
    record = compute_model_identity(model, level="none")
    assert record["hub_identity"] == "deadbeef"
    assert record["hub_identity_source"] == "config._commit_hash"


def test_compare_model_identity_level_rules() -> None:
    """The D6 comparison matrix: same-level compares, cross-level refuses."""

    measured_a = {
        "level": "measured",
        "digest": "blake2b:aa",
        "algorithm_id": "x",
        "algorithm_version": 1,
    }
    measured_b = {
        "level": "measured",
        "digest": "blake2b:bb",
        "algorithm_id": "x",
        "algorithm_version": 1,
    }
    assert compare_model_identity(measured_a, dict(measured_a)) is None
    assert compare_model_identity(measured_a, measured_b) == "digest"
    assert compare_model_identity(measured_a, {"level": "none"}) == "level"
    assert compare_model_identity({"level": "unavailable"}, measured_a) == "unavailable"
    assert compare_model_identity({"level": "none"}, {"level": "none"}) is None
    asserted_a = {"level": "asserted", "assertion": {"checkpoint": "a"}}
    asserted_b = {"level": "asserted", "assertion": {"checkpoint": "b"}}
    assert compare_model_identity(asserted_a, dict(asserted_a)) is None
    assert compare_model_identity(asserted_a, asserted_b) == "assertion"
    assert compare_model_identity("junk", measured_a) == "level"
