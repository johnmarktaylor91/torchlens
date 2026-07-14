"""Barcode RNG must track ``set_random_seed`` yet stay off the global host stream.

Regression guard for the facet-rerun ``[param_xrefs]`` breakage: internal tensor
barcodes draw from a process-private ``random.Random`` (so they never masquerade as
user Python/NumPy RNG consumption during a forward), but a fixed capture seed still has
to yield a reproducible barcode sequence. A fork replay reuses the original capture
seed, and matching barcodes are what keep tensor/op/param cross-references consistent
between the original and replayed captures. If ``set_random_seed`` stopped seeding the
barcode RNG, the replay drew different barcodes and metadata invariants failed.
"""

import random

import numpy as np
import pytest

from torchlens.utils.hashing import make_random_barcode, seed_barcode_rng
from torchlens.utils.rng import set_random_seed


@pytest.mark.smoke
def test_set_random_seed_makes_barcodes_reproducible() -> None:
    """A fixed seed yields the same barcode sequence (fork-replay determinism)."""
    set_random_seed(1234)
    first = [make_random_barcode() for _ in range(5)]
    set_random_seed(1234)
    second = [make_random_barcode() for _ in range(5)]
    assert first == second

    set_random_seed(9999)
    third = [make_random_barcode() for _ in range(5)]
    assert third != first  # a different seed gives a different sequence


@pytest.mark.smoke
def test_seed_barcode_rng_direct_is_reproducible() -> None:
    """The private seeding helper alone reproduces a sequence without touching seeds."""
    seed_barcode_rng(7)
    a = [make_random_barcode() for _ in range(4)]
    seed_barcode_rng(7)
    b = [make_random_barcode() for _ in range(4)]
    assert a == b


@pytest.mark.smoke
def test_barcode_draws_do_not_advance_global_host_rng() -> None:
    """Drawing barcodes must not perturb global ``random`` / NumPy (host-RNG honesty).

    The runnable capture path brackets the user forward with host-RNG snapshots to
    detect *user* Python/NumPy control flow. TorchLens's own barcode draws must stay on
    the private stream so they never register as user host-RNG consumption.
    """
    set_random_seed(42)
    py_before = random.getstate()
    np_before = np.random.get_state()
    for _ in range(50):
        make_random_barcode()
    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
