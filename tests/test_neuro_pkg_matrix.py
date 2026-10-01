"""Lane F22 rsatoolbox convention matrix (neuro MEMO item 16, rows T6/T11/T12).

The entire provenance claim rests on a foreign package's behaviour across
the declared version range, so these rows are the BOTH-VERSIONS gate: the
suite must pass with rsatoolbox 0.1.5 (the declared floor) AND current
0.3.x installed. Pins:

- T6: Dataset descriptors PROMOTE into calc_rdm output (descriptors are
  the product); the condensed ordering equals ``triu_indices(n, k=1)``;
  ``dissimilarity_measure`` is recorded; hand-built RDMs construct; the
  recorded version comes from ``importlib.metadata`` (neither validated
  version exposes ``__version__``).
- T11: rsatoolbox's ``calc_rdm(descriptor=...)`` returns rows in SORTED
  descriptor order -- with non-sort-stable string ids the presentation
  order is asserted RECOVERABLE from our always-written
  ``tl_presentation_index`` (D10), under both calling conventions.
- T12: repeated ids under ``calc_rdm(descriptor=...)`` average
  MEASUREMENTS within condition and then compute dissimilarity -- the
  composition asserts a NUMBER against a manual mean-then-correlate, not
  an intention (D11).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl

rsatoolbox = pytest.importorskip("rsatoolbox")

from torchlens.neuro._datasets import datasets  # noqa: E402
from torchlens.neuro._handoff import rsatoolbox_version  # noqa: E402
from torchlens.neuro._rdms import rdms  # noqa: E402


def _relu_trace(n_stimuli: int = 6) -> tl.Trace:
    """Small trace with one saved relu site."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU()).eval()
    x = torch.randn(n_stimuli, 3)
    return tl.trace(model, x, save=tl.func("relu"))


@pytest.mark.smoke
def test_t6_version_recording_via_importlib_metadata() -> None:
    """Neither validated version exposes __version__; metadata serves."""

    version = rsatoolbox_version()
    assert version not in ("", "unknown")
    assert version[0].isdigit()


@pytest.mark.heavy
def test_t6_descriptor_promotion_into_calc_rdm() -> None:
    """Everything the handoff writes survives into the user's RDMs."""

    log = _relu_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site_datasets = datasets(log, ["relu_1_2"])
    result = rsatoolbox.rdm.calc_rdm(site_datasets["relu_1_2"], method="correlation")
    promoted = {**result.descriptors, **result.rdm_descriptors}
    for key in ("site", "pool", "tl_version", "rsatoolbox_version", "source_dtype"):
        assert key in promoted, key
    assert result.descriptors.get("pool") == "flatten" or list(
        result.rdm_descriptors.get("pool", [])
    ) == ["flatten"]


@pytest.mark.heavy
def test_t6_condensed_ordering_matches_rsatoolbox() -> None:
    """Our condensed converter equals rsatoolbox's own vector ordering."""

    log = _relu_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site_datasets = datasets(log, ["relu_1_2"])
        ours = rdms(log, ["relu_1_2"])
    theirs = rsatoolbox.rdm.calc_rdm(site_datasets["relu_1_2"], method="correlation")
    square = theirs.get_matrices()[0]
    assert np.allclose(theirs.dissimilarities[0], square[np.triu_indices(square.shape[0], k=1)])
    assert np.allclose(theirs.dissimilarities[0], ours.dissimilarities[0], atol=1e-6)


@pytest.mark.smoke
def test_t6_hand_built_rdms_construction() -> None:
    """Matrix mode's RDMs constructor contract holds at this version."""

    square = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    result = rdms({"external": square}, dissimilarity_measure="declared_metric")
    assert result.n_rdm == 1 and result.n_cond == 3
    assert result.dissimilarity_measure == "declared_metric"
    assert np.allclose(result.get_matrices()[0], square)


@pytest.mark.heavy
def test_t11_sorted_output_recoverable_from_presentation_index() -> None:
    """D10: non-sort-stable ids permute rows; our index recovers them."""

    log = _relu_trace()
    ids = ["s10", "s2", "s1", "s30", "s4", "s3"]  # sorts differently
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site_datasets = datasets(log, ["relu_1_2"], stimulus_ids=ids)
        presentation_order = rdms(log, ["relu_1_2"])
    dataset = site_datasets["relu_1_2"]

    # Calling convention 1: descriptor= (sorted output, the trap).
    sorted_result = rsatoolbox.rdm.calc_rdm(dataset, descriptor="stimulus_id", method="correlation")
    permutation = np.asarray(sorted_result.pattern_descriptors["tl_presentation_index"], dtype=int)
    assert list(permutation) != list(range(6)), "ids chosen non-sort-stable"
    inverse = np.argsort(permutation)
    recovered = sorted_result.get_matrices()[0][np.ix_(inverse, inverse)]
    assert np.allclose(recovered, presentation_order.get_matrices()[0], atol=1e-6)

    # Calling convention 2: no descriptor= (presentation order preserved).
    plain_result = rsatoolbox.rdm.calc_rdm(dataset, method="correlation")
    assert np.allclose(
        plain_result.get_matrices()[0],
        presentation_order.get_matrices()[0],
        atol=1e-6,
    )


@pytest.mark.heavy
def test_t12_condition_averaging_is_mean_then_dissimilarity() -> None:
    """D11: repeated ids average measurements, then compute -- a NUMBER."""

    log = _relu_trace()
    repeated = ["a", "a", "b", "b", "c", "c"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site_datasets = datasets(log, ["relu_1_2"], stimulus_ids=repeated)
    dataset = site_datasets["relu_1_2"]
    averaged = rsatoolbox.rdm.calc_rdm(
        dataset, descriptor="stimulus_id", method="correlation"
    ).get_matrices()[0]

    measurements = dataset.measurements
    means = np.stack(
        [
            measurements[0:2].mean(axis=0),
            measurements[2:4].mean(axis=0),
            measurements[4:6].mean(axis=0),
        ]
    )
    from torchlens.repgeom import rdm as tl_rdm

    manual = tl_rdm(means, metric="correlation")
    assert np.allclose(averaged, manual, atol=1e-12)
