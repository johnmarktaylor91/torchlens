"""Lane F22 neuro.rdms two-mode contract (neuro MEMO 4.2/D8/D9, rows T5/T15).

Pins:

- SOURCE MODE computes through the canonical repgeom arithmetic, defaults
  to ``metric="correlation"``, records ``measure_source="computed"`` and
  the ``measure_convention`` formula (D9), and hands rsatoolbox a stack
  whose condensed rows equal its own ``calc_rdm`` output exactly.
- The T5 oracles on real features: correlation EQUALITY with rsatoolbox,
  the euclidean RELATION (theirs == ours^2 / n_features), and the
  fails-loudly pin that rsatoolbox has no cosine RDM method.
- MATRIX MODE requires the declared measure, validates
  square/symmetry/diagonal/finite/shape agreement, never invents pattern
  identity, and both modes reject mixed arguments.
- T15 anti-divergence: source-mode site selection is IDENTICAL (modulo the
  annotation-key prefix) to ``tl.repgeom.rdm_evolution`` on the same
  trace, and the same exception classes fire on the same bad inputs.
- The reserved ``chunked=`` execution seam refuses typed at launch.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.neuro._handoff import NeuroHandoffError

rsatoolbox = pytest.importorskip("rsatoolbox")

from torchlens.neuro._datasets import datasets  # noqa: E402
from torchlens.neuro._rdms import rdms  # noqa: E402


class _BNModel(nn.Module):
    """Toy model whose BatchNorm mints buffer sites on save-everything."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.bn(self.conv(x)))
        return self.fc(hidden.mean(dim=(2, 3)))


def _full_trace(n_stimuli: int = 6) -> tl.Trace:
    """Save-everything trace with buffer sites present."""

    torch.manual_seed(0)
    model = _BNModel().eval()
    x = torch.randn(n_stimuli, 3, 8, 8)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


def _matrices() -> dict[str, np.ndarray]:
    """Two valid square symmetric zero-diagonal matrices."""

    a = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    b = np.array([[0.0, 0.5, 0.2], [0.5, 0.0, 0.1], [0.2, 0.1, 0.0]])
    return {"site_a": a, "site_b": b}


@pytest.mark.heavy
def test_source_mode_defaults_and_provenance() -> None:
    """D8/D9: correlation default, computed measure, convention recorded."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rdms(log)
    assert result.dissimilarity_measure == "correlation"
    assert result.descriptors["measure_source"] == "computed"
    assert "Pearson" in result.descriptors["measure_convention"]
    assert result.descriptors["pattern_identity"] == "synthetic_positional"
    assert list(result.pattern_descriptors["tl_presentation_index"]) == list(range(6))
    assert "site" in result.rdm_descriptors and "pool" in result.rdm_descriptors
    assert all(pool == "flatten" for pool in result.rdm_descriptors["pool"])
    assert len(result.tl_ledger) >= result.n_rdm
    assert not any("buffer" in site for site in result.rdm_descriptors["site"])


@pytest.mark.heavy
def test_t5_oracles_correlation_equality_and_euclidean_relation() -> None:
    """T5: equality pinned for correlation; the RELATION for euclidean."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site_datasets = datasets(log, ["relu_1_3"])
        ours_corr = rdms(log, ["relu_1_3"])
        ours_eucl = rdms(log, ["relu_1_3"], metric="euclidean")
    dataset = site_datasets["relu_1_3"]
    n_features = dataset.measurements.shape[1]

    theirs_corr = rsatoolbox.rdm.calc_rdm(dataset, method="correlation")
    assert np.allclose(theirs_corr.dissimilarities[0], ours_corr.dissimilarities[0], atol=1e-6)

    theirs_eucl = rsatoolbox.rdm.calc_rdm(dataset, method="euclidean")
    assert np.allclose(
        theirs_eucl.dissimilarities[0],
        ours_eucl.dissimilarities[0] ** 2 / n_features,
        atol=1e-6,
    )


@pytest.mark.heavy
def test_t5_oracle_rsatoolbox_has_no_cosine_rdm() -> None:
    """T5: our "they have no cosine RDM" claim fails loudly if added."""

    dataset = rsatoolbox.data.Dataset(measurements=np.random.default_rng(0).normal(size=(4, 5)))
    with pytest.raises(NotImplementedError):
        rsatoolbox.rdm.calc_rdm(dataset, method="cosine")


@pytest.mark.heavy
def test_t15_anti_divergence_with_rdm_evolution() -> None:
    """T15: same site selection and exception classes as the repgeom verb."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verb_result = tl.repgeom.rdm_evolution(log, metric="correlation")
        neuro_result = rdms(log)
    verb_sites = [key.split(":", 1)[1] for key in verb_result]
    assert list(neuro_result.rdm_descriptors["site"]) == verb_sites

    # Same exception class on the same explicit ineligible site.
    with pytest.raises(ValueError) as verb_error:
        tl.repgeom.rdm_evolution(log, save=tl.label("buffer_1"))
    with pytest.raises(ValueError) as neuro_error:
        rdms(log, ["buffer_1"])
    assert type(verb_error.value) is type(neuro_error.value)

    # Same exception class on an explicitly requested UNSAVED site.
    torch.manual_seed(0)
    partial = tl.trace(_BNModel().eval(), torch.randn(6, 3, 8, 8), save=tl.func("relu"))
    with pytest.raises(ValueError) as verb_unsaved:
        tl.repgeom.rdm_evolution(partial, save=tl.label("conv2d_1_1"))
    with pytest.raises(ValueError) as neuro_unsaved:
        rdms(partial, ["conv2d_1_1"])
    assert type(verb_unsaved.value) is type(neuro_unsaved.value)


@pytest.mark.heavy
def test_t15_values_match_rdm_evolution() -> None:
    """The compute path IS repgeom's: identical matrices per site."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verb_result = tl.repgeom.rdm_evolution(log, metric="correlation")
        neuro_result = rdms(log)
    for row_index, (key, square) in enumerate(zip(verb_result, verb_result.values(), strict=True)):
        square = verb_result[key]
        condensed = square[np.triu_indices(square.shape[0], k=1)]
        assert np.allclose(neuro_result.dissimilarities[row_index], condensed, atol=1e-12), key


@pytest.mark.heavy
def test_source_mode_never_mutates_annotations() -> None:
    """The handoff is a treaty desk: no annotation blobs are written."""

    log = _full_trace()
    before = dict(getattr(log, "_annotation_blobs", None) or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rdms(log)
    after = dict(getattr(log, "_annotation_blobs", None) or {})
    assert before == after


@pytest.mark.smoke
def test_matrix_mode_requires_declared_measure() -> None:
    """D8: a bare matrix has forgotten its metric; declaration is REQUIRED."""

    with pytest.raises(NeuroHandoffError) as excinfo:
        rdms(_matrices())
    assert excinfo.value.fields["code"] == "neuro_rdms_measure_required"


@pytest.mark.smoke
def test_matrix_mode_happy_path_and_positional_disclosure() -> None:
    """Matrix mode converts, declares, and discloses positional identity."""

    result = rdms(_matrices(), dissimilarity_measure="euclidean")
    assert result.n_rdm == 2 and result.n_cond == 3
    assert result.dissimilarity_measure == "euclidean"
    assert result.descriptors["measure_source"] == "declared"
    assert result.descriptors["pattern_identity"] == "positional"
    assert list(result.pattern_descriptors["tl_pattern_index"]) == [0, 1, 2]
    assert list(result.rdm_descriptors["site"]) == ["site_a", "site_b"]
    expected = _matrices()["site_a"][np.triu_indices(3, k=1)]
    assert np.allclose(result.dissimilarities[0], expected)


@pytest.mark.smoke
def test_matrix_mode_user_pattern_descriptors_validated() -> None:
    """Supplied pattern descriptors are length-checked and win over index."""

    result = rdms(
        _matrices(),
        dissimilarity_measure="correlation",
        pattern_descriptors={"stimulus_id": ["x", "y", "z"]},
    )
    assert result.descriptors["pattern_identity"] == "user_supplied"
    assert list(result.pattern_descriptors["stimulus_id"]) == ["x", "y", "z"]
    with pytest.raises(NeuroHandoffError) as excinfo:
        rdms(
            _matrices(),
            dissimilarity_measure="correlation",
            pattern_descriptors={"stimulus_id": ["x", "y"]},
        )
    assert excinfo.value.fields["code"] == "neuro_obs_descriptor_invalid"


@pytest.mark.smoke
def test_matrix_mode_validation_refusals() -> None:
    """Square/symmetry/diagonal/finite/shape-agreement all refuse typed."""

    cases: list[dict[str, np.ndarray]] = [
        {"bad": np.zeros((2, 3))},
        {"bad": np.zeros((1, 1))},
        {"bad": np.array([[0.0, 1.0], [2.0, 0.0]])},
        {"bad": np.array([[1.0, 2.0], [2.0, 1.0]])},
        {"bad": np.array([[0.0, np.nan], [np.nan, 0.0]])},
        {"a": np.zeros((3, 3)), "b": np.zeros((4, 4))},
        {},
    ]
    for mapping in cases:
        with pytest.raises(NeuroHandoffError) as excinfo:
            rdms(mapping, dissimilarity_measure="euclidean")
        assert excinfo.value.fields["code"] == "neuro_matrix_invalid", mapping
    # The shape-agreement case fires on the SECOND matrix; zeros are a
    # degenerate-but-symmetric valid diagonal, so pin one passing zeros row.
    zeros = rdms({"a": np.zeros((3, 3))}, dissimilarity_measure="euclidean")
    assert zeros.n_rdm == 1


@pytest.mark.smoke
def test_mode_conflicts_reject_mixed_arguments() -> None:
    """D8: the modes reject mixed arguments in both directions."""

    for kwargs in (
        {"metric": "correlation", "dissimilarity_measure": "x"},
        {"sites": ["a"], "dissimilarity_measure": "x"},
        {"pool": "flatten", "dissimilarity_measure": "x"},
        {"obs": {"condition": [1, 2, 3]}, "dissimilarity_measure": "x"},
    ):
        with pytest.raises(NeuroHandoffError) as excinfo:
            rdms(_matrices(), **kwargs)  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "neuro_rdms_mode_conflict", kwargs


@pytest.mark.heavy
def test_source_mode_rejects_declared_measure() -> None:
    """Source mode computes; a hand-typed measure label refuses typed."""

    log = _full_trace()
    with pytest.raises(NeuroHandoffError) as excinfo:
        rdms(log, dissimilarity_measure="correlation")
    assert excinfo.value.fields["code"] == "neuro_rdms_mode_conflict"


@pytest.mark.heavy
def test_source_mode_inconsistent_row_counts_refuse() -> None:
    """Sites disagreeing on stimulus rows cannot share one RDMs stack."""

    class TwoStream(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(3, 4)
            self.right = nn.Linear(3, 4)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a = torch.relu(self.left(x))
            b = torch.sigmoid(self.right(y))
            return a.sum() + b.sum()

    torch.manual_seed(0)
    log = tl.trace(
        TwoStream().eval(),
        (torch.randn(6, 3), torch.randn(4, 3)),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    with pytest.raises(NeuroHandoffError) as excinfo, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rdms(log, ["relu_1_2", "sigmoid_1_4"])
    assert excinfo.value.fields["code"] == "neuro_sites_inconsistent_rows"
    assert excinfo.value.fields["row_counts"] == [4, 6]


@pytest.mark.smoke
def test_chunked_seam_refuses_typed() -> None:
    """The reserved chunked= execution seam refuses at launch (eager)."""

    with pytest.raises(NeuroHandoffError) as excinfo:
        rdms(_matrices(), dissimilarity_measure="euclidean", chunked=8)
    assert excinfo.value.fields["code"] == "neuro_rdms_chunked_unavailable"


@pytest.mark.heavy
def test_source_mode_gaussian_metric_records_convention() -> None:
    """The new gaussian metric flows through with its convention recorded."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rdms(log, ["relu_1_3"], metric="gaussian")
    assert result.dissimilarity_measure == "gaussian"
    assert "thingsvision" in result.descriptors["measure_convention"]
    assert float(result.dissimilarities.max()) < 1.0


@pytest.mark.heavy
def test_source_mode_unsupported_metric_matches_repgeom_class() -> None:
    """An unsupported metric raises repgeom's ValueError class."""

    log = _full_trace()
    with (
        pytest.raises(ValueError, match="Unsupported activation distance metric"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        rdms(log, ["relu_1_3"], metric="minkowski")
