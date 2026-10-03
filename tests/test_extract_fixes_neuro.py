"""Lane A11 neuro fails-open fixes (neuro MEMO item 4 defect slice, D4-D7).

Pins:

- The *_evolution default sweep processes only stimulus-indexed sites: buffer
  overwrites and leading-axis mismatches are SKIPPED with one summarized
  disclosure (previously a stock resnet18 sweep fabricated 78 buffer
  pseudo-RDMs into core trace state).
- An EXPLICITLY selected ineligible site refuses actionably, naming the site,
  the evidence, and the remediation (D5).
- Annotation writes are GATED and ATOMIC across the verbs sharing the write
  helper: a failure at site k leaves ZERO annotations behind (D4; a forced
  late failure used to leave exactly the fabricated blobs).
- Shared-helper diagnostics name the verb actually called, never
  ``mds_evolution`` from inside ``rdm_evolution`` (D-shared-text), and
  zero-norm metric errors name the failing site and stimulus rows.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import repgeom
from torchlens.errors._base import TorchLensWarning


def _bn_trace(n_stimuli: int = 6) -> tl.Trace:
    """Trace a BatchNorm toy with everything saved (buffer sites included)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU()).eval()
    x = torch.randn(n_stimuli, 3)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


class _ZeroTail(nn.Module):
    """Second stage outputs all zeros: a zero-norm site for angular metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 4)
        self.second = nn.Linear(4, 4)
        with torch.no_grad():
            self.second.weight.zero_()
            self.second.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(x)))


@pytest.mark.smoke
def test_default_rdm_sweep_skips_buffer_sites_with_one_disclosure() -> None:
    """Buffer sites are excluded from the sweep and disclosed exactly once."""

    trace = _bn_trace()
    buffer_labels = {layer.layer_label for layer in trace.layers if bool(layer.is_buffer_source)}
    assert buffer_labels, "fixture lost its buffer sites; rebuild with BatchNorm"

    with pytest.warns(TorchLensWarning, match="not stimulus-indexed") as record:
        matrices = repgeom.rdm_evolution(trace, min_n=2)

    skip_warnings = [entry for entry in record if "not stimulus-indexed" in str(entry.message)]
    assert len(skip_warnings) == 1, "the skip disclosure must be ONE summarized warning"
    disclosure = skip_warnings[0].message
    assert isinstance(disclosure, TorchLensWarning)
    assert disclosure.fields["code"] == "annotation_sweep_sites_skipped"
    assert disclosure.fields["remedy"].startswith("select stimulus-indexed sites")
    for label in buffer_labels:
        assert f"layer:{label}" not in matrices
        assert (
            trace._annotation_blobs is None or f"rdm:layer:{label}" not in trace._annotation_blobs
        )
    assert matrices, "eligible stimulus-indexed sites must still be swept"
    for key, matrix in matrices.items():
        assert matrix.shape[0] == 6, f"{key}: RDM axis is not the stimulus count"


def test_explicit_buffer_site_selection_refuses_actionably() -> None:
    """An explicit request for a buffer site refuses with teaching (D5)."""

    trace = _bn_trace()
    buffer_label = next(layer.layer_label for layer in trace.layers if bool(layer.is_buffer_source))
    with pytest.raises(ValueError, match="buffer overwrite") as excinfo:
        repgeom.rdm_evolution(trace, save=tl.label(buffer_label), min_n=2)
    message = str(excinfo.value)
    assert "rdm_evolution" in message
    assert buffer_label in message
    assert trace._annotation_blobs is None or not any(
        key.startswith("rdm:") for key in trace._annotation_blobs
    )


@pytest.mark.smoke
def test_forced_late_failure_leaves_zero_annotations() -> None:
    """Atomicity (D4): a failure at a later site commits nothing at all."""

    torch.manual_seed(0)
    model = _ZeroTail().eval()
    trace = tl.trace(
        model, torch.randn(6, 3), capture=tl.options.CaptureOptions(layers_to_save="all")
    )

    with pytest.raises(ValueError, match="zero-norm"):
        repgeom.rdm_evolution(trace, metric="cosine", min_n=2)
    assert trace._annotation_blobs is None or not any(
        key.startswith("rdm:") for key in trace._annotation_blobs
    ), "a failed sweep left partial annotations behind"

    with pytest.raises(ValueError, match="zero-norm"):
        repgeom.scree_evolution(trace, metric="cosine", min_n=3)
    assert trace._annotation_blobs is None or not any(
        key.startswith("scree:") for key in trace._annotation_blobs
    )

    with pytest.raises(ValueError, match="zero-norm"):
        repgeom.mds_evolution(trace, metric="cosine", min_n=3)
    assert trace._annotation_blobs is None or not any(
        key.startswith("mds:") for key in trace._annotation_blobs
    )


def test_zero_norm_error_names_site_and_stimulus_rows() -> None:
    """Zero-norm metric errors name the failing site and stimulus rows."""

    torch.manual_seed(0)
    model = _ZeroTail().eval()
    trace = tl.trace(
        model, torch.randn(6, 3), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    with pytest.raises(ValueError, match="cosine distance is undefined for zero-norm stimuli"):
        try:
            repgeom.rdm_evolution(trace, metric="cosine", min_n=2)
        except ValueError as exc:
            message = str(exc)
            assert "failed for site" in message
            assert "stimulus rows" in message
            raise


def test_shared_helper_errors_name_the_calling_verb() -> None:
    """rdm/scree diagnostics name their own verb, never mds_evolution."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh()).eval()
    trace = tl.trace(model, torch.randn(6, 3), save=tl.func("tanh"))

    with pytest.raises(ValueError) as rdm_excinfo:
        repgeom.rdm_evolution(trace, save=tl.func("linear"), min_n=2)
    assert "rdm_evolution" in str(rdm_excinfo.value)
    assert "mds_evolution" not in str(rdm_excinfo.value)

    with pytest.raises(ValueError) as scree_excinfo:
        repgeom.scree_evolution(trace, save=tl.func("linear"))
    assert "scree_evolution" in str(scree_excinfo.value)
    assert "mds_evolution" not in str(scree_excinfo.value)


def test_shape_mismatch_site_refuses_on_explicit_selection() -> None:
    """A site whose leading axis is not the stimulus count refuses, teaching."""

    class _Transposer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.transpose(torch.tanh(self.linear(x)), 0, 1)

    torch.manual_seed(0)
    trace = tl.trace(
        _Transposer().eval(),
        torch.randn(6, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    with pytest.raises(ValueError, match="leading axis") as excinfo:
        repgeom.rdm_evolution(trace, save=tl.func("transpose"), min_n=2)
    message = str(excinfo.value)
    assert "6 stimuli" in message
    assert "rdm_evolution" in message


@pytest.mark.smoke
def test_successful_sweep_still_annotates_and_returns() -> None:
    """The gate does not disturb the happy path: annotations + returns intact."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh()).eval()
    trace = tl.trace(model, torch.randn(8, 3), save=tl.func("tanh"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        matrices = repgeom.rdm_evolution(trace, min_n=2)
        spectra = repgeom.scree_evolution(trace, min_n=3)
        coords = repgeom.mds_evolution(trace, min_n=8)

    assert trace._annotation_blobs is not None
    for key, matrix in matrices.items():
        assert torch.equal(trace._annotation_blobs[f"rdm:{key}"], torch.from_numpy(matrix))
    for key, eigenvalues in spectra.items():
        assert torch.equal(trace._annotation_blobs[f"scree:{key}"], torch.from_numpy(eigenvalues))
    for key, coordinate in coords.items():
        assert torch.equal(trace._annotation_blobs[f"mds:{key}"], torch.from_numpy(coordinate))


@pytest.mark.real_model
@pytest.mark.heavy
def test_resnet18_class_rdm_sweep_fabricates_zero_buffer_pseudo_rdms() -> None:
    """R0 ResNet row: the stock-resnet18 sweep that fabricated 78 pseudo-RDMs.

    Real torchvision resnet18 class, random init, real image-shaped stimuli.
    Every returned matrix must be stimulus-indexed (N x N over the stimulus
    count) and no buffer site may be annotated.
    """

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    n_stimuli = 6
    stimuli = torch.randn(n_stimuli, 3, 64, 64)
    trace = tl.trace(model, stimuli, capture=tl.options.CaptureOptions(layers_to_save="all"))

    buffer_labels = {layer.layer_label for layer in trace.layers if bool(layer.is_buffer_source)}
    assert buffer_labels, "resnet18 trace lost its buffer sites"

    with pytest.warns(TorchLensWarning, match="not stimulus-indexed"):
        matrices = repgeom.rdm_evolution(trace, min_n=2)

    assert matrices, "no stimulus-indexed sites survived the gate"
    for key, matrix in matrices.items():
        assert matrix.shape == (n_stimuli, n_stimuli), (
            f"{key}: fabricated non-stimulus-indexed RDM of shape {matrix.shape}"
        )
    assert trace._annotation_blobs is not None
    for label in buffer_labels:
        assert f"rdm:layer:{label}" not in trace._annotation_blobs, (
            f"buffer site {label} received a fabricated pseudo-RDM annotation"
        )
