"""Round-21 honesty regressions for MHA/transformer numeric attestation.

Two fail-closed attestation defects on very common eval-mode layers:

* F1 (HIGH): a plain eval ``nn.MultiheadAttention`` + ``include_activations``
  raised ``NumericAttestationError`` on a faithful replay. Capture records the
  in-proj ``F.linear`` under autograd (a grad-specialized BLAS reduction order);
  the pre-r37 run path recomputed it with DETACHED state clones, so the two
  reduction orders differed by ~1 dtype ULP and a ``not_applicable`` carve-out
  was the honest floor. Since r37 (corr2-7/R13) replay restores the recorded
  per-slot trainable bit, reproduces the capture-time grad-specialized
  reduction, and byte-ATTESTS; the carve-out remains armed as a narrow,
  provably-benign fail-safe that never blesses a corruption.
* F2 (LOW): eval-mode / ``p == 0`` dropout was unconditionally seeded-RNG-tagged
  and over-triggered ``not_applicable`` although its replay is byte-exact. The
  RNG tag is now keyed off actual consumption (``training`` and ``p``), so a
  genuinely RNG-inert dropout attests byte-exact while a real training dropout
  (``p > 0``) stays ``not_applicable``.

The byte-exact tripwire stays armed: a tampered activation archive still raises
``numeric_attestation_failed`` and a plain CNN still attests byte-exact.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.runnable import NumericAttestationError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _save(model: nn.Module, capture_input: torch.Tensor, path: Path, **save_kwargs: object) -> Path:
    """Capture, save, and return one runnable ``.tlspec`` path.

    Parameters
    ----------
    model:
        Module to trace.
    capture_input:
        Tensor input used for capture.
    path:
        Destination path for the runnable artifact.
    save_kwargs:
        Additional keyword arguments forwarded to ``Trace.save``.

    Returns
    -------
    Path
        Saved runnable artifact path.
    """

    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", **save_kwargs)
    return path


def _run(path: Path, x: torch.Tensor) -> tl.RunResult:
    """Load and run one sparse runnable artifact returning divergence in the report.

    Parameters
    ----------
    path:
        Runnable artifact path.
    x:
        Runtime tensor input.

    Returns
    -------
    tl.RunResult
        Sparse replay result.
    """

    return tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


class DropNet(nn.Module):
    """Linear + ReLU + dropout, exercising the dropout RNG-tag decision."""

    def __init__(self, p: float) -> None:
        """Build a dropout-terminated MLP.

        Parameters
        ----------
        p:
            Dropout probability.
        """

        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dropout applied to a ReLU'd linear.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Module output.
        """

        return self.drop(torch.relu(self.lin(x)))


class MHANet(nn.Module):
    """Batch-first self-attention wrapper for the F1 in-proj replay case."""

    def __init__(self) -> None:
        """Build a small batch-first ``nn.MultiheadAttention`` with no dropout."""

        super().__init__()
        self.mha = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attended output of a self-attention call.

        Parameters
        ----------
        x:
            Input sequence tensor ``(batch, seq, embed)``.

        Returns
        -------
        torch.Tensor
            Attention output.
        """

        out, _ = self.mha(x, x, x)
        return out


class TELNet(nn.Module):
    """Thin wrapper around a single ``nn.TransformerEncoderLayer``."""

    def __init__(self, layer: nn.Module) -> None:
        """Store the wrapped encoder layer.

        Parameters
        ----------
        layer:
            The transformer encoder layer to run.
        """

        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoder layer applied to the input.

        Parameters
        ----------
        x:
            Input sequence tensor.

        Returns
        -------
        torch.Tensor
            Encoder layer output.
        """

        return self.layer(x)


@pytest.mark.smoke
def test_eval_dropout_include_activations_attests_byte_exact(tmp_path: Path) -> None:
    """F2: an eval-mode dropout replays byte-exact and attests (not not_applicable)."""

    torch.manual_seed(0)
    model = DropNet(0.5)
    model.eval()
    x = torch.randn(2, 4)
    path = _save(
        model, x, tmp_path / "drop_eval.tlspec", include_weights=True, include_activations=True
    )

    result = _run(path, x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_zero_p_dropout_include_activations_attests_byte_exact(tmp_path: Path) -> None:
    """F2: a ``p == 0`` dropout is RNG-inert and attests byte-exact."""

    torch.manual_seed(0)
    model = DropNet(0.0)
    model.eval()
    x = torch.randn(2, 4)
    path = _save(
        model, x, tmp_path / "drop_p0.tlspec", include_weights=True, include_activations=True
    )

    result = _run(path, x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_training_dropout_stays_not_applicable(tmp_path: Path) -> None:
    """F2 boundary: a genuine training dropout (p>0) keeps seeded-RNG not_applicable."""

    torch.manual_seed(0)
    model = DropNet(0.5)
    model.train()
    x = torch.randn(2, 4)
    path = _save(
        model, x, tmp_path / "drop_train.tlspec", include_weights=True, include_activations=True
    )

    result = _run(path, x)

    # A training dropout actually draws from the RNG, so a byte-exact attestation
    # would dishonestly bless a run that only happens to reproduce the seed.
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_eval_mha_include_activations_is_honest_no_raise(tmp_path: Path) -> None:
    """F1 (updated r37): a plain eval MHA + include_activations now honestly ATTESTS.

    The original F1 premise -- "the capture-time grad context is not recorded, so
    the no-grad run path cannot byte-reproduce the grad-specialized in-proj
    ``F.linear`` BLAS reduction" -- is obsolete: the per-call grad context has been
    recorded since r35, and r37 (corr2-7/R13 clone fidelity) restores the RECORDED
    per-slot trainable bit on state clones, so replay recomputes the in-proj under
    the same grad-specialized reduction as capture and byte-MATCHES the archive.
    ``attested`` is only reachable when EVERY recomputed slot digest equals the
    recorded and archived digests (the byte-exact tripwire is unchanged), and the
    eval capture consumes no host RNG (monitor: zero channels). Verified
    empirically: 3/3 independent load+run cycles ATTESTED, 28 archived members, 0
    digest mismatches; stripping the trainable restore reproduces the old
    ``not_applicable`` via the benign-BLAS fallback (which stays armed as a
    fail-safe and, per the tamper tests below, never masks corruption).
    """

    torch.manual_seed(0)
    model = MHANet()
    model.eval()
    x = torch.randn(2, 3, 8)
    path = _save(
        model, x, tmp_path / "mha_eval.tlspec", include_weights=True, include_activations=True
    )

    result = _run(path, x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_eval_transformer_encoder_layer_attests_byte_exact(tmp_path: Path) -> None:
    """A ``TransformerEncoderLayer(dropout=0.0)`` honestly ATTESTS (updated r37).

    Its dropout does not taint (F2: eval/p==0 consumes no RNG), and its contained
    MHA in-proj ``F.linear`` byte-reproduces since r37 restores the recorded
    trainable bit on state clones (see the eval-MHA test above for the mechanism
    and byte evidence). ``attested`` structurally requires every recomputed slot
    to byte-match the recorded and archived digests -- the tripwire is unchanged.
    """

    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
    )
    layer.eval()
    x = torch.randn(2, 3, 8)
    path = _save(
        TELNet(layer),
        x,
        tmp_path / "tel_eval.tlspec",
        include_weights=True,
        include_activations=True,
    )

    result = _run(path, x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_plain_cnn_include_activations_still_attests(tmp_path: Path) -> None:
    """No regression: a plain eval CNN + include_activations still attests byte-exact."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 5))
    model.eval()
    x = torch.randn(1, 3, 8, 8)
    path = _save(model, x, tmp_path / "cnn.tlspec", include_weights=True, include_activations=True)

    result = _run(path, x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_tampered_activation_archive_still_raises(tmp_path: Path) -> None:
    """Tripwire: a corrupted activation archive still raises numeric_attestation_failed."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 5))
    model.eval()
    x = torch.randn(1, 3, 8, 8)
    path = _save(
        model, x, tmp_path / "cnn_tamper.tlspec", include_weights=True, include_activations=True
    )

    loaded = tl.load(path)
    archive = loaded.__dict__["_runnable_archived_activations"]
    key = next(iter(archive))
    record = archive[key]
    archive[key] = dataclasses.replace(record, value=record.value + 1.0)

    with pytest.raises(NumericAttestationError):
        loaded.run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


def test_large_blas_slot_corruption_is_not_masked_by_fallback(tmp_path: Path) -> None:
    """Tripwire: a large corruption on a BLAS-produced slot still raises, not not_applicable.

    The layout-nonreproducible fallback is narrow: it only excuses a tight ULP
    divergence on an intact archive. A grossly corrupted archived linear slot
    breaks both the intact-archive guard and the ULP bound, so it must still
    raise numeric_attestation_failed rather than being silently excused.
    """

    torch.manual_seed(0)
    model = MHANet()
    model.eval()
    x = torch.randn(2, 3, 8)
    path = _save(
        model, x, tmp_path / "mha_tamper.tlspec", include_weights=True, include_activations=True
    )

    loaded = tl.load(path)
    archive = loaded.__dict__["_runnable_archived_activations"]
    linear_keys = [key for key in archive if key.startswith("slot:linear_1_2:1")]
    assert linear_keys, "expected an archived in-proj linear slot"
    record = archive[linear_keys[0]]
    archive[linear_keys[0]] = dataclasses.replace(record, value=record.value + 0.5)

    with pytest.raises(NumericAttestationError):
        loaded.run(inputs=x.clone(), seed=0, on_divergence="return_diverged")
