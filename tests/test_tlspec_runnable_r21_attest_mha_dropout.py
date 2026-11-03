"""Round-21 honesty regressions for MHA/transformer numeric attestation.

Two fail-closed attestation defects on very common eval-mode layers:

* F1 (HIGH): a plain eval ``nn.MultiheadAttention`` + ``include_activations``
  raised ``ReplayValueMismatchError`` on a faithful replay. Capture records the
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
``replay_value_mismatch`` and a plain CNN still attests byte-exact.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.errors._runnable import ReplayValueMismatchError
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(
    replayable=True,
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

    return tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return")


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


# Kept: the catch of its regression fault.
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


# Kept: sole branch coverage of _runnable/attestation.py, _runnable/verification.py.
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


# Kept: sole branch coverage of _runnable/path_faithfulness.py, _runnable/transaction.py, _runnable/verification.py; a sole mutation kill.
def test_large_blas_slot_corruption_is_not_masked_by_fallback(tmp_path: Path) -> None:
    """Tripwire: a large corruption on a BLAS-produced slot still raises, not not_applicable.

    The layout-nonreproducible fallback is narrow: it only excuses a tight ULP
    divergence on an intact archive. A grossly corrupted archived linear slot
    breaks both the intact-archive guard and the ULP bound, so it must still
    raise replay_value_mismatch rather than being silently excused.
    """

    torch.manual_seed(0)
    model = MHANet()
    model.eval()
    x = torch.randn(2, 3, 8)
    path = _save(
        model, x, tmp_path / "mha_tamper.tlspec", include_weights=True, include_activations=True
    )

    loaded = tl.load(path)
    archive = loaded._runnable.archived_activations
    linear_keys = [key for key in archive if key.startswith("slot:linear_1_2:1")]
    assert linear_keys, "expected an archived in-proj linear slot"
    record = archive[linear_keys[0]]
    archive[linear_keys[0]] = dataclasses.replace(record, value=record.value + 0.5)

    with pytest.raises(ReplayValueMismatchError):
        loaded.run(inputs=x.clone(), seed=0, on_divergence="return")
