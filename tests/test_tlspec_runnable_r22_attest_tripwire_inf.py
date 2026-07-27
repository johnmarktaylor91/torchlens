"""Round-22 runnable attestation and non-finite literal regressions."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import struct

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import _encode_literal
from torchlens._runnable_execution import _decode_literal
from torchlens.errors.runnable import NumericAttestationError, RunPreconditionError
from torchlens.options import CaptureOptions
from torchlens.runnable import LiteralAtom, LiteralAtomKind, NumericAttestationStatus

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class MhaLinearNet(nn.Module):
    """Self-attention followed by a final linear projection."""

    def __init__(self) -> None:
        """Build the MHA + Linear regression module."""

        super().__init__()
        self.mha = nn.MultiheadAttention(8, 2, batch_first=True)
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final linear projection of attended tokens.

        Parameters
        ----------
        x:
            Input sequence tensor.

        Returns
        -------
        torch.Tensor
            Projected attention output.
        """

        attended, _ = self.mha(x, x, x)
        return self.fc(attended)


class MaskedMha(nn.Module):
    """Small MHA using the conventional ``-inf`` attention mask literal."""

    def __init__(self) -> None:
        """Build deterministic masked attention."""

        super().__init__()
        self.mha = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return masked self-attention output.

        Parameters
        ----------
        x:
            Input sequence tensor.

        Returns
        -------
        torch.Tensor
            Masked attention output.
        """

        mask = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        mask[0, 1] = float("-inf")
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return out


def _save_runnable(
    model: nn.Module,
    capture_input: torch.Tensor,
    path: Path,
    **save_kwargs: object,
) -> Path:
    """Capture and save one runnable trace.

    Parameters
    ----------
    model:
        Module to capture.
    capture_input:
        Tensor input for capture.
    path:
        Destination ``.tlspec`` path.
    save_kwargs:
        Extra keyword arguments forwarded to ``Trace.save``.

    Returns
    -------
    Path
        Saved runnable artifact path.
    """

    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", **save_kwargs)
    return path


def test_mha_benign_fallback_still_checks_later_corrupted_slot(tmp_path: Path) -> None:
    """Corruption after the benign in-proj member still raises fail-closed."""

    torch.manual_seed(1234)
    model = MhaLinearNet()
    model.eval()
    x = torch.randn(2, 4, 8)
    path = _save_runnable(
        model,
        x.clone(),
        tmp_path / "mha_linear.tlspec",
        include_weights=True,
        include_activations=True,
    )

    loaded = tl.load(path)
    archive = loaded.__dict__["_runnable_archived_activations"]
    victim = "slot:softmax_1_20:1:out"
    assert victim in archive
    record = archive[victim]
    archive[victim] = dataclasses.replace(record, value=record.value + 0.5)

    with pytest.raises(NumericAttestationError):
        loaded.run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


def test_masked_mha_nonfinite_attention_mask_saves_loads_runs(tmp_path: Path) -> None:
    """A genuine ``-inf`` MHA mask literal is runnable-representable."""

    torch.manual_seed(1234)
    model = MaskedMha().eval()
    x = torch.randn(2, 3, 8)
    path = _save_runnable(
        model,
        x.clone(),
        tmp_path / "masked_mha.tlspec",
        include_weights=True,
        include_activations=True,
    )

    result = tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")

    assert result.output.shape == x.shape
    assert result.report.numeric_attestation in {
        NumericAttestationStatus.NOT_APPLICABLE,
        NumericAttestationStatus.ATTESTED,
    }


@pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan")])
def test_nonfinite_float_literal_round_trips(value: float) -> None:
    """Non-finite float atoms encode with string payloads and decode by bit pattern."""

    encoded = _encode_literal(value)
    decoded = _decode_literal(encoded)

    assert isinstance(encoded, LiteralAtom)
    assert encoded.kind is LiteralAtomKind.NONFINITE_FLOAT
    assert isinstance(encoded.value, str)
    assert isinstance(decoded, float)
    if math.isnan(value):
        assert math.isnan(decoded)
    else:
        assert decoded == value
    assert struct.pack(">d", decoded) == struct.pack(">d", value)


def test_invalid_nonfinite_float_payload_is_rejected() -> None:
    """Untrusted non-finite float atom payloads remain closed."""

    literal = LiteralAtom(LiteralAtomKind.NONFINITE_FLOAT, "not-a-float")

    with pytest.raises(RunPreconditionError):
        _decode_literal(literal)
