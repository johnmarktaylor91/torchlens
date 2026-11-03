"""B1-02: the semantic-output scratch never survives onto a settled product.

Two session-time fields are written at capture entry (``user_funcs.py``) and
consumed ONLY by ``decode_outputs_for_trace``:

    _output_tokenizer  _semantic_output_metadata

Both pin LIVE USER OBJECTS -- an attached HF tokenizer
(``bridge/hf.py``) and a model-derived metadata key -- so a copy surviving onto
the escaping Trace is both a retention leak and a privacy leak: a plain
``pickle`` / ``torch.save`` of the Trace serializes the tokenizer's
vocab/merges into an artifact the user believes is a graph.

The pop block used to live ONLY on the normal forward-return arm, so the
HALTED axis (and the failure/interrupt axes) exited with both still
pinned, and neither was declared in ``Trace._FIELD_POLICY`` /
``MODEL_LOG_FIELD_ORDER`` -- which is why the runtime-declaration lockstep gate
never saw them.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.trace import _SEMANTIC_OUTPUT_TRANSIENT_FIELDS
from torchlens._data_classes.field_policy import FieldPolicy


class _Tokenizer:
    """Stand-in for an attached HF tokenizer: a live, picklable user object."""

    def __init__(self) -> None:
        # A recognizable payload, standing in for vocab/merges.
        self.vocab = {f"tok{index}": index for index in range(64)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:  # noqa: D102
        return " ".join(f"tok{int(index)}" for index in token_ids)


class _HaltableClassifier(nn.Module):
    """Two-stage model whose relu is a usable halt frontier."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.relu(self.fc(x)) * 2.0


def _model_with_tokenizer() -> _HaltableClassifier:
    model = _HaltableClassifier().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    return model


def _leaked(trace: tl.Trace) -> list[str]:
    """Return the semantic-output scratch names still pinned to ``trace``."""

    return sorted(name for name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS if name in trace.__dict__)


# ---------------------------------------------------------------------------
# The four capture axes
# ---------------------------------------------------------------------------


def test_completed_capture_drops_the_semantic_output_scratch() -> None:
    """The historical normal-return arm keeps working."""

    trace = tl.trace(_model_with_tokenizer(), torch.ones(1, 3))
    assert _leaked(trace) == []


def test_halt_does_not_break_decoding_on_the_completed_path() -> None:
    """The drop never runs before the sole consumer.

    ``decode_outputs_for_trace`` is the only reader; if the drop moved ahead
    of it, decoding would silently stop working. This pins that it does not.
    """

    class Text(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # A registered buffer is a KNOWN provenance source; a closed-over
            # tensor would trip the unattributed-tensor-arg warning.
            self.register_buffer("logits", torch.tensor([[[0.0, 5.0], [6.0, 1.0]]]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            return self.logits + x.sum() * 0.0

    model = Text().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(model, torch.ones(1, 2))
    assert trace.provenance.output_postprocessor is not None
    assert trace.provenance.output_postprocessor.style == "hf_text"
    assert trace._decoded_output is not None
    assert _leaked(trace) == []


# ---------------------------------------------------------------------------
# Declaration + serialization boundaries
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_transient_fields_are_declared_drop_on_trace() -> None:
    """Declared, so the runtime-declaration lockstep gate can see them."""

    from torchlens._constants import MODEL_LOG_FIELD_ORDER

    policy = tl.Trace._FIELD_POLICY
    for name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS:
        assert name in policy, name
        assert policy[name].portable_policy is FieldPolicy.DROP, name
        # Session-time, never user-facing portable schema.
        assert name not in MODEL_LOG_FIELD_ORDER, name
        assert policy[name].user_facing is False, name
