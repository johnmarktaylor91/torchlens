"""Persistence honesty on REAL models (lane A08; realism rule: toy-only
validation disqualifies).

Reuses the R0 roster's config-built, offline, seeded families (a shrunk
GPT2LMHeadModel and a shrunk BERT with registered buffers) to pin the A-IV
fixes where they shipped broken: real HF architectures, not toy Sequentials.
The harvested-corpus round-trip (genuine resnet18 artifacts from released
writers) rides tests/test_persist_honesty_corpus_roundtrip.py.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.capture.outcome import CaptureStatus
from torchlens.options import CaptureOptions

pytest.importorskip("transformers")

from tests.real_model.r0.families import _token_ids, build_bert, build_gpt2  # noqa: E402

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def gpt2_and_inputs():
    torch.manual_seed(0)
    return build_gpt2("eager").eval(), _token_ids()


def test_streamed_gpt2_capture_loads_complete(tmp_path, gpt2_and_inputs):
    """Item 18 on a real HF causal LM: the streamed artifact carries the
    settled COMPLETE attestation."""

    model, input_ids = gpt2_and_inputs
    path = tmp_path / "gpt2_streamed.tlspec"
    trace = tl.trace(model, [], {"input_ids": input_ids}, storage=tl.to_disk(str(path)))
    assert trace.outcome.status is CaptureStatus.COMPLETE
    loaded = tl.load(path)
    assert loaded.outcome.status is CaptureStatus.COMPLETE
    assert loaded.layer_list


def test_gpt2_executable_level_respects_explicit_opt_out(tmp_path, gpt2_and_inputs):
    """Item 19 on a real model: the opt-out refusal, never a silent override
    that ships the real token ids as raw payloads."""

    model, input_ids = gpt2_and_inputs
    trace = tl.trace(model, [], {"input_ids": input_ids})
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.save(
            trace,
            tmp_path / "bundle",
            level="executable_with_callables",
            include_saved_args=False,
        )
    assert excinfo.value.fields["code"] == "save_payload_level_conflict"


def test_bert_structure_only_save_round_trips(tmp_path):
    """W3 on a real buffer-holding architecture (the weightsfree memo's L6
    named BERT explicitly): structure-only save-then-load works and ships no
    value payloads."""

    torch.manual_seed(0)
    model = build_bert("eager").eval()
    trace = tl.trace(
        model,
        [],
        {"input_ids": _token_ids()},
        capture=CaptureOptions(structure_only=True),
    )
    path = tmp_path / "bert_structure.tlspec"
    tl.save(trace, path)
    assert list((path / "blobs").iterdir()) == []
    loaded = tl.load(path)
    assert loaded.structure_only is True
    assert loaded.layer_list
