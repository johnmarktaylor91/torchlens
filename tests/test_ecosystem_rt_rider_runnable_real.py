"""F32 folded rider: listA rows 17/21 evidence at REAL HF causal-LM tier.

Both rows shipped with toy-only evidence and the excuse "real-model runnable
saves blocked by row 2 Cache defect". FIX-A (lane A05) landed, so a runnable
save of a real HF causal-LM works; this file upgrades the evidence:

* Row 17 (weight-free runnable warns): the R0 roster's shrunk-config
  ``GPT2LMHeadModel`` saves ``level="runnable"`` with the default
  ``include_weights=False``, loads, and ``run()`` executes on RANDOM
  role-init state with the loud S-18 ``runnable_random_init_run`` warning.
  The ``include_weights=True`` sibling settles VERIFIED and reproduces the
  captured logits bit-faithfully -- the FIX-A dividend the rows waited on.
* Row 21 (loaded-backward refuses before mutation; outcome gate learns
  Recording): the same real architecture, not a toy Sequential.

Unblocking defect fixed in the same change and pinned here: the sparse-run
structure witness's ``_container_kind`` dispatched ``dataclasses.is_dataclass``
BEFORE ``_is_hf_model_output``, and HF ``ModelOutput`` subclasses ARE
dataclasses -- so a real GPT-2's recorded ``hf_model_output`` witness compared
against a runtime "dataclass" kind and every honest identical run
false-DIVERGED with ``OUTPUT_STRUCTURE_MISMATCH`` (same bug class as the r67
C2 registered-branch inversion).
"""

from __future__ import annotations

import warnings

import pytest
import torch

import torchlens as tl
from torchlens.capture.outcome import (
    CaptureStatus,
    outcome_for,
    require_capture_capability,
)
from torchlens.errors import RunCapabilityUnavailableError, TorchLensWarning
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, StateSource

pytest.importorskip("transformers")

from tests.real_model.r0.families import _token_ids, build_gpt2  # noqa: E402

pytestmark = [pytest.mark.heavy, pytest.mark.real_model]


@pytest.fixture(scope="module")
def gpt2_and_inputs():
    torch.manual_seed(0)
    return build_gpt2("eager").eval(), _token_ids()


@pytest.fixture(scope="module")
def runnable_trace(gpt2_and_inputs):
    model, input_ids = gpt2_and_inputs
    trace = tl.trace(model, input_ids, capture=CaptureOptions(intervention_ready=True))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_gpt2_weight_free_runnable_random_run_warns(tmp_path, gpt2_and_inputs, runnable_trace):
    """Row 17 at real-model tier: the default weight-free runnable artifact of
    a real HF causal-LM loads, runs on RANDOM role-init state, and warns with
    the contracted S-18 code naming both remedies."""

    _, input_ids = gpt2_and_inputs
    path = tmp_path / "gpt2_runnable"
    tl.save(runnable_trace, path, level="runnable")
    loaded = tl.load(path)
    with pytest.warns(TorchLensWarning, match="RANDOM") as record:
        result = loaded.run(inputs=input_ids, seed=0)
    assert result.report.state_source is StateSource.RANDOM_INITIALIZATION
    warnings_seen = [w.message for w in record if issubclass(w.category, TorchLensWarning)]
    # S-18 contract: consumers branch on fields["code"], never message text.
    assert any(
        getattr(w, "fields", {}).get("code") == "runnable_random_init_run" for w in warnings_seen
    )
    messages = [str(w) for w in warnings_seen]
    assert any("include_weights=True" in m and "load_state_dict" in m for m in messages)


def test_gpt2_embedded_weights_runnable_run_settles_verified(
    tmp_path, gpt2_and_inputs, runnable_trace
):
    """The FIX-A dividend row 17 waited on: a real HF causal-LM runnable save
    with embedded weights loads, re-executes its taken-path DAG, settles
    VERIFIED, and reproduces the captured logits through the real
    ``ModelOutput`` container."""

    model, input_ids = gpt2_and_inputs
    path = tmp_path / "gpt2_runnable_weights"
    tl.save(runnable_trace, path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = loaded.run(inputs=input_ids)
    assert result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    random_warnings = [
        w for w in caught if issubclass(w.category, TorchLensWarning) and "RANDOM" in str(w.message)
    ]
    assert random_warnings == []
    # The runtime output is the real HF container, not a lossy substitute.
    assert type(result.output).__name__ == "CausalLMOutputWithCrossAttentions"
    live = model(input_ids=input_ids)
    assert torch.allclose(result.output.logits, live.logits, atol=1e-5)


def test_structure_witness_classifies_model_output_as_hf_kind(gpt2_and_inputs):
    """Regression pin for the witness classifier-order defect this rider
    unblocked: an HF ``ModelOutput`` (which IS a dataclass) must classify as
    ``hf_model_output``, mirroring the capture-side spec builder's dispatch
    order -- the inverted order false-DIVERGED every honest run above."""

    from torchlens._runnable_witness_contracts import _container_kind

    model, input_ids = gpt2_and_inputs
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert _container_kind(out) == "hf_model_output"


def test_gpt2_loaded_log_backward_refuses_before_mutation(tmp_path, gpt2_and_inputs):
    """Row 21 at real-model tier: ``log_backward`` on a bundle-loaded real HF
    trace refuses typed BEFORE mutating backward state."""

    model, input_ids = gpt2_and_inputs
    trace = tl.trace(model, input_ids, capture=CaptureOptions(backward_ready=True))
    tl.save(trace, tmp_path / "gpt2_bundle")
    loaded = tl.load(tmp_path / "gpt2_bundle")
    live_loss = model(input_ids=input_ids).logits.sum()
    with pytest.raises(RunCapabilityUnavailableError, match="bundle-loaded"):
        loaded.log_backward(live_loss)
    assert len(getattr(loaded, "backward_passes", []) or []) == 0
    assert len(loaded.grad_fns) == 0
    with pytest.raises(RunCapabilityUnavailableError, match="bundle-loaded"):
        loaded.recording_backward()


def test_gpt2_recording_outcome_gate_reads_complete(gpt2_and_inputs):
    """Row 21 at real-model tier: the N1-N5 chokepoint reads a slots-backed
    real-model ``Recording``'s settled COMPLETE outcome without the false
    hand-built-object warning."""

    model, input_ids = gpt2_and_inputs
    rec = tl.record(model, input_ids, save=tl.func("softmax"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        settled = outcome_for(rec)
        gate_outcome = require_capture_capability(rec, "save_analysis")
    assert settled is not None and settled.status is CaptureStatus.COMPLETE
    assert gate_outcome.status is CaptureStatus.COMPLETE
    hand_built = [w for w in caught if "hand-built" in str(w.message)]
    assert hand_built == []
