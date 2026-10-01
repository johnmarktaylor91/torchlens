"""Cheap universal-property sweep bundles (compo memo stratum 2; F36 waves A-D).

Each bundle is one universal property driven over the shared fixtures:
container-protocol agreement, refusal transactionality, post-success
contamination, and sink hygiene. Measured asymmetries are pinned as ledgered
cells (the fix must flip them deliberately); nothing here infers a property
from a name -- every claim executes (foldB s7: "every protective claim
executed, never inferred from names").

Ground truth probed live on the merged tree, 2026-08-30.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "transactionality/contamination sweeps measure CONSTRUCTION side effects (post-trace pickle, global state) on private models"


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()


def _x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 4)


@pytest.fixture(scope="module")
def sweep_trace() -> Any:
    import torchlens as tl

    trace = tl.trace(_model(), _x(), capture=tl.options.CaptureOptions(intervention_ready=True))
    yield trace
    trace.cleanup()


# ---------------------------------------------------------------- container


def test_container_len_iter_getitem_agree_on_the_op_domain(sweep_trace: Any) -> None:
    """len(), iter(), and getitem agree on ONE element domain (ops)."""

    labels = sweep_trace.layer_labels
    assert len(sweep_trace) == len(labels)
    iterated = list(iter(sweep_trace))
    assert len(iterated) == len(labels)
    for label in labels:
        assert sweep_trace[label] is not None


def test_container_contains_disagreement_is_pinned_sg12(sweep_trace: Any) -> None:
    """LEDGERED CELL SG#12 (live, measured): ``label in trace`` reads False
    for EVERY label that getitem resolves. Pinned exactly so the
    ``__contains__`` fix flips this test instead of landing silently."""

    labels = sweep_trace.layer_labels
    misses = [label for label in labels if label not in sweep_trace]
    assert misses == list(labels), (
        f"the SG#12 contains/getitem disagreement CHANGED (misses={misses}):"
        " if __contains__ now delegates to lookup, replace this pin with the"
        " parity assertion (every resolvable label is contained)"
    )


# ----------------------------------------------------------- transactional


def test_refused_edit_is_transactional(sweep_trace: Any) -> None:
    """A refused do() leaves payloads, outputs, and usability untouched."""

    import torchlens as tl

    fork = sweep_trace.fork()
    try:
        site_before = fork["relu_1_2"].out.clone()
        output_before = fork.output_ops[0].out.clone()
        with pytest.raises(Exception, match="hook|tensor|signature"):
            fork.do("relu_1_2", lambda value: "not a tensor")
        assert torch.equal(fork["relu_1_2"].out, site_before), (
            "a REFUSED edit mutated the site payload (transactionality broken)"
        )
        assert torch.equal(fork.output_ops[0].out, output_before)
        # The fork is still fully usable: a sanctioned edit lands cleanly.
        fork.do("relu_1_2", tl.zero_ablate())
        assert torch.count_nonzero(fork["relu_1_2"].out) == 0
    finally:
        fork.cleanup()


# ------------------------------------------------------ post-success state


def test_trace_preserves_torch_global_state_and_model_behavior() -> None:
    """After a successful capture: grad mode, default dtype, and the model's
    own forward are exactly what they were before."""

    import torchlens as tl

    model = _model()
    example = _x()
    grad_before = torch.is_grad_enabled()
    dtype_before = torch.get_default_dtype()
    with torch.no_grad():
        forward_before = model(example).clone()
    trace = tl.trace(model, example)
    try:
        assert torch.is_grad_enabled() == grad_before
        assert torch.get_default_dtype() == dtype_before
        with torch.no_grad():
            assert torch.equal(model(example), forward_before), (
                "the model computes DIFFERENTLY after a capture"
            )
    finally:
        trace.cleanup()
        tl.release_model(model)


def test_post_trace_pickle_gap_is_pinned_and_the_remedy_works() -> None:
    """LEDGERED CELL SG#33 (live, measured): after tl.trace the user's model
    cannot be pickled (instance-level forward attrs). Both halves executed:
    the gap is pinned AND the documented remedy (release_model) is verified
    to actually restore picklability -- remedy efficacy, never prose."""

    import torchlens as tl

    model = _model()
    trace = tl.trace(model, _x())
    trace.cleanup()
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(model)
    tl.release_model(model)
    restored = pickle.loads(pickle.dumps(model))
    with torch.no_grad():
        assert torch.equal(restored(_x()), model(_x())), (
            "release_model restored picklability but the round-tripped model computes differently"
        )


# ------------------------------------------------------------- sink hygiene


def test_text_sinks_carry_no_terminal_escapes(sweep_trace: Any) -> None:
    """No ANSI/OSC escapes in non-tty text sinks (explain, summary, repr)."""

    import torchlens as tl

    for name, text in (
        ("explain", tl.report.explain(sweep_trace)),
        ("summary", str(sweep_trace.summary())),
        ("repr", repr(sweep_trace)),
    ):
        assert "\x1b" not in text, f"{name} embeds terminal escapes in a text sink"


def test_agent_json_sink_is_serializable_and_self_identifying(sweep_trace: Any) -> None:
    """The agent payload round-trips through json.dumps and self-identifies
    its schema (payload self-identification, compo stratum 2)."""

    payload = sweep_trace.to_agent_json()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped
    schema_markers = [
        value
        for key, value in payload.items()
        if "schema" in key.lower() or "version" in key.lower()
    ]
    assert schema_markers, (
        f"agent JSON carries no schema/version self-identification; top-level"
        f" keys: {sorted(payload)[:12]}"
    )
