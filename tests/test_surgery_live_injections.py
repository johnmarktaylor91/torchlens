"""Lane F01 injections checkpoint: ``log_injections`` stages 0-1.

Surgery memo 3.5 / foldA D12: computation performed inside intervention
hooks becomes first-class injected-op records with anchored identity.
Covered here, with THE MISFIRE TEST first (the highest-priority test in the
build -- both halves of the no-ordinal-consumption mechanism measured):
injected ops never shift a later target's live label and never consume
site-key cohort ordinals; anchored provenance is VERBATIM the C07
``injection_provenance`` slot grammar (host_site_key, spec_rule_id,
host_pass, firing_index, nesting_path, local_op_ordinal, output_slot); the
query split (``trace.injected_ops`` / ``trace.model_ops``); selectors never
firing on injected ops; stage-1 save refusing typed and naming lane F44;
exception cleanup; and the release gate -- ordinary validation behaves
IDENTICALLY on logged and unlogged intervened traces (no new exemption,
ever).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.injection import InjectedOp, InjectionProvenance

_LOGGED = tl.options.CaptureOptions(log_injections=True)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3: early-inject / late-target substrate."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


class _TwoRelu(nn.Module):
    """Two relu fires for one rule: the firing-index substrate."""

    def __init__(self) -> None:
        """Build two linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fire relu twice."""

        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


def _sae_like(out: torch.Tensor, *, hook) -> torch.Tensor:
    """SAE-style injected computation: encode, gate, decode (5 torch calls)."""

    z = torch.relu(out @ torch.eye(out.shape[-1]))
    return torch.sigmoid(z) * 2.0


@pytest.fixture()
def chain():
    """Seeded chain model + input."""

    torch.manual_seed(0)
    return _Chain().eval(), torch.randn(3, 4)


def test_injection_misfire_two_edit_labels_never_shift(chain) -> None:
    """THE MISFIRE TEST: early-site injections never move a later target.

    Both halves measured: (a) live labels, type indexes, and raw indexes of
    every model op are IDENTICAL with and without ``log_injections`` (no
    global label-counter consumption); (b) every model op's site key is
    IDENTICAL (no site-key cohort ordinal consumption). The second edit
    targets the LATER site by the same spelling in both runs and fires.
    """

    model, x = chain
    spec = tl.when(tl.func("relu"), _sae_like) & tl.when(tl.func("tanh"), tl.scale(0.5))
    base = tl.trace(model, x, intervene=spec)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    assert [op.label for op in logged.layer_list] == [op.label for op in base.layer_list]
    assert [op.raw_label for op in logged.layer_list] == [op.raw_label for op in base.layer_list]
    assert [op.site_key for op in logged.layer_list] == [op.site_key for op in base.layer_list]
    # the LATER edit fired identically: same tanh output in both runs
    assert torch.allclose(logged["tanh_1_4"].out, base["tanh_1_4"].out)
    assert len(logged.injected_ops) > 0
    assert len(base.injected_ops) == 0


def test_injection_anchored_provenance_c07_grammar(chain) -> None:
    """Provenance is verbatim the C07 slot: anchored key + display label."""

    model, x = chain
    spec = tl.when(tl.func("relu"), _sae_like)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    records = logged.injected_ops
    assert len(records) == 5  # eye, matmul, relu, sigmoid, mul
    first = records[0]
    assert isinstance(first, InjectedOp)
    prov = first.provenance
    assert isinstance(prov, InjectionProvenance)
    # the C07 entry-dark slot's exact field set, no more, no fewer
    assert sorted(vars(prov)) == [
        "firing_index",
        "host_pass",
        "host_site_key",
        "local_op_ordinal",
        "nesting_path",
        "output_slot",
        "spec_rule_id",
    ]
    host = logged["relu_1_2"].ops[0]
    assert prov.host_site_key == host.site_key
    assert prov.spec_rule_id == spec.rules[0].rule_id  # PERSISTED rule id
    assert prov.host_pass == 1
    assert prov.firing_index == 1
    assert prov.nesting_path == ()
    assert [r.provenance.local_op_ordinal for r in records] == [1, 2, 3, 4, 5]
    # the human label anchors to the host's FINAL label (display sugar)
    assert first.label == f"{host.label}/inj_1"
    assert first.host_label == host.label
    assert records[0].func_name == "eye"
    assert isinstance(first.out, torch.Tensor)


def test_injection_firing_index_advances_per_fire() -> None:
    """A rule firing at two sites gets firing_index 1 then 2."""

    torch.manual_seed(0)
    model = _TwoRelu().eval()
    x = torch.randn(2, 4)
    spec = tl.when(tl.func("relu"), lambda out, *, hook: torch.sigmoid(out))
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    firing_indexes = sorted({r.provenance.firing_index for r in logged.injected_ops})
    assert firing_indexes == [1, 2]
    hosts = {r.host_label for r in logged.injected_ops}
    assert hosts == {"relu_1_2:1", "relu_2_4:1"}


@pytest.mark.smoke
def test_injection_query_split_and_model_family_untouched(chain) -> None:
    """model_ops is exactly the ordinary op family; injected ops sit apart."""

    model, x = chain
    logged = tl.trace(model, x, intervene=tl.when(tl.func("relu"), _sae_like), capture=_LOGGED)
    assert [op.label for op in logged.model_ops] == [op.label for op in logged.layer_list]
    injected_labels = {record.label for record in logged.injected_ops}
    model_labels = {op.label for op in logged.model_ops}
    assert injected_labels and not (injected_labels & model_labels)
    # injected ops never enter lookup or the module hierarchy
    from torchlens._errors import InvalidArgumentError

    for record in logged.injected_ops:
        with pytest.raises(InvalidArgumentError):
            logged[record.label]


def test_injection_selectors_never_fire_on_injected_ops(chain) -> None:
    """A save= selector's selection is identical with injections logged.

    The hook injects a ``relu`` call while the save predicate selects relu
    ops: the injected relu must never satisfy the selector (injected ops
    are outside the op stream by construction), so exactly the MODEL relu
    is selected, logged or not.
    """

    model, x = chain
    spec = tl.when(tl.func("relu"), _sae_like)  # injects relu (+ sigmoid, mul, ...)
    logged = tl.trace(model, x, intervene=spec, save=tl.func("relu"), capture=_LOGGED)
    base = tl.trace(model, x, intervene=spec, save=tl.func("relu"))

    def _saved_labels(trace) -> list[str]:
        """Labels whose activation payload was retained by the selector."""

        from torchlens._errors import PayloadUnavailableError

        saved = []
        for op in trace.layer_list:
            try:
                if op.out is not None:
                    saved.append(op.label)
            except PayloadUnavailableError:
                continue
        return saved

    saved_logged = _saved_labels(logged)
    saved_base = _saved_labels(base)
    assert saved_logged == saved_base
    assert not any("inj" in label for label in saved_logged)
    assert any(r.func_name == "relu" for r in logged.injected_ops)


def test_injection_save_persists_at_analysis_level(chain, tmp_path) -> None:
    """Stage 2 (F44): analysis saves persist the injected family; runnable refuses.

    The stage-1 blanket refusal is superseded by the persistence codec:
    a logged trace saves at the default analysis level and loads with its
    injected records intact (degrade-to-unattested is pinned in the
    test_log_injections_s2_* files). The RUNNABLE product still cannot
    carry the family -- its sparse core is the taken-path DAG -- so that
    one level keeps a typed refusal.
    """

    model, x = chain
    logged = tl.trace(model, x, intervene=tl.when(tl.func("relu"), _sae_like), capture=_LOGGED)
    tl.save(logged, str(tmp_path / "logged.tlspec"))
    loaded = tl.load(str(tmp_path / "logged.tlspec"))
    assert len(loaded.injected_ops) == len(logged.injected_ops)
    with pytest.raises(Exception) as excinfo:
        tl.save(logged, str(tmp_path / "run.tlspec"), level="runnable")
    assert excinfo.value.fields["code"] == "injection_logged_runnable_unsupported"
    # an armed-but-unfired capture (no injected ops) saves normally
    clean = tl.trace(model, x, capture=_LOGGED)
    assert len(clean.injected_ops) == 0
    tl.save(clean, str(tmp_path / "clean.tlspec"))


def test_injection_validation_tripwire_unchanged(chain) -> None:
    """RELEASE GATE: ordinary validation identical on logged vs unlogged.

    No surgery tolerance, no new validation exemption, ever: the logged
    trace's forward-pass validation verdict must equal the unlogged
    intervened trace's verdict (like for like; an intervened run diverges
    from the plain oracle in BOTH cases identically).
    """

    model, x = chain
    spec = tl.when(tl.func("relu"), _sae_like)
    base = tl.trace(model, x, intervene=spec)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert logged.validate_forward_pass(model(x)) == base.validate_forward_pass(model(x))
    # and a PLAIN armed capture validates exactly like a plain unarmed one
    plain_logged = tl.trace(model, x, capture=_LOGGED)
    plain_base = tl.trace(model, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logged_verdict = plain_logged.validate_forward_pass(model(x))
        base_verdict = plain_base.validate_forward_pass(model(x))
    assert str(logged_verdict) == str(base_verdict)


def test_injection_hook_exception_cleans_recorder(chain) -> None:
    """A raising hook tears the recorder down; later torch ops stay clean."""

    model, x = chain

    def _boom(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Inject one op, then raise."""

        torch.sigmoid(out)
        raise RuntimeError("hook boom")

    with pytest.raises(Exception, match="hook boom"):
        tl.trace(model, x, intervene=tl.when(tl.func("relu"), _boom), capture=_LOGGED)
    # no leftover TorchFunctionMode: plain torch runs unintercepted
    assert torch.relu(torch.tensor([-1.0, 2.0])).tolist() == [0.0, 2.0]


def test_injection_off_by_default_and_plain_capture_unaffected(chain) -> None:
    """The option defaults OFF; plain captures carry an empty injected family."""

    model, x = chain
    assert tl.options.CaptureOptions().log_injections is False
    plain = tl.trace(model, x)
    assert plain.injected_ops == ()
    assert [op.label for op in plain.model_ops] == [op.label for op in plain.layer_list]


def test_intervene_fire_counter_is_a_declared_session_transient() -> None:
    """FIX-2 enrollment pin: the intervene fire counter has a real scrub policy.

    ``_tl_intervene_selector_fire_count`` sat in the external-write ledger
    with NO ``PORTABLE_STATE_SPEC`` row, so any save reached while the
    counter was live (streamed ``to_disk`` finalize runs before the settle
    epilogue pops it; deferred backward selectors keep it past ``trace()``)
    refused with ``TorchLensIOError`` and downgraded runnable honesty
    verdicts through the witness-coverage path. It is now enrolled exactly
    like its sibling ``_tl_save_selector_fire_count``: a declared
    ``FieldPolicy.DROP`` session field, owned, with the dead ledger row
    deleted (declared and exempted are disjoint by gate).
    """

    from torchlens.data_classes._trace_components import (
        TRACE_EXTERNAL_WRITE_EXEMPTIONS,
        TRACE_FIELD_OWNERSHIP,
    )
    from torchlens.data_classes.field_policy import FieldPolicy
    from torchlens.data_classes.trace import Trace

    name = "_tl_intervene_selector_fire_count"
    assert Trace.PORTABLE_STATE_SPEC[name] is FieldPolicy.DROP
    assert TRACE_FIELD_OWNERSHIP[name] == "session"
    assert name not in TRACE_EXTERNAL_WRITE_EXEMPTIONS


def test_streamed_to_disk_intervene_capture_settles_and_loads(chain, tmp_path) -> None:
    """FIX-2 regression: streamed ``to_disk`` + ``intervene=`` captures save.

    The streaming finalize scrubs the trace while the fire counter is still
    attached; before the enrollment this refused the whole capture at the
    save boundary. The bundle must settle, carry the intervention audit,
    and load back with the session counter dropped, never persisted.
    """

    model, x = chain
    path = str(tmp_path / "streamed_intervened.tlspec")
    log = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        storage=tl.to_disk(path),
    )
    assert log.outcome.status.name == "COMPLETE"
    assert len(log.intervention_audit) == 1
    loaded = tl.load(path)
    # The streamed descriptor is finalized before the settle epilogue appends
    # the audit row, so audit parity across the stream boundary is not pinned
    # here -- only the refusal class this fix removed: the capture settles,
    # the artifact loads, and the session counter never persists.
    assert loaded.outcome.status.name == "COMPLETE"
    assert getattr(loaded, "_tl_intervene_selector_fire_count", None) is None
