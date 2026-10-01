"""Lane F44 stage 2: injected-op persistence, degrade-to-unattested, replay.

foldA MEMO s5 item 12: the persistence codec rides the trusted-callable
resolver and the C07 ``Op.injection_provenance`` slot (no schema bump);
loads degrade to ``unattested`` -- never refusing on an unresolvable
callable; :func:`torchlens.intervention.injection.attest_injected_ops`
replays trusted injected callables and byte-compares; the runnable save
level refuses typed. Ordinary validation is UNCHANGED (release gate).
"""

from __future__ import annotations

import dataclasses
import sys
import types
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.injection import (
    InjectedOp,
    attest_injected_ops,
    injection_state,
)
from torchlens.intervention.types import FunctionRegistryKey

_LOGGED = tl.options.CaptureOptions(log_injections=True)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3: the stage-1 substrate."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


def _sae_like(out: torch.Tensor, *, hook) -> torch.Tensor:
    """SAE-style injected computation (5 recordable torch calls)."""

    z = torch.relu(out @ torch.eye(out.shape[-1]))
    return torch.sigmoid(z) * 2.0


@pytest.fixture()
def chain():
    """Seeded chain model + input."""

    torch.manual_seed(0)
    return _Chain().eval(), torch.randn(3, 4)


def _logged_trace(chain, hook=_sae_like):
    """One logged intervened capture."""

    model, x = chain
    return tl.trace(model, x, intervene=tl.when(tl.func("relu"), hook), capture=_LOGGED)


@pytest.mark.smoke
def test_round_trip_preserves_identity_payloads_and_query_split(chain, tmp_path) -> None:
    """Analysis save/load round-trips the injected family verbatim.

    Durable provenance, labels, callable keys, arg snapshots, and output
    payloads survive byte-exactly; loaded model_ops equals the live one;
    layer_list never carries an injection_provenance row after load; and
    injected labels stay outside lookup exactly like live.
    """

    logged = _logged_trace(chain)
    live = logged.injected_ops
    path = str(tmp_path / "logged.tlspec")
    tl.save(logged, path)
    loaded = tl.load(path)
    restored = loaded.injected_ops
    assert len(restored) == len(live) == 5
    for live_record, loaded_record in zip(live, restored, strict=True):
        assert loaded_record.label == live_record.label
        assert loaded_record.host_label == live_record.host_label
        assert loaded_record.func_name == live_record.func_name
        assert loaded_record.layer_type == live_record.layer_type
        assert loaded_record.provenance == live_record.provenance
        assert loaded_record.callable_ref == live_record.callable_ref
        assert isinstance(loaded_record.callable_ref, FunctionRegistryKey)
        assert torch.equal(loaded_record.out, live_record.out)
        assert loaded_record.fire_device == live_record.fire_device
        assert len(loaded_record.saved_args or ()) == len(live_record.saved_args or ())
    assert [op.label for op in loaded.model_ops] == [op.label for op in logged.model_ops]
    assert [op.label for op in loaded.model_ops] == [op.label for op in loaded.layer_list]
    assert not any(
        getattr(op, "injection_provenance", None) is not None for op in loaded.layer_list
    )
    from torchlens._errors import InvalidArgumentError

    for record in restored:
        with pytest.raises(InvalidArgumentError):
            loaded[record.label]


@pytest.mark.smoke
def test_load_degrades_to_unattested_and_replay_attests(chain, tmp_path) -> None:
    """Loading NEVER attests; the replay door promotes trusted matches.

    Live records read ``recorded``; loaded ones ``unattested`` with no
    reason (the callable resolves through the fixed trusted namespaces and
    merely awaits replay); ``attest_injected_ops`` re-executes each trusted
    callable on its snapshotted args and promotes every byte-exact match.
    """

    logged = _logged_trace(chain)
    assert {record.attestation for record in logged.injected_ops} == {"recorded"}
    live_report = attest_injected_ops(logged)
    assert live_report.passed
    assert {row.status for row in live_report.rows} == {"attested"}
    path = str(tmp_path / "logged.tlspec")
    tl.save(logged, path)
    loaded = tl.load(path)
    assert {record.attestation for record in loaded.injected_ops} == {"unattested"}
    assert {record.attestation_reason for record in loaded.injected_ops} == {None}
    report = attest_injected_ops(loaded)
    assert report.passed
    assert {row.status for row in report.rows} == {"attested"}
    assert {record.attestation for record in loaded.injected_ops} == {"attested"}


@pytest.mark.smoke
def test_resave_of_loaded_logged_trace_round_trips(chain, tmp_path) -> None:
    """A loaded logged trace re-saves with its injected family intact."""

    logged = _logged_trace(chain)
    first = str(tmp_path / "one.tlspec")
    second = str(tmp_path / "two.tlspec")
    tl.save(logged, first)
    loaded = tl.load(first)
    tl.save(loaded, second)
    reloaded = tl.load(second)
    assert [r.provenance for r in reloaded.injected_ops] == [
        r.provenance for r in logged.injected_ops
    ]
    report = attest_injected_ops(reloaded)
    assert report.passed and len(report.attested) == 5


@pytest.mark.smoke
def test_runnable_save_refuses_typed(chain, tmp_path) -> None:
    """The runnable level cannot carry the injected family and refuses."""

    logged = _logged_trace(chain)
    with pytest.raises(Exception) as excinfo:
        tl.save(logged, str(tmp_path / "run.tlspec"), level="runnable")
    assert excinfo.value.fields["code"] == "injection_logged_runnable_unsupported"
    # an unlogged capture keeps its historical runnable behavior
    model, x = chain
    plain = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    tl.save(plain, str(tmp_path / "plain_run.tlspec"), level="runnable")


@pytest.mark.smoke
def test_untrusted_and_missing_callable_arms_degrade_never_refuse(chain, tmp_path) -> None:
    """Foreign and unresolvable callables load fine, disclosed unattested.

    The untrusted arm resolves only under the explicit resolver trust
    opt-in and then attests via real replay; the missing arm (a trusted-
    namespace name this torch build does not expose) stays unattested.
    """

    module = types.ModuleType("f44_fake_user_module")

    def user_sigmoid(value: torch.Tensor) -> torch.Tensor:
        """Foreign stand-in for the recorded torch.sigmoid call."""

        return torch.sigmoid(value)

    module.user_sigmoid = user_sigmoid
    sys.modules["f44_fake_user_module"] = module
    try:
        logged = _logged_trace(chain)
        state = injection_state(logged)
        records = list(logged.injected_ops)
        # forge record 3 (sigmoid) FOREIGN and record 2 (relu) MISSING
        foreign = FunctionRegistryKey(
            namespace="custom",
            qualname="user_sigmoid",
            dispatch_kind="function",
            import_path="f44_fake_user_module:user_sigmoid",
        )
        missing = FunctionRegistryKey(
            namespace="torch", qualname="f44_not_a_real_torch_name", dispatch_kind="function"
        )
        records[3] = dataclasses.replace(records[3], callable_ref=foreign)
        records[2] = dataclasses.replace(records[2], callable_ref=missing)
        state["records"] = records
        path = str(tmp_path / "arms.tlspec")
        tl.save(logged, path)
        loaded = tl.load(path)
        reasons = {r.label: r.attestation_reason for r in loaded.injected_ops}
        assert reasons[records[3].label] == "callable_untrusted"
        assert reasons[records[2].label] == "callable_missing"
        report = attest_injected_ops(loaded)
        assert report.passed  # unattested rows never fail the report
        by_label = {row.label: row for row in report.rows}
        assert by_label[records[3].label].reason == "callable_untrusted"
        assert by_label[records[2].label].reason == "callable_missing"
        # the explicit trust opt-in resolves the foreign callable and replays it
        trusted_report = attest_injected_ops(
            loaded, allowed_custom_callable_modules={"f44_fake_user_module"}
        )
        assert trusted_report.passed
        assert {row.status for row in trusted_report.rows if row.label == records[3].label} == {
            "attested"
        }
    finally:
        sys.modules.pop("f44_fake_user_module", None)


@pytest.mark.smoke
def test_replay_divergence_is_the_tripwire_verdict(chain, tmp_path) -> None:
    """A trusted replay contradicting the recorded output fails the report."""

    logged = _logged_trace(chain)
    state = injection_state(logged)
    records = list(logged.injected_ops)
    records[2] = dataclasses.replace(records[2], out=records[2].out + 1.0)
    state["records"] = records
    report = attest_injected_ops(logged)
    assert not report.passed
    assert [row.reason for row in report.diverged] == ["replay_value_mismatch"]
    # the tampered claim also fails after a save/load cycle
    path = str(tmp_path / "tampered.tlspec")
    tl.save(logged, path)
    loaded_report = attest_injected_ops(tl.load(path))
    assert not loaded_report.passed


@pytest.mark.smoke
def test_include_outs_false_persists_identity_only(chain, tmp_path) -> None:
    """A payload-free save keeps the identity family, disclosed unattested."""

    logged = _logged_trace(chain)
    path = str(tmp_path / "noouts.tlspec")
    tl.save(logged, path, include_outs=False)
    loaded = tl.load(path)
    restored = loaded.injected_ops
    assert len(restored) == 5
    assert {record.out for record in restored} == {None}
    assert {record.saved_args for record in restored} == {None}
    report = attest_injected_ops(loaded)
    assert report.passed
    assert {row.reason for row in report.rows} == {"payload_unavailable"}


@pytest.mark.smoke
def test_nondeterministic_injected_callable_never_reads_diverged(chain) -> None:
    """RNG-consuming injected calls disclose unattested, not diverged."""

    def noisy(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Inject one dropout call (fresh RNG per replay)."""

        return torch.dropout(out, 0.5, True)

    logged = _logged_trace(chain, hook=noisy)
    assert [r.layer_type for r in logged.injected_ops] == ["dropout"]
    report = attest_injected_ops(logged)
    assert report.passed
    assert [row.reason for row in report.rows] == ["nondeterministic_callable"]


@pytest.mark.smoke
def test_multi_output_slots_round_trip_and_attest(chain, tmp_path) -> None:
    """A multi-output injected call keeps per-slot records through the codec."""

    def chunky(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Inject one chunk call with two tensor outputs."""

        first, second = torch.chunk(out, 2, dim=0)
        return torch.cat([first, second], dim=0)

    logged = _logged_trace(chain, hook=chunky)
    slots = sorted(r.provenance.output_slot for r in logged.injected_ops if "chunk" in r.func_name)
    assert slots == [0, 1]
    path = str(tmp_path / "chunk.tlspec")
    tl.save(logged, path)
    loaded = tl.load(path)
    report = attest_injected_ops(loaded)
    assert report.passed
    chunk_rows = [row for row in report.rows if "chunk" in row.label or ":1" in row.label]
    assert {row.status for row in report.rows} == {"attested"}
    assert len(loaded.injected_ops) == len(logged.injected_ops)
    assert chunk_rows


@pytest.mark.smoke
def test_ordinary_validation_unchanged_after_round_trip(chain, tmp_path) -> None:
    """RELEASE GATE: forward validation identical for logged vs unlogged saves."""

    model, x = chain
    spec = tl.when(tl.func("relu"), _sae_like)
    base = tl.trace(model, x, intervene=spec)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    tl.save(logged, str(tmp_path / "l.tlspec"))
    tl.save(base, str(tmp_path / "b.tlspec"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logged_verdict = logged.validate_forward_pass(model(x))
        base_verdict = base.validate_forward_pass(model(x))
    assert str(logged_verdict) == str(base_verdict)


@pytest.mark.smoke
def test_unanchored_record_refuses_at_save(chain, tmp_path) -> None:
    """A record with no resolved host site key cannot persist (typed)."""

    logged = _logged_trace(chain)
    state = injection_state(logged)
    records = list(logged.injected_ops)
    records[0] = dataclasses.replace(
        records[0], provenance=dataclasses.replace(records[0].provenance, host_site_key=None)
    )
    state["records"] = records
    with pytest.raises(Exception) as excinfo:
        tl.save(logged, str(tmp_path / "unanchored.tlspec"))
    assert excinfo.value.fields["code"] == "injection_persist_unanchored"


@pytest.mark.smoke
def test_loaded_record_types_are_the_live_types(chain, tmp_path) -> None:
    """Loaded records are ordinary InjectedOp values (one surface, two origins)."""

    logged = _logged_trace(chain)
    path = str(tmp_path / "types.tlspec")
    tl.save(logged, path)
    loaded = tl.load(path)
    for record in loaded.injected_ops:
        assert isinstance(record, InjectedOp)
        assert isinstance(record.out, torch.Tensor)
        assert record.saved_args is not None
