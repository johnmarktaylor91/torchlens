"""W051-BIND fix wave: AUD-CODE 2.3a-d / 3.7a-d regression pins for the live-surgery stack.

Each test re-runs the audit's reproducer shape against the fixed tree:
in-place injected ops attest (2.3a), module-boundary firings anchor and
persist under the spec rule id (2.3b), re-anchored/forged provenance refuses
at load (2.3c), bind site keys are byte-parity with capture (2.3d), the bind
fire record derives ``replaced`` from identity and discloses in-place targets
(3.7a), NaN-producing honest injected ops attest (3.7b), region edits run the
ONE payload gate (3.7c), and aliased submodules are taught, never guessed
(3.7d).
"""

from __future__ import annotations

import dataclasses
import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BindingPreflightError
from torchlens.intervention.injection import attest_injected_ops, injected_ops

_LOGGED = tl.options.CaptureOptions(log_injections=True)
_READY = tl.options.CaptureOptions(intervention_ready=True)


class _MLP(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(torch.relu(self.fc1(x)))


class _Nested(nn.Module):
    """Submodule-contained ops plus a twice-called Sequential (site-key parity substrate)."""

    def __init__(self) -> None:
        """Build fc1, fc2, and the reused block."""

        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
        self.blk = nn.Sequential(nn.Linear(2, 2), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1/relu/fc2 then the block twice."""

        h = torch.relu(self.fc1(x))
        o = self.fc2(h)
        o = self.blk(o)
        return self.blk(o)


class _Shared(nn.Module):
    """One module object registered under two names (alias substrate)."""

    def __init__(self) -> None:
        """Register enc and its alias dec."""

        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = self.enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the shared module at two sites."""

        return self.dec(torch.relu(self.enc(x)))


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3 (regions / forgery substrate)."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


class _InPlace(nn.Module):
    """A forward whose activation is an in-place relu with the return ignored."""

    def __init__(self) -> None:
        """Build one linear."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate h in place and return the mutated storage."""

        h = self.fc(x)
        torch.relu_(h)
        return h


def _seeded(
    cls: type[nn.Module], shape: tuple[int, ...] = (3, 4)
) -> tuple[nn.Module, torch.Tensor]:
    """Deterministic model + input."""

    torch.manual_seed(0)
    return cls().eval(), torch.randn(*shape)


def _sae_like(out: torch.Tensor, *, hook) -> torch.Tensor:
    """Multi-call injected computation."""

    z = torch.relu(out @ torch.eye(out.shape[-1]))
    return torch.sigmoid(z)


def _save_tamper_load(tmp_path, trace, tamper):
    """Save, mutate the pickled metadata in place, and load back."""

    path = tmp_path / "forged.tlspec"
    tl.save(trace, str(path))
    metadata_path = path / "metadata.pkl"
    with metadata_path.open("rb") as handle:
        state = pickle.load(handle)
    tamper(state)
    with metadata_path.open("wb") as handle:
        pickle.dump(state, handle)
    return tl.load(str(path))


def _injected_rows(state):
    """Injected op rows of a pickled artifact state."""

    return [
        op for op in state["layer_list"] if getattr(op, "injection_provenance", None) is not None
    ]


def _model_rows(state):
    """Model op rows of a pickled artifact state."""

    return [op for op in state["layer_list"] if getattr(op, "injection_provenance", None) is None]


# ---------------------------------------------------------------------------
# 2.3a -- in-place injected ops snapshot their inputs BEFORE the call
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_inplace_injected_ops_attest_on_honest_capture() -> None:
    """mul_/relu_ inside a hook: the pre-call snapshot replays to the recorded out."""

    model, x = _seeded(_MLP)

    def inplace_hook(out: torch.Tensor, *, hook) -> torch.Tensor:
        y = out.clone()
        y.mul_(0.5)
        torch.relu_(y)
        return y

    logged = tl.trace(model, x, intervene=tl.when(tl.func("relu"), inplace_hook), capture=_LOGGED)
    records = injected_ops(logged)
    by_func = {record.func_name: record for record in records}
    assert {"clone", "mul_", "relu_"} <= set(by_func)
    mul_record = by_func["mul_"]
    assert mul_record.saved_args is not None
    # the snapshot is the INPUT (pre-mutation), never the mutated output
    assert not torch.equal(mul_record.saved_args[0], mul_record.out)
    assert torch.equal(mul_record.saved_args[0] * 0.5, mul_record.out)
    report = attest_injected_ops(logged)
    assert report.passed, [(row.label, row.status, row.reason) for row in report.rows]
    assert {row.status for row in report.rows} == {"attested"}


@pytest.mark.smoke
def test_functional_inplace_kwarg_injected_op_attests() -> None:
    """F.relu(inplace=True) inside a hook attests too (kwarg-spelled in-place)."""

    model, x = _seeded(_MLP)

    def hook(out: torch.Tensor, *, hook) -> torch.Tensor:
        y = out.clone() - 0.5
        return torch.nn.functional.relu(y, inplace=True)

    logged = tl.trace(model, x, intervene=tl.when(tl.func("relu"), hook), capture=_LOGGED)
    report = attest_injected_ops(logged)
    assert report.passed, [(row.label, row.status, row.reason) for row in report.rows]


# ---------------------------------------------------------------------------
# 3.7b -- NaN-producing honest injected ops attest; real mismatches still diverge
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_nan_output_injected_op_attests_and_value_mismatch_still_diverges() -> None:
    """NaN == NaN positionally is honest evidence; any finite mismatch is not."""

    model, x = _seeded(_MLP)

    def nan_hook(out: torch.Tensor, *, hook) -> torch.Tensor:
        torch.log(out - 100.0)  # NaN wherever out < 100 (every element here)
        return out * 0.5

    logged = tl.trace(model, x, intervene=tl.when(tl.func("relu"), nan_hook), capture=_LOGGED)
    records = injected_ops(logged)
    log_record = next(record for record in records if record.func_name == "log")
    assert bool(torch.isnan(log_record.out).any())
    report = attest_injected_ops(logged)
    assert report.passed, [(row.label, row.status, row.reason) for row in report.rows]

    # tripwire intact: a finite value edit on the recorded out diverges
    from torchlens.intervention.injection import injection_state

    state = injection_state(logged)
    tampered = []
    for record in state["records"]:
        if record.func_name == "mul":
            record = dataclasses.replace(record, out=record.out + 1.0)
        tampered.append(record)
    state["records"] = tampered
    report = attest_injected_ops(logged)
    assert not report.passed
    mul_row = next(row for row in report.rows if "inj" in row.label and row.status == "diverged")
    assert mul_row.reason == "replay_value_mismatch"

    # tripwire intact: NaN moved onto a finite position diverges
    state["records"] = [
        dataclasses.replace(record, out=torch.nan_to_num(record.out, nan=0.0))
        if record.func_name == "log"
        else record
        for record in injected_ops(logged)
    ]
    report = attest_injected_ops(logged)
    assert any(row.status == "diverged" for row in report.rows)


# ---------------------------------------------------------------------------
# 2.3c -- forged / re-anchored provenance refuses at load (host label, pass, rule)
# ---------------------------------------------------------------------------


def _logged_chain():
    """A logged capture with a spec rule on relu."""

    model, x = _seeded(_Chain, (2, 4))
    spec = tl.when(tl.func("relu"), _sae_like)
    return tl.trace(model, x, intervene=spec, capture=_LOGGED), spec


@pytest.mark.smoke
def test_reanchored_rows_refuse_at_load(tmp_path) -> None:
    """Moving rows onto another op's site key + label + pass refuses typed."""

    logged, _spec = _logged_chain()

    def reanchor(state) -> None:
        tanh = next(op for op in _model_rows(state) if op.layer_label.startswith("tanh"))
        for row in _injected_rows(state):
            envelope = row.annotations["injection_codec_v1"]
            suffix = row.layer_label[len(envelope["host_label"]) :]
            row.injection_provenance["host_site_key"] = tanh.site_key
            row.injection_provenance["host_pass"] = 7
            row.injection_provenance["spec_rule_id"] = "r-forged"
            envelope["host_label"] = f"{tanh.layer_label}:{tanh.pass_index}"
            row.layer_label = f"{tanh.layer_label}:{tanh.pass_index}{suffix}"

    with pytest.raises(Exception) as excinfo:
        _save_tamper_load(tmp_path, logged, reanchor)
    assert excinfo.value.fields["code"] == "artifact_injection_codec_invalid"
    assert excinfo.value.fields["reason"] in {"host_pass_mismatch", "rule_unrecorded"}


@pytest.mark.smoke
def test_site_key_label_disagreement_refuses_at_load(tmp_path) -> None:
    """A host site key that names a DIFFERENT op than host_label refuses."""

    logged, _spec = _logged_chain()

    def inconsistent(state) -> None:
        tanh = next(op for op in _model_rows(state) if op.layer_label.startswith("tanh"))
        for row in _injected_rows(state):
            row.injection_provenance["host_site_key"] = tanh.site_key

    with pytest.raises(Exception) as excinfo:
        _save_tamper_load(tmp_path, logged, inconsistent)
    assert excinfo.value.fields["code"] == "artifact_injection_codec_invalid"
    assert excinfo.value.fields["reason"] == "host_label_mismatch"


@pytest.mark.smoke
def test_forged_host_pass_refuses_at_load(tmp_path) -> None:
    """host_pass must be the host op's own pass index."""

    logged, _spec = _logged_chain()

    def bump_pass(state) -> None:
        for row in _injected_rows(state):
            row.injection_provenance["host_pass"] = 7

    with pytest.raises(Exception) as excinfo:
        _save_tamper_load(tmp_path, logged, bump_pass)
    assert excinfo.value.fields["code"] == "artifact_injection_codec_invalid"
    assert excinfo.value.fields["reason"] == "host_pass_mismatch"


@pytest.mark.smoke
def test_unrecorded_rule_id_refuses_at_load(tmp_path) -> None:
    """spec_rule_id must be a rule the artifact's intervention record ran."""

    logged, _spec = _logged_chain()

    def forge_rule(state) -> None:
        for row in _injected_rows(state):
            row.injection_provenance["spec_rule_id"] = "r-forged"

    with pytest.raises(Exception) as excinfo:
        _save_tamper_load(tmp_path, logged, forge_rule)
    assert excinfo.value.fields["code"] == "artifact_injection_codec_invalid"
    assert excinfo.value.fields["reason"] == "rule_unrecorded"


@pytest.mark.smoke
def test_honest_artifact_still_loads_and_attests(tmp_path) -> None:
    """The anchor checks never refuse an honest round trip."""

    logged, spec = _logged_chain()
    path = tmp_path / "honest.tlspec"
    tl.save(logged, str(path))
    loaded = tl.load(str(path))
    records = injected_ops(loaded)
    assert records and {record.provenance.spec_rule_id for record in records} == {
        spec.rules[0].rule_id
    }
    assert attest_injected_ops(loaded).passed


# ---------------------------------------------------------------------------
# 2.3d -- bind site keys are byte-parity with capture site keys
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bind_site_keys_match_capture_for_submodule_ops() -> None:
    """Every bind fire's live site key is a key the capture minted for that op type."""

    model, x = _seeded(_Nested)
    trace = tl.trace(model, x)
    capture_keys: dict[str, list[str]] = {}
    for op in trace.ops:
        capture_keys.setdefault(op.layer_type, []).append(op.site_key)
    for func_name in ("linear", "relu"):
        bound = tl.when(tl.func(func_name), tl.scale(1.0)).bind(model, on_zero_fire="disclose")
        bound(x)
        bind_keys = sorted(fire["site_key"] for fire in bound.last_report.fires)
        assert bind_keys == sorted(capture_keys[func_name]), (func_name, bind_keys)
        assert not any("|/" in key for key in bind_keys)


@pytest.mark.smoke
def test_bind_in_module_pass_qualifier_still_resolves_after_root_skip() -> None:
    """Root pass counting survives the root frame leaving the module stack."""

    model, x = _seeded(_Nested)
    bound = tl.when(tl.in_module("blk:2") & tl.func("relu"), tl.scale(0.0)).bind(model)
    bound(x)
    assert [fire["site_key"] for fire in bound.last_report.fires] == ["s1|blk/blk.1|relu||1"]
    assert bound.last_report.fire_count == 1


# ---------------------------------------------------------------------------
# 3.7a -- bind fire records derive `replaced` from identity; in-place targets disclosed
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bind_identity_hook_reads_replaced_false() -> None:
    """An identity hook fired but replaced nothing."""

    model, x = _seeded(_MLP)
    bound = tl.when(tl.func("relu"), lambda out, *, hook: out).bind(model)
    bound(x)
    (record,) = bound.last_report.fire_records
    (fire,) = bound.last_report.fires
    assert record.replaced is False
    assert fire["replaced"] is False
    assert fire["in_place_op"] is False

    scaled = tl.when(tl.func("relu"), tl.scale(0.5)).bind(model)
    scaled(x)
    assert scaled.last_report.fire_records[0].replaced is True
    assert scaled.last_report.fires[0]["replaced"] is True


@pytest.mark.smoke
def test_bind_module_boundary_identity_hook_reads_replaced_false() -> None:
    """Boundary fires derive replaced by identity too."""

    model, x = _seeded(_MLP)
    bound = tl.when(tl.module("fc1"), lambda out, *, hook: out).bind(model)
    bound(x)
    assert bound.last_report.fire_records[0].replaced is False
    assert bound.last_report.fires[0]["replaced"] is False


@pytest.mark.smoke
def test_bind_inplace_target_disclosed() -> None:
    """relu_ / F.relu(inplace=True) targets carry in_place_op=True on the fire."""

    model, x = _seeded(_InPlace)
    bound = tl.when(tl.func("relu_"), tl.scale(0.0)).bind(model)
    out = bound(x)
    (fire,) = bound.last_report.fires
    assert fire["in_place_op"] is True
    assert fire["replaced"] is True  # the RETURNED handle was replaced ...
    # ... and the disclosure matters: the caller kept the mutated storage
    assert not bool((out == 0).all())

    plain, x2 = _seeded(_MLP)
    plain_bound = tl.when(tl.func("relu"), tl.scale(0.0)).bind(plain)
    plain_bound(x2)
    assert plain_bound.last_report.fires[0]["in_place_op"] is False


# ---------------------------------------------------------------------------
# 3.7c -- region edits run validate_hook_output
# ---------------------------------------------------------------------------


@pytest.fixture()
def chain_fork():
    """An intervention-ready chain capture plus a fork and reference model."""

    model, x = _seeded(_Chain)
    log = tl.trace(model, x, capture=_READY)
    return model, x, log


@pytest.mark.smoke
def test_region_edit_shape_change_refuses_typed(chain_fork) -> None:
    """out[:1] on a region exit is a downstream lie, refused like node-level do."""

    _model, _x, log = chain_fork
    fork = log.fork()
    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    with pytest.raises(Exception) as excinfo:
        fork.do(target, lambda out, *, hook: out[:1])
    assert excinfo.value.fields["code"] == "intervention_replacement_invalid"


@pytest.mark.smoke
def test_region_edit_dtype_change_refuses_typed(chain_fork) -> None:
    """.double() on a region exit refuses typed instead of a raw downstream RuntimeError."""

    _model, _x, log = chain_fork
    fork = log.fork()
    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    with pytest.raises(Exception) as excinfo:
        fork.do(target, lambda out, *, hook: out.double())
    assert excinfo.value.fields["code"] == "intervention_replacement_invalid"


@pytest.mark.smoke
def test_region_edit_same_shape_still_lowers(chain_fork) -> None:
    """The gate admits a well-formed replacement (no false refusal)."""

    model, x, log = chain_fork
    fork = log.fork()
    target = fork.between(fork["linear_1_1"], fork["tanh_1_4"]).as_region()
    fork.do(target, tl.scale(0.5))
    reference = model.fc3(torch.tanh(model.fc2(torch.relu(model.fc1(x)))) * 0.5)
    assert torch.allclose(fork["linear_3_5"].out, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# 3.7d -- aliased submodules under bind: taught, never guessed
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bind_alias_address_refuses_teaching_canonical_name() -> None:
    """tl.module('dec') on self.dec = self.enc names the alias and its canonical name."""

    model, x = _seeded(_Shared)
    with pytest.raises(BindingPreflightError) as excinfo:
        tl.when(tl.module("dec"), tl.scale(0.0)).bind(model)
    assert excinfo.value.fields["code"] == "bind_static_anchor_unresolved"
    assert "'dec' is an alias of 'enc'" in str(excinfo.value)
    assert "EVERY call site" in str(excinfo.value)


@pytest.mark.smoke
def test_bind_canonical_address_discloses_aliases_and_fires_every_call_site() -> None:
    """The report discloses the alias map; the shared module fires at both sites."""

    model, x = _seeded(_Shared)
    bound = tl.when(tl.module("enc"), tl.scale(0.0)).bind(model)
    out = bound(x)
    report = bound.last_report
    assert report.module_aliases == {"enc": ("dec",)}
    assert [(fire["target"], fire["pass_index"]) for fire in report.fires] == [
        ("enc:1", 1),
        ("enc:2", 2),
    ]
    assert bool((out == 0).all())

    plain, x2 = _seeded(_MLP)
    plain_bound = tl.when(tl.module("fc1"), tl.scale(0.5)).bind(plain)
    plain_bound(x2)
    assert plain_bound.last_report.module_aliases == {}


# ---------------------------------------------------------------------------
# 2.3b -- module-boundary firings anchor to the module-exit op and persist
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_module_boundary_injected_ops_anchor_and_round_trip(tmp_path) -> None:
    """tl.module(...) firings resolve a host site key and survive save/load."""

    model, x = _seeded(_MLP)
    spec = tl.when(tl.module("fc1"), _sae_like)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    records = injected_ops(logged)
    assert records
    host = next(op for op in logged.ops if getattr(op, "intervention_replaced", False))
    assert "fc1:1" in host.output_of_module_calls
    for record in records:
        assert record.host_label == host.label
        assert record.provenance.host_site_key == host.site_key
        assert record.provenance.host_pass == host.pass_index
    path = tmp_path / "module_boundary.tlspec"
    tl.save(logged, str(path))
    loaded = tl.load(str(path))
    reloaded = injected_ops(loaded)
    assert [(r.label, r.provenance.host_site_key) for r in reloaded] == [
        (r.label, r.provenance.host_site_key) for r in records
    ]
    assert attest_injected_ops(loaded).passed


@pytest.mark.smoke
def test_module_boundary_reused_block_anchors_each_call(tmp_path) -> None:
    """A twice-called block anchors each firing to its own module-exit op."""

    model, x = _seeded(_Nested)
    logged = tl.trace(model, x, intervene=tl.when(tl.module("blk"), tl.scale(0.5)), capture=_LOGGED)
    hosts = {record.host_label for record in injected_ops(logged)}
    replaced = {op.label for op in logged.ops if getattr(op, "intervention_replaced", False)}
    assert len(hosts) == 2 and hosts == replaced
    assert all(record.provenance.host_site_key for record in injected_ops(logged))
    tl.save(logged, str(tmp_path / "blk.tlspec"))


@pytest.mark.smoke
def test_module_boundary_rule_id_from_stamped_plan_entry(monkeypatch) -> None:
    """The live hook door anchors a plan entry's stamped ``rule_id`` (in-fence half).

    The capture entry lowers a single-rule ``tl.module(...)`` spec into a
    static hook plan (``user_funcs``); stamping ``metadata["rule_id"]`` there
    is the out-of-fence half of AUD-CODE 2.3b. This test simulates that stamp
    on the lowering call and pins that the door consumes it, so the persisted
    id lands instead of ``adhoc:<helper>``.
    """

    import torchlens.user_funcs as user_funcs
    from torchlens.intervention import hooks as hooks_mod

    model, x = _seeded(_MLP)
    spec = tl.when(tl.module("fc1"), tl.scale(0.5))
    real = hooks_mod.normalize_hook_plan

    def stamped(*args, **kwargs):
        entries = real(*args, **kwargs)
        return [
            dataclasses.replace(
                entry, metadata={**dict(entry.metadata), "rule_id": spec.rules[0].rule_id}
            )
            for entry in entries
        ]

    monkeypatch.setattr(user_funcs, "normalize_hook_plan", stamped)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    records = injected_ops(logged)
    assert records
    assert {record.provenance.spec_rule_id for record in records} == {spec.rules[0].rule_id}
    # and the door resets: a later plain capture's records never inherit the stamp
    monkeypatch.setattr(user_funcs, "normalize_hook_plan", real)
    plain = tl.trace(model, x, intervene=tl.when(tl.func("relu"), _sae_like), capture=_LOGGED)
    relu_rule = tl.when(tl.func("relu"), _sae_like).rules[0].rule_id
    assert {record.provenance.spec_rule_id for record in injected_ops(plain)} == {relu_rule}
