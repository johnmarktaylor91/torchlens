"""Lane F44 stage 2: forged injected-op provenance REFUSES at load.

Every structural violation of the persisted injected family -- the C07
``Op.injection_provenance`` grammar, the codec envelope, host anchoring,
graph entanglement, and the intervention-replacement carve-out theft -- is
a typed load refusal, never a degradation. Degrade-to-unattested is for
UNRESOLVABLE CALLABLES only; forged identity always refuses (foldA MEMO s5
item 12: "forged provenance refuses").
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

import torchlens as tl

_LOGGED = tl.options.CaptureOptions(log_injections=True)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh: minimal injected-op substrate."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return torch.tanh(self.fc2(torch.relu(self.fc1(x))))


def _sae_like(out: torch.Tensor, *, hook) -> torch.Tensor:
    """SAE-style injected computation (multiple recordable calls)."""

    z = torch.relu(out @ torch.eye(out.shape[-1]))
    return torch.sigmoid(z)


def _tampered_load(tmp_path, tamper):
    """Save a logged capture, tamper its pickled state, and load it back.

    ``tamper(state)`` mutates the raw metadata mapping in place; the return
    value is whatever ``tl.load`` does with the forged artifact.
    """

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(2, 4)
    logged = tl.trace(model, x, intervene=tl.when(tl.func("relu"), _sae_like), capture=_LOGGED)
    path = tmp_path / "forged.tlspec"
    tl.save(logged, str(path))
    metadata_path = path / "metadata.pkl"
    with metadata_path.open("rb") as handle:
        state = pickle.load(handle)
    tamper(state)
    with metadata_path.open("wb") as handle:
        pickle.dump(state, handle)
    return tl.load(str(path))


def _injected_rows(state):
    """The forged artifact's injected op rows."""

    return [
        op for op in state["layer_list"] if getattr(op, "injection_provenance", None) is not None
    ]


def _model_rows(state):
    """The forged artifact's ordinary model op rows."""

    return [op for op in state["layer_list"] if getattr(op, "injection_provenance", None) is None]


def _assert_refuses(tmp_path, tamper, code: str, reason: str | None = None) -> None:
    """Assert one tamper refuses with the expected typed code (and reason)."""

    with pytest.raises(Exception) as excinfo:
        _tampered_load(tmp_path, tamper)
    fields = getattr(excinfo.value, "fields", {}) or {}
    assert fields.get("code") == code, (fields, str(excinfo.value)[:300])
    if reason is not None:
        assert fields.get("reason") == reason, fields


@pytest.mark.smoke
def test_malformed_grammar_refuses(tmp_path) -> None:
    """A provenance record missing a grammar field refuses record_schema."""

    def tamper(state) -> None:
        """Drop one closed-grammar key."""

        del _injected_rows(state)[0].injection_provenance["output_slot"]

    _assert_refuses(tmp_path, tamper, "artifact_injection_provenance_invalid", "record_schema")


@pytest.mark.smoke
def test_negative_ordinal_refuses(tmp_path) -> None:
    """A negative firing index refuses record_ordinals."""

    def tamper(state) -> None:
        """Forge an impossible ordinal."""

        _injected_rows(state)[0].injection_provenance["firing_index"] = -3

    _assert_refuses(tmp_path, tamper, "artifact_injection_provenance_invalid", "record_ordinals")


@pytest.mark.smoke
def test_host_site_key_must_exist_in_the_artifact(tmp_path) -> None:
    """A provenance anchor naming no retained model op refuses host_missing."""

    def tamper(state) -> None:
        """Point the anchor at a site the artifact does not contain."""

        for row in _injected_rows(state):
            row.injection_provenance["host_site_key"] = "s1|nowhere/relu/0/0"

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "host_missing")


@pytest.mark.smoke
def test_duplicate_durable_keys_refuse(tmp_path) -> None:
    """Two rows sharing one durable identity key refuse record_duplicate."""

    def tamper(state) -> None:
        """Copy row 0's whole identity onto row 1."""

        rows = _injected_rows(state)
        rows[1].injection_provenance = dict(rows[0].injection_provenance)

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "record_duplicate")


@pytest.mark.smoke
def test_carve_out_theft_refuses(tmp_path) -> None:
    """An injected row wearing the intervention_replacement identity refuses.

    The metadata-invariant exemption stays scoped to GENUINE user
    interventions (the 2026-06-02 incident law): the injected family can
    never launder a row into the placeholder carve-out.
    """

    def tamper(state) -> None:
        """Dress an injected row as an intervention placeholder."""

        row = _injected_rows(state)[0]
        row.func_name = "intervention_replacement"

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "record_func")


@pytest.mark.smoke
def test_replaced_stamp_on_injected_row_refuses(tmp_path) -> None:
    """An injected row claiming intervention_replaced refuses record_func."""

    def tamper(state) -> None:
        """Forge the replacement stamp."""

        _injected_rows(state)[0].intervention_replaced = True

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "record_func")


@pytest.mark.smoke
def test_graph_entangled_injected_row_refuses(tmp_path) -> None:
    """An injected row claiming model dataflow parents refuses."""

    def tamper(state) -> None:
        """Forge a dataflow edge onto the injected row."""

        model_label = _model_rows(state)[0].layer_label
        _injected_rows(state)[0].parents = [model_label]

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "record_graph_entangled")


@pytest.mark.smoke
def test_model_op_referencing_injected_row_refuses(tmp_path) -> None:
    """A model op wired to an injected label refuses record_referenced.

    This also closes the op-hiding hole: stamping injection_provenance
    onto a REAL model op cannot silently remove it from the graph, because
    its neighbors still reference it.
    """

    def tamper(state) -> None:
        """Wire a model op's children at an injected row's label."""

        injected_label = _injected_rows(state)[0].layer_label
        model_row = _model_rows(state)[0]
        model_row.children = [*list(model_row.children or ()), injected_label]

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "record_referenced")


@pytest.mark.smoke
def test_missing_codec_envelope_refuses(tmp_path) -> None:
    """The identity slot only ships with its codec envelope (fail-closed)."""

    def tamper(state) -> None:
        """Strip the codec envelope."""

        _injected_rows(state)[0].annotations = {}

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "codec_missing")


@pytest.mark.smoke
def test_envelope_key_set_is_closed(tmp_path) -> None:
    """An extra envelope key refuses codec_schema."""

    def tamper(state) -> None:
        """Smuggle an extra envelope key."""

        row = _injected_rows(state)[0]
        row.annotations["injection_codec_v1"]["extra"] = 1

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "codec_schema")


@pytest.mark.smoke
def test_unknown_envelope_version_refuses(tmp_path) -> None:
    """A future envelope version refuses codec_version (never normalizes)."""

    def tamper(state) -> None:
        """Claim an unknown codec version."""

        _injected_rows(state)[0].annotations["injection_codec_v1"]["version"] = 2

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "codec_version")


@pytest.mark.smoke
def test_label_must_extend_host_label(tmp_path) -> None:
    """An injected label detached from its host label refuses codec_label."""

    def tamper(state) -> None:
        """Rename the row off its host."""

        _injected_rows(state)[0].layer_label = "orphan/inj_1"

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "codec_label")


@pytest.mark.smoke
def test_malformed_callable_key_refuses(tmp_path) -> None:
    """A callable reference outside the closed registry-key shape refuses."""

    def tamper(state) -> None:
        """Forge a string-typed callable reference."""

        row = _injected_rows(state)[0]
        row.annotations["injection_codec_v1"]["callable"] = "os:system"

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "codec_callable")


@pytest.mark.smoke
def test_malformed_encoded_arg_refuses(tmp_path) -> None:
    """An encoded replay arg outside the closed grammar refuses codec_args."""

    def tamper(state) -> None:
        """Forge an unknown arg encoding kind."""

        row = _injected_rows(state)[0]
        envelope = row.annotations["injection_codec_v1"]
        if envelope["args"]:
            envelope["args"][0] = {"kind": "pickle", "value": b"boom"}
        else:  # pragma: no cover - the fixture always snapshots args
            envelope["args"] = [{"kind": "pickle", "value": b"boom"}]

    _assert_refuses(tmp_path, tamper, "artifact_injection_codec_invalid", "codec_args")


@pytest.mark.smoke
def test_model_op_wearing_the_slot_cannot_hide(tmp_path) -> None:
    """Stamping the slot onto a connected model op refuses, never hides it."""

    def tamper(state) -> None:
        """Claim a mid-graph model op is injected."""

        rows = _injected_rows(state)
        model_row = next(
            op for op in _model_rows(state) if (op.parents or ()) and (op.children or ())
        )
        model_row.injection_provenance = dict(rows[0].injection_provenance)
        model_row.injection_provenance["local_op_ordinal"] = 99

    with pytest.raises(Exception) as excinfo:
        _tampered_load(tmp_path, tamper)
    assert (getattr(excinfo.value, "fields", {}) or {}).get("code") in {
        "artifact_injection_codec_invalid",
        "artifact_injection_provenance_invalid",
    }


@pytest.mark.smoke
def test_internal_codec_guards_are_typed(tmp_path) -> None:
    """The two defensive codec guards raise their declared codes.

    Neither is reachable from lawful user input (the scrub always emits a
    layer_list; the fire-time snapshotter admits only encodable values), so
    they are provoked directly.
    """

    from torchlens._io.injection_codec import _encode_value, append_injected_op_rows

    torch.manual_seed(0)
    model = _Chain().eval()
    logged = tl.trace(
        model,
        torch.randn(2, 4),
        intervene=tl.when(tl.func("relu"), _sae_like),
        capture=_LOGGED,
    )
    with pytest.raises(Exception) as excinfo:
        append_injected_op_rows(
            logged, {"layer_list": None}, [], include_outs=True, backend_name="torch"
        )
    assert excinfo.value.fields["code"] == "injection_persist_state_invalid"
    with pytest.raises(Exception) as excinfo:
        _encode_value(object(), "x/inj_1", [], [0], "torch")
    assert excinfo.value.fields["code"] == "injection_persist_arg_unencodable"


@pytest.mark.smoke
def test_stream_finalize_door_refuses_on_records() -> None:
    """The streamed-writer door refuses a trace carrying injected records.

    End-to-end streaming of an intervened capture fails earlier today on
    the pre-existing fire-counter portability gap, so the door is armed
    defense-in-depth and provoked directly here.
    """

    from torchlens.intervention.injection import (
        injection_state,
        refuse_injection_logged_stream_finalize,
    )

    class _Holder:
        """Minimal trace stand-in carrying one injected record."""

    holder = _Holder()
    injection_state(holder)["records"] = ["one-record"]
    with pytest.raises(Exception) as excinfo:
        refuse_injection_logged_stream_finalize(holder)
    assert excinfo.value.fields["code"] == "injection_logged_stream_unsupported"
    clean = _Holder()
    refuse_injection_logged_stream_finalize(clean)
