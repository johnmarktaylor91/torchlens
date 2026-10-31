"""Bound-method roots on REAL models (lane F41; foldA MEMO s6 F41 plan).

The realism arms the ruling names, on the R0 roster's REAL classes
(vendored tiny configs, zero network): ``tl.trace(model.generate, ids,
episode=...)`` on distilgpt2, greedy and sampled; a BERT bound-method
encoder; ``model.forward`` as the trivial case; tensor / tuple /
ModelOutput returns; the identity arm (capture ``owner.generate``,
attempt rerun supplying ``owner`` -- must refuse, never report verified
fidelity on a different entry point); and save/load + replay.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import ModelMismatchError

pytest.importorskip("transformers")

from tests.real_model.r0.families import SEED, _token_ids, build_bert, build_distilgpt2

# heavy, not smoke: real distilgpt2 generate() captures measure over the
# 5 s smoke budget on the dev box (tiered honestly, like the F40 real arms).
pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

N_NEW = 3


def _generate_kwargs(do_sample: bool) -> dict:
    return {"max_new_tokens": N_NEW, "do_sample": do_sample, "pad_token_id": 0}


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_distilgpt2_generate_greedy_episode() -> None:
    model = build_distilgpt2("eager")
    ids = _token_ids()
    torch.manual_seed(SEED)
    expected = model.generate(ids, **_generate_kwargs(False))
    torch.manual_seed(SEED)
    log = tl.trace(
        model.generate,
        ids,
        input_kwargs=_generate_kwargs(False),
        episode=tl.options.EpisodeSpec(n_steps=N_NEW),
    )
    episode = log.annotations["episode"]
    # stepped_module DEFAULTED to the bound method's owner.
    assert episode["header"]["stepped_module"] == "owner"
    assert len(episode["rows"]) == N_NEW
    assert log.root_entry_point.endswith("GPT2LMHeadModel.generate")
    assert log.model_class_name == "GPT2LMHeadModel"
    # The captured root output IS what real generate() returns.
    captured = log[log.output_layers[0]].out
    assert torch.equal(captured, expected)


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_distilgpt2_generate_sampled_episode() -> None:
    model = build_distilgpt2("eager")
    ids = _token_ids()
    log = tl.trace(
        model.generate,
        ids,
        input_kwargs=_generate_kwargs(True),
        episode=tl.options.EpisodeSpec(n_steps=N_NEW),
    )
    episode = log.annotations["episode"]
    assert len(episode["rows"]) == N_NEW
    assert all(row["status"] == "complete" for row in episode["rows"])
    captured = log[log.output_layers[0]].out
    assert captured.shape[1] == ids.shape[1] + N_NEW


def test_distilgpt2_generate_plain_capture_without_episode() -> None:
    model = build_distilgpt2("eager")
    ids = _token_ids()
    log = tl.trace(model.generate, ids, input_kwargs=_generate_kwargs(False))
    assert log.root_entry_point.endswith("GPT2LMHeadModel.generate")
    assert "episode" not in (log.annotations or {})
    for output_label in log.output_layers:
        for op in log[output_label].ops:
            assert op.tl_authored_root is True


def test_bert_bound_method_encoder() -> None:
    model = build_bert("eager")
    ids = _token_ids()
    with torch.no_grad():
        hidden = model.embeddings(ids)
    log = tl.trace(model.encoder.forward, hidden)
    assert log.model_class_name == "BertEncoder"
    assert log.root_entry_point.endswith("BertEncoder.forward")
    assert log.outcome.status.name == "COMPLETE"


def test_bert_forward_trivial_case_modeloutput_return() -> None:
    model = build_bert("eager")
    ids = _token_ids()
    log = tl.trace(model.forward, ids)
    assert log.model_class_name == "BertModel"
    assert log.root_entry_point.endswith("BertModel.forward")
    # BertModel.forward returns a ModelOutput; the root output structure
    # captures per-slot output ops rather than refusing.
    assert len(log.output_layers) >= 1


def test_bert_forward_tuple_return() -> None:
    model = build_bert("eager")
    ids = _token_ids()
    log = tl.trace(model.forward, ids, input_kwargs={"return_dict": False})
    assert log.root_entry_point.endswith("BertModel.forward")
    assert len(log.output_layers) >= 1


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_identity_arm_rerun_with_owner_refuses() -> None:
    """The D11 silent-wrongness door: owner passes class+weights, refuses."""

    model = build_distilgpt2("eager")
    ids = _token_ids()
    log = tl.trace(
        model.generate,
        ids,
        input_kwargs=_generate_kwargs(False),
        episode=tl.options.EpisodeSpec(n_steps=N_NEW),
    )
    with pytest.raises(ModelMismatchError) as excinfo:
        log._validate_supplied_model_matches_capture(model)
    assert excinfo.value.fields["code"] == "rerun_entry_point_unsupported"
    assert "generate" in str(excinfo.value)


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_bound_capture_save_load(tmp_path) -> None:
    model = build_distilgpt2("eager")
    ids = _token_ids()
    log = tl.trace(
        model.generate,
        ids,
        input_kwargs=_generate_kwargs(False),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    target = tmp_path / "gpt2_generate.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    assert loaded.root_entry_point == log.root_entry_point
    with pytest.raises(ModelMismatchError):
        loaded._validate_supplied_model_matches_capture(model)


def test_replay_engine_on_real_bound_capture() -> None:
    # Replay-engine intervention works on a live bound capture (single-pass
    # BERT root: replay inside a multi-pass generate() graph is the F42
    # coupling lane's territory, and episode x intervene refuses per D5).
    bert = build_bert("eager")
    bert_log = tl.trace(
        bert.forward,
        _token_ids(),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    fork = bert_log.fork()
    target_op = next(op for op in bert_log.layer_list if op.label.startswith("linear"))
    zero_index = tuple(0 for _ in target_op.out.shape)
    fork.do(tl.units(target_op.label, [zero_index]).resolve(fork), tl.zero_ablate())
    assert len(fork.intervention_audit) == 1
