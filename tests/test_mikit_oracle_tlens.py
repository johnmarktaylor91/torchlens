"""Gate 2 -- the TransformerLens oracle (mikit D13/D14; legs G2a/G2b/G2c).

LIVE in-process oracle on the transformers-5.x band, pinned transformer-lens
3.8.0 (test-only extra; packaging request filed). Revision-pinned real gpt2
from the local HF cache; the module skips without the pin installed and runs
OFFLINE (no test-body downloads).

Blocking law (D14, amended from the brief by measurement): the IDENTITY is
the gate; TLens agreement is per-gauge rows. G2a/G2b residual halves assert
``torch.equal``; G2b's logit-diff DLA parity row is BLOCKING (the 2-1); the
raw-gauge ABSOLUTE DLA row is a convention diagnostic that can never gate
(TLens's own absolute logit_attrs does not sum to their native logit on
unprocessed weights -- verified from their source; ours does).
"""

from __future__ import annotations

import os

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._anchors import resolve_lm_head

transformer_lens = pytest.importorskip("transformer_lens")

pytestmark = [pytest.mark.slow, pytest.mark.real_model]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROMPT = "The Eiffel Tower is located in the city of"


@pytest.fixture(scope="module")
def raw_leg():
    """G2a fixture: TLens no_processing gpt2, its cache, and our trace of IT."""

    model = transformer_lens.HookedTransformer.from_pretrained_no_processing("gpt2")
    model.eval()
    tokens = model.to_tokens(PROMPT)  # explicit ids to BOTH sides (BOS measured)
    _, cache = model.run_with_cache(tokens)
    log = tl.trace(model, tokens, capture=tl.options.CaptureOptions(layers_to_save="all"))
    try:
        yield model, tokens, cache, log
    finally:
        log.cleanup()


def _tlens_states(model, cache):
    """TLens's own residual states in execution order (pre0, mid/post per layer)."""

    states = [cache["resid_pre", 0]]
    for layer in range(model.cfg.n_layers):
        states.append(cache["resid_mid", layer])
        states.append(cache["resid_post", layer])
    return states


def test_g2a_same_object_raw_torch_equal(raw_leg):
    """G2a: 25 accumulated + 26 decomposed rows, asserted at torch.equal."""

    model, _tokens, cache, log = raw_leg
    acc = mi.residual_accumulation(log, include_mid=True)
    states = _tlens_states(model, cache)
    assert len(acc) == len(states) == 25
    for ours, theirs in zip(acc.rows, states, strict=True):
        assert torch.equal(ours.value, theirs)

    dec = mi.residual_decomposition(log)
    stack, labels = cache.decompose_resid(layer=-1, return_labels=True)
    assert len(dec) == len(labels) == 26
    for index in range(len(dec)):
        assert torch.equal(dec.rows[index].value, stack[index])
    assert dec.identity_receipt["result"] == "bitwise_equal"


def test_g2a_norm_scale_bit_exact(raw_leg):
    """The scale definition vs TLens's own ln_final.hook_scale: bit-exact."""

    _model, _tokens, cache, log = raw_leg
    norm = resolve_lm_head(log).norm_reconstruction()
    assert norm.kind == "layernorm_affine"
    assert torch.equal(norm.scale, cache["ln_final.hook_scale"])


def test_g2a_raw_gauge_dla_identity(raw_leg):
    """Raw-gauge DLA: OUR identity closes (TLens's absolute form cannot).

    Convention-diagnostic row -- it can never gate TLens agreement; what it
    pins is that our sum(rows) + constant equals the model's own logit.
    """

    model, _tokens, _cache, log = raw_leg
    answer = int(model.to_tokens(" Paris", prepend_bos=False)[0, 0])
    vs = int(model.to_tokens(" London", prepend_bos=False)[0, 0])
    dla = mi.direct_logit_contributions(log, answer=answer, vs=vs)
    assert dla.identity_receipt["result"] == "verified"


def test_g2b_folded_gauge_blocking_rows():
    """G2b: folded/community gauge -- residual torch.equal + BLOCKING DLA parity.

    Exercises the ``normalize_only`` norm kind end-to-end (LayerNormPre).
    """

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
    model.eval()
    tokens = model.to_tokens(PROMPT)
    _, cache = model.run_with_cache(tokens)
    log = tl.trace(model, tokens, capture=tl.options.CaptureOptions(layers_to_save="all"))

    acc = mi.residual_accumulation(log, include_mid=True)
    for ours, theirs in zip(acc.rows, _tlens_states(model, cache), strict=True):
        assert torch.equal(ours.value, theirs)

    norm = resolve_lm_head(log).norm_reconstruction()
    assert norm.kind == "normalize_only" and norm.centered

    answer = int(model.to_tokens(" Paris", prepend_bos=False)[0, 0])
    vs = int(model.to_tokens(" London", prepend_bos=False)[0, 0])
    dla = mi.direct_logit_contributions(log, answer=answer, vs=vs)
    assert dla.identity_receipt["result"] == "verified"

    # BLOCKING parity: our logit-diff rows vs TLens's own fold of its own stack.
    stack, _labels = cache.decompose_resid(layer=-1, return_labels=True)
    folded = cache.apply_ln_to_stack(stack, layer=-1)[:, 0, -1, :]
    direction = model.W_U[:, answer] - model.W_U[:, vs]
    theirs = (folded @ direction).reshape(-1)
    ours = dla.values[:, 0, 0, 0].detach()
    assert ours.shape == theirs.shape
    gap = float((ours - theirs).abs().max())
    assert gap < 5e-6, f"G2b DLA logit-diff parity broke: {gap}"


def test_g2c_cross_model_on_the_model_users_run():
    """G2c: torchlens on REAL HF gpt2 vs TLens on its no_processing copy.

    The only leg saying we agree with them ON THE MODEL USERS RUN; the
    tolerance is the measured per-depth residual noise floor (memo class
    2.4e-04; re-measured here and recorded in the assertion message).
    """

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    tlens = transformer_lens.HookedTransformer.from_pretrained_no_processing("gpt2")
    tlens.eval()
    tokens = tlens.to_tokens(PROMPT)
    _, cache = tlens.run_with_cache(tokens)
    log = tl.trace(hf, tokens, capture=tl.options.CaptureOptions(layers_to_save="all"))

    acc = mi.residual_accumulation(log, include_mid=True)
    states = _tlens_states(tlens, cache)
    assert len(acc) == len(states)
    worst = max(
        float((ours.value - theirs).abs().max())
        for ours, theirs in zip(acc.rows, states, strict=True)
    )
    assert worst < 2e-3, f"cross-model residual drift {worst} (noise-floor class 2.4e-04)"

    dec = mi.residual_decomposition(log)
    assert dec.identity_receipt["result"] == "bitwise_equal"
    answer = int(tlens.to_tokens(" Paris", prepend_bos=False)[0, 0])
    dla = mi.direct_logit_contributions(log, answer=answer)
    assert dla.identity_receipt["result"] == "verified"


def test_convention_tripwire_apply_ln_is_pre_affine():
    """D13's pin-move tripwire: TLens 3.8.0's apply_ln_to_stack applies NO
    gamma/beta (the fold lives in processed weights). A version bump that
    changes this convention must fail LOUDLY here, not drift silently.
    """

    model = transformer_lens.HookedTransformer.from_pretrained_no_processing("gpt2")
    model.eval()
    tokens = model.to_tokens("Paris is a city")
    _, cache = model.run_with_cache(tokens)
    resid = cache["resid_post", model.cfg.n_layers - 1]  # [batch, pos, d]
    folded = cache.apply_ln_to_stack(resid.unsqueeze(0), layer=-1)[0, 0, -1, :]
    scale = cache["ln_final.hook_scale"][0, -1]
    last = resid[0, -1]
    manual_pre_affine = (last - last.mean()) / scale
    assert torch.allclose(folded, manual_pre_affine, atol=1e-5), (
        "TLens's apply_ln_to_stack convention changed (gamma/beta now applied?): "
        "re-adjudicate the G2b parity row before moving the pin"
    )
