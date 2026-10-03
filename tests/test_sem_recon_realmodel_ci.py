"""Semantic-tier REAL-MODEL CI test (A03 row gate; WT1 A-I bundle requirement).

The entire semantic appliance tier used to be validated only on hand-rolled
zero-parameter toys, which is how five independent real-model breakages
shipped (silent no-op patches, the attribution autograd crash, the 5.x
logit-lens refusal). These rows run the appliances against the R0 config-built
GPT-2 (REAL upstream class, REAL vendored config, zero network) and pin the
honest settled state of each surface:

* a surface either WORKS with verifiable numbers or refuses/warns TYPED --
  a silent flat table or a bare crash fails these rows in either direction;
* rows are written either-or where a sibling lane owns the remaining repair
  (A01 per-head facets, A02 facet re-homing), so they stay green when those
  lanes land and keep asserting the honesty contract.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

import torchlens as tl
from tests.real_model.r0.families import _token_ids, build_gpt2
from torchlens.errors._base import TorchLensWarning
from torchlens.semantic.logit_lens import (
    PROVENANCE_NATIVE,
    LogitLensError,
    logit_lens,
    logit_lens_predictions,
)
from torchlens.semantic.patching import PatchApplicationError

# File-level: the real_model selection marker only. Cost tiers are per-test:
# the logit-lens/attribution rows are smoke (sub-second on the cached capture);
# the activation-patch rows re-trace inside the helpers (~5-12 s) and ride the
# heavy tier per the marker lint's 5 s smoke budget.
pytestmark = pytest.mark.real_model

_CAPTURE = tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True)


@pytest.fixture(scope="module")
def gpt2() -> Any:
    """One config-built GPT-2 (sdpa) for the whole module, built before tracing."""

    return build_gpt2("sdpa")


@pytest.fixture(scope="module")
def gpt2_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Clean/corrupted token batches differing at the last position."""

    clean = _token_ids()
    corrupted = clean.clone()
    corrupted[0, -1] = (corrupted[0, -1] + 7) % 512
    return clean, corrupted


@pytest.fixture(scope="module")
def gpt2_log(gpt2: Any, gpt2_inputs: tuple[torch.Tensor, torch.Tensor]) -> Any:
    """One cached exhaustive capture of the clean batch."""

    log = tl.trace(gpt2, gpt2_inputs[0], capture=_CAPTURE)
    yield log
    log.cleanup()


def _user_lens(gpt2: Any) -> Any:
    """The model's REAL final norm + unembedding as an explicit lens."""

    ln_f = gpt2.transformer.ln_f
    weight = gpt2.lm_head.weight

    def lens(hidden: torch.Tensor) -> torch.Tensor:
        normed = torch.nn.functional.layer_norm(
            hidden, ln_f.normalized_shape, ln_f.weight, ln_f.bias, ln_f.eps
        )
        return normed @ weight.T

    return lens


def _last_token_metric(log: Any) -> torch.Tensor:
    return log[log.output_layers[0]].out[0, -1, 0]


def test_logit_lens_default_route_settles_typed_or_validated(gpt2_log: Any) -> None:
    """The flagship 5.x failure: the default route may refuse TYPED, never lie.

    Today the final-norm anchor does not survive the ``logits_to_keep``
    getitem between ln_f and lm_head (recipe hop rule, lane A02), so the
    default reconstruction refuses with the facet named. Once A02 lands, the
    same call must return a VALIDATED result. Both settled states are honest;
    anything else -- an unvalidated success, a bare crash -- is a defect.
    """

    try:
        result = logit_lens(gpt2_log)
    except LogitLensError as exc:
        assert "final_norm_kind" in str(exc)
        assert "lens=" in str(exc)  # the teaching remedy is named
    else:
        assert result.validated
        assert result.entries and result.entries[-1].logits.shape[-1] == 512


def test_logit_lens_user_lens_route_projects_real_numbers(gpt2: Any, gpt2_log: Any) -> None:
    """The lens= socket works on real GPT-2 TODAY, with verifiable numbers.

    Projecting the last block's captured resid_post through the model's OWN
    ln_f + lm_head must reproduce the model's captured output logits exactly
    (suffix-aware: 5.x captures may keep only the last positions).
    """

    result = logit_lens(gpt2_log, lens=_user_lens(gpt2))
    assert result.lens_source == "user" and not result.validated
    assert [entry.address for entry in result.entries] == [
        "transformer.h.0",
        "transformer.h.1",
    ]
    final = result.final_logits
    assert final is not None
    last = result.entries[-1].logits
    kept = final.shape[-2]
    assert torch.allclose(last[..., -kept:, :], final, atol=1e-4, rtol=1e-4)


def test_logit_lens_predictions_stream_on_real_model(gpt2: Any, gpt2_log: Any) -> None:
    """FIX-L on a real model: streamed rows, native-output honesty, no vocab retention."""

    preds = logit_lens_predictions(gpt2_log, k=5, tokens=[0, 17], lens=_user_lens(gpt2))
    native = preds.rows[-1]
    assert native.provenance == PROVENANCE_NATIVE
    captured = gpt2_log.modules["self"].facets["logits"].value.float()
    assert torch.allclose(native.logsumexp, torch.logsumexp(captured, dim=-1), atol=1e-5)
    for row in preds.rows:
        assert row.top_ids.shape[-1] == 5
        assert 512 not in tuple(row.logsumexp.shape)
        assert row.token_ranks.dtype == torch.int64
        assert bool((row.token_ranks >= 1).all())


@pytest.mark.heavy
def test_activation_patch_residual_stream_never_silently_flat(
    gpt2: Any, gpt2_inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The measured silent no-op: a flat baseline-equal table with NO signal fails.

    Today ``resid_pre`` is mis-homed on a mask-derived op (lane A02 owns the
    re-homing), so every patch replaces an identical value and the table
    equals the corrupted baseline -- that state must be DISCLOSED (typed
    refusal or the all-identical campaign warning). Post-A02, patches land on
    the real stream and at least one cell must move. A silent flat table --
    the publishable null result -- fails this row in every world.
    """

    clean, corrupted = gpt2_inputs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            table = tl.facets.patching.activation_patch_residual_stream(
                gpt2, clean, corrupted, _last_token_metric, patch_positions=False
            )
        except PatchApplicationError:
            return  # typed refusal is an honest settled state
    baseline_log = tl.trace(gpt2, corrupted, capture=_CAPTURE)
    try:
        baseline = _last_token_metric(baseline_log).detach()
    finally:
        baseline_log.cleanup()
    disclosed = any(
        isinstance(warning.message, TorchLensWarning) and "IDENTICAL" in str(warning.message)
        for warning in caught
    )
    moved = bool((table != baseline).any())
    assert disclosed or moved, "flat baseline-equal patch table published with no disclosure"


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("helper_name", "kwargs"),
    [
        (
            "activation_patch_residual_stream",
            {"facet_name": "resid_post", "patch_positions": False},
        ),
        ("activation_patch_attention_output", {}),
        ("activation_patch_mlp_output", {}),
    ],
)
def test_activation_patch_surfaces_settle_typed_or_effective(
    gpt2: Any,
    gpt2_inputs: tuple[torch.Tensor, torch.Tensor],
    helper_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Every activation-patch surface on real GPT-2: typed refusal or real effect.

    Measured today (all typed, all honest -- pre-fix every one of these was a
    SILENT flat table or an unprotected wrong patch):

    * ``attn_out``/MLP ``output`` homes sit on aliasing ops the live-hook
      engine refuses to replace -- the fire ledger turns that into
      ``PatchApplicationError`` (the measured DIGEST mechanism).
    * ``resid_post`` homes on the recurrence-grouped residual adds; the
      hook path addresses them by bare layer label and refuses
      ``SiteAmbiguityError`` (pass-qualified addressing, lane A04).

    When the sibling lanes land their re-homing/addressing repairs, a table
    comes out instead -- then at least one cell must move (clean and
    corrupted inputs differ at the last token) or the campaign must disclose
    all-identical values. A silent baseline-equal table fails in every world.
    """

    from torchlens.intervention.errors import SiteAmbiguityError

    clean, corrupted = gpt2_inputs
    helper = getattr(tl.facets.patching, helper_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            table = helper(gpt2, clean, corrupted, _last_token_metric, **kwargs)
        except PatchApplicationError as exc:
            assert "replaced=False" in str(exc) or "never fired" in str(exc)
            return
        except SiteAmbiguityError as exc:
            assert "pass-qualified" in str(exc)
            return
    baseline_log = tl.trace(gpt2, corrupted, capture=_CAPTURE)
    try:
        baseline = _last_token_metric(baseline_log).detach()
    finally:
        baseline_log.cleanup()
    disclosed = any(
        isinstance(warning.message, TorchLensWarning) and "IDENTICAL" in str(warning.message)
        for warning in caught
    )
    moved = bool((table != baseline).any())
    assert disclosed or moved, "flat baseline-equal patch table published with no disclosure"


def test_attribution_patch_heads_refuses_typed_on_sdpa(
    gpt2: Any, gpt2_inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Attribution on sdpa GPT-2: reconstruction facets refuse grads TYPED.

    A01's registration made per-head facets exist on the sdpa build, but they
    come from the reconstruction recipe (computed read-only hypotheses with no
    captured gradient home), so attribution must refuse with the teaching
    message naming the facet and module rather than publish silent numbers.
    Pre-A01 this row pinned the facet-absence ValueError instead.
    """

    clean, corrupted = gpt2_inputs
    with pytest.raises(RuntimeError, match="requires grad capture.*not grad-capable"):
        tl.facets.patching.attribution_patch_attention_heads(
            gpt2, clean, corrupted, _last_token_metric
        )


@pytest.mark.heavy
def test_attribution_patch_heads_finite_on_eager() -> None:
    """Attribution on eager GPT-2: op-anchored ``z`` facets give finite numbers.

    The eager build exposes op-anchored per-head facets (A01's anchors), so
    the attribution path exercises the param-restore fix end-to-end on a real
    parameterized model and must return finite values instead of the
    historical autograd version-counter crash. The default ``result`` facet
    stays a computed derivation (grad-refusing) even on eager, so this row
    addresses the grad-capable ``z`` facet explicitly.
    """

    model = build_gpt2("eager")
    clean = _token_ids()
    corrupted = clean.clone()
    corrupted[0, -1] = (corrupted[0, -1] + 7) % 512
    table = tl.facets.patching.attribution_patch_attention_heads(
        model, clean, corrupted, _last_token_metric, facet_name="z"
    )
    assert table.shape == (2, 2)
    assert bool(torch.isfinite(table).all())
