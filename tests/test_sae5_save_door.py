"""Facet selectors at the capture-time save door, ``tl.trace(model, x, save=selector)``.

``save=tl.head(i, "q")`` or ``save=tl.facet(name).latent(k)`` keeps the ops a
later ``trace[selector]`` read needs: an SAE latent's registered site, or, for
a head, every op inside a module whose class has a recipe declaring the facet
(a superset of the facet's homes, since a recipe picks its home ops from the
finished module record). The read afterwards equals the closed form and the
read on a fully saved trace.

The save door is save-only. ``intervene=``, ``bind`` and capture-time
``hooks=`` evaluate the selector itself at capture, where it still refuses
typed, so no facet edit can land on a whole home tensor; the list spelling
``save=[selector]`` still refuses too. Tolerances, fixed before the first run:
``atol=1e-6, rtol=0``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_sae4_latent_facets import TOL, X, _HandSAE, _latent_column, _Model, _TwoSites
from tests.test_sae5_read_doors import _AttnModel, _q_head
from torchlens._errors import ArgumentTypeError
from torchlens.intervention.errors import SelectorCapabilityError, SiteResolutionError


def _saved(trace: tl.Trace) -> set[str]:
    return {
        str(op.layer_label) for op in trace.layer_list if getattr(op, "has_saved_activation", False)
    }


@pytest.fixture()
def attn() -> tuple[_AttnModel, torch.Tensor]:
    torch.manual_seed(0)
    return _AttnModel().eval(), torch.randn(2, 3, 8)


@pytest.fixture()
def latent_setup() -> Iterator[tuple[_Model, torch.Tensor]]:
    model, sae = _Model().eval(), _HandSAE()
    tl.bridge.sae.register("res", sae, site="fc1")
    try:
        yield model, torch.tensor(X)
    finally:
        tl.bridge.sae.unregister("res")


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


# --- the door works ----------------------------------------------------------------


def test_a_head_save_keeps_the_attention_interior_and_reads_the_slice(attn) -> None:
    model, x = attn
    selector = tl.head(0, "q")
    trace = tl.trace(model, x, save=selector)
    full = tl.trace(model, x)
    torch.testing.assert_close(trace[selector].value, _q_head(model, x, 0), **TOL)
    torch.testing.assert_close(trace[selector].value, full[selector].value, **TOL)
    interior = {
        str(op.layer_label)
        for op in full.layer_list
        if any(str(m).split(":")[0].startswith("attn") for m in (op.modules or ()))
    }
    assert interior and interior <= _saved(trace)


def test_a_latent_save_keeps_its_site_and_reads_the_closed_form(latent_setup) -> None:
    model, x = latent_setup
    selector = tl.facet("res").latent(1)
    trace = tl.trace(model, x, save=selector)
    assert trace["fc1"].layer_label in _saved(trace)
    torch.testing.assert_close(trace[selector].value, _latent_column(1), **TOL)
    torch.testing.assert_close(trace["fc1"].facets["res[1]"].value, _latent_column(1), **TOL)


def test_a_latent_registered_at_a_selector_site_saves_every_home() -> None:
    torch.manual_seed(0)
    model, x, sae = _TwoSites().eval(), torch.randn(2, 4), _HandSAE()
    tl.bridge.sae.register("both", sae, site=tl.module("fc_a") | tl.module("fc_b"))
    try:
        trace = tl.trace(model, x, save=tl.facet("both").latent(2))
        assert {trace["fc_a"].layer_label, trace["fc_b"].layer_label} <= _saved(trace)
        for site in ("fc_a", "fc_b"):
            expected = sae.encode(trace[site].out)[..., 2].detach()
            torch.testing.assert_close(trace[site].facets["both[2]"].value, expected, **TOL)
    finally:
        tl.bridge.sae.unregister("both")


def test_a_structural_facet_save_keeps_every_op() -> None:
    """``tl.facet("out")`` is served by every op, so its save keeps every op."""

    torch.manual_seed(0)
    model, x = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval(), torch.randn(2, 4)
    trace = tl.trace(model, x, save=tl.facet("out"))
    full = tl.trace(model, x)
    assert _saved(trace) == _saved(full)
    for op in trace.layer_list:
        torch.testing.assert_close(op.facets["out"].value, full[op.label].out, **TOL)


@pytest.mark.parametrize("make", [lambda: tl.head(0), lambda: tl.facet("q")], ids=["head", "facet"])
def test_a_facet_save_on_a_model_without_the_facet_warns_it_matched_nothing(make) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU()).eval()
    selector = make()
    with pytest.warns(
        UserWarning, match=rf"save selector {re.escape(repr(selector))} matched zero"
    ):
        trace = tl.trace(model, torch.randn(1, 1, 5, 5), save=selector)
    assert all(trace[label].is_output for label in _saved(trace))


# --- the doors that still refuse ---------------------------------------------------


@pytest.mark.parametrize("which", ["head", "latent"])
def test_intervene_still_refuses_a_facet_selector(attn, latent_setup, which) -> None:
    model, x = attn if which == "head" else latent_setup
    selector = tl.head(0, "q") if which == "head" else tl.facet("res").latent(1)
    with pytest.raises(SiteResolutionError, match="resolve through intervention mutators"):
        tl.trace(model, x, intervene=tl.when(selector, tl.zero_ablate()))


@pytest.mark.parametrize("which", ["head", "latent"])
def test_bind_still_refuses_a_facet_selector(attn, latent_setup, which) -> None:
    model, x = attn if which == "head" else latent_setup
    selector = tl.head(0, "q") if which == "head" else tl.facet("res").latent(1)
    bound = tl.when(selector, tl.zero_ablate()).bind(model)
    with pytest.raises(Exception) as caught:
        bound(x)
    assert any(isinstance(error, SelectorCapabilityError) for error in _chain(caught.value))


@pytest.mark.parametrize("which", ["head", "latent"])
def test_capture_hooks_key_is_hashable_and_still_refuses_typed(attn, latent_setup, which) -> None:
    """A facet selector keys a ``hooks=`` mapping by its spelling; the door refuses typed."""

    model, x = attn if which == "head" else latent_setup
    selector = tl.head(0, "q") if which == "head" else tl.facet("res").latent(1)
    assert hash(selector) == hash(repr(selector))
    hooks = {selector: tl.zero_ablate()}
    with pytest.raises(SiteResolutionError, match="resolve through intervention mutators"):
        tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(intervention_ready=True, hooks=hooks),
        )


def test_the_list_save_spelling_still_refuses(latent_setup) -> None:
    model, x = latent_setup
    with pytest.raises(ArgumentTypeError) as caught:
        tl.trace(model, x, save=[tl.facet("res").latent(1)])
    assert caught.value.fields["code"] == "save_predicate_type_invalid"
    assert "save=tl.func('relu')" in caught.value.fields["remedy"]
