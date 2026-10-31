"""Facet selectors at the finished-trace read doors: ``trace[...]`` and ``resolve_sites``.

``tl.head(i, "q")`` and ``tl.facet(name).latent(k)`` already address a facet at
the write doors (``Trace.set``, ``attach_hooks``, ``do``) and at the facet read
door (``record.facets[...]``). On a finished trace, ``trace[selector]`` and
``trace.resolve_sites(selector)`` now return the same facets: each entry is the
``Facet`` the record's view serves, so its ``.value`` equals the read door's
value and the activation an edit at the write doors receives. A selector that
matches several facets keeps its broadcast and reports the count, as an
ordinary multi-site selector does.

Expected values are closed forms: the head slice is computed from the
attention block's own ``q_lin`` in plain torch, and the latent from SAE-4's
hand-set SAE in plain Python. Tolerances, fixed before the first run:
``atol=1e-6, rtol=0``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_sae4_latent_facets import (
    TOL,
    X,
    _HandSAE,
    _latent_column,
    _latents,
    _Model,
    _recording_hook,
    _TwoSites,
)
from tests.test_sae4_multi_call_modules import MultiHeadSelfAttention
from torchlens.intervention.errors import (
    AmbiguousSiteError,
    MultiMatchWarning,
    SelectorCapabilityError,
    SiteResolutionError,
)
from torchlens.semantic.facets import Facet

_READY = tl.options.CaptureOptions(replayable=True)


class _AttnModel(nn.Module):
    """One DistilBERT-shaped attention block (2 heads of width 4) and a readout."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention()
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(x))


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


def _q_head(model: _AttnModel, x: torch.Tensor, head: int) -> torch.Tensor:
    """Query head ``head`` in plain torch: ``q_lin(x)`` split into 2 heads of width 4."""

    with torch.no_grad():
        q = x @ model.attn.q_lin.weight.T + model.attn.q_lin.bias
    return q.view(2, 3, 2, 4)[:, :, head]


def _q_label(trace: tl.Trace) -> str:
    """Layer label of ``q_lin``'s output: the first linear in execution order."""

    return next(op.layer_label for op in trace._layer_list if op.func_name == "linear")


def _set_door_value(trace: tl.Trace, selector) -> torch.Tensor:
    """The activation the ``attach_hooks`` write door hands an edit for ``selector``."""

    seen: list[torch.Tensor] = []
    fork = trace.fork("door")
    fork.attach_hooks(selector, _recording_hook(seen))
    fork.push()
    assert seen, "the write door never handed the edit a value"
    return seen[0]


# --- trace[selector] ----------------------------------------------------------------


def test_trace_getitem_reads_a_head_slice(attn) -> None:
    model, x = attn
    trace = tl.trace(model, x, capture=_READY)
    facet = trace[tl.head(0, "q")]
    assert isinstance(facet, Facet)
    expected = _q_head(model, x, 0)
    torch.testing.assert_close(facet.value, expected, **TOL)
    read_door = trace.modules["attn"].facets.head(0)["q"].value
    torch.testing.assert_close(facet.value, read_door, **TOL)
    torch.testing.assert_close(facet.value, _set_door_value(trace, tl.head(0, "q")), **TOL)


def test_trace_getitem_reads_a_latent(latent_setup) -> None:
    model, x = latent_setup
    trace = tl.trace(model, x, capture=_READY)
    facet = trace[tl.facet("res").latent(1)]
    torch.testing.assert_close(facet.value, _latent_column(1), **TOL)
    torch.testing.assert_close(facet.value, trace["fc1"].facets["res[1]"].value, **TOL)
    torch.testing.assert_close(
        facet.value, _set_door_value(trace, tl.facet("res").latent(1)), **TOL
    )


def test_trace_getitem_reads_a_latent_set_as_one_facet(latent_setup) -> None:
    model, x = latent_setup
    trace = tl.trace(model, x, capture=_READY)
    expected = torch.tensor([[_latents(r)[0], _latents(r)[2]] for r in X])
    torch.testing.assert_close(trace[tl.facet("res").latent([0, 2])].value, expected, **TOL)


def test_trace_getitem_refuses_a_broadcast_head_selector(attn) -> None:
    """``tl.head(0)`` reads several facets; ``trace[...]`` needs one, as for any selector."""

    model, x = attn
    trace = tl.trace(model, x, capture=_READY)
    with pytest.raises(AmbiguousSiteError), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace[tl.head(0)]


# --- resolve_sites ------------------------------------------------------------------


def test_resolve_sites_names_the_head_facet_home(attn) -> None:
    model, x = attn
    trace = tl.trace(model, x, capture=_READY)
    table = trace.resolve_sites(tl.head(0, "q"))
    assert len(table) == 1
    assert table.labels() == (_q_label(trace),)
    torch.testing.assert_close(table.first().value, _q_head(model, x, 0), **TOL)


def test_resolve_sites_names_the_latent_home(latent_setup) -> None:
    model, x = latent_setup
    trace = tl.trace(model, x, capture=_READY)
    table = trace.resolve_sites(tl.facet("res").latent(2))
    assert table.labels() == (trace["fc1"].layer_label,)
    torch.testing.assert_close(table.first().value, _latent_column(2), **TOL)


def test_a_broadcast_head_read_reports_the_count_the_write_door_reports(attn) -> None:
    """forks.9: the broadcast stays, and the read count equals the write count."""

    model, x = attn
    trace = tl.trace(model, x, capture=_READY)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = trace.resolve_sites(tl.head(1))
        trace.fork("w").do(tl.head(1), tl.zero_ablate())
    read_counts = [w for w in caught if issubclass(w.category, MultiMatchWarning)]
    write_counts = [
        getattr(w.message, "fields", {}).get("site_count")
        for w in caught
        if getattr(w.message, "fields", {}).get("code") == "facet_selector_fanout"
    ]
    assert len(table) > 1
    assert read_counts and f"matched {len(table)} sites" in str(read_counts[0].message)
    assert write_counts == [len(table)]
    at_q = [
        facet
        for label, facet in zip(table.labels(), table, strict=True)
        if label == _q_label(trace)
    ]
    assert at_q
    expected = _q_head(model, x, 1)
    assert any(torch.allclose(f.value, expected, atol=1e-6, rtol=0.0) for f in at_q)


def test_a_latent_registered_at_two_layers_reads_both_and_counts_them() -> None:
    torch.manual_seed(0)
    model, x, sae = _TwoSites().eval(), torch.randn(2, 4), _HandSAE()
    tl.bridge.sae.register("both", sae, site=tl.module("fc_a") | tl.module("fc_b"))
    try:
        trace = tl.trace(model, x, capture=_READY)
        with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
            table = trace.resolve_sites(tl.facet("both").latent(0))
        assert table.labels() == (trace["fc_a"].layer_label, trace["fc_b"].layer_label)
        for site, facet in zip(("fc_a", "fc_b"), table, strict=True):
            expected = sae.encode(trace[site].out)[..., 0]
            torch.testing.assert_close(facet.value, expected.detach(), **TOL)
    finally:
        tl.bridge.sae.unregister("both")


def test_a_facet_no_record_serves_refuses_with_a_running_remedy() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    trace = tl.trace(model, torch.randn(2, 4), capture=_READY)
    with pytest.raises(SiteResolutionError) as caught:
        trace.resolve_sites(tl.head(0, "q"))
    assert caught.value.fields["code"] == "facet_selector_no_match"
    remedy = caught.value.fields["remedy"]
    first = trace.layer_labels[0]
    assert f"trace[{first!r}].facets.keys()" in remedy
    assert isinstance(trace[first].facets.keys(), list)


def test_find_sites_and_composites_still_refuse_naming_the_read_doors(attn) -> None:
    model, x = attn
    trace = tl.trace(model, x, capture=_READY)
    with pytest.raises(SelectorCapabilityError, match=r"trace\.resolve_sites\(selector\)"):
        trace.find_sites(tl.head(0, "q"))
    with pytest.raises(SelectorCapabilityError, match="composite selectors"):
        trace.resolve_sites(tl.head(0, "q") | tl.label("output_1"))


def test_ordinary_selectors_resolve_as_before(attn) -> None:
    """A non-facet selector still resolves to op records, unchanged."""

    model, x = attn
    trace = tl.trace(model, x, capture=_READY)
    with pytest.warns(MultiMatchWarning, match="matched 5 sites"):
        table = trace.resolve_sites(tl.func("linear"), max_fanout=10)
    assert all(not isinstance(site, Facet) for site in table)
    assert len(table) == 5
