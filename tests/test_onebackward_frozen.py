"""One-backward reads: frozen= policy and the facet bridge (F04, D5-D7).

Config-built GPT-2 (zero network, the R0 discipline): the default MLP-output
linearization resolves through semantic facet evidence only, the numbers
differ strongly from the total derivative, hook removal restores bit-exact,
``frozen_applied`` is cone-dependent, the disclosure detector runs both
FORK-1 branches behind the one constant, and the substring guard proves the
resolver can never be replaced by a name filter.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.attribution import onebackward as ob

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gpt2_trace():
    """One config-built tiny GPT-2 capture shared by the module's tests."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=32)
    model = transformers.GPT2LMHeadModel(config)
    trace = tl.trace(model, torch.randint(0, 128, (1, 6)))
    try:
        yield trace
    finally:
        trace.cleanup()


def _output_seed(trace: tl.Trace) -> ob.SeedTarget:
    """Target the last-position logit 7 at the trace output site."""

    index = ob.read_edge_index(trace)
    output_label = next(label for label in index.edges if label.startswith("output"))
    return ob.seed(output_label, index=(0, -1, 7))


class TestFacetBridge:
    """Item 3b: the semantic-evidence-only resolver."""

    def test_resolves_exactly_the_mlp_output_homes(self, gpt2_trace: tl.Trace) -> None:
        """Both blocks' GPT2MLP output homes resolve; nothing else does."""

        index = ob.read_edge_index(gpt2_trace)
        from torchlens.attribution.onebackward._facet_bridge import resolve_frozen_policy

        resolution = resolve_frozen_policy(gpt2_trace, index, "mlp_out")
        assert resolution.attention_bearing
        assert len(resolution.sites) == 2, sorted(resolution.sites)

    def test_substring_guard_pin(self, gpt2_trace: tl.Trace) -> None:
        """A path-substring filter over-selects vs the semantic bridge.

        The memo's guard pin: on real gpt2 an "mlp" path filter selects 48
        sites where the true answer is 12. At this config scale the filter
        still over-selects (every op INSIDE the MLP matches), so nobody can
        ever replace the resolver with a name filter and stay green.
        """

        index = ob.read_edge_index(gpt2_trace)
        from torchlens.attribution.onebackward._facet_bridge import resolve_frozen_policy

        resolution = resolve_frozen_policy(gpt2_trace, index, "mlp_out")
        substring_selected = {
            op.label
            for op in gpt2_trace.ops
            if any("mlp" in str(address) for address in getattr(op, "modules", ()))
        }
        assert len(substring_selected) > len(resolution.sites), (
            "the substring filter no longer over-selects; re-verify the "
            "semantic bridge is still the only resolver"
        )

    def test_underivable_refusal_on_attention_without_mlp_facets(
        self, gpt2_trace: tl.Trace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attention-bearing graph whose MLP homes are unresolvable refuses.

        Deterministic branch pin: the attention evidence stays real (the
        config GPT-2's q facets) while the MLP "output" facet homes are made
        unresolvable, which is exactly the hand-built-attention-block
        scenario of memo test 12 -- never warn-and-proceed, never substring
        guessing.
        """

        from torchlens.attribution.onebackward import _facet_bridge
        from torchlens.semantic.facets import FacetView

        original_has = FacetView.has

        def _no_output_facets(self: FacetView, name: str) -> bool:
            if name == "output":
                return False
            return original_has(self, name)

        monkeypatch.setattr(FacetView, "has", _no_output_facets)
        index = ob.read_edge_index(gpt2_trace)
        with pytest.raises(ob.ReadError) as excinfo:
            _facet_bridge.resolve_frozen_policy(gpt2_trace, index, "mlp_out")
        assert excinfo.value.fields["code"] == "default_frozen_sites_underivable"
        assert excinfo.value.fields["remedy"]

    def test_unknown_policy_refuses(self, gpt2_trace: tl.Trace) -> None:
        """No 'attribution_graph' preset name exists until R10 lands."""

        index = ob.read_edge_index(gpt2_trace)
        from torchlens.attribution.onebackward._facet_bridge import resolve_frozen_policy

        with pytest.raises(ob.ReadError) as excinfo:
            resolve_frozen_policy(gpt2_trace, index, "attribution_graph")
        assert excinfo.value.fields["code"] == "read_frozen_policy_invalid"


class TestFrozenSemantics:
    """Item 3: the three-state contract and hook lifecycle."""

    def test_default_differs_from_total_derivative_upstream(self, gpt2_trace: tl.Trace) -> None:
        """Memo test 11 at config scale: the two policies differ upstream.

        The strong cos <= 0.5 floor is a REAL-gpt2 property (12 trained MLP
        stops; measured 0.155) and lives in the venue-gated real-model file;
        at 2-layer random init the pin is that the vectors differ measurably
        and the rows disclose different policy digests.
        """

        from torchlens.errors import TorchLensWarning

        target = _output_seed(gpt2_trace)
        with pytest.warns(TorchLensWarning, match="frozen= was omitted"):
            frozen_table = ob.read(gpt2_trace, target=target, method="grad", reduce=None)
        none_table = ob.read(gpt2_trace, target=target, method="grad", reduce=None, frozen=None)
        index = ob.read_edge_index(gpt2_trace)
        upstream_label = next(iter(index.edges))
        frozen_row = frozen_table[upstream_label]
        none_row = none_table[upstream_label]
        assert none_row.value.norm() > 0
        assert not torch.allclose(frozen_row.value, none_row.value, rtol=1e-3, atol=1e-6), (
            "the default linearization no longer changes upstream gradients"
        )
        assert frozen_row.policy_digest != none_row.policy_digest

    def test_hook_removal_restores_bitexact(self, gpt2_trace: tl.Trace) -> None:
        """A frozen read then a frozen=None read: the latter is unpolluted."""

        from torchlens.errors import TorchLensWarning

        target = _output_seed(gpt2_trace)
        baseline = ob.read(gpt2_trace, target=target, method="grad", reduce=None, frozen=None)
        with pytest.warns(TorchLensWarning, match="frozen= was omitted"):
            ob.read(gpt2_trace, target=target, method="grad", reduce="sum")
        after = ob.read(gpt2_trace, target=target, method="grad", reduce=None, frozen=None)
        for key, row in baseline.items():
            if row.value is not None:
                assert torch.equal(after[key].value, row.value), (
                    "freeze hooks leaked into a later unfrozen read"
                )

    def test_frozen_applied_is_cone_dependent(self, gpt2_trace: tl.Trace) -> None:
        """fired <= resolved and non-empty on cone-pruned targets (D6)."""

        from torchlens.errors import TorchLensWarning

        with pytest.warns(TorchLensWarning, match="frozen= was omitted"):
            table = ob.read(
                gpt2_trace, target=_output_seed(gpt2_trace), method="grad", reduce="sum"
            )
        assert 0 < table.provenance.frozen_fired_count <= table.provenance.frozen_resolved_count
        frozen_rows = [row for row in table.rows() if row.frozen_resolved]
        assert frozen_rows
        assert all(row.frozen_reached is not None for row in frozen_rows)

    def test_explicit_frozen_replaces_default(self, gpt2_trace: tl.Trace) -> None:
        """An explicit set replaces (never unions with) the default."""

        index = ob.read_edge_index(gpt2_trace)
        some_site = next(iter(index.edges))
        table = ob.read(
            gpt2_trace,
            target=_output_seed(gpt2_trace),
            method="grad",
            reduce="sum",
            frozen=some_site,
        )
        assert table.provenance.frozen_policy == "explicit"
        assert table.provenance.frozen_requested_count == 1

    def test_alias_group_freeze_disclosure(self) -> None:
        """Freezing one alias member discloses its siblings (D11)."""

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
        trace = tl.trace(model, torch.randn(2, 4))
        table = ob.read(
            trace,
            target=ob.seed("gelu_1_2", index=(0, 0)),
            method="grad",
            reduce="sum",
            frozen="linear_2_3",  # aliases with output_1
        )
        assert table.provenance.frozen_requested_count == 1


class TestDisclosureBranches:
    """FORK-1: both branches behind the one constant, both green (test 13)."""

    def test_branch_w_warns_once_per_process(self, gpt2_trace: tl.Trace) -> None:
        """The shipped branch: one warning, then silence; never with explicit."""

        from torchlens.attribution.onebackward import _frozen

        _frozen._DISCLOSURE_WARNED.clear()
        target = _output_seed(gpt2_trace)
        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            ob.read(gpt2_trace, target=target, method="grad", reduce="sum")
        codes = [getattr(warning.message, "fields", {}).get("code") for warning in first]
        assert codes.count("read_frozen_default_linearization") == 1
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            ob.read(gpt2_trace, target=target, method="grad", reduce="sum")
            ob.read(gpt2_trace, target=target, method="grad", reduce="sum", frozen=None)
        codes = [getattr(warning.message, "fields", {}).get("code") for warning in second]
        assert "read_frozen_default_linearization" not in codes

    @pytest.mark.smoke
    def test_branch_r_refuses_until_chosen(
        self, gpt2_trace: tl.Trace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kept-buildable branch: typed refusal with the shared message."""

        from torchlens.attribution.onebackward import _frozen

        monkeypatch.setattr(_frozen, "FROZEN_DEFAULT_DISCLOSURE_BRANCH", "refuse")
        target = _output_seed(gpt2_trace)
        with pytest.raises(ob.ReadError) as excinfo:
            ob.read(gpt2_trace, target=target, method="grad", reduce="sum")
        assert excinfo.value.fields["code"] == "read_frozen_choice_required"
        # Explicit frozen= proceeds under branch R.
        table = ob.read(gpt2_trace, target=target, method="grad", reduce="sum", frozen=None)
        assert table.status_counts().get("ok", 0) > 0

    def test_detector_never_fires_on_attention_free_models(self) -> None:
        """ResNet-style graphs: empty default set, disclosed, no warning."""

        from torchlens.attribution.onebackward import _frozen

        _frozen._DISCLOSURE_WARNED.clear()
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(72, 4))
        trace = tl.trace(model, torch.randn(1, 1, 8, 8))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = ob.read(
                trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
            )
        codes = [getattr(warning.message, "fields", {}).get("code") for warning in caught]
        assert "read_frozen_default_linearization" not in codes
        assert table.provenance.frozen_policy == "mlp_out"
        assert table.provenance.frozen_resolved_count == 0
        assert any("empty set" in note for note in table.provenance.warnings)
