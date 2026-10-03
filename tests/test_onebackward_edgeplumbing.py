"""One-backward reads: deferred-EAP plumbing (F04 items 7+9, D16).

The GradInputUseMap is fail-closed session-time plumbing: ``exact`` only for
provably-unambiguous correspondences, candidates listed but never picked
from, and the no-positional-zipping census pinned. EAP remains honestly
blocked after this build. The origin-namespaced pass stamps and the
deferred-sibling seams (``sample_id``, the method vocabulary tuple) ride
here too.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.attribution import onebackward as ob
from torchlens.utils._torch_compat import get_gradient_edge_support

_requires_gradient_edge = pytest.mark.skipif(
    not get_gradient_edge_support(),
    reason="one-backward reads require torch.autograd.graph.GradientEdge (2.4+)",
)


def _trace(intervention_ready: bool = True) -> tl.Trace:
    """Trace the toy MLP with edge provenance on by default."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    return tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=intervention_ready),
    )


class TestGradInputUseMap:
    """The fail-closed correspondence contract."""

    @_requires_gradient_edge
    def test_statuses_are_closed_and_fail_closed(self) -> None:
        trace = _trace()
        use_map = ob.mint_grad_input_use_map(trace)
        assert use_map.entries, "the map minted no entries"
        for use in use_map.entries.values():
            assert use.status in ("exact", "ambiguous", "unsupported")
            if use.status == "exact":
                assert len(use.addresses) == 1
            elif use.status == "unsupported":
                assert use.addresses == ()

    @pytest.mark.smoke
    @_requires_gradient_edge
    def test_no_positional_zipping_census(self) -> None:
        """Multi-input nodes stay unmated -- the census that kills zipping.

        On real gpt2 the number is 155/594 backward nodes without a direct
        forward mate; at toy scale the linear ops (weight/bias/input slots)
        are the unmated population, and the pin is that it is NON-EMPTY: a
        future 'simplification' that force-mates every node goes red here.
        """

        trace = _trace()
        use_map = ob.mint_grad_input_use_map(trace)
        assert use_map.unmated_sites, "every node force-mated: positional zipping risk"
        assert use_map.census["unsupported"] > 0

    @_requires_gradient_edge
    def test_provenance_gate_refuses_typed(self) -> None:
        trace = _trace(intervention_ready=False)
        with pytest.raises(ob.ReadError) as excinfo:
            ob.mint_grad_input_use_map(trace)
        assert excinfo.value.fields["code"] == "read_edge_provenance_unavailable"
        assert "intervention_ready" in excinfo.value.fields["remedy"]


class TestPassOriginStamps:
    """Origin-namespaced backward-pass stamping (write-once, closed vocab)."""

    def test_stamp_and_default_user_origin(self) -> None:
        trace = _trace()
        stamp = ob.stamp_backward_pass_origin(trace, 1, "read", target_ids=("t0",), chunk_id=0)
        assert stamp.origin == "read"
        origins = ob.backward_pass_origins(trace)
        assert origins[1].target_ids == ("t0",)

    @pytest.mark.smoke
    def test_unstamped_recorded_passes_default_to_user(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
        trace = tl.trace(
            model,
            torch.randn(2, 4, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        trace.log_backward(trace["output_1"].out.sum(), retain_graph=True)
        origins = ob.backward_pass_origins(trace)
        assert origins[1].origin == "user"

    def test_closed_vocabulary_and_write_once(self) -> None:
        trace = _trace()
        with pytest.raises(ob.ReadError) as excinfo:
            ob.stamp_backward_pass_origin(trace, 1, "mystery")
        assert excinfo.value.fields["code"] == "read_pass_origin_invalid"
        ob.stamp_backward_pass_origin(trace, 2, "internal")
        with pytest.raises(ob.ReadError) as excinfo:
            ob.stamp_backward_pass_origin(trace, 2, "read")
        assert excinfo.value.fields["code"] == "read_pass_origin_invalid"


class TestDeferredSiblingSeams:
    """Item 9: the seams later waves land on without redesign."""

    @_requires_gradient_edge
    def test_sample_id_is_nullable_from_day_one(self) -> None:
        trace = _trace()
        table = ob.read(
            trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum"
        )
        assert all(row.sample_id is None for row in table.rows())
        assert table.provenance.sample_count == 1

    def test_method_vocabulary_is_the_extension_seam(self) -> None:
        """The closed tuple is importable and the R4 spelling stays absent."""

        assert ob.METHODS == ("activation_x_grad", "grad", "activation")
        assert "position" not in ob.METHODS, "R4 semantics must not ship in v1"

    def test_edge_kind_admitted_on_the_carrier(self) -> None:
        """ReadRow accepts EDGE with kind-specific addresses (no ACT stuffing)."""

        row = ob.ReadRow(
            target_id="t0",
            kind="EDGE",
            address=(7, "positional", (0,)),
            site_key=None,
            alias_group=None,
            method="grad",
            reduction="sum",
            grain="site",
            score=1.0,
            value=None,
            shape=None,
            dtype=None,
            device="cpu",
            status="ok",
            status_reason=None,
            differentiable=True,
            retention=None,
            capture_status=None,
            resolution="exact",
            frozen_requested=False,
            frozen_resolved=False,
            frozen_reached=None,
            policy_digest=None,
            detached=True,
            sample_id=None,
            rescorable=True,
        )
        assert row.key == ("t0", "EDGE", (7, "positional", (0,)))

    def test_frozen_policy_registry_is_the_r10_seam(self) -> None:
        """New policies register once; duplicates refuse typed."""

        from torchlens.attribution.onebackward import _facet_bridge

        with pytest.raises(ob.ReadError) as excinfo:
            _facet_bridge.register_frozen_policy("mlp_out", lambda trace, index: None)
        assert excinfo.value.fields["code"] == "read_frozen_policy_invalid"
