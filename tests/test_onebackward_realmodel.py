"""One-backward reads on real checkpoints (F04 gate rows RG06/RG11-flavored).

Real ``openai-community/gpt2`` (the literature's model, RG06's artifact) and
torchvision PRETRAINED ResNet-18 (RG11's artifact): the substrate census,
target-kind-keyed reachability, split-slot discipline on the fused QKV, the
mid-stack linearization floor, zero-retention serving, and the VISCNN
abs-of-sum oracle. Everything runs OFFLINE from local caches; out of the
offline venue the gpt2 rows skip with the standard venue gate (in venue they
can never skip -- a missing artifact is a loud load failure).
"""

from __future__ import annotations

import os

import pytest
import torch
from support.r1_venue import IN_OFFLINE_VENUE

import torchlens as tl
from torchlens.attribution import onebackward as ob

pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

transformers = pytest.importorskip("transformers")
torchvision = pytest.importorskip("torchvision")


requires_offline_venue = pytest.mark.skipif(
    not IN_OFFLINE_VENUE,
    reason=(
        "GATE R1_OFFLINE_VENUE: not in the offline preflighted venue; export "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 with a warmed cache. In "
        "venue this suite can NEVER skip."
    ),
)


@pytest.fixture(scope="module")
def gpt2_trace():
    """Real gpt2, one capture shared by the module (offline venue only)."""

    model = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.eval()
    tokens = torch.tensor([[464, 3290, 318, 257, 922, 3290]])
    trace = tl.trace(model, tokens)
    try:
        yield trace
    finally:
        trace.cleanup()


@requires_offline_venue
class TestRealGPT2:
    """RG06-flavored rows on the literature's model."""

    def test_substrate_census(self, gpt2_trace: tl.Trace) -> None:
        """Memo test 1: the registry covers the walkable sites, no holes.

        The panel measured 538/567 op sites with live handles on this exact
        checkpoint; the census here asserts the shape of that fact (every
        site is either addressable or refused with a closed reason, and the
        addressable set dominates) rather than the raw constant, which moves
        with transformers versions.
        """

        index = ob.read_edge_index(gpt2_trace)
        total = len(index.edges) + len(index.unaddressable)
        assert len(index.edges) >= 500, (len(index.edges), total)
        assert len(index.edges) / total > 0.9
        assert set(index.unaddressable.values()) <= {"no_grad_fn", "handle_dead"}

    def test_split_slots_and_aliases(self, gpt2_trace: tl.Trace) -> None:
        """Memo test 8: fused-QKV split ops carry distinct nonzero slots."""

        index = ob.read_edge_index(gpt2_trace)
        nonzero_slots = [edge for edge in index.edges.values() if edge.slot != 0]
        assert len(nonzero_slots) >= 24, len(nonzero_slots)
        by_node: dict[int, set[int]] = {}
        for edge in index.edges.values():
            by_node.setdefault(id(edge.node), set()).add(edge.slot)
        split_nodes = {node for node, slots in by_node.items() if len(slots) > 1}
        assert len(split_nodes) >= 12, "the q/k/v split substrate disappeared"
        assert index.alias_groups, "gpt2 lost its (node, slot) alias groups"

    def test_target_kind_keyed_reachability(self, gpt2_trace: tl.Trace) -> None:
        """Memo test 5: output target reaches everything; intermediate ~81% not."""

        index = ob.read_edge_index(gpt2_trace)
        output_label = next(label for label in index.edges if label.startswith("output"))
        output_table = ob.read(
            gpt2_trace,
            target=ob.seed(output_label, index=(0, -1, 464)),
            method="grad",
            reduce="sum",
            frozen=None,
        )
        assert output_table.provenance.excluded_counts.get("not_upstream_of_target", 0) == 0
        mlp_sites = [
            edge.label
            for edge in index.edges.values()
            if edge.label.startswith("dropout") and edge.pass_index == 1
        ]
        mid_label = mlp_sites[len(mlp_sites) // 2]
        mid_edge = index.edges[mid_label]
        intermediate_table = ob.read(
            gpt2_trace,
            target=ob.seed(mid_label, index=tuple(0 for _ in mid_edge.shape)),
            method="grad",
            reduce="sum",
            frozen=None,
        )
        excluded = intermediate_table.provenance.excluded_counts["not_upstream_of_target"]
        assert excluded > len(intermediate_table), (
            "an intermediate target should exclude most of the population "
            f"(excluded={excluded}, served={len(intermediate_table)})"
        )
        assert all(row.score is not None for row in intermediate_table.rows()), (
            "0 fabricated zeros: every served row carries a real gradient"
        )

    def test_frozen_resolver_finds_all_twelve_mlp_homes(self, gpt2_trace: tl.Trace) -> None:
        """Memo test 12: 12/12 GPT2MLP output homes, semantic evidence only."""

        from torchlens.attribution.onebackward._facet_bridge import resolve_frozen_policy

        index = ob.read_edge_index(gpt2_trace)
        resolution = resolve_frozen_policy(gpt2_trace, index, "mlp_out")
        assert len(resolution.sites) == 12, sorted(resolution.sites)

    def test_midstack_linearization_floor(self, gpt2_trace: tl.Trace) -> None:
        """Memo test 11: default vs frozen=None at cos <= 0.5 mid-stack."""

        index = ob.read_edge_index(gpt2_trace)
        output_label = next(label for label in index.edges if label.startswith("output"))
        target = ob.seed(output_label, index=(0, -1, 464))
        candidates = [
            edge.label
            for edge in index.edges.values()
            if edge.layer_label.startswith("add_") and edge.shape == (1, 6, 768)
        ]
        mid_site = candidates[len(candidates) // 2]
        from torchlens.errors import TorchLensWarning

        with pytest.warns(TorchLensWarning, match="frozen= was omitted"):
            frozen_table = ob.read(
                gpt2_trace, target=target, method="grad", reduce=None, within=mid_site
            )
        none_table = ob.read(
            gpt2_trace,
            target=target,
            method="grad",
            reduce=None,
            frozen=None,
            within=mid_site,
        )
        frozen_vector = frozen_table[mid_site].value.reshape(-1)
        none_vector = none_table[mid_site].value.reshape(-1)
        cosine = float(torch.nn.functional.cosine_similarity(frozen_vector, none_vector, dim=0))
        assert abs(cosine) <= 0.5, (
            f"the published linearization stopped mattering (cos={cosine} at {mid_site})"
        )

    def test_zero_retention_serves_gradients(self) -> None:
        """Memo test 10 at real scale: layers_to_save=[] serves method='grad'."""

        model = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.eval()
        tokens = torch.tensor([[464, 3290, 318, 257]])
        trace = tl.trace(model, tokens, capture=tl.options.CaptureOptions(layers_to_save=[]))
        index = ob.read_edge_index(trace)
        assert len(index.edges) >= 500
        output_label = next(label for label in index.edges if label.startswith("output"))
        table = ob.read(
            trace,
            target=ob.seed(output_label, index=(0, -1, 464)),
            method="grad",
            reduce="sum",
            frozen=None,
        )
        assert table.status_counts()["ok"] == len(table)
        assert all(row.retention == "unsaved" for row in table.rows())


@pytest.fixture(scope="module")
def resnet_trace():
    """Pretrained ResNet-18 on a fixed synthetic image batch."""

    weights_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    model = torchvision.models.resnet18()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    torch.manual_seed(0)
    trace = tl.trace(model, torch.randn(1, 3, 64, 64))
    try:
        yield trace
    finally:
        trace.cleanup()


class TestRealResNet18:
    """RG11-flavored rows: pretrained torchvision weights from the local cache."""

    def test_empty_default_frozen_set_disclosed(self, resnet_trace: tl.Trace) -> None:
        """Memo test 16: no transformer MLPs -> empty set, disclosed, no warning."""

        table = ob.read(
            resnet_trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="sum",
        )
        assert table.provenance.frozen_policy == "mlp_out"
        assert table.provenance.frozen_resolved_count == 0
        assert any("empty set" in note for note in table.provenance.warnings)

    def test_viscnn_abs_of_sum_oracle(self, resnet_trace: tl.Trace) -> None:
        """Memo test 16: abs-of-sum equals a hand-rolled Taylor importance.

        The VISCNN quantity is ``|sum(act * grad)|`` per channel after
        spatial reduction; the read's site-grain ``abs_of_sum`` over the
        whole site must equal the hand-rolled value from the same payload
        and a plain-hooks native-autograd gradient.
        """

        site = "relu_1_3:1"
        table = ob.read(
            resnet_trace,
            target=ob.seed("output_1", index=(0, 7)),
            method="activation_x_grad",
            reduce="abs_of_sum",
            within="relu_1_3",
        )
        grad_table = ob.read(
            resnet_trace,
            target=ob.seed("output_1", index=(0, 7)),
            method="grad",
            reduce=None,
            within="relu_1_3",
        )
        payload = resnet_trace["relu_1_3"].out
        manual = float((payload * grad_table[site].value).sum().abs())
        assert table[site].score == pytest.approx(manual, rel=1e-6)

    def test_grad_only_read_retains_nothing(self, resnet_trace: tl.Trace) -> None:
        """D14 headline at real scale: gradient reads carry no payload bytes."""

        table = ob.read(
            resnet_trace,
            target=ob.seed("output_1", index=(0, 0)),
            method="grad",
            reduce="l2",
        )
        assert table.provenance.result_bytes == 0
        assert table.status_counts()["ok"] == len(table)
