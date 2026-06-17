"""Tests for the semantic facets registry and view surface."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import FacetRecipe, FacetSpec, FacetView, MissingFacetError, MissingGradient
from torchlens.semantic import facets as facets_mod
from torchlens.semantic.recipes import BUILTIN_FACET_CAPABILITY_INVENTORY
from torchlens.intervention.errors import SiteResolutionError


class _Record:
    """Simple record-like object for facet tests."""

    class_name = "UnitFacetRecord"
    class_qualname = "tests.UnitFacetRecord"


def _trace_output(log: Any) -> torch.Tensor:
    """Return the first saved model output tensor for a trace.

    Parameters
    ----------
    log:
        TorchLens trace.

    Returns
    -------
    torch.Tensor
        First output tensor.
    """

    return log[log.output_layers[0]].out


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Restore the global facet registry after each test."""

    original = list(facets_mod._REGISTRY)
    yield
    facets_mod._REGISTRY[:] = original


def test_facet_view_dict_and_attribute_access_uniformity() -> None:
    """Attribute and item access return the same cached value."""

    @tl.facets.register(class_name="UnitFacetRecord")
    def unit_recipe(record: Any) -> dict[str, Any]:
        """Return a test facet."""

        return {"x": 10}

    view = FacetView(_Record())
    assert view.x == 10
    assert view["x"] == 10


def test_lazy_compute_keys_has_and_cache() -> None:
    """Available-now dictionary methods compute once and reuse the cache."""

    calls = 0

    @tl.facets.register(class_name="UnitFacetRecord")
    def lazy_recipe(record: Any) -> dict[str, Any]:
        """Return a lazily computed facet."""

        nonlocal calls
        calls += 1
        return {"x": 10}

    view = FacetView(_Record())
    assert view.keys() == ["x"]
    assert view.has("x")
    assert list(view) == ["x"]
    assert len(view) == 1
    assert calls == 1
    assert view.x == 10
    assert view.x == 10
    assert calls == 1


def test_invalidate_clears_cached_values() -> None:
    """Invalidation causes the next access to re-run the recipe."""

    calls = 0

    @tl.facets.register(class_name="UnitFacetRecord")
    def invalidated_recipe(record: Any) -> dict[str, Any]:
        """Return a monotonically increasing test facet."""

        nonlocal calls
        calls += 1
        return {"x": calls}

    view = FacetView(_Record())
    assert view.x == 1
    view.invalidate()
    assert view.x == 2


def test_empty_facet_view() -> None:
    """Unmatched records have an empty but usable view."""

    view = FacetView(object())
    assert view.keys() == []
    assert not view.has("x")
    assert list(view) == []
    assert len(view) == 0
    assert view.recipe_source is None


def test_multi_recipe_merge_warns_and_last_wins() -> None:
    """Same-tier facet collisions warn and choose a deterministic winner."""

    @tl.facets.register(class_name="UnitFacetRecord")
    def first_recipe(record: Any) -> dict[str, Any]:
        """Return the first test facets."""

        return {"x": 1, "y": 2}

    @tl.facets.register(class_name="UnitFacetRecord")
    def second_recipe(record: Any) -> dict[str, Any]:
        """Return overriding test facets."""

        return {"x": 3, "z": 4}

    view = FacetView(_Record())
    assert view.recipe_source == ("first_recipe", "second_recipe")
    with pytest.warns(UserWarning, match="ambiguous same-tier recipes"):
        assert view.x == 3
    assert view.y == 2
    assert view.z == 4


def test_trace_snapshot_is_immune_to_later_registry_mutation() -> None:
    """A trace uses the recipe snapshot captured during its forward pass."""

    class Tiny(nn.Module):
        """Tiny module with one named child."""

        def __init__(self) -> None:
            """Initialize the child linear layer."""

            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the child layer."""

            return self.linear(x)

    @tl.facets.register(class_name="Linear")
    def first_linear(record: Any) -> dict[str, Any]:
        """Return the first snapshot value."""

        return {"snapshot_value": 1}

    log = tl.trace(Tiny(), torch.randn(1, 2), layers_to_save="all")

    @tl.facets.register(class_name="Linear")
    def second_linear(record: Any) -> dict[str, Any]:
        """Return a later value that must not affect old traces."""

        return {"snapshot_value": 2}

    assert log.modules["linear"].facets.snapshot_value == 1


def test_trace_recipes_and_using_are_capture_time_additive() -> None:
    """Per-trace and context recipes affect snapshots only at capture time."""

    class Tiny(nn.Module):
        """Tiny single-module model."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return an incremented tensor."""

            return x + 1

    def local_recipe(record: Any) -> dict[str, Any]:
        """Return a local facet for any record."""

        return {"local_value": "present"}

    model = Tiny()
    x = torch.randn(1, 2)
    per_trace = tl.trace(model, x, layers_to_save="all", recipes=[local_recipe])
    assert per_trace.modules["self"].facets.local_value == "present"

    with tl.facets.using(local_recipe):
        contextual = tl.trace(model, x, layers_to_save="all")
    assert contextual.modules["self"].facets.local_value == "present"

    outside = tl.trace(model, x, layers_to_save="all")
    assert not outside.modules["self"].facets.has("local_value")


def test_structural_output_facets_expose_names_and_method_collisions() -> None:
    """Existing container output names are available through item access."""

    class DictBlock(nn.Module):
        """Return dict keys that include a FacetView method name."""

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Return named tensors."""

            return {"keys": x + 1, "hidden": x + 2}

    class Model(nn.Module):
        """Wrapper with a named child module."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.block = DictBlock()

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Return the child output."""

            return self.block(x)

    x = torch.randn(2, 3)
    log = tl.trace(Model(), x, layers_to_save="all")
    facets = log.modules["block"].facets

    assert torch.equal(facets["keys"], x + 1)
    assert torch.equal(facets["hidden"], x + 2)
    assert callable(facets.keys)


def test_structural_output_facets_expose_namedtuple_and_dataclass_names() -> None:
    """NamedTuple and dataclass output names surface as structural facets."""

    Pair = namedtuple("Pair", ["values", "indices"])

    @dataclass
    class DataOut:
        """Dataclass output container."""

        first: torch.Tensor
        second: torch.Tensor

    class StructBlock(nn.Module):
        """Return nested named containers."""

        def forward(self, x: torch.Tensor) -> tuple[Pair, DataOut]:
            """Return tuple, namedtuple, and dataclass outputs."""

            return Pair(x + 1, x + 2), DataOut(x + 3, x + 4)

    class Model(nn.Module):
        """Wrapper exposing the structured output as a child module."""

        def __init__(self) -> None:
            """Initialize the structured child."""

            super().__init__()
            self.block = StructBlock()

        def forward(self, x: torch.Tensor) -> tuple[Pair, DataOut]:
            """Return the child output."""

            return self.block(x)

    log = tl.trace(Model(), torch.randn(2, 3), layers_to_save="all")
    facets = log.modules["block"].facets

    assert "out0.values" in facets.keys()
    assert "out0.indices" in facets.keys()
    assert "out1.first" in facets.keys()
    assert "out1.second" in facets.keys()


def test_structseq_output_facets_expose_torch_return_type_names() -> None:
    """Torch return_types names surface as structural facets when preserved."""

    class MaxBlock(nn.Module):
        """Return a torch structseq output."""

        def forward(self, x: torch.Tensor) -> Any:
            """Return max values and indices."""

            return torch.max(x, dim=1)

    class Model(nn.Module):
        """Wrapper exposing the structseq return as a child module."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.block = MaxBlock()

        def forward(self, x: torch.Tensor) -> Any:
            """Return the child output."""

            return self.block(x)

    x = torch.randn(2, 3)
    log = tl.trace(Model(), x, layers_to_save="all")
    facets = log.modules["block"].facets
    expected = torch.max(x, dim=1)

    assert torch.equal(facets["values"], expected.values)
    assert torch.equal(facets["indices"], expected.indices)


def test_keys_has_get_are_available_now_for_unsaved_op_backed_facets() -> None:
    """Op-backed recipe facets are absent from keys when their home payload is unsaved."""

    class Model(nn.Module):
        """Tiny model with an unsaved LayerNorm output under predicate capture."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.norm = nn.LayerNorm(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run normalization followed by a saved ReLU."""

            return torch.relu(self.norm(x))

    log = tl.trace(Model(), torch.randn(2, 4), save=tl.func("relu"))
    facets = log.modules["norm"].facets

    assert "normalized" not in facets.keys()
    assert "normalized" not in facets
    assert not facets.has("normalized")
    assert facets.get("normalized", "default") == "default"
    assert facets.menu()["normalized"].status == "needs_capture"
    with pytest.raises(MissingFacetError, match="save=.*including"):
        facets["normalized"]
    with pytest.raises(MissingFacetError, match="save=.*including"):
        facets.normalized


def test_structurally_absent_declared_facet_raises_plain_keyerror() -> None:
    """Structurally absent built-in facets are omitted and raise plain KeyError."""

    class RMSNorm(nn.Module):
        """Tiny RMSNorm-like module with no bias parameter."""

        def __init__(self) -> None:
            """Initialize the weight parameter."""

            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a minimal RMS normalization."""

            denom = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            return x / denom * self.weight

    class Model(nn.Module):
        """Wrapper exposing an RMSNorm child."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.norm = RMSNorm()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the child module."""

            return self.norm(x)

    log = tl.trace(Model(), torch.randn(2, 4), layers_to_save="all")
    facets = log.modules["norm"].facets

    assert "beta" not in facets.keys()
    assert "beta" not in facets
    assert facets.menu()["beta"].status == "structurally_absent"
    with pytest.raises(KeyError) as exc_info:
        facets["beta"]
    assert not isinstance(exc_info.value, MissingFacetError)
    with pytest.raises(AttributeError):
        facets.beta


def test_lstm_multi_output_facets_preserve_single_call_roles() -> None:
    """LSTM outputs expose role names without becoming recurrent passes."""

    class LSTMModel(nn.Module):
        """Wrapper exposing an LSTM child."""

        def __init__(self) -> None:
            """Initialize the LSTM child."""

            super().__init__()
            self.lstm = nn.LSTM(3, 5)

        def forward(self, x: torch.Tensor) -> Any:
            """Return the LSTM output tuple."""

            return self.lstm(x)

    x = torch.randn(4, 2, 3)
    log = tl.trace(LSTMModel(), x, layers_to_save="all")
    lstm = log.modules["lstm"]

    assert lstm.num_calls == 1
    assert [log.ops[label].multi_output_name for label in lstm.calls[0].output_ops] == [
        "output",
        "h_n",
        "c_n",
    ]
    assert torch.equal(lstm.facets["output"], lstm.outs[0])
    assert torch.equal(lstm.facets["h_n"], lstm.outs[1])
    assert torch.equal(lstm.facets["c_n"], lstm.outs[2])


def test_facetspec_read_and_default_missing_gradient() -> None:
    """FacetSpec read works and default traces return MissingGradient."""

    class Tiny(nn.Module):
        """Tiny model with a named linear home."""

        def __init__(self) -> None:
            """Initialize the linear layer."""

            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the linear output."""

            return self.linear(x)

    @tl.facets.register(class_name="Linear", target_scope="module", facets=("first_feature",))
    def first_feature(module: Any) -> dict[str, Any]:
        """Return the first feature as an op-anchored FacetSpec."""

        op = module.trace.ops[module.calls[0].output_ops[0]]
        return {"first_feature": FacetSpec.from_home(op, recipe_id="first_feature").select(-1, 0)}

    log = tl.trace(Tiny(), torch.randn(4, 3), layers_to_save="all")
    facet = log.modules["linear"].facets["first_feature"]

    assert torch.equal(facet, log.modules["linear"].out[..., 0])
    missing = facet.grad
    assert isinstance(missing, MissingGradient)
    assert "backward_ready=True" in missing.reason
    assert "save_grads" in missing.reason
    assert "home_label='linear_1_1:1'" in repr(facet)
    assert "save_grads" in repr(missing)
    with pytest.raises(RuntimeError, match="Facet gradient unavailable"):
        torch.add(missing, 1)


def test_facetspec_grad_matches_manual_slice_when_saved() -> None:
    """FacetSpec grad applies the same transform chain to the home op grad."""

    class Tiny(nn.Module):
        """Tiny model with a named linear home."""

        def __init__(self) -> None:
            """Initialize the linear layer."""

            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the linear output."""

            return self.linear(x)

    @tl.facets.register(class_name="Linear", target_scope="module", facets=("first_feature",))
    def first_feature(module: Any) -> dict[str, Any]:
        """Return the first feature as an op-anchored FacetSpec."""

        op = module.trace.ops[module.calls[0].output_ops[0]]
        return {"first_feature": FacetSpec.from_home(op, recipe_id="first_feature").select(-1, 0)}

    log = tl.trace(Tiny(), torch.randn(4, 3), layers_to_save="all", save_grads="all")
    log.log_backward(log[log.output_layers[0]].out.sum())
    facet = log.modules["linear"].facets["first_feature"]
    home = log.ops[log.modules["linear"].calls[0].output_ops[0]]

    assert torch.equal(facet.grad, home.grad.select(-1, 0))


def test_facetspec_grad_missing_for_unselected_home() -> None:
    """Selective gradients return MissingGradient for an unselected home op."""

    class Tiny(nn.Module):
        """Tiny model with linear and relu homes."""

        def __init__(self) -> None:
            """Initialize the layers."""

            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return relu(linear(x))."""

            return self.relu(self.linear(x))

    @tl.facets.register(class_name="Linear", target_scope="module", facets=("first_feature",))
    def first_feature(module: Any) -> dict[str, Any]:
        """Return the first feature as an op-anchored FacetSpec."""

        op = module.trace.ops[module.calls[0].output_ops[0]]
        return {"first_feature": FacetSpec.from_home(op, recipe_id="first_feature").select(-1, 0)}

    log = tl.trace(Tiny(), torch.randn(4, 3), layers_to_save="all", save_grads=["relu"])
    log.log_backward(log[log.output_layers[0]].out.sum())
    missing = log.modules["linear"].facets["first_feature"].grad

    assert isinstance(missing, MissingGradient)
    assert "home op has no saved gradient" in missing.reason


def test_parameter_builtin_facets_are_read_only_parameter_homes() -> None:
    """Built-in norm parameter facets use parameter homes, not op grads."""

    class Model(nn.Module):
        """Wrapper exposing a LayerNorm module."""

        def __init__(self) -> None:
            """Initialize LayerNorm."""

            super().__init__()
            self.norm = nn.LayerNorm(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run normalization."""

            return self.norm(x)

    model = Model()
    log = tl.trace(model, torch.randn(2, 3), layers_to_save="all")
    gamma = log.modules["norm"].facets["gamma"]
    beta = log.modules["norm"].facets["beta"]

    assert gamma.spec.home_kind == "parameter"
    assert beta.spec.home_kind == "parameter"
    assert gamma.spec.capability_flags.grad is False
    assert torch.equal(gamma, model.norm.weight)
    assert torch.equal(beta, model.norm.bias)
    assert isinstance(gamma.grad, MissingGradient)


def test_builtin_capability_inventory_is_recorded_as_data() -> None:
    """Every built-in inventory entry records one allowed capability class."""

    allowed = {
        "op_structural",
        "parameter",
        "module_input",
        "module_output",
        "computed_read_only",
        "missing",
    }
    assert BUILTIN_FACET_CAPABILITY_INVENTORY
    assert BUILTIN_FACET_CAPABILITY_INVENTORY["attention"]["q"] == "op_structural"
    assert BUILTIN_FACET_CAPABILITY_INVENTORY["mlp"]["intermediate"] == "computed_read_only"
    assert BUILTIN_FACET_CAPABILITY_INVENTORY["embedding"]["weight"] == "parameter"
    for facet_map in BUILTIN_FACET_CAPABILITY_INVENTORY.values():
        assert facet_map
        assert set(facet_map.values()) <= allowed


def test_list_info_glob_and_recipe_record() -> None:
    """Discoverability functions expose registered recipe metadata."""

    @tl.facets.register(class_name=("UnitFacetRecord", "UnitAttention"))
    def discovery_recipe(record: Any) -> dict[str, Any]:
        """Return discoverable facets."""

        return {"q": 1}

    recipes = tl.facets.list(class_name="Unit*")
    assert len(recipes) == 1
    assert isinstance(recipes[0], FacetRecipe)
    assert recipes[0].recipe_name == "discovery_recipe"
    assert tl.facets.info("UnitFacetRecord") == {
        "recipes": ["discovery_recipe"],
        "facets": ["q"],
    }


def test_register_preserves_function_object_and_docstring() -> None:
    """The register decorator returns the original function unchanged."""

    def original_recipe(record: Any) -> dict[str, Any]:
        """Original docstring."""

        return {"x": 1}

    registered = tl.facets.register(class_name="UnitFacetRecord")(original_recipe)
    assert registered is original_recipe
    assert registered.__doc__ == "Original docstring."


class _FacetP2Model(nn.Module):
    """Wrapper exposing one named block for facet intervention tests."""

    def __init__(self, block: nn.Module) -> None:
        """Initialize with a block under a stable module address."""

        super().__init__()
        self.attn = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped block."""

        return self.attn(x)


class MultiHeadSelfAttention(nn.Module):
    """Tiny attention block matching the DistilBERT recipe class name."""

    def __init__(self) -> None:
        """Initialize projection children used by the built-in recipe."""

        super().__init__()
        self.n_heads = 2
        self.dim = 8
        self.q_lin = nn.Linear(8, 8)
        self.k_lin = nn.Linear(8, 8)
        self.v_lin = nn.Linear(8, 8)
        self.out_lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run projection children so q/k/v facets are op-anchored."""

        return self.out_lin(self.q_lin(x) + self.k_lin(x) + self.v_lin(x))


class GPT2Attention(nn.Module):
    """Tiny GPT-2-like fused-QKV block for shared-home writes."""

    def __init__(self) -> None:
        """Initialize fused c_attn and output projection children."""

        super().__init__()
        self.num_heads = 2
        self.embed_dim = 8
        self.c_attn = nn.Linear(8, 24)
        self.c_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fused q/k/v projection and consume all three slices."""

        q, k, v = self.c_attn(x).split(8, dim=-1)
        return self.c_proj(q + k + v)


class LlamaAttention(nn.Module):
    """Tiny GQA-style attention block with fewer KV heads than Q heads."""

    def __init__(self) -> None:
        """Initialize projections matching the GQA built-in recipe."""

        super().__init__()
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 2
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 4)
        self.v_proj = nn.Linear(8, 4)
        self.o_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run GQA projections and repeat KV values for the output."""

        q = self.q_proj(x)
        k = self.k_proj(x).repeat_interleave(2, dim=-1)
        v = self.v_proj(x).repeat_interleave(2, dim=-1)
        return self.o_proj(q + k + v)


class GPT2MLP(nn.Module):
    """Tiny GPT-2-like MLP block for computed facet refusal."""

    def __init__(self) -> None:
        """Initialize child modules used by the built-in MLP recipe."""

        super().__init__()
        self.c_fc = nn.Linear(8, 16)
        self.c_proj = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a GELU MLP."""

        return self.c_proj(torch.nn.functional.gelu(self.c_fc(x)))


class _MLPModel(nn.Module):
    """Wrapper exposing one named MLP block."""

    def __init__(self) -> None:
        """Initialize the MLP child."""

        super().__init__()
        self.mlp = GPT2MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP child."""

        return self.mlp(x)


def test_facet_head_zero_and_patch_rerun_changes_output_and_validates() -> None:
    """Head zeroing and static patching scatter back through rerun."""

    torch.manual_seed(0)
    model = _FacetP2Model(MultiHeadSelfAttention())
    x = torch.randn(2, 3, 8)
    clean = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    clean_out = _trace_output(clean).clone()

    zeroed = clean.fork("zero_q_head")
    zeroed.attach_hooks(tl.head(0, "q"), tl.zero_ablate())
    zeroed.rerun(model, x)

    assert not torch.allclose(_trace_output(zeroed), clean_out)
    assert torch.count_nonzero(zeroed.modules["attn"].facets.head(0).q) == 0
    assert zeroed.validate_forward_pass([_trace_output(zeroed)])

    patched = clean.fork("patch_q_head")
    replacement = torch.zeros_like(clean.modules["attn"].facets.head(1).q)
    patched.set(tl.facet("q").head(1), replacement)
    patched.rerun(model, x)

    assert not torch.allclose(_trace_output(patched), clean_out)
    assert torch.count_nonzero(patched.modules["attn"].facets.head(1).q) == 0
    assert patched.validate_forward_pass([_trace_output(patched)])


def test_gpt2_fused_c_attn_facet_edits_compose_and_conflicts_error() -> None:
    """Fused q/k edits compose on one c_attn home and same-region writes fail."""

    torch.manual_seed(1)
    model = _FacetP2Model(GPT2Attention())
    x = torch.randn(2, 3, 8)
    clean = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    clean_out = _trace_output(clean).clone()

    edited = clean.fork("fused_qk")
    edited.attach_hooks(
        [
            (tl.head(0, "q"), tl.zero_ablate()),
            (tl.head(1, "k"), tl.zero_ablate()),
        ]
    )
    edited.rerun(model, x)

    assert not torch.allclose(_trace_output(edited), clean_out)
    assert torch.count_nonzero(edited.modules["attn"].facets.head(0).q) == 0
    assert torch.count_nonzero(edited.modules["attn"].facets.head(1).k) == 0
    assert edited.validate_forward_pass([_trace_output(edited)])

    conflict = clean.fork("fused_conflict")
    with pytest.raises(SiteResolutionError, match="Facet intervention conflict"):
        conflict.attach_hooks(
            [
                (tl.head(0, "q"), tl.zero_ablate()),
                (tl.facet("q").head(0), tl.zero_ablate()),
            ]
        )


def test_gqa_kv_aliasing_write_refuses_but_read_and_grad_work() -> None:
    """Aliasing GQA K/V head writes refuse while read and grad remain usable."""

    torch.manual_seed(2)
    model = _FacetP2Model(LlamaAttention())
    x = torch.randn(2, 3, 8)
    log = tl.trace(
        model,
        x,
        layers_to_save="all",
        save_grads="all",
        save_arg_values=True,
    )
    log.log_backward(_trace_output(log).sum())
    k_head = log.modules["attn"].facets.head(3).k

    assert torch.equal(k_head, log.modules["attn"].facets.k[:, :, 1, :])
    assert not isinstance(k_head.grad, MissingGradient)
    assert torch.equal(k_head.grad, k_head.spec.apply(k_head.spec.home.grad))

    edited = log.fork("gqa_refuse")
    with pytest.raises(RuntimeError, match="aliasing_selection.*alias group"):
        edited.attach_hooks(tl.head(3, "k"), tl.zero_ablate())


def test_computed_facet_write_refuses() -> None:
    """Computed MLP intermediate facets are read-only for intervention."""

    torch.manual_seed(3)
    model = _MLPModel()
    log = tl.trace(model, torch.randn(2, 3, 8), layers_to_save="all", save_arg_values=True)

    with pytest.raises(RuntimeError, match="computed facets are read-only"):
        log.fork("computed_refuse").attach_hooks(tl.facet("intermediate"), tl.zero_ablate())


def test_in_place_or_view_version_facet_write_refuses() -> None:
    """Facets marked with non-raw value versions are not intervention-safe."""

    class Tiny(nn.Module):
        """Tiny model exposing a linear child."""

        def __init__(self) -> None:
            """Initialize the child."""

            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the linear child."""

            return self.linear(x)

    @tl.facets.register(class_name="Linear", target_scope="module", facets=("unsafe",))
    def unsafe_version(module: Any) -> dict[str, Any]:
        """Return a facet marked as a non-raw value version."""

        op = module.trace.ops[module.calls[0].output_ops[0]]
        spec = FacetSpec.from_home(op, recipe_id="unsafe_version").select(-1, 0)
        return {"unsafe": replace(spec, value_version="out_versions_by_child")}

    log = tl.trace(Tiny(), torch.randn(2, 4), layers_to_save="all", save_arg_values=True)

    with pytest.raises(RuntimeError, match="not intervention-safe"):
        log.fork("unsafe_refuse").attach_hooks(tl.facet("unsafe"), tl.zero_ablate())


def test_whole_model_head_selector_ablation_reruns_and_validates() -> None:
    """``tl.head(i)`` ablates default q/k/v head facets across the model."""

    torch.manual_seed(4)
    model = _FacetP2Model(MultiHeadSelfAttention())
    x = torch.randn(2, 3, 8)
    clean = tl.trace(model, x, layers_to_save="all", save_arg_values=True)
    clean_out = _trace_output(clean).clone()

    edited = clean.fork("head_all")
    edited.attach_hooks(tl.head(1), tl.zero_ablate())
    edited.rerun(model, x)

    assert not torch.allclose(_trace_output(edited), clean_out)
    assert torch.count_nonzero(edited.modules["attn"].facets.head(1).q) == 0
    assert torch.count_nonzero(edited.modules["attn"].facets.head(1).k) == 0
    assert torch.count_nonzero(edited.modules["attn"].facets.head(1).v) == 0
    assert edited.validate_forward_pass([_trace_output(edited)])
