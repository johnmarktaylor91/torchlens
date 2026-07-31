"""The extended declarative R1 grammar admits pinned-library shapes and refuses laundering.

The R1 rung's defining machine property is that no authored text executes
(``recipe.py`` loads a closed JSON recipe without evaluating source). These
tests pin the grammar extension in BOTH directions:

- The three previously inexpressible pinned-library shapes -- two-step
  construction, constructed non-tensor input leaves, and non-``forward``
  entrypoints -- plus post-construction configuration are now expressible with
  ``code_path: null`` and an empty code manifest.
- The failing direction: a model composed from generic primitives is refused by
  the runtime provenance tripwire, generic containers are refused as root or
  construct nodes, and the grammar exposes no slot for arbitrary computation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from menagerie.crawler.constants import RunMode
from menagerie.crawler.recipe import (
    MAX_CONSTRUCT_DEPTH,
    MAX_CONSTRUCT_NODES,
    DeclarativeRecipe,
    RecipeError,
    _reject_container_constructor,
    assert_model_provenance,
    load_declarative_recipe,
    resolve_input_constructor,
)
from menagerie.crawler.schema import PayloadValidationError, validate_payload
from menagerie.crawler.tests.conftest import make_author_proposal, make_model
from menagerie.crawler.worker import WorkerRequest, run_worker


def _torch_recipe(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid torch-distribution declarative recipe mapping.

    Parameters
    ----------
    **overrides:
        Recipe fields replacing the Linear baseline.

    Returns
    -------
    dict[str, Any]
        Declarative recipe mapping.
    """

    recipe: dict[str, Any] = {
        "distribution": "torch",
        "version": torch.__version__,
        "module": "torch.nn",
        "symbol": "Linear",
        "kwargs": {"in_features": 4, "out_features": 2},
        "pretrained_disable_fields": [],
    }
    recipe.update(overrides)
    return recipe


# ---------------------------------------------------------------------------
# Admission: the previously inexpressible shapes, zero authored bytes.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_two_step_construction_builds_pinned_library_object() -> None:
    """A construct-node kwarg expresses constructor composition with zero authored bytes."""

    loaded = load_declarative_recipe(
        _torch_recipe(
            symbol="TransformerEncoder",
            kwargs={
                "encoder_layer": {
                    "__construct__": {
                        "module": "torch.nn",
                        "symbol": "TransformerEncoderLayer",
                        "kwargs": {
                            "d_model": 8,
                            "nhead": 2,
                            "dim_feedforward": 16,
                            "batch_first": True,
                        },
                    }
                },
                "num_layers": 1,
            },
        )
    )
    model = loaded.build_model()

    assert loaded.kind == "declarative-library"
    assert loaded.adapter_sha256 is None
    assert type(model) is torch.nn.TransformerEncoder
    assert tuple(model(torch.zeros(1, 3, 8)).shape) == (1, 3, 8)


def test_two_step_construction_expresses_hf_config_first_models() -> None:
    """The HuggingFace config-then-model two-step lands as pure declarative JSON."""

    transformers = pytest.importorskip("transformers")
    loaded = load_declarative_recipe(
        {
            "distribution": "transformers",
            "version": transformers.__version__,
            "module": "transformers",
            "symbol": "MixtralForCausalLM",
            "kwargs": {
                "config": {
                    "__construct__": {
                        "module": "transformers",
                        "symbol": "MixtralConfig",
                        "kwargs": {
                            "hidden_size": 16,
                            "intermediate_size": 32,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "num_local_experts": 2,
                            "num_experts_per_tok": 1,
                            "vocab_size": 64,
                            "max_position_embeddings": 32,
                        },
                    }
                }
            },
            "pretrained_disable_fields": [],
        }
    )
    model = loaded.build_model()

    assert type(model).__module__.startswith("transformers.models.mixtral")
    output = model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    assert tuple(output.logits.shape) == (1, 4, 64)


def test_post_construct_restores_upstream_train_contract_at_r1() -> None:
    """The PiT case: ``set_distilled_training(True)`` is declaratively expressible.

    This is the motivating post-construction shape: the upstream train-mode
    output contract (a distillation tuple) is restored through a documented
    public configuration method of the pinned installed class, with zero
    authored bytes.
    """

    timm = pytest.importorskip("timm")
    loaded = load_declarative_recipe(
        {
            "distribution": "timm",
            "version": timm.__version__,
            "module": "timm.models.pit",
            "symbol": "pit_xs_distilled_224",
            "kwargs": {"pretrained": False},
            "pretrained_disable_fields": ["pretrained"],
            "post_construct": [
                {"method": "set_distilled_training", "args": [True], "kwargs": {}}
            ],
        }
    )
    model = loaded.build_model()

    assert model.distilled_training is True
    model.train()
    output = model(torch.zeros(1, 3, 224, 224))
    assert isinstance(output, tuple) and len(output) == 2
    assert tuple(output[0].shape) == (1, 1000)
    assert tuple(output[1].shape) == (1, 1000)


@pytest.mark.smoke
def test_post_construct_applies_bounded_json_configuration() -> None:
    """A plain torch module accepts a bounded public JSON-argument configuration call."""

    loaded = load_declarative_recipe(
        _torch_recipe(post_construct=[{"method": "train", "args": [True], "kwargs": {}}])
    )
    model = loaded.build_model()

    assert isinstance(model, torch.nn.Linear)
    assert model.training is True


@pytest.mark.smoke
def test_constructed_input_leaf_runs_through_the_worker(tmp_path: Path) -> None:
    """A ``distribution: constructor`` input leaf materializes without authored code."""

    contract = {
        "builder_symbol": "make_dummy_call",
        "seed": 0,
        "semantic_description": "One constructed input object.",
        "source_basis": ["evidence-1"],
        "smallest_valid_probe_rationale": "Smallest valid batch.",
        "args": [
            {
                "path": "args[0]",
                "kind": "constructed",
                "semantic_role": "input object",
                "shape": [],
                "dtype": "object",
                "device_policy": "cpu",
                "distribution": "constructor",
                "constraints": [],
                "source_evidence_ids": ["evidence-1"],
                "constructor": {
                    "module": "torch",
                    "symbol": "zeros",
                    "kwargs": {"size": [1, 4]},
                },
            }
        ],
        "kwargs": [],
        "non_tensor_values": [],
        "masks_state_and_control": [],
        "expected_output_semantics": "projected features",
    }
    request = WorkerRequest(
        stable_id="m_constructed_input",
        recipe={"kind": "declarative-library", "recipe": _torch_recipe()},
        modality=None,
        input_spec=None,
        input_contract=contract,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
    )

    receipt = run_worker(request)

    mode_receipt = receipt["per_mode"]["eval"]
    assert mode_receipt["forward_completed"] is True
    assert mode_receipt["input_kind"] == "standard-constructed-input"
    assert "constructed input via torch.zeros" in mode_receipt["input_note"]
    assert receipt["observed_adapter_sha256"] is None


def test_declarative_entrypoint_delegates_through_transparent_adapter(
    tmp_path: Path,
) -> None:
    """A declared non-``forward`` entrypoint is receipted as the delegated method."""

    timm = pytest.importorskip("timm")
    tensor_leaf = {
        "path": "args[0]",
        "kind": "tensor",
        "semantic_role": "image",
        "shape": [1, 3, 64, 64],
        "dtype": "float32",
        "device_policy": "cpu",
        "distribution": "normal",
        "constraints": [],
        "source_evidence_ids": ["evidence-1"],
    }
    contract = {
        "builder_symbol": "make_dummy_call",
        "seed": 0,
        "semantic_description": "One small RGB image for the feature entrypoint.",
        "source_basis": ["evidence-1"],
        "smallest_valid_probe_rationale": "Smallest fully-convolutional extent.",
        "args": [tensor_leaf],
        "kwargs": [],
        "non_tensor_values": [],
        "masks_state_and_control": [],
        "expected_output_semantics": "backbone feature map",
    }
    request = WorkerRequest(
        stable_id="m_entrypoint",
        recipe={
            "kind": "declarative-library",
            "recipe": {
                "distribution": "timm",
                "version": timm.__version__,
                "module": "timm.models.resnet",
                "symbol": "resnet18",
                "kwargs": {"pretrained": False},
                "pretrained_disable_fields": ["pretrained"],
                "entrypoint": "forward_features",
            },
        },
        modality=None,
        input_spec=None,
        input_contract=contract,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
    )

    receipt = run_worker(request)

    assert set(receipt["per_mode"]) == {"train", "eval"}
    for mode_receipt in receipt["per_mode"].values():
        assert mode_receipt["forward_completed"] is True
        assert mode_receipt["delegated_method"] == "forward_features"
    eval_leaves = receipt["per_mode"]["eval"]["output_signature"]["leaves"]
    assert eval_leaves[0]["shape"] == [1, 512, 2, 2]


@pytest.mark.smoke
def test_pre_extension_recipes_keep_their_exact_revision_payload() -> None:
    """Recipes without grammar-extension fields emit the exact historical payload."""

    recipe = DeclarativeRecipe.from_mapping(_torch_recipe())

    assert set(recipe.to_dict()) == {
        "distribution",
        "version",
        "artifact_sha256",
        "module",
        "symbol",
        "kwargs",
        "pretrained_disable_fields",
    }


# ---------------------------------------------------------------------------
# Refusal: composition laundering, containers, and computation slots.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_composed_from_primitives_model_fails_the_provenance_tripwire() -> None:
    """A generic torch primitive claimed under another distribution is refused at build."""

    loaded = load_declarative_recipe(
        _torch_recipe(distribution="timm", version="1.0.28")
    )

    with pytest.raises(RecipeError) as caught:
        loaded.build_model()
    assert str(caught.value) == (
        "constructed model type torch.nn.modules.linear.Linear is not defined by "
        "the pinned distribution 'timm': attributed to ['torch']"
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "symbol", ["Sequential", "ModuleList", "ModuleDict", "ParameterList", "Module"]
)
def test_generic_container_root_is_refused_at_parse(symbol: str) -> None:
    """Generic containers are denied as the declarative recipe root symbol."""

    with pytest.raises(RecipeError) as caught:
        load_declarative_recipe(_torch_recipe(symbol=symbol, kwargs={}))
    assert str(caught.value) == (
        f"generic torch.nn containers are refused in declarative R1 recipes: torch.nn.{symbol}"
    )


@pytest.mark.smoke
def test_generic_container_is_refused_as_construct_node() -> None:
    """Generic containers are denied inside the construct graph, not only at the root."""

    with pytest.raises(RecipeError) as caught:
        load_declarative_recipe(
            _torch_recipe(
                symbol="TransformerEncoder",
                kwargs={
                    "encoder_layer": {
                        "__construct__": {
                            "module": "torch.nn",
                            "symbol": "Sequential",
                            "kwargs": {},
                        }
                    },
                    "num_layers": 1,
                },
            )
        )
    assert str(caught.value) == (
        "generic torch.nn containers are refused in declarative R1 recipes: "
        "torch.nn.Sequential"
    )


@pytest.mark.smoke
def test_container_identity_is_refused_even_through_reexport() -> None:
    """The runtime identity check refuses the actual container class regardless of its path."""

    with pytest.raises(RecipeError) as caught:
        _reject_container_constructor(torch.nn.Sequential, "laundered.alias")
    assert str(caught.value) == (
        "generic torch.nn containers are refused in declarative R1 recipes: laundered.alias"
    )


@pytest.mark.smoke
def test_constructed_container_instance_is_refused_even_for_torch_distribution() -> None:
    """A built ``nn.Sequential`` cannot claim R1 even when torch itself is the distribution."""

    composed = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    with pytest.raises(RecipeError) as caught:
        assert_model_provenance(composed, "torch")
    assert str(caught.value) == (
        "declarative recipe constructed a generic container "
        "torch.nn.modules.container.Sequential; a composed container cannot claim R1"
    )


@pytest.mark.smoke
def test_provenance_tripwire_passes_the_pinned_distribution_class() -> None:
    """The tripwire's accepting direction: a torch-defined class under distribution torch."""

    assert_model_provenance(torch.nn.Linear(2, 2), "torch")


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("label", "recipe", "message"),
    [
        (
            "construct node with sibling keys",
            _torch_recipe(
                kwargs={
                    "in_features": {
                        "__construct__": {
                            "module": "torch",
                            "symbol": "zeros",
                            "kwargs": {},
                        },
                        "extra": 1,
                    },
                    "out_features": 2,
                }
            ),
            "construct node at kwargs.in_features must carry only the '__construct__' key",
        ),
        (
            "construct spec with an extra field",
            _torch_recipe(
                kwargs={
                    "in_features": {
                        "__construct__": {
                            "module": "torch",
                            "symbol": "zeros",
                            "kwargs": {},
                            "code": "evil()",
                        }
                    },
                    "out_features": 2,
                }
            ),
            "construct node at kwargs.in_features must declare exactly module, symbol, and kwargs",
        ),
        (
            "non-identifier construct module",
            _torch_recipe(
                kwargs={
                    "in_features": {
                        "__construct__": {
                            "module": "torch; import os",
                            "symbol": "zeros",
                            "kwargs": {},
                        }
                    },
                    "out_features": 2,
                }
            ),
            "construct node at kwargs.in_features module must be a dotted identifier",
        ),
        (
            "private post_construct method",
            _torch_recipe(
                post_construct=[{"method": "__setattr__", "args": ["x", 1], "kwargs": {}}]
            ),
            "post_construct[0] method must be a public name, not '__setattr__'",
        ),
        (
            "construct node smuggled into post_construct args",
            _torch_recipe(
                post_construct=[
                    {
                        "method": "train",
                        "args": [
                            {
                                "__construct__": {
                                    "module": "torch",
                                    "symbol": "zeros",
                                    "kwargs": {},
                                }
                            }
                        ],
                        "kwargs": {},
                    }
                ]
            ),
            "post_construct[0].args[0] must be plain JSON without construct nodes",
        ),
        (
            "private entrypoint",
            _torch_recipe(entrypoint="_forward_impl"),
            "entrypoint must be a public method, not '_forward_impl'",
        ),
        (
            "dunder entrypoint",
            _torch_recipe(entrypoint="__call__"),
            "entrypoint must be a public method, not '__call__'",
        ),
    ],
)
def test_the_grammar_has_no_slot_for_arbitrary_computation(
    label: str, recipe: dict[str, Any], message: str
) -> None:
    """Every attempted computation slot in the extended grammar is a typed refusal.

    Parameters
    ----------
    label:
        Human-readable attack name.
    recipe:
        Malformed declarative recipe.
    message:
        Exact expected refusal.
    """

    with pytest.raises(RecipeError) as caught:
        load_declarative_recipe(recipe)
    assert str(caught.value) == message, label


@pytest.mark.smoke
def test_construct_graph_depth_and_node_bounds_are_enforced() -> None:
    """Nesting beyond the closed depth bound and node budget is refused."""

    def nested(levels: int) -> dict[str, Any]:
        """Build a construct chain of the given depth.

        Parameters
        ----------
        levels:
            Construct-node nesting depth.

        Returns
        -------
        dict[str, Any]
            Nested construct-node value.
        """

        value: dict[str, Any] = {
            "__construct__": {"module": "torch", "symbol": "zeros", "kwargs": {}}
        }
        for _level in range(levels - 1):
            value = {
                "__construct__": {
                    "module": "torch",
                    "symbol": "zeros",
                    "kwargs": {"size": value},
                }
            }
        return value

    with pytest.raises(RecipeError) as too_deep:
        load_declarative_recipe(
            _torch_recipe(kwargs={"in_features": nested(MAX_CONSTRUCT_DEPTH + 1)})
        )
    assert str(too_deep.value) == (
        f"construct graph exceeds the maximum depth of {MAX_CONSTRUCT_DEPTH}"
    )

    flat_nodes = {
        f"key_{index}": {
            "__construct__": {"module": "torch", "symbol": "zeros", "kwargs": {}}
        }
        for index in range(MAX_CONSTRUCT_NODES + 1)
    }
    with pytest.raises(RecipeError) as too_many:
        load_declarative_recipe(_torch_recipe(kwargs=flat_nodes))
    assert str(too_many.value) == (
        f"construct graph exceeds the maximum of {MAX_CONSTRUCT_NODES} nodes"
    )


@pytest.mark.smoke
def test_post_construct_unknown_method_fails_at_build() -> None:
    """A declared configuration method absent from the model is a typed build refusal."""

    loaded = load_declarative_recipe(
        _torch_recipe(
            post_construct=[{"method": "set_distilled_training", "args": [True], "kwargs": {}}]
        )
    )

    with pytest.raises(RecipeError) as caught:
        loaded.build_model()
    assert str(caught.value) == (
        "post_construct[0] method 'set_distilled_training' is not callable "
        "on the constructed model"
    )


@pytest.mark.smoke
def test_input_constructor_refusals() -> None:
    """The input-constructor grammar refuses containers, malformed specs, and None."""

    with pytest.raises(RecipeError) as container:
        resolve_input_constructor(
            {"module": "torch.nn", "symbol": "Sequential", "kwargs": {}}
        )
    assert str(container.value) == (
        "generic torch.nn containers are refused in declarative R1 recipes: "
        "torch.nn.Sequential"
    )

    with pytest.raises(RecipeError) as malformed:
        resolve_input_constructor({"module": "torch", "symbol": "zeros"})
    assert str(malformed.value) == (
        "construct node at input_contract constructor must declare exactly "
        "module, symbol, and kwargs"
    )

    with pytest.raises(RecipeError) as produced_none:
        resolve_input_constructor(
            {"module": "gc", "symbol": "disable", "kwargs": {}}
        )
    assert str(produced_none.value) == "input constructor gc.disable produced None"


@pytest.mark.smoke
def test_worker_refuses_a_constructor_leaf_with_a_laundered_kind(tmp_path: Path) -> None:
    """A constructor-distribution leaf that claims another kind fails before execution."""

    contract = {
        "builder_symbol": "make_dummy_call",
        "seed": 0,
        "semantic_description": "One mislabeled constructed leaf.",
        "source_basis": ["evidence-1"],
        "smallest_valid_probe_rationale": "Smallest valid batch.",
        "args": [
            {
                "path": "args[0]",
                "kind": "tensor",
                "semantic_role": "input object",
                "shape": [1, 4],
                "dtype": "float32",
                "device_policy": "cpu",
                "distribution": "constructor",
                "constraints": [],
                "source_evidence_ids": ["evidence-1"],
                "constructor": {
                    "module": "torch",
                    "symbol": "zeros",
                    "kwargs": {"size": [1, 4]},
                },
            }
        ],
        "kwargs": [],
        "non_tensor_values": [],
        "masks_state_and_control": [],
        "expected_output_semantics": "projected features",
    }
    request = WorkerRequest(
        stable_id="m_laundered_kind",
        recipe={"kind": "declarative-library", "recipe": _torch_recipe()},
        modality=None,
        input_spec=None,
        input_contract=contract,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
    )

    receipt = run_worker(request)

    assert receipt["per_mode"] == {}
    assert receipt["error"]["message"] == (
        "input_contract.args constructor leaf must declare kind 'constructed'"
    )


# ---------------------------------------------------------------------------
# Schema: the extended grammar validates, misplacement is refused.
# ---------------------------------------------------------------------------


def _constructed_leaf() -> dict[str, Any]:
    """Return one schema-valid constructed input leaf.

    Returns
    -------
    dict[str, Any]
        Constructor-distribution leaf payload.
    """

    return {
        "path": "args[0]",
        "kind": "constructed",
        "semantic_role": "graph input",
        "shape": [],
        "dtype": "object",
        "device_policy": "cpu",
        "distribution": "constructor",
        "constraints": [],
        "source_evidence_ids": ["evidence-1"],
        "constructor": {
            "module": "dgl",
            "symbol": "rand_graph",
            "kwargs": {"num_nodes": 100, "num_edges": 500},
        },
    }


def test_schema_admits_the_extended_recipe_and_constructed_leaves() -> None:
    """Model and proposal records carrying the extended grammar validate."""

    model = make_model(accepted=True)
    recipe = model["implementation"]["library_recipe"]
    recipe["entrypoint"] = "run"
    recipe["post_construct"] = [
        {"method": "set_distilled_training", "args": [True], "kwargs": {}}
    ]
    model["input_contract"]["args"][0] = _constructed_leaf()
    validate_payload(model)

    proposal = make_author_proposal()
    proposal_recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    proposal_recipe["entrypoint"] = "run"
    proposal_recipe["post_construct"] = [
        {"method": "set_distilled_training", "args": [True], "kwargs": {}}
    ]
    proposal["proposed_facts"]["input_contract"]["args"][0] = _constructed_leaf()
    validate_payload(proposal)


def test_schema_refuses_misplaced_or_missing_input_constructors() -> None:
    """A constructor on a tensor leaf and a constructor-less constructed leaf both fail."""

    misplaced = make_model(accepted=True)
    misplaced["input_contract"]["args"][0]["constructor"] = {
        "module": "torch",
        "symbol": "zeros",
        "kwargs": {},
    }
    with pytest.raises(PayloadValidationError):
        validate_payload(misplaced)

    missing = make_model(accepted=True)
    leaf = _constructed_leaf()
    del leaf["constructor"]
    missing["input_contract"]["args"][0] = leaf
    with pytest.raises(PayloadValidationError):
        validate_payload(missing)

    wrong_kind = make_model(accepted=True)
    mislabeled = _constructed_leaf()
    mislabeled["kind"] = "tensor"
    wrong_kind["input_contract"]["args"][0] = mislabeled
    with pytest.raises(PayloadValidationError):
        validate_payload(wrong_kind)


def test_schema_refuses_a_malformed_post_construct_call() -> None:
    """Post-construct calls missing their closed fields are refused by the schema."""

    model = make_model(accepted=True)
    model["implementation"]["library_recipe"]["post_construct"] = [{"method": "train"}]
    with pytest.raises(PayloadValidationError):
        validate_payload(model)

    unknown_field = deepcopy(make_model(accepted=True))
    unknown_field["implementation"]["library_recipe"]["post_construct"] = [
        {"method": "train", "args": [], "kwargs": {}, "code": "evil()"}
    ]
    with pytest.raises(PayloadValidationError):
        validate_payload(unknown_field)
