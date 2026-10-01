"""Transforms substrate contract tests (transforms memo B1-B4 + P2; lane C04).

Covers the frozen :class:`TransformSpec` plan/apply protocol, canonical JSON
as the resume identity, the single ``coerce_transform`` door, the registry
(closed builtin set, versioned customs), declaration-based ctx dispatch (P2:
never ``inspect.signature`` — both measured failure directions provoked), the
evidence-based axis-role vocabulary, and the role-free kernels.
"""

from __future__ import annotations

import functools
import json

import pytest
import torch

import torchlens as tl
from torchlens.transforms import (
    BTD,
    DEFAULT,
    ContextTransform,
    OpaqueStep,
    RoleDeclaration,
    TensorSpec,
    TransformContext,
    TransformContractError,
    TransformDefinition,
    TransformPipeline,
    TransformSpec,
    axis_for_role,
    canonical_json,
    cast,
    chain,
    coerce_transform,
    coerce_transform_mapping,
    flatten,
    lookup_transform,
    magnitude,
    pipeline_from_record,
    pipeline_record,
    reduce,
    register_transform,
    registered_transform_names,
    take_index,
    unit_norm,
    wants_context,
    with_context,
)

pytestmark = pytest.mark.smoke


# --- canonical JSON: the one byte form -------------------------------------


def test_canonical_json_is_byte_stable_across_key_order() -> None:
    """Two equal records serialize byte-identically (string equality IS equality)."""

    a = canonical_json({"b": [1, 2], "a": {"y": 1, "x": 2}})
    b = canonical_json({"a": {"x": 2, "y": 1}, "b": [1, 2]})
    assert a == b == '{"a":{"x":2,"y":1},"b":[1,2]}'


def test_canonical_json_refuses_nan_and_non_portable_values() -> None:
    """NaN/Infinity and non-JSON values refuse typed (transform_params_invalid)."""

    with pytest.raises(TransformContractError) as excinfo:
        canonical_json(float("nan"))
    assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        canonical_json(torch.zeros(1))
    assert excinfo.value.fields["code"] == "transform_params_invalid"


# --- TransformSpec: seed pairing + version discipline -----------------------


def test_spec_seed_and_source_are_recorded_together_or_not_at_all() -> None:
    """T-C4: a realized seed and its source travel together."""

    with pytest.raises(TransformContractError) as excinfo:
        TransformSpec(name="x", version=1, seed=3, seed_source=None)
    assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        TransformSpec(name="x", version=1, seed=None, seed_source="explicit")
    assert excinfo.value.fields["code"] == "transform_params_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        TransformSpec(name="x", version=1, seed=3, seed_source="vibes")
    assert excinfo.value.fields["code"] == "transform_params_invalid"


def test_spec_version_mismatch_refuses_never_substitutes() -> None:
    """T-C13: a version change is numerics-visible, never silently swapped."""

    stale = TransformSpec(name="flatten", version=99)
    with pytest.raises(TransformContractError) as excinfo:
        stale.plan(TensorSpec(shape=(None, 4), dtype="torch.float32"))
    assert excinfo.value.fields["code"] == "transform_version_mismatch"
    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform(stale)
    assert excinfo.value.fields["code"] == "transform_version_mismatch"


# --- kernels: plan predicts apply (T-C6 self-consistency) --------------------


@pytest.mark.parametrize(
    ("spec", "shape"),
    [
        (flatten(), (3, 2, 4)),
        (flatten(start_axis=2), (3, 2, 4)),
        (cast(torch.float16), (3, 5)),
        (magnitude(), (3, 5)),
        (unit_norm(axis=-1), (3, 5)),
        (take_index(axis=1, index=2), (3, 4, 2)),
        (reduce("mean", axis=1), (3, 4, 2)),
        (reduce("max", axis=(1, 2)), (3, 4, 2)),
    ],
)
def test_kernel_plan_predicts_apply(spec: TransformSpec, shape: tuple[int, ...]) -> None:
    """Every role-free kernel's plan matches its apply on shape and dtype."""

    torch.manual_seed(0)
    tensor = torch.randn(*shape)
    planned = spec.plan(TensorSpec.of(tensor))
    result = spec.apply(tensor)
    assert result.shape[0] == tensor.shape[0], "T-C2: stimulus axis inviolable"
    assert tuple(result.shape[1:]) == planned.output.shape[1:]
    assert str(result.dtype) == planned.output.dtype


def test_kernel_param_validation_refuses_typed() -> None:
    """Bad kernel params refuse with transform_params_invalid at build time."""

    for build in (
        lambda: cast(torch.int64),  # T-C7: float-only cast
        lambda: reduce("median", axis=1),  # closed op set
        lambda: reduce("mean", axis=0),  # never the stimulus axis
        lambda: flatten(start_axis=0),  # never folds the stimulus axis
    ):
        with pytest.raises(TransformContractError) as excinfo:
            build()
        assert excinfo.value.fields["code"] == "transform_params_invalid"


def test_kernel_plan_refuses_out_of_range_axes() -> None:
    """Axis resolution against a concrete rank refuses typed, never guesses."""

    with pytest.raises(TransformContractError) as excinfo:
        take_index(axis=3, index=0).plan(TensorSpec(shape=(None, 4), dtype="torch.float32"))
    assert excinfo.value.fields["code"] == "transform_plan_invalid"


# --- the coercion door -------------------------------------------------------


def test_coerce_none_and_pipeline_pass_through() -> None:
    """``None`` coerces to ``None``; a pipeline passes through unchanged."""

    assert coerce_transform(None) is None
    pipeline = chain(flatten())
    assert coerce_transform(pipeline) is pipeline


def test_bare_strings_resolve_only_to_zero_param_presets() -> None:
    """Strings are never import paths or a parameter mini-language."""

    pipeline = coerce_transform("magnitude")
    assert pipeline is not None and isinstance(pipeline.steps[0], TransformSpec)
    assert pipeline.steps[0].name == "magnitude"
    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform("cast")  # takes params: not a preset
    assert excinfo.value.fields["code"] == "transform_name_not_preset"
    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform("os.system")
    assert excinfo.value.fields["code"] == "transform_name_unknown"


def test_chain_and_sequence_normalize_to_the_same_object() -> None:
    """Decision 3: ``chain(a, b)`` and ``[a, b]`` are one object, one record."""

    a = chain(flatten(), "magnitude")
    b = coerce_transform([flatten(), "magnitude"])
    assert b is not None
    assert a.canonical_chain() == b.canonical_chain()
    nested = coerce_transform([[flatten()], ["magnitude"]])
    assert nested is not None
    assert nested.canonical_chain() == a.canonical_chain()


def test_unordered_containers_refuse() -> None:
    """T-C8: an unordered container has no left or right."""

    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform({flatten()})
    assert excinfo.value.fields["code"] == "transform_coercion_invalid"


def test_uncoercible_values_refuse_typed() -> None:
    """Non-callable non-spec values refuse with the door's full menu."""

    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform(42)
    assert excinfo.value.fields["code"] == "transform_coercion_invalid"


def test_mapping_refuses_at_the_single_chain_door() -> None:
    """Per-site Mappings resolve engine-side, never as one chain."""

    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform({"relu": flatten()})
    assert excinfo.value.fields["code"] == "transform_coercion_invalid"


def test_mapping_resolution_per_output_key_with_default() -> None:
    """Decision 13: per-site Mapping resolves per label BEFORE coercion."""

    resolved = coerce_transform_mapping({"relu": "magnitude", DEFAULT: flatten()}, ["relu", "fc"])
    assert set(resolved) == {"relu", "fc"}
    relu_chain = resolved["relu"]
    fc_chain = resolved["fc"]
    assert relu_chain is not None and relu_chain.steps[0].name == "magnitude"
    assert fc_chain is not None and fc_chain.steps[0].name == "flatten"
    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform_mapping({"typo": "magnitude"}, ["relu", "fc"])
    assert excinfo.value.fields["code"] == "transform_mapping_key_unknown"
    assert excinfo.value.fields["unknown_keys"] == ["typo"]


# --- registry: closed builtins, versioned customs ----------------------------


def test_builtin_names_are_closed_and_non_replaceable() -> None:
    """Registering over a builtin refuses transform_builtin_shadowed."""

    definition = TransformDefinition(
        name="flatten",
        version=2,
        normalize_params=dict,
        plan_fn=lambda spec, input_spec, ctx: None,  # type: ignore[arg-type,return-value]
        apply_fn=lambda spec, tensor, ctx: tensor,
    )
    with pytest.raises(TransformContractError) as excinfo:
        register_transform(definition)
    assert excinfo.value.fields["code"] == "transform_builtin_shadowed"


def test_custom_registration_and_rehydration_round_trip() -> None:
    """A registered custom rebuilds from its record without importing code."""

    from torchlens.transforms import PlannedStep

    name = "test_c04_double"
    if name not in registered_transform_names():
        register_transform(
            TransformDefinition(
                name=name,
                version=3,
                normalize_params=dict,
                plan_fn=lambda spec, input_spec, ctx: PlannedStep(
                    name=spec.name,
                    version=spec.version,
                    output=input_spec,
                    stream_safe=True,
                    may_alias=False,
                    context_capable=False,
                ),
                apply_fn=lambda spec, tensor, ctx: tensor * 2,
            )
        )
    assert name in registered_transform_names()
    assert lookup_transform(name).version == 3
    record = {
        "schema": "tl_transform_pipeline_v1",
        "steps": [
            {
                "kind": "spec",
                "name": name,
                "version": 3,
                "params": {},
                "seed": None,
                "seed_source": None,
            }
        ],
    }
    pipeline = pipeline_from_record(record)
    assert torch.equal(pipeline.apply(torch.ones(2, 3)), torch.full((2, 3), 2.0))


def test_lookup_unknown_names_the_registry() -> None:
    """Refusals name the registry and the registered names (S4 shape)."""

    with pytest.raises(TransformContractError) as excinfo:
        lookup_transform("no_such_transform")
    assert excinfo.value.fields["code"] == "transform_name_unknown"
    assert "flatten" in excinfo.value.fields["registered"]


# --- P2: ctx dispatch by DECLARATION, never inspect.signature ----------------


def test_wants_context_is_false_for_every_sniffing_counterexample() -> None:
    """The measured counterexamples: C-ops and modules report (*args, **kwargs).

    ``inspect.signature``-based dispatch would call each of these WITH ctx
    (TypeError at batch 1); declaration-based dispatch keeps them unary.
    """

    assert wants_context(torch.abs) is False
    assert wants_context(torch.nn.ReLU()) is False
    assert wants_context(torch.nn.Flatten(start_dim=1)) is False
    assert wants_context(lambda t: t) is False


def test_wants_context_is_false_for_keyword_only_ctx_partials() -> None:
    """The other measured direction: a kw-only ctx partial reports unary.

    Signature sniffing would call it unary and silently run mask-blind;
    declaration keeps that a conscious choice — undeclared means unary.
    """

    def masked(tensor: torch.Tensor, *, ctx: TransformContext | None = None) -> torch.Tensor:
        return tensor

    partial = functools.partial(masked)
    assert wants_context(partial) is False


def test_with_context_declares_and_receives_ctx() -> None:
    """The one legal door to ctx: explicit declaration."""

    seen: list[TransformContext | None] = []

    def fn(tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
        seen.append(ctx)
        return tensor

    declared = with_context(fn)
    assert wants_context(declared) is True
    ctx = TransformContext(site_label="relu")
    pipeline = coerce_transform(declared)
    assert pipeline is not None
    pipeline.apply(torch.ones(2, 2), ctx)
    assert seen == [ctx]
    with pytest.raises(TransformContractError) as excinfo:
        with_context("not callable")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "transform_coercion_invalid"


def test_context_none_is_always_legal() -> None:
    """Every chain behaves identically under ctx=None and an empty context."""

    pipeline = chain(flatten(), "magnitude")
    torch.manual_seed(0)
    tensor = torch.randn(3, 2, 2)
    assert torch.equal(pipeline.apply(tensor, None), pipeline.apply(tensor, TransformContext()))


# --- axis roles: evidence-based, rank is never evidence ----------------------


def test_role_evidence_vocabulary_is_closed_and_inferred_reserved() -> None:
    """``inferred`` is reserved: rank-based guessing is never evidence."""

    with pytest.raises(TransformContractError) as excinfo:
        RoleDeclaration(axes=("batch", "token"), evidence="inferred")
    assert excinfo.value.fields["code"] == "transform_role_evidence_invalid"
    with pytest.raises(TransformContractError) as excinfo:
        RoleDeclaration(axes=(), evidence="declared")
    assert excinfo.value.fields["code"] == "transform_role_declaration_invalid"


def test_axis_for_role_resolves_declared_and_refuses_with_both_remedies() -> None:
    """Semantic resolution needs recorded roles at the right rank, or teaches."""

    assert axis_for_role(BTD, rank=3, role="token") == 1
    with pytest.raises(TransformContractError) as excinfo:
        axis_for_role(BTD, rank=4, role="token")  # rank mismatch: no guessing
    assert excinfo.value.fields["code"] == "transform_axis_roles_unavailable"
    message = str(excinfo.value)
    assert "declare axis roles" in message and "explicit-axis" in message
    with pytest.raises(TransformContractError) as excinfo:
        axis_for_role(None, rank=3, role="token")
    assert excinfo.value.fields["code"] == "transform_axis_roles_unavailable"


# --- pipeline: one record, the resume identity --------------------------------


def test_canonical_chain_is_the_resume_identity() -> None:
    """Byte equality of the static chain IS chain equality; params change it."""

    a = chain(flatten(), cast(torch.float16))
    b = pipeline_from_record(pipeline_record(chain(flatten(), cast(torch.float16))))
    assert a.canonical_chain() == b.canonical_chain()
    changed = chain(flatten(), cast(torch.float32))
    assert a.canonical_chain() != changed.canonical_chain(), "numerics-visible change mismatches"


def test_pipeline_record_shape_and_verifiability() -> None:
    """The tl_transform_pipeline_v1 record carries steps + identity + verdict."""

    record = pipeline_record(chain(flatten(), torch.abs))
    assert record is not None
    assert record["schema"] == "tl_transform_pipeline_v1"
    assert record["steps"][0]["kind"] == "spec"
    assert record["steps"][1]["kind"] == "opaque"
    assert record["steps"][1]["identity"] == "identification_only"
    assert record["resume_verifiable"] is False
    assert json.loads(record["canonical_chain"])[1]["qualname"] == "abs"
    assert pipeline_record(None) is None


def test_opaque_rehydration_refuses_application_typed() -> None:
    """Identification is a disclosure, not code: no import, no execution."""

    record = pipeline_record(chain(torch.abs))
    assert record is not None
    rebuilt = pipeline_from_record(record)
    assert rebuilt.resume_verifiable is False
    opaque = rebuilt.steps[0]
    assert isinstance(opaque, OpaqueStep) and opaque.fn is None
    with pytest.raises(TransformContractError) as excinfo:
        rebuilt.apply(torch.ones(2, 2))
    assert excinfo.value.fields["code"] == "transform_opaque_unresolvable"


def test_malformed_pipeline_records_refuse_typed() -> None:
    """Wrong schema, missing steps, and unknown kinds all refuse typed."""

    for record in (
        {"schema": "wrong"},
        {"schema": "tl_transform_pipeline_v1"},
        {"schema": "tl_transform_pipeline_v1", "steps": [{"kind": "mystery"}]},
        {"schema": "tl_transform_pipeline_v1", "steps": [17]},
    ):
        with pytest.raises(TransformContractError) as excinfo:
            pipeline_from_record(record)  # type: ignore[arg-type]
        assert excinfo.value.fields["code"] == "transform_pipeline_record_invalid"


def test_plan_stops_predicting_after_an_opaque_step() -> None:
    """Post-opaque spec steps are unplanned: a guessed plan is worse than none."""

    pipeline = chain(flatten(), torch.abs, cast(torch.float16))
    planned = pipeline.plan(TensorSpec(shape=(None, 2, 3), dtype="torch.float32"))
    assert planned[0] is not None and planned[0].output.shape == (None, 6)
    assert planned[1] is None, "opaque step has no plan"
    assert planned[2] is None, "post-opaque spec steps cannot be planned"


def test_chain_applies_left_to_right() -> None:
    """T-C8: composition is ordered; the record order is the numeric order."""

    tensor = torch.tensor([[-2.0, 2.0]])
    forward = chain(magnitude(), unit_norm(axis=-1)).apply(tensor)
    expected = torch.abs(tensor) / torch.linalg.vector_norm(torch.abs(tensor))
    assert torch.allclose(forward, expected)


def test_hand_built_specs_renormalize_through_the_door() -> None:
    """A hand-built spec cannot smuggle unvalidated params into a record."""

    smuggled = TransformSpec(name="reduce", version=1, params=(("axis", 0), ("op", "mean")))
    with pytest.raises(TransformContractError) as excinfo:
        coerce_transform(smuggled)
    assert excinfo.value.fields["code"] == "transform_params_invalid"


class _DeclaredDouble(ContextTransform):
    """Subclass declaration form of a context-capable transform."""

    def __call__(self, tensor: torch.Tensor, ctx: TransformContext | None) -> torch.Tensor:
        """Double the tensor, ignoring the context.

        Parameters
        ----------
        tensor:
            Batch tensor.
        ctx:
            Ignored.

        Returns
        -------
        torch.Tensor
            ``tensor * 2``.
        """

        return tensor * 2


def test_context_transform_subclass_is_declared_and_opaque_in_records() -> None:
    """A ContextTransform subclass is context-capable and records as opaque."""

    step = _DeclaredDouble()
    assert wants_context(step) is True
    pipeline = coerce_transform(step)
    assert pipeline is not None
    assert torch.equal(pipeline.apply(torch.ones(1, 2)), torch.full((1, 2), 2.0))
    record = pipeline_record(pipeline)
    assert record is not None and record["steps"][0]["kind"] == "opaque"


def test_transforms_facade_is_lazily_reachable() -> None:
    """tl.transforms is the (documented-unstable) lazy home of the library."""

    assert tl.transforms.flatten is flatten
    assert isinstance(TransformPipeline(steps=()), TransformPipeline)
