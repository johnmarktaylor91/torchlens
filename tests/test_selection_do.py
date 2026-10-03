"""L6 stage 2: Selection-consuming do() under the mask-application contract.

Pins the normative contract rows: edit-then-scatter on a fresh tensor
(helpers stay mask-oblivious, the engine owns masking), the whole-site
short-circuit, the closed ``selection_apply_invalid`` reason set
(shape | dtype | device | broadcast | not_maskable), site eligibility, the
audit record, spec-persistence honesty (the mask never enters a KEEP field;
``selection_recipe`` is DROP-gated), the ``tl.Edit`` public alias, and the
``tl.patch_from`` helper (identity + discriminating behavior; portability
``opaque_audit``).
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.types import HelperSpec
from torchlens.selection import SelectionError

_replay_module = importlib.import_module("torchlens.intervention.replay")


class _TwoConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


@pytest.fixture(scope="module")
def log():
    torch.manual_seed(0)
    trace = tl.trace(
        _TwoConv(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


def test_edit_is_public_type_and_helper_spec_is_alias():
    """tl.Edit is the public edit-object type; HelperSpec the deprecated alias."""

    assert tl.Edit is HelperSpec
    assert isinstance(tl.zero_ablate(), tl.Edit)


def test_interior_masked_edit_scatter_contract(log):
    """Element-masked interior edit: masked elements edited, rest bit-intact."""

    selection = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)])
    mask = selection.resolve(log)[0].mask
    fork = log.fork()
    fork.do(selection, tl.zero_ablate())
    edited = fork["relu_1_2"].out
    baseline = log["relu_1_2"].out
    assert bool((edited[mask] == 0).all())
    assert torch.equal(edited[~mask], baseline[~mask])
    # downstream propagation is real
    assert not torch.equal(fork["output_1"].out, log["output_1"].out)
    # stored capture truth on the SOURCE trace is never written through
    assert torch.equal(log["relu_1_2"].out, baseline)


def test_leaf_site_masked_edit_via_rf_intersection(log):
    """The flagship intersection ablation at the model-input (leaf) site."""

    u1 = log["relu_2_4"]
    u2 = log["conv2d_2_3"]
    inter = u1.receptive_field.at((3, 3)) & u2.receptive_field.at((5, 5))
    mask = inter.resolve(log)[0].mask
    fork = log.fork()
    fork.do(inter, tl.zero_ablate())
    patched = fork["input_1"].out
    assert bool((patched[mask] == 0).all())
    assert torch.equal(patched[~mask], log["input_1"].out[~mask])
    assert not torch.equal(fork["output_1"].out, log["output_1"].out)
    # FireRecord disclosure matches the hook path
    records = list(fork["input_1"].interventions)
    assert records and records[0].helper_name == "zero_ablate" and records[0].replaced
    assert fork["input_1"].intervention_replaced


def test_whole_site_short_circuit(log):
    """A whole-site mask skips the scatter: exactly today's behavior."""

    fork_selection = log.fork()
    fork_selection.do(log["relu_1_2"].__selection__(), tl.zero_ablate())
    fork_shipped = log.fork()
    fork_shipped.do("relu_1_2", tl.zero_ablate())
    assert torch.equal(fork_selection["relu_1_2"].out, fork_shipped["relu_1_2"].out)
    assert torch.equal(fork_selection["output_1"].out, fork_shipped["output_1"].out)


def test_raw_value_routes_through_replace_with(log):
    """A non-callable replacement value obeys the same scatter contract."""

    selection = tl.units("relu_1_2", [(0, 0, 1, 1)])
    mask = selection.resolve(log)[0].mask
    replacement = torch.full_like(log["relu_1_2"].out, 7.0)
    fork = log.fork()
    fork.do(selection, replacement)
    edited = fork["relu_1_2"].out
    assert bool((edited[mask] == 7.0).all())
    assert torch.equal(edited[~mask], log["relu_1_2"].out[~mask])


def test_apply_refusal_axes(log):
    """The closed selection_apply_invalid reason axes, each pinned."""

    selection = tl.units("relu_1_2", [(0, 0, 1, 1)])
    out_shape = log["relu_1_2"].out.shape

    def bad_shape(out, *, hook):
        return torch.zeros(3, 5)

    def bad_broadcast(out, *, hook):
        return torch.zeros(out_shape[-1])  # broadcastable to the site shape

    def bad_dtype(out, *, hook):
        return torch.zeros(tuple(out.shape), dtype=torch.float64)

    def non_tensor(out, *, hook):
        return "junk"

    for hook_fn, reason in (
        (bad_shape, "shape"),
        (bad_broadcast, "broadcast"),
        (bad_dtype, "dtype"),
        (non_tensor, "not_maskable"),
    ):
        fork = log.fork()
        with pytest.raises(SelectionError) as excinfo:
            fork.do(selection, hook_fn)
        assert excinfo.value.fields["code"] == "selection_apply_invalid"
        assert excinfo.value.fields["reason"] == reason


def test_param_and_mixed_plan_refusals(log):
    """PARAM edits ride parameter substitution on the replay engine (the
    2026-08-17 param-operand decision supersedes the D3 typed-refusal default
    there; tests/test_param_substitution.py pins the substitution behavior).
    Off-replay engines and mixed plans still refuse typed."""

    from torchlens.intervention.errors import EngineDispatchError

    with pytest.raises(SelectionError) as excinfo:
        log.fork().do(
            tl.params("c1.weight"),
            tl.zero_ablate(),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert excinfo.value.fields["code"] == "param_substitution_engine_unsupported"

    mixed = tl.units("input_1", [(0, 0, 0, 0)]) | tl.units("relu_1_2", [(0, 0, 0, 0)])
    with pytest.raises(EngineDispatchError):
        log.fork().do(mixed, tl.zero_ablate())

    with pytest.raises(ValueError, match="requires an edit"):
        log.fork().do(tl.units("relu_1_2", [(0, 0, 0, 0)]))


def test_empty_selection_do_is_noop_with_audit(log):
    """Empty resolutions attach nothing (disclosure, never an error)."""

    fork = log.fork()
    fork.do(tl.label("no_such_layer").__selection__(), tl.zero_ablate())
    assert torch.equal(fork["output_1"].out, log["output_1"].out)
    assert fork.intervention_audit[-1]["sites"] == []


def test_audit_record_contents(log):
    """The audit record carries repr + digest + per-site relation/counts."""

    selection = tl.units("relu_1_2", [(0, 0, 1, 1)])
    resolved = selection.resolve(log)
    fork = log.fork()
    fork.do(selection, tl.zero_ablate())
    record = fork.intervention_audit[-1]
    assert record["kind"] == "ACT"
    assert record["resolve_digest"] == resolved.resolve_digest
    assert record["edit"] == "zero_ablate"
    assert record["sites"][0]["relation"] == "exact"
    assert record["sites"][0]["selected"] == 1


def test_mask_never_enters_keep_fields_and_recipe_policy(log):
    """Persistence honesty: derived specs carry masks only in the DROP factory;
    the recipe rides the selection_recipe family (BLOB_RECURSIVE as of the
    tlspec v8 coordinated bump — never smuggled through the KEEP args/kwargs
    fields). Args/kwargs/metadata stay tensor-free."""

    selection = tl.units("relu_1_2", [(0, 0, 1, 1)])
    fork = log.fork()
    fork.do(selection, tl.zero_ablate())
    spec = fork._intervention_spec
    hook_specs = [
        entry.helper for entry in spec.hook_specs if getattr(entry, "helper", None) is not None
    ]
    assert hook_specs, "the derived edit must ride the hook plan as a HelperSpec"
    derived = hook_specs[-1]
    assert derived.helper_name == "zero_ablate"  # helper identity preserved
    assert derived.selection_recipe is not None
    assert derived.selection_recipe["resolve_digest"]
    # the mask lives only in the runtime factory closure, which never persists
    assert HelperSpec.PORTABLE_STATE_SPEC["factory"].name == "DROP"
    assert HelperSpec.PORTABLE_STATE_SPEC["selection_recipe"].name == "BLOB_RECURSIVE"

    def _no_tensor_leaves(value):
        if isinstance(value, torch.Tensor):
            return False
        if isinstance(value, (list, tuple, set)):
            return all(_no_tensor_leaves(item) for item in value)
        if isinstance(value, dict):
            return all(_no_tensor_leaves(item) for item in value.values())
        return True

    assert _no_tensor_leaves(derived.args)
    assert _no_tensor_leaves(dict(derived.kwargs))
    assert _no_tensor_leaves(dict(derived.metadata))
    # element-masked derived specs disclose opaque_audit portability
    assert derived.portability == "opaque_audit"


def test_patch_from_identity_and_discriminating_twin(log):
    """patch_from: identity patch is exact; a perturbed source is consumed."""

    selection = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)])
    mask = selection.resolve(log)[0].mask

    # (a) IDENTITY: patching the trace's own values is exact end to end.
    fork = log.fork()
    fork.do(selection.resolve(fork), tl.patch_from(log))
    assert torch.equal(fork["output_1"].out, log["output_1"].out)

    # (b) DISCRIMINATING TWIN: a documented-epsilon perturbation propagates.
    source = log.fork()
    perturbed = source["relu_1_2"].out.clone()
    perturbed[0, 0, 1, 1] += 10.0
    _replay_module._commit_replay_updates(source, {"relu_1_2": perturbed}, {})
    fork2 = log.fork()
    fork2.do(selection.resolve(fork2), tl.patch_from(source))
    assert torch.isclose(fork2["relu_1_2"].out[0, 0, 1, 1], perturbed[0, 0, 1, 1])
    assert torch.equal(fork2["relu_1_2"].out[~mask], log["relu_1_2"].out[~mask])
    assert not torch.equal(fork2["output_1"].out, log["output_1"].out)

    spec = tl.patch_from(log)
    assert spec.portability == "opaque_audit"
    assert all(not isinstance(value, torch.Tensor) for _, value in spec.kwargs)
    audit = fork2.intervention_audit[-1]
    assert audit["patch_source"]["source_model_class"].endswith("_TwoConv")


def test_receiver_sugar_op_do(log):
    """op.do(edit) / layer.do(edit) receiver sugar composes with selections."""

    fork = log.fork()
    fork["relu_1_2"].do(tl.zero_ablate())
    assert bool((fork["relu_1_2"].out == 0).all())
