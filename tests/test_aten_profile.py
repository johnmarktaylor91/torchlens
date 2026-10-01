"""Wave-0 gates for the private DROP-gated ATen execution profile."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._io import TorchLensIOError
from torchlens._io.prerelease import activate_prerelease_fields
from torchlens.backends.torch._aten_capture import _activate_aten_recording_for_tests
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.constants import PRIMITIVE_OP_FIELD_ORDER
from torchlens.data_classes.aten_op import AtenOp, OpRef
from torchlens.intervention import SuperAtenOp
from torchlens.ir.events import _AtenExecutionContext, _AtenTensorFact
from torchlens.validation._invariants_primitive_ops import _check_primitive_op_invariants
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants

pytestmark = pytest.mark.smoke

_DOCUMENTED_UNSTABLE_ATEN_TOKENS = {
    "AtenOp",
    "OpRef",
    "SuperAtenOp",
    "primitive_op",
    "mode_paused_interior",
    "exact_via_aten",
    "heuristic",
    "PRIMITIVE_OP_FIELD_ORDER",
    *PRIMITIVE_OP_FIELD_ORDER,
    "kind",
    "sequence_before",
    "sequence_after",
    "reason",
    "op_row_index",
    "op_label",
    "func_call_id",
    *_AtenTensorFact.PORTABLE_STATE_SPEC,
    *_AtenExecutionContext.PORTABLE_STATE_SPEC,
    "comparison_status",
    "has_observation_gap",
    "primitive_op_invariants",
    "non_torch_primitive_op_inert",
    "primitive_op_schema_invalid",
    "primitive_op_fk_invalid",
    "forward",
    "backward",
    "setup",
    "none",
    "in_place",
    "out_variant",
    "metadata_only",
    "unknown",
    "view",
    "copy",
    "alias",
    "forward_op",
    "backward_grad_fn_call",
    "orphan",
    "unresolved",
    "linked",
    "unlinked",
    "conflict",
    "not_applicable",
    "returned",
    "raised",
    "formula_exact",
    "estimated",
    "unsupported",
    "all_present_same_schema",
    "all_present_different_schema",
    "sparse",
    "coverage_indeterminate",
    "forced_eager",
    "strict_subclass_constructor",
    "aten_<sequence>",
}


class _TwoOpModel(nn.Module):
    """Small deterministic model with two wrapped operation owners."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run add followed by relu."""

        return torch.relu(x + 1)


class _PlainSubclass(torch.Tensor):
    """Strict Tensor subclass whose constructor requires mode pausing."""


class _SubclassCtorModel(nn.Module):
    """Exercise the strict subclass-constructor pause bracket."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Construct a strict subclass and keep its value contribution zero."""

        scratch = _PlainSubclass(2, 3)
        return x + scratch.sum() * 0


class _SlashModuleModel(nn.Module):
    """Use a legal slash-containing ModuleDict address."""

    def __init__(self) -> None:
        """Install one module beneath a slash-containing key."""

        super().__init__()
        self.blocks = nn.ModuleDict({"a/b": nn.ReLU()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the slash-addressed child module."""

        return self.blocks["a/b"](x)


class _CopySlicesModel(nn.Module):
    """Create a CopySlices autograd node through indexed assignment."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assign one differentiable slice into a cloned tensor."""

        source = x * 2
        result = source.clone()
        result[:, :1] = source[:, 1:2]
        return result.sum()


class _InplaceViewModel(nn.Module):
    """Mutate a differentiable view in place."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply an in-place add through a view."""

        result = x.clone()
        result[:, :1].add_(1)
        return result.sum()


class _InplaceParameterSliceModel(nn.Module):
    """Mutate a parameter slice before using the parameter."""

    def __init__(self) -> None:
        """Create one small trainable matrix."""

        super().__init__()
        self.weight = nn.Parameter(torch.eye(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update one parameter slice without adding an autograd mutation edge."""

        self.weight.detach()[:1].add_(0.25)
        return (x @ self.weight).sum()


class _IndexPutModel(nn.Module):
    """Exercise explicit ``index_put_`` lineage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write a differentiable source through ``index_put_``."""

        result = x.clone()
        indices = torch.tensor([0], device=x.device)
        result.index_put_((indices,), x[:1] * 3)
        return result.sum()


@pytest.fixture
def _isolated_witness_mode() -> Iterator[None]:
    """Restore process-level wrapper diagnostic configuration after each test."""

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _armed_trace(*, backward_ready: bool = False) -> tl.Trace:
    """Return a private-recorder trace for the deterministic two-op model.

    Parameters
    ----------
    backward_ready
        Whether the capture must retain its autograd graph.

    Returns
    -------
    Trace
        Trace with a present primitive profile.
    """

    capture = tl.options.CaptureOptions(backward_ready=backward_ready)
    kwargs = {"save_mode": "reference"} if backward_ready else {}
    with _activate_aten_recording_for_tests():
        return tl.trace(
            _TwoOpModel(),
            torch.ones(2, requires_grad=backward_ready),
            capture=capture,
            **kwargs,
        )


def test_documented_unstable_aten_surface_matches_glossary_index() -> None:
    """Every introduced public spelling has the required unstable marker."""

    glossary = (Path(__file__).parents[1] / "docs/reference/glossary.md").read_text()
    indexed = glossary.split("<!-- ATEN-UNSTABLE-INDEX:START -->", 1)[1].split(
        "<!-- ATEN-UNSTABLE-INDEX:END -->", 1
    )[0]
    assert set(re.findall(r"`([^`]+)`", indexed)) == _DOCUMENTED_UNSTABLE_ATEN_TOKENS
    surface_rows = [line for line in indexed.splitlines() if line.startswith("|")][2:]
    assert surface_rows
    assert all("unstable -- no deprecation shim owed" in row for row in surface_rows)


def test_private_recorder_materializes_value_free_forward_rows() -> None:
    """Armed forward calls become opaque-label AtenOp rows with exact Op FKs."""

    trace = _armed_trace()
    profile = trace._primitive_op_profile

    assert profile is not None
    assert profile.primitive_ops
    assert all(isinstance(row, AtenOp) for row in profile.primitive_ops)
    assert all(row.label == f"aten_{row.sequence}" for row in profile.primitive_ops)
    assert all("/" not in row.label for row in profile.primitive_ops)
    assert any(row.owner_status == "forward_op" for row in profile.primitive_ops)
    assert all(
        not isinstance(fact, torch.Tensor)
        for row in profile.primitive_ops
        for facts in (row.input_tensor_facts, row.output_tensor_facts)
        for fact in facts
    )
    assert check_metadata_invariants(trace)


def test_recording_off_never_constructs_primitive_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary capture path never reaches the ATen retention constructor."""

    from torchlens.backends.torch import _aten_capture

    def fail_if_called(*args: object, **kwargs: object) -> None:
        """Fail if the gated retention path is called."""

        del args, kwargs
        raise AssertionError("ATen retention constructor ran while recording was off")

    monkeypatch.setattr(_aten_capture, "_prepare_aten_call", fail_if_called)
    trace = tl.trace(_TwoOpModel(), torch.ones(2))
    assert trace._primitive_op_profile is None


def test_mode_paused_interior_is_one_lower_bound_gap_not_a_synthetic_row() -> None:
    """Strict subclass construction records one typed gap with exact parentage."""

    with _activate_aten_recording_for_tests():
        trace = tl.trace(_SubclassCtorModel(), torch.ones(2, 3))
    profile = trace._primitive_op_profile

    assert len(profile.mode_paused_interior) == 1
    gap = profile.mode_paused_interior[0]
    assert gap.kind == "mode_paused_interior"
    assert gap.reason == "strict_subclass_constructor"
    assert gap.sequence_after == gap.sequence_before + 1
    assert gap.parent_op_refs
    assert all(row.sequence != gap.sequence_after for row in profile.primitive_ops)
    assert check_metadata_invariants(trace)


def test_backward_rows_resolve_to_grad_fn_calls() -> None:
    """The shared observer attributes engine dispatches to active GradFn calls."""

    trace = _armed_trace(backward_ready=True)
    with _activate_aten_recording_for_tests():
        trace.log_backward(trace.output_ops[0].out.sum())

    backward_rows = [
        row for row in trace._primitive_op_profile.primitive_ops if row.capture_phase == "backward"
    ]
    assert backward_rows
    assert any(row.owner_status == "backward_grad_fn_call" for row in backward_rows)
    assert all(row.backward_epoch_index == 1 for row in backward_rows)
    assert check_metadata_invariants(trace)


@pytest.mark.parametrize(
    ("model", "expected_operator"),
    [
        (_CopySlicesModel(), "copy_"),
        (_InplaceViewModel(), "add_"),
        (_InplaceParameterSliceModel(), "add_"),
        (_IndexPutModel(), "index_put_"),
    ],
)
def test_delayed_grad_fn_lineage_covers_required_mutation_breakers(
    model: nn.Module,
    expected_operator: str,
) -> None:
    """Delayed links remain typed across CopySlices and in-place/view cases."""

    capture = tl.options.CaptureOptions(backward_ready=True)
    with _activate_aten_recording_for_tests():
        trace = tl.trace(
            model,
            torch.ones(2, 3, requires_grad=True),
            capture=capture,
            save_mode="reference",
        )
        trace.log_backward(trace.output_ops[0].out)

    rows = trace._primitive_op_profile.primitive_ops
    assert any(row.operator == expected_operator for row in rows)
    linked = [row for row in rows if row.grad_fn_link_status == "linked"]
    assert linked
    assert all(row.grad_fn_ref is not None for row in linked)
    assert all(row.grad_fn_link_provenance in {"exact_via_aten", "heuristic"} for row in linked)
    _check_primitive_op_invariants(trace)


def test_primitive_rows_retain_no_tensor_grad_fn_or_callable_objects() -> None:
    """The value-free profile holds no strong runtime object references."""

    trace = _armed_trace(backward_ready=True)
    with _activate_aten_recording_for_tests():
        trace.log_backward(trace.output_ops[0].out.sum())

    def assert_value_free(value: object) -> None:
        """Recursively reject tensors, autograd nodes, and callables.

        Parameters
        ----------
        value
            Primitive field value to inspect.
        """

        assert not isinstance(value, torch.Tensor)
        assert not callable(value)
        if isinstance(value, tuple | list):
            for item in value:
                assert_value_free(item)
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                assert_value_free(getattr(value, name))

    for row in trace._primitive_op_profile.primitive_ops:
        for field_name in PRIMITIVE_OP_FIELD_ORDER:
            assert_value_free(getattr(row, field_name))


def test_retention_does_not_change_completeness_census_counts(
    _isolated_witness_mode: None,
) -> None:
    """The primitive tier leaves the independent completeness census unchanged."""

    del _isolated_witness_mode
    wrap_torch(completeness_witness=True)
    torch.manual_seed(7)
    plain = tl.trace(_TwoOpModel(), torch.randn(4))
    torch.manual_seed(7)
    with _activate_aten_recording_for_tests():
        armed = tl.trace(_TwoOpModel(), torch.randn(4))

    fields = (
        "completeness_witness_event_count",
        "completeness_witness_accounted_count",
        "completeness_witness_expected_opaque_count",
        "completeness_witness_unaccounted_count",
    )
    assert tuple(getattr(plain, name) for name in fields) == tuple(
        getattr(armed, name) for name in fields
    )


def test_slash_module_address_does_not_enter_parent_or_opaque_aten_labels() -> None:
    """Slash remains legal in module addresses and absent from wave-0 parent labels."""

    with _activate_aten_recording_for_tests():
        trace = tl.trace(_SlashModuleModel(), torch.ones(2))

    module_addresses = list(trace.modules.keys())
    assert any("a/b" in address for address in module_addresses)
    for row in trace._primitive_op_profile.primitive_ops:
        assert "/" not in row.label
        assert all("/" not in ref.op_label for ref in row.parent_op_refs)


def test_plain_v8_round_trip_persists_the_profile(tmp_path: Path) -> None:
    """tlspec v8: the primitive profile persists on a PLAIN save/load."""

    trace = _armed_trace()
    default_path = tmp_path / "default.tlspec"
    tl.save(trace, default_path)
    loaded = tl.load(default_path)
    assert len(loaded._primitive_op_profile.primitive_ops) == len(
        trace._primitive_op_profile.primitive_ops
    )
    assert check_metadata_invariants(loaded)


@pytest.mark.parametrize("tamper_kind", ["dangling", "forged"])
def test_prerelease_load_refuses_dangling_and_forged_op_fks(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    """Switch-active loads reject both out-of-range and valid-row forged FKs."""

    trace = _armed_trace()
    owned_rows = [row for row in trace._primitive_op_profile.primitive_ops if row.parent_op_refs]
    row = owned_rows[0]
    ref = row.parent_op_refs[0]
    if tamper_kind == "dangling":
        replacement = OpRef(
            op_row_index=len(trace.ops) + 10,
            op_label=ref.op_label,
            func_call_id=ref.func_call_id,
        )
    else:
        target = next(op_ref for other in owned_rows[1:] for op_ref in other.parent_op_refs)
        replacement = OpRef(
            op_row_index=target.op_row_index,
            op_label=target.op_label,
            func_call_id=target.func_call_id,
        )
        row.owner_func_call_id = target.func_call_id
        evidence = dict(trace._primitive_op_profile._event_owner_evidence)
        evidence[row.sequence] = target.func_call_id
        trace._primitive_op_profile._event_owner_evidence = tuple(sorted(evidence.items()))
    row.parent_op_refs = (replacement,)
    path = tmp_path / f"{tamper_kind}.tlspec"
    with activate_prerelease_fields():
        tl.save(trace, path)
        with pytest.raises(TorchLensIOError) as exc_info:
            tl.load(path)
    assert exc_info.value.fields["code"] == "primitive_op_fk_invalid"


def test_live_fk_tamper_names_primitive_invariant() -> None:
    """In-memory FK corruption reports the owning invariant contract."""

    trace = _armed_trace()
    row = next(row for row in trace._primitive_op_profile.primitive_ops if row.parent_op_refs)
    ref = row.parent_op_refs[0]
    row.parent_op_refs = (
        OpRef(
            op_row_index=len(trace.ops) + 1,
            op_label=ref.op_label,
            func_call_id=ref.func_call_id,
        ),
    )

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "primitive_op_invariants"


@pytest.mark.parametrize(
    ("members", "member_names", "incomplete", "expected"),
    [
        ({"a": None, "b": None}, ["a", "b"], False, "all_present_same_schema"),
        ({"a": None}, ["a", "b"], False, "sparse"),
        ({"a": None}, ["a", "b"], True, "coverage_indeterminate"),
    ],
)
def test_super_aten_op_alignment_statuses(
    members: dict[str, None],
    member_names: list[str],
    incomplete: bool,
    expected: str,
) -> None:
    """SuperAtenOp preserves positional absence and coverage uncertainty."""

    trace = _armed_trace()
    row = trace._primitive_op_profile.primitive_ops[0]
    concrete = dict.fromkeys(members, row)
    super_row = SuperAtenOp(
        label="slot_0",
        decomposition_slot=0,
        members=concrete,
        bundle_member_names=member_names,
        has_observation_gap=incomplete,
    )
    assert super_row.comparison_status == expected
