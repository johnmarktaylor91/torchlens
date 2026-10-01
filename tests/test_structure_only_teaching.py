"""Teaching-refusal battery for structure-only capture (L7a memo sec 2/9.3).

Covers: provocations per consumer kind with exact user source lines,
DEVICE-NEUTRAL refusals (the sol r2 B1 contract — real tensors refuse like
meta ones), the lifted float-truthiness gate (in-mode only), the
missing-meta-kernel backstop with the delegated-helper exact-line pin,
user-exception honesty (C-HONESTY), per-belt immunizer liveness (a dead layer
must be caught by the other; both dead proves the tests measure the belts),
and the five-set belt-census meta-test with an install census.
"""

from __future__ import annotations

import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.structure_only import (
    MetaKernelUnavailableError,
    ValueDependentBranchError,
)
from torchlens.options import CaptureOptions
from torchlens.utils._torch_compat import get_meta_item_guard_support

smoke = pytest.mark.smoke

STRUCTURE = CaptureOptions(structure_only=True)


def _structure_trace(model: nn.Module, x: torch.Tensor):
    return tl.trace(model, x, capture=CaptureOptions(structure_only=True))


# ---------------------------------------------------------------------------
# Provocations per consumer kind (memo 2.3 closed vocabulary)
# ---------------------------------------------------------------------------
# Module-level models so ast_branches can classify (one arm per line).


class IfBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        if h.sum() > 0:
            return torch.relu(h)
        return h


class ElifBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        if x.shape[0] == 0:
            return h
        elif h.mean() > 0:
            return torch.relu(h)
        return h


class TernaryBranch(nn.Module):
    def forward(self, x):
        return torch.relu(x) if x.sum().item() > 0 else x


class WhileBranch(nn.Module):
    def forward(self, x):
        while x.norm() > 1e6:
            x = x / 2
        return x * 2


class AssertBranch(nn.Module):
    def forward(self, x):
        assert torch.isfinite(x).all()
        return x * 2


class BoolCast(nn.Module):
    def forward(self, x):
        keep = bool((x > 0).any())
        return x * 2 if keep else x


class BareItem(nn.Module):
    def forward(self, x):
        scale = x.abs().max().item()
        return x / (scale + 1.0)


class FloatTruthiness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        if h.mean():
            return torch.relu(h)
        return h


_CONSUMER_CASES = [
    (IfBranch, "if_test"),
    (ElifBranch, "elif_test"),
    (TernaryBranch, "ifexp"),
    (WhileBranch, "while"),
    (AssertBranch, "assert"),
    (BoolCast, "bool_cast"),
    (BareItem, "scalar_escape"),
]


@pytest.mark.parametrize(("model_cls", "expected_kind"), _CONSUMER_CASES)
@smoke
def test_value_branch_refuses_with_consumer_kind_and_user_line(
    model_cls: type, expected_kind: str
) -> None:
    with pytest.raises(ValueDependentBranchError) as excinfo:
        _structure_trace(model_cls(), torch.randn(2, 4))
    err = excinfo.value
    assert err.fields["code"] == "value_dependent_branch_unsupported"
    assert err.fields["consumer_kind"] == expected_kind
    # The exact user source line: the offense points into THIS file.
    assert err.file_path == __file__
    assert isinstance(err.line_no, int) and err.line_no > 0
    offense = err.fields["offenses"][0]
    assert offense["file"] == __file__
    assert offense["escape_method"] in {
        "item",
        "__bool__",
        "__int__",
        "__float__",
        "equal",
        "allclose",
        "is_nonzero",
    } or offense["escape_method"].startswith("torch.")


@smoke
def test_device_neutral_real_tensor_refusal_carries_substrate_real() -> None:
    """sol r2 B1: a REAL tensor driving a value branch refuses identically —
    a value-selected graph must never launder as structure-only."""

    with pytest.raises(ValueDependentBranchError) as excinfo:
        _structure_trace(IfBranch(), torch.randn(2, 4))
    assert excinfo.value.fields["offenses"][0]["substrate"] == "real"


class MetaContextBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        with torch.device("meta"):
            mask = torch.ones(4, 4)
        if mask.sum() > 0:
            return torch.relu(h)
        return h


@smoke
def test_meta_tensor_refusal_carries_substrate_meta() -> None:
    with pytest.raises(ValueDependentBranchError) as excinfo:
        _structure_trace(MetaContextBranch(), torch.randn(2, 4))
    assert excinfo.value.fields["offenses"][0]["substrate"] == "meta"


@smoke
def test_float_truthiness_gate_is_lifted_in_mode_only() -> None:
    """The dtype-is-bool gate stays a documented false negative on the
    DEFAULT path (its own pinning test governs); in-mode the escalated belt
    refuses float truthiness with the same teaching quality."""

    with pytest.raises(ValueDependentBranchError) as excinfo:
        _structure_trace(FloatTruthiness(), torch.randn(2, 4))
    assert excinfo.value.fields["consumer_kind"] == "if_test"
    # Companion: the default path still captures this model without raising.
    log = tl.trace(FloatTruthiness(), torch.randn(2, 4))
    assert log.structure_only is False


@smoke
def test_internal_frame_reads_do_not_trip_the_belt() -> None:
    """PROVENANCE carve: torchlens internals touch real-tensor storage during
    an E-1 capture (aliasing/dedup/hashing) with the belt armed — a
    value-branch-free capture must complete."""

    class Clean(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return torch.relu(self.fc(x))

    log = _structure_trace(Clean(), torch.randn(2, 4))
    assert log.structure_only is True
    assert len(log.layer_list) >= 3


# ---------------------------------------------------------------------------
# Missing meta kernels (memo 2.4) + delegated-helper exact-line pin
# ---------------------------------------------------------------------------


class DelegatedHelperKernel(nn.Module):
    def forward(self, x):
        with torch.device("meta"):
            m = torch.ones(5, dtype=torch.long)

        def helper(t):
            return torch.bincount(t)  # HELPER_LINE anchor

        return helper(m), x


_HELPER_LINE = next(
    offset
    for offset, line in enumerate(Path(__file__).read_text(encoding="utf-8").splitlines(), start=1)
    if "HELPER_LINE anchor" in line
)


@smoke
def test_missing_meta_kernel_refuses_typed_at_the_helper_line() -> None:
    """sol r2 M2: the traceback walk takes the INNERMOST external frame — the
    helper's failing line, never the outer forward callsite."""

    with pytest.raises(MetaKernelUnavailableError) as excinfo:
        _structure_trace(DelegatedHelperKernel(), torch.randn(2))
    err = excinfo.value
    assert err.fields["code"] == "meta_kernel_unavailable"
    assert err.file_path == __file__
    assert err.line_no == _HELPER_LINE
    assert isinstance(err.__cause__, NotImplementedError)
    assert err.fields.get("entry_frame", "").startswith(__file__)


class UserRaisesNIE(nn.Module):
    def forward(self, x):
        raise NotImplementedError("user's own refusal")


@smoke
def test_user_raised_notimplementederror_propagates_unchanged() -> None:
    """C-HONESTY: user exceptions stay user exceptions — annotated (3.11+)
    or warned (3.10), never wrapped."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match="user's own refusal") as excinfo:
            _structure_trace(UserRaisesNIE(), torch.randn(2))
    if sys.version_info >= (3, 11):
        notes = getattr(excinfo.value, "__notes__", [])
        assert any("structure-only" in note for note in notes)
    else:
        assert any("structure-only" in str(w.message) for w in caught)


class UnenumeratedMetaDeath(nn.Module):
    def forward(self, x):
        with torch.device("meta"):
            m = torch.ones(3)
        return torch.tensor(m.tolist()), x


@smoke
def test_backstop_types_meta_deaths_the_belt_enumerates_or_not() -> None:
    """An unenumerated meta-mechanism death is typed via the backstop; the
    enumerated tolist spelling is caught by Layer 1 first (belt precedence).
    Either way the failure is TYPED, never torch's bare error."""

    with pytest.raises((ValueDependentBranchError, MetaKernelUnavailableError)):
        _structure_trace(UnenumeratedMetaDeath(), torch.randn(2))


# ---------------------------------------------------------------------------
# Per-belt immunizer liveness (memo 2.5; opus B3d)
# ---------------------------------------------------------------------------


def _neutralize_layer1(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens.backends.torch import backend as backend_module

    monkeypatch.setattr(backend_module, "structure_only_escape_belt", lambda trace: nullcontext())


def _neutralize_layer2(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens import capture as capture_pkg  # noqa: F401
    from torchlens.capture import trace as capture_trace_module

    monkeypatch.setattr(
        capture_trace_module,
        "_structure_only_forward_boundary",
        lambda trace: nullcontext(),
    )


@smoke
@pytest.mark.skipif(
    not get_meta_item_guard_support(),
    reason="torch 2.1-2.2 have no Tensor.item()-cannot-be-called-on-meta-tensors guard: "
    "the same MetaContextBranch forward instead dies in the aten dispatcher with a "
    "NotImplementedError, which layer 2 classifies as the sibling "
    "MetaKernelUnavailableError (meta_kernel_unavailable), not ValueDependentBranchError",
)
def test_layer2_backstop_catches_when_layer1_is_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _neutralize_layer1(monkeypatch)
    with pytest.raises(ValueDependentBranchError) as excinfo:
        _structure_trace(MetaContextBranch(), torch.randn(2, 4))
    assert excinfo.value.fields["consumer_kind"] == "unclassified_escape"


@smoke
def test_layer1_belt_catches_when_layer2_is_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _neutralize_layer2(monkeypatch)
    with pytest.raises(ValueDependentBranchError) as excinfo:
        _structure_trace(MetaContextBranch(), torch.randn(2, 4))
    assert excinfo.value.fields["consumer_kind"] == "if_test"


@smoke
def test_both_layers_dead_returns_the_bare_torch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the liveness tests measure the belts, not incidental behavior."""

    _neutralize_layer1(monkeypatch)
    _neutralize_layer2(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        _structure_trace(MetaContextBranch(), torch.randn(2, 4))
    assert not isinstance(excinfo.value, ValueDependentBranchError)


# ---------------------------------------------------------------------------
# Belt census (memo 2.5): the surface IS the five-set union, and every
# member is actually escalated in-mode (install census, not set equality).
# ---------------------------------------------------------------------------


@smoke
def test_escape_surface_is_the_union_of_all_five_constituents() -> None:
    from torchlens.backends.torch.completeness_witness import (
        HOST_VALUE_ESCAPE_METHODS,
        HOST_VALUE_ESCAPE_MODULE_FUNCS,
        INVISIBLE_HOST_ESCAPE_FUNCS,
        INVISIBLE_HOST_ESCAPE_PROPERTIES,
        STORAGE_BRIDGE_ESCAPE_FUNCS,
    )
    from torchlens.backends.torch.structure_only_belt import (
        STRUCTURE_ONLY_ESCAPE_SURFACE,
    )

    assert STRUCTURE_ONLY_ESCAPE_SURFACE == (
        HOST_VALUE_ESCAPE_METHODS
        | HOST_VALUE_ESCAPE_MODULE_FUNCS
        | INVISIBLE_HOST_ESCAPE_FUNCS
        | STORAGE_BRIDGE_ESCAPE_FUNCS
        | INVISIBLE_HOST_ESCAPE_PROPERTIES
    ), (
        "the mode belt's surface drifted from the five constituent escape "
        "sets — when another lane adds an escape spelling, the mode's "
        "coverage must grow with it, never silently shrink into the backstop"
    )


@smoke
def test_install_census_every_surface_member_is_escalated_in_mode() -> None:
    import inspect as inspect_module

    from torchlens.backends.torch import structure_only_belt as belt

    class _FakeTrace:
        structure_only = True

    tensor_originals = {
        name: getattr(torch.Tensor, name, None) for name in belt._TENSOR_METHOD_SURFACE
    }
    module_originals = {
        name: getattr(torch, name, None) for name in belt.HOST_VALUE_ESCAPE_MODULE_FUNCS
    }
    with belt.structure_only_escape_belt(_FakeTrace()):
        for name, original in sorted(tensor_originals.items()):
            if original is None or not callable(original):
                continue
            assert getattr(torch.Tensor, name) is not original, name
        for name, original in sorted(module_originals.items()):
            if original is None or not callable(original):
                continue
            assert getattr(torch, name) is not original, f"torch.{name}"
        for name in belt.INVISIBLE_HOST_ESCAPE_PROPERTIES:
            descriptor = inspect_module.getattr_static(torch.Tensor, name, None)
            assert isinstance(descriptor, property), name
    # Shadow-aware restore: every patch is gone after exit.
    for name, original in tensor_originals.items():
        assert getattr(torch.Tensor, name, None) is original, name
    for name, original in module_originals.items():
        assert getattr(torch, name, None) is original, f"torch.{name}"


@smoke
def test_teaching_refusals_fire_identically_under_predicate_composition() -> None:
    """Memo 2.5 combination row: composing the mode with a (value-free)
    predicate surface must not soften the belts."""

    with pytest.raises(ValueDependentBranchError) as excinfo:
        tl.trace(
            IfBranch(),
            torch.randn(2, 4),
            capture=CaptureOptions(structure_only=True),
            halt=tl.func("nonexistent_op_name"),
        )
    assert excinfo.value.fields["consumer_kind"] == "if_test"
    assert excinfo.value.fields["code"] == "value_dependent_branch_unsupported"
