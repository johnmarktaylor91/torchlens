"""Phase 4 capture-unification retroactive-save regression tests."""

from __future__ import annotations

import warnings
from collections import defaultdict, namedtuple
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import get_tensor_label, set_tensor_label
from torchlens.backends.torch.ops import _get_parent_output_version_snapshot
from torchlens.utils.arg_handling import copy_arg_tree
from torchlens.utils.tensor_utils import _clone_tensor_payload, copy_tensor_payload


class PartialConvRelu(nn.Module):
    """Conv model where only one conv output feeds a relu."""

    def __init__(self) -> None:
        """Initialize the two conv branches."""

        super().__init__()
        self.conv_relu = nn.Conv2d(1, 1, 1)
        self.conv_plain = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one relu-fed conv and one plain conv."""

        relu_parent = self.conv_relu(x)
        plain = self.conv_plain(x)
        return torch.relu(relu_parent) + plain


class ReusedConvRelu(nn.Module):
    """Conv output consumed by multiple relu successors."""

    def __init__(self) -> None:
        """Initialize the shared conv."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed one conv output to two relus."""

        conv = self.conv(x)
        return torch.relu(conv) + torch.relu(conv)


class ConvWithoutRelu(nn.Module):
    """Conv model with no matching successor."""

    def __init__(self) -> None:
        """Initialize the conv."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conv without a relu successor."""

        return self.conv(x) + 1


class EvictedParentRelu(nn.Module):
    """Conv parent is evicted before its relu successor appears."""

    def __init__(self) -> None:
        """Initialize the conv."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Insert an unrelated op between the conv and relu."""

        conv = self.conv(x)
        _ = conv + 1
        return torch.relu(conv)


class BranchyAddParents(nn.Module):
    """Two conv parents feed one branchy successor."""

    def __init__(self) -> None:
        """Initialize both conv branches."""

        super().__init__()
        self.left = nn.Conv2d(1, 1, 1)
        self.right = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed two matching conv parents into one add successor."""

        return self.left(x) + self.right(x)


class CandidateEvictionModel(nn.Module):
    """Model that creates more candidates than the lookback payload window."""

    def __init__(self) -> None:
        """Initialize two conv candidates."""

        super().__init__()
        self.first = nn.Conv2d(1, 1, 1)
        self.second = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two conv candidates without a matching successor."""

        return self.first(x) + self.second(x)


class CopyDuringLoggingModel(nn.Module):
    """Model that calls copy helpers while TorchLens logging is active."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy an intermediate through both policies and return a simple output."""

        y = x + 1
        _ = copy_tensor_payload(y, save_mode="copy", detach_tensor=True)
        _ = copy_arg_tree({"nested": [y]})
        return y * 2


class ForeachNestedMutationModel(nn.Module):
    """Model that mutates a tensor nested inside a list argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate a nested tensor argument in-place through a foreach op."""

        y = x + 1
        items = [y]
        torch._foreach_add_(items, 3.0)
        return y * 2


class CustomTensorBox:
    """Custom wrapper used to verify copy_arg_tree preserves object identity."""

    tensor: torch.Tensor


class VersionSnapshotTraceStub:
    """Minimal trace stub for version-snapshot helper tests."""

    save_arg_values = True


def _conv_ops(log: tl.Trace) -> list[tl.Op]:
    """Return conv ops in execution order."""

    return [op for op in log.layer_list if op.layer_type == "conv2d"]


def _saved_conv_ops(log: tl.Trace) -> list[tl.Op]:
    """Return saved conv ops in execution order."""

    return [op for op in _conv_ops(log) if op.has_saved_activation]


def _retro_trace(model: nn.Module, x: torch.Tensor, successor: object) -> tl.Trace:
    """Run a retroactive conv save trace."""

    return tl.trace(
        model,
        x,
        save=tl.func("conv2d") & tl.followed_by(successor),  # type: ignore[arg-type]
        lookback=4,
        lookback_payload_policy="detached_raw",
        random_seed=123,
    )


def _assert_tensor_payload_matches(source: torch.Tensor, copied: torch.Tensor) -> None:
    """Assert copied tensor metadata and values match a source tensor where data exists."""

    assert copied.shape == source.shape
    assert copied.dtype == source.dtype
    if source.is_meta:
        assert copied.is_meta
    elif source.is_sparse:
        source_coalesced = source.coalesce()
        copied_coalesced = copied.coalesce()
        assert torch.equal(copied_coalesced.indices(), source_coalesced.indices())
        assert torch.equal(copied_coalesced.values(), source_coalesced.values())
    elif source.is_quantized:
        assert copied.qscheme() == source.qscheme()
        assert torch.equal(copied.int_repr(), source.int_repr())
    else:
        assert torch.equal(copied, source)


def _tensor_zoo() -> list[tuple[str, torch.Tensor]]:
    """Return representative tensors for copy policy coverage."""

    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([1.0, 2.0]),
        (2, 2),
    )
    return [
        ("dense", torch.randn(2, 3)),
        ("sparse", sparse),
        ("meta", torch.empty(2, 3, device="meta")),
        ("quantized", torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), 0.1, 10, torch.quint8)),
        ("complex", torch.tensor([1 + 2j, 3 - 4j])),
    ]


def test_copy_policies_pause_logging_for_internal_tensor_ops() -> None:
    """Copy helpers called during active logging do not create spurious log entries."""

    log = tl.trace(
        CopyDuringLoggingModel(),
        torch.ones(2),
        layers_to_save="all",
        save_arg_values=True,
        random_seed=123,
    )

    layer_types = [op.layer_type for op in log.layer_list]
    assert layer_types == ["input", "add", "mul", "output"]


@pytest.mark.parametrize("save_mode", ["copy", "reference", "view", "cpu_async"])
def test_copy_tensor_payload_tensor_zoo(save_mode: str) -> None:
    """copy_tensor_payload handles representative tensor kinds under each save mode."""

    for tensor_name, tensor in _tensor_zoo():
        if tensor_name == "meta" and save_mode == "cpu_async":
            with pytest.raises(NotImplementedError, match="Cannot copy out of meta tensor"):
                copy_tensor_payload(tensor, save_mode=save_mode, detach_tensor=True)
            continue
        copied = copy_tensor_payload(tensor, save_mode=save_mode, detach_tensor=True)
        assert isinstance(copied, torch.Tensor)
        _assert_tensor_payload_matches(tensor, copied)


def test_copy_policies_preserve_raw_labels_and_parameters() -> None:
    """Tensor copies preserve raw labels, with input-specific parameter downgrading."""

    tensor = torch.randn(2, 2)
    set_tensor_label(tensor, "raw_label_for_copy")
    parameter = nn.Parameter(torch.ones(2))

    payload_copy = copy_tensor_payload(tensor, save_mode="copy", detach_tensor=True)
    arg_copy = copy_arg_tree({"x": [tensor]})["x"][0]
    input_parameter_copy = copy_arg_tree(parameter)
    snapshot_parameter_copy = _clone_tensor_payload(
        parameter,
        detach_tensor=False,
        save_mode="copy",
    )

    assert get_tensor_label(payload_copy) == "raw_label_for_copy"
    assert get_tensor_label(arg_copy) == "raw_label_for_copy"
    assert isinstance(input_parameter_copy, torch.Tensor)
    assert not isinstance(input_parameter_copy, nn.Parameter)
    assert input_parameter_copy.requires_grad
    assert isinstance(snapshot_parameter_copy, nn.Parameter)


def test_copy_arg_tree_recurses_builtin_containers_and_preserves_custom_identity() -> None:
    """copy_arg_tree clones nested tensors but keeps custom objects by reference."""

    Pair = namedtuple("Pair", ["left", "right"])
    tensor = torch.tensor([1.0])
    custom = CustomTensorBox()
    custom.tensor = tensor
    value: dict[str, Any] = {
        "list": [tensor],
        "tuple": (tensor,),
        "namedtuple": Pair(tensor, 3),
        "defaultdict": defaultdict(list, {"tensor": tensor}),
        "custom": custom,
    }

    copied = copy_arg_tree(value)

    assert copied is not value
    assert copied["list"][0] is not tensor
    assert copied["tuple"][0] is not tensor
    assert copied["namedtuple"].left is not tensor
    assert isinstance(copied["namedtuple"], Pair)
    assert isinstance(copied["defaultdict"], defaultdict)
    assert copied["defaultdict"].default_factory is list
    assert copied["defaultdict"]["tensor"] is not tensor
    assert copied["custom"] is custom


def test_nested_container_inplace_arg_snapshot_preserves_pre_call_value() -> None:
    """Wrapper snapshots clone nested tensor args before an in-place mutation runs."""

    log = tl.trace(
        ForeachNestedMutationModel(),
        torch.ones(2),
        layers_to_save="all",
        save_arg_values=True,
        random_seed=123,
    )

    foreach = log["foreachadd_1_2"]
    assert torch.equal(foreach.saved_args[0][0], torch.full((2,), 2.0))
    assert torch.equal(foreach.out, torch.full((2,), 5.0))


@pytest.mark.parametrize(
    ("args", "kwargs", "parent_positions", "contract_positions"),
    [
        (([torch.full((2,), 2.0)],), {}, {"args": {(0, 0): "parent"}, "kwargs": {}}, ((0, 0),)),
        (
            (),
            {"payload": {"x": torch.full((2,), 2.0)}},
            {"args": {}, "kwargs": {("payload", "x"): "parent"}},
            (("payload", "x"),),
        ),
    ],
)
def test_nested_container_version_snapshot_preserves_pre_call_value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parent_positions: dict[str, dict[Any, str]],
    contract_positions: tuple[object, ...],
) -> None:
    """Version capture reads nested pre-call copies after live tensors mutate."""

    arg_copies = tuple(copy_arg_tree(arg) for arg in args)
    kwarg_copies = {key: copy_arg_tree(value) for key, value in kwargs.items()}
    live_tensor = args[0][0] if args else kwargs["payload"]["x"]
    live_tensor.add_(3.0)

    snapshot = _get_parent_output_version_snapshot(
        VersionSnapshotTraceStub(),  # type: ignore[arg-type]
        "parent",
        parent_positions,
        contract_positions,
        (),
        False,
        None,
        arg_copies,
        kwarg_copies,
    )

    assert torch.equal(snapshot, torch.full((2,), 2.0))


def test_followed_by_saves_only_conv_feeding_relu_with_matching_payload() -> None:
    """followed_by saves exactly relu-feeding conv payloads."""

    model = PartialConvRelu()
    x = torch.randn(1, 1, 4, 4)
    full = tl.trace(model, x.clone(), layers_to_save="all", random_seed=123)
    log = _retro_trace(model, x.clone(), tl.func("relu"))
    saved = _saved_conv_ops(log)
    assert [op.label for op in saved] == [_conv_ops(log)[0].label]
    assert torch.equal(saved[0].out, full[saved[0].label].out)
    assert not _conv_ops(log)[1].has_saved_activation
    with pytest.raises(ValueError, match="not saved"):
        _ = _conv_ops(log)[1].out


def test_followed_by_multiple_successors_are_idempotent() -> None:
    """Multiple matching successors mark the same retained parent once."""

    log = _retro_trace(ReusedConvRelu(), torch.randn(1, 1, 4, 4), tl.func("relu"))
    saved = _saved_conv_ops(log)
    assert len(saved) == 1
    assert saved[0].has_saved_activation


def test_followed_by_never_appears_does_not_save_candidate() -> None:
    """Candidates are dropped unsaved when no matching successor appears."""

    log = _retro_trace(ConvWithoutRelu(), torch.randn(1, 1, 4, 4), tl.func("relu"))
    assert _saved_conv_ops(log) == []


def test_followed_by_out_of_window_parent_warns() -> None:
    """A parent evicted before its successor emits an explicit warning."""

    with pytest.warns(RuntimeWarning, match="outside the lookback window"):
        log = tl.trace(
            EvictedParentRelu(),
            torch.randn(1, 1, 4, 4),
            save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
            lookback=1,
            lookback_payload_policy="detached_raw",
            random_seed=123,
        )
    assert _saved_conv_ops(log) == []


def test_followed_by_branchy_successor_marks_all_matching_parents() -> None:
    """A multi-parent successor marks all matching parents in the window."""

    log = _retro_trace(BranchyAddParents(), torch.randn(1, 1, 4, 4), tl.func("add"))
    saved = _saved_conv_ops(log)
    assert [op.label for op in saved] == [op.label for op in _conv_ops(log)]


def test_lookback_payload_window_is_bounded_and_evicts_candidates() -> None:
    """Payload retention holds no more than K candidates and evicts older ones."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log = tl.trace(
            CandidateEvictionModel(),
            torch.randn(1, 1, 4, 4),
            save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
            lookback=1,
            lookback_payload_policy="detached_raw",
            random_seed=123,
        )
    candidates = getattr(log, "_predicate_lookback_candidates")
    assert len(candidates) <= 1
    assert tuple(candidate.raw_label for candidate in candidates) == (_conv_ops(log)[1]._label_raw,)
    assert _saved_conv_ops(log) == []
