"""Index-domain perturbations stay inside ``[0, domain)`` on every device.

Forward validation proves each saved parent influences its op by perturbing
it. For a parent in an index/target slot (gather, index_select, the scatter
family, embedding, cross_entropy) a uniform value step moves the extreme index
out of the domain: on CPU that raised and was absorbed, proving nothing, and on
CUDA it is a fatal device-side assert (V-JEPA 2 repro). These tests pin that
every index fed to a replayed index-domain op is in range, that the index edge
is proven by an in-domain move rather than an absorbed exception, and that a
corrupted saved index or output still fails validation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.validation import core
from torchlens.validation._index_domain import (
    _INDEX_DOMAIN_ARG_SPECS,
    _index_domain_size,
    _parent_is_index_domain_arg,
)


class GatherByIndex(nn.Module):
    """Gather projected rows by an index computed in the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = torch.argsort(x.sum(-1), dim=1)
        return torch.gather(self.proj(x), 1, index.unsqueeze(-1).expand(-1, -1, 4))


class NllByTarget(nn.Module):
    """nll_loss whose target is computed in the forward (5 classes, targets in [0, 4))."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 5)
        self.ignore_index = ignore_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = torch.argmax(x, dim=-1)
        log_probs = torch.log_softmax(self.proj(x), dim=-1)
        return F.nll_loss(log_probs, target, ignore_index=self.ignore_index)


class TakeAlongDim(nn.Module):
    """take_along_dim with sort indices computed in the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.take_along_dim(self.proj(x), torch.argsort(x, dim=1), dim=1)


class BoolMaskSelect(nn.Module):
    """Boolean-mask indexing: a mask is not an index domain."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)[x.sum(-1) > 0]


#: A gather whose source is constant along the gather dim reads the same values
#: for any in-range index; the pre-existing exemption names that case exactly.
_VJEPA2_EXEMPTIONS = frozenset({"index_domain_value_irrelevant"})


def _tiny_vjepa2(frames: int = 4):
    transformers = pytest.importorskip("transformers")
    config = transformers.VJEPA2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        pred_hidden_size=32,
        pred_num_hidden_layers=2,
        pred_num_attention_heads=4,
        crop_size=32,
        image_size=32,
        frames_per_clip=frames,
        patch_size=16,
        tubelet_size=2,
        mlp_ratio=2,
        pred_mlp_ratio=2.0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return transformers.VJEPA2Model(config).eval()


def _index_arg(layer, input_args: dict) -> torch.Tensor | None:
    position, kwarg_names = _INDEX_DOMAIN_ARG_SPECS[layer.func_name]
    args, kwargs = input_args.get("args", ()), input_args.get("kwargs", {}) or {}
    if len(args) > position and isinstance(args[position], torch.Tensor):
        return args[position]
    return next((kwargs[n] for n in kwarg_names if isinstance(kwargs.get(n), torch.Tensor)), None)


@pytest.fixture
def replay_record(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    """Record every replayed index-domain call and every index-edge verdict."""

    record: dict[str, list[dict]] = {"calls": [], "verdicts": []}
    execute = core._execute_func_with_restored_state
    check = core._check_whether_func_on_saved_parents_yields_saved_tensor

    def spy_execute(layer, input_args, layers_to_perturb, layer_label, verbose):
        output = execute(layer, input_args, layers_to_perturb, layer_label, verbose)
        if getattr(layer, "func_name", None) in _INDEX_DOMAIN_ARG_SPECS:
            index, size = _index_arg(layer, input_args), _index_domain_size(layer)
            ignore = (layer.saved_kwargs or {}).get("ignore_index")
            saved = _index_arg(
                layer, {"args": layer.saved_args or (), "kwargs": layer.saved_kwargs or {}}
            )
            ignore_kept = True
            if isinstance(ignore, int) and index is not None and saved is not None:
                ignore_kept = bool(torch.equal(index == ignore, saved == ignore))
            record["calls"].append(
                {
                    "ignore_kept": ignore_kept,
                    "func": layer.func_name,
                    "perturbs_index": any(
                        _parent_is_index_domain_arg(layer, label) for label in layers_to_perturb
                    ),
                    "in_range": index is None
                    or size is None
                    or bool(((index >= 0) & (index < size)).all()),
                    "raised": output is None,
                }
            )
        return output

    def spy_check(self, label, *args, **kwargs):
        result = check(self, label, *args, **kwargs)
        layer = core._op_for_validation_label(self, label)
        labels = kwargs.get("layers_to_perturb") or ()
        if any(_parent_is_index_domain_arg(layer, label) for label in labels):
            record["verdicts"].append((layer.func_name, result.decision, result.reason))
        return result

    monkeypatch.setattr(core, "_execute_func_with_restored_state", spy_execute)
    monkeypatch.setattr(core, "_check_whether_func_on_saved_parents_yields_saved_tensor", spy_check)
    return record


def _assert_index_edges_proven_in_domain(
    record: dict[str, list[dict]], allowed_exemptions: frozenset[str] = frozenset()
) -> None:
    index_calls = [call for call in record["calls"] if call["perturbs_index"]]
    assert index_calls, "the index edge was never perturbed: the instrument saw nothing"
    assert all(call["in_range"] for call in record["calls"]), [
        call for call in record["calls"] if not call["in_range"]
    ]
    assert not any(call["raised"] for call in index_calls), "an index perturbation raised"
    assert record["verdicts"], "no index-edge verdict was recorded"
    assert any(decision == "validated" for _, decision, _ in record["verdicts"])
    assert all(
        decision == "validated" or (decision == "exempted" and reason in allowed_exemptions)
        for _, decision, reason in record["verdicts"]
    ), record["verdicts"]


def test_gather_index_edge_is_proven_in_domain(replay_record) -> None:
    torch.manual_seed(0)
    model = GatherByIndex().eval()
    assert tl.validate(model, torch.randn(2, 6, 4), scope="forward") is True
    _assert_index_edges_proven_in_domain(replay_record)


def test_tiny_vjepa2_validates_with_in_domain_index_perturbations(replay_record) -> None:
    model = _tiny_vjepa2()
    inputs = {"pixel_values_videos": torch.randn(1, 4, 3, 32, 32)}
    assert tl.validate(model, (), inputs, scope="forward") is True
    _assert_index_edges_proven_in_domain(replay_record, _VJEPA2_EXEMPTIONS)
    assert any(call["func"] == "gather" for call in replay_record["calls"])


@pytest.mark.parametrize(
    ("model_factory", "func"),
    [
        (NllByTarget, "nll_loss"),
        (lambda: NllByTarget(ignore_index=1), "nll_loss"),
        (TakeAlongDim, "take_along_dim"),
    ],
    ids=["nll_loss", "nll_loss-ignore_index-in-domain", "take_along_dim"],
)
def test_registered_index_op_is_proven_in_domain(replay_record, model_factory, func) -> None:
    torch.manual_seed(0)
    model = model_factory().eval()
    assert tl.validate(model, torch.randn(6, 4), scope="forward") is True
    _assert_index_edges_proven_in_domain(replay_record)
    assert any(call["func"] == func for call in replay_record["calls"]), "op not registered"
    assert all(call["ignore_kept"] for call in replay_record["calls"]), (
        "a target moved onto or off ignore_index"
    )


@pytest.mark.parametrize(
    ("model_factory", "index_func"),
    [(NllByTarget, "argmax"), (TakeAlongDim, "argsort")],
    ids=["nll_loss", "take_along_dim"],
)
def test_corrupted_saved_index_fails_for_registered_ops(model_factory, index_func) -> None:
    torch.manual_seed(0)
    model = model_factory().eval()
    x = torch.randn(6, 4)
    with torch.no_grad():
        ground_truth = model(x)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(save_arg_values=True, save_rng_states=True),
    )
    try:
        index_op = next(op for op in trace.layer_list if op.func_name == index_func)
        index_op.out.copy_(index_op.out.roll(1, 0))
        assert trace.validate_forward_pass(ground_truth) is False
    finally:
        trace.cleanup()


def test_boolean_mask_indexing_stays_on_its_current_path() -> None:
    """A mask changes the output shape when perturbed, so it is never an index domain."""

    assert "__getitem__" not in _INDEX_DOMAIN_ARG_SPECS
    assert "index" not in _INDEX_DOMAIN_ARG_SPECS
    torch.manual_seed(0)
    assert tl.validate(BoolMaskSelect().eval(), torch.randn(6, 4), scope="forward") is True


@pytest.mark.parametrize("corrupt", ["saved_gather_output", "saved_index"])
def test_corrupted_saved_gather_still_fails_validation(corrupt: str) -> None:
    """Negative control: the in-domain-only perturbation leaves the tripwire armed."""

    torch.manual_seed(0)
    model = GatherByIndex().eval()
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        ground_truth = model(x)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(save_arg_values=True, save_rng_states=True),
    )
    try:
        gather = next(op for op in trace.layer_list if op.func_name == "gather")
        if corrupt == "saved_gather_output":
            gather.out.add_(1.0)
        else:
            index_op = next(op for op in trace.layer_list if op.func_name == "argsort")
            index_op.out.copy_(index_op.out.flip(1))
        assert trace.validate_forward_pass(ground_truth) is False
    finally:
        trace.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA leg: run on a GPU venue")
def test_tiny_vjepa2_validates_on_cuda_without_a_device_assert(replay_record) -> None:
    model = _tiny_vjepa2().cuda()
    inputs = {"pixel_values_videos": torch.randn(1, 4, 3, 32, 32, device="cuda")}
    assert tl.validate(model, (), inputs, scope="forward") is True
    _assert_index_edges_proven_in_domain(replay_record, _VJEPA2_EXEMPTIONS)
