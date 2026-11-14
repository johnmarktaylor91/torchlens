"""Phase 6 saved-DAG replay engine tests."""

from __future__ import annotations

import importlib
import pickle
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import torchlens as tl
from torchlens.intervention import _replay_sites
from torchlens.intervention._replay_sites import _unmet_log_precondition
from torchlens.intervention.errors import (
    ControlFlowDivergenceWarning,
    MultiMatchWarning,
    ReplayPreconditionError,
)
from torchlens.io import TraceState
from torchlens.options import InterventionOptions

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword

_CONFIRM = InterventionOptions()
_DIFFERENTIABLE = InterventionOptions(differentiable=True)

replay_mod = importlib.import_module("torchlens.intervention._replay")


def _zero_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    """Return a zero ablation for hook tests.

    Parameters
    ----------
    out:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zeroed out.
    """

    return out * 0


def _identity_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    """Return an unchanged out.

    Parameters
    ----------
    out:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Original out.
    """

    return out


class ResidualRelu(torch.nn.Module):
    """Small residual model with a relu cone feeding the output."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual relu graph.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        h = self.linear(x)
        return torch.relu(h) + h


class SplitModel(torch.nn.Module):
    """Model that records a multi-output torch call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a multi-output ``torch.max`` call.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Values plus indices coerced to float.
        """

        values, indices = torch.max(x, dim=1)
        return values + indices.float()


class RecurrentRelu(torch.nn.Module):
    """Model with repeated relu operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent relu-style operations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final tensor.
        """

        for _idx in range(3):
            x = torch.relu(x)
        return x


class ReplayPatchModel(torch.nn.Module):
    """Tiny model for differentiable replay patching tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a relu cone with a scalar output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar output depending on the relu activation.
        """

        hidden = torch.relu(x)
        return (hidden * 3.0).sum()


def _interventions(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """Capture an intervention-ready model log.

    Parameters
    ----------
    model:
        Model to log.
    x:
        Input tensor.

    Returns
    -------
    Any
        TorchLens model log.
    """

    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def _first_func(log: Any, func_name: str) -> Any:
    """Return the first site with a function name.

    Parameters
    ----------
    log:
        Model log.
    func_name:
        Function name to find.

    Returns
    -------
    Any
        Matching layer pass.
    """

    return next(layer for layer in log._layer_list if layer.func_name == func_name)


def test_differentiable_replay_patching_captures_backward_grads() -> None:
    """Differentiable replay patches one run into another and captures grads."""

    clean_x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    corrupt_x = torch.tensor([2.0, -4.0, 1.0], requires_grad=True)
    clean = tl.trace(
        ReplayPatchModel(),
        clean_x,
        capture=tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True),
        save="all",
    )
    corrupt = tl.trace(
        ReplayPatchModel(),
        corrupt_x,
        capture=tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True),
        save="all",
    )
    clean_relu = _first_func(clean, "relu")
    patch = clean_relu.out

    def patch_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Replace the replayed activation with a tensor from another run.

        Parameters
        ----------
        out:
            Replay-computed activation.
        hook:
            TorchLens hook context.

        Returns
        -------
        torch.Tensor
            Activation from the clean source run.
        """

        del out, ctx
        return patch

    replayed = corrupt.do(tl.func("relu"), patch_hook, intervention=_DIFFERENTIABLE)
    replay_relu = _first_func(replayed, "relu")
    replay_output = replayed[replayed.output_layers[0]].out

    reference_patch = patch.detach().clone().requires_grad_()
    reference_output = (reference_patch * 3.0).sum()
    reference_output.backward()
    replay_output.backward()

    assert replayed is not corrupt
    assert corrupt.autograd.has_backward_pass is False
    assert replay_relu.layer_label in replayed._replay_frontier
    assert replayed._replay_frontier[replay_relu.layer_label].is_leaf
    assert replayed._replay_frontier[replay_relu.layer_label].requires_grad
    assert torch.equal(replay_output, reference_output.detach())
    assert reference_patch.grad is not None
    assert replay_relu.grad is not None
    frontier_grad = replayed._replay_frontier[replay_relu.layer_label].grad
    assert frontier_grad is not None
    replay_relu_grad = replay_relu.grad
    reference_grad = reference_patch.grad
    assert replay_relu_grad is not None
    assert reference_grad is not None
    assert torch.equal(replay_relu_grad, reference_grad)
    assert torch.equal(frontier_grad, reference_grad)
    assert clean_x.grad is None
    assert corrupt_x.grad is None
    last_run = cast(dict[str, Any], replayed.last_run)
    assert last_run["differentiable"] is True


def test_differentiable_replay_forks_isolated_records() -> None:
    """Differentiable replay forks COW records isolated from the source.

    The M11 COW fork replaced the deep-cone/shallow-rest forkcopier split:
    every fork Op is a fresh facade over shared frozen storage, so the
    history record now carries the replay cone for provenance only, and the
    isolation contract holds uniformly inside and outside the cone.
    """

    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    log = tl.trace(
        ReplayPatchModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True),
        save="all",
    )

    replayed = log.do(tl.func("relu"), _identity_hook, intervention=_DIFFERENTIABLE)

    fork_record = next(
        record
        for record in reversed(log.state_history)
        if isinstance(record, dict) and record.get("op") == "fork"
    )
    fork_payload = cast(dict[str, Any], fork_record)
    last_run = cast(dict[str, Any], replayed.last_run)
    source_cone_labels = set(fork_payload["source_cone_labels"])
    replay_cone_labels = set(last_run["cone"])
    all_labels = {site.layer_label for site in log._layer_list}
    outside_cone_label = next(iter(all_labels - replay_cone_labels))

    assert source_cone_labels == replay_cone_labels
    assert source_cone_labels < all_labels
    assert replayed[outside_cone_label] is not log[outside_cone_label]
    assert replayed[outside_cone_label].source_trace is replayed
    assert replayed[outside_cone_label].edits is not log[outside_cone_label].edits


def test_do_value_preserves_origin_out_and_recomputes_children() -> None:
    """``do(site, value)`` keeps the set origin value and recomputes its children."""

    torch.manual_seed(1)
    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    replacement = torch.full_like(relu_site.out, 2.0)

    log.do(tl.label(relu_site.layer_label), replacement, intervention=_CONFIRM)

    assert torch.equal(relu_site.out, replacement)
    assert log.state is TraceState.REPLAY_PROPAGATED
    assert relu_site.layer_label in log.last_run["origins"]


def test_replay_executes_multi_output_func_call_group_once() -> None:
    """Replay executes one saved function call for all output siblings."""

    log = _interventions(SplitModel(), torch.randn(4, 5))
    max_sites = [layer for layer in log._layer_list if layer.func_name == "max"]
    original_func: Callable[..., Any] = max_sites[0].func
    calls = {"count": 0}

    def counted_max(*args: Any, **kwargs: Any) -> Any:
        """Count replay executions and delegate to the captured function."""

        calls["count"] += 1
        return original_func(*args, **kwargs)

    for site in max_sites:
        site._internal_set("func", counted_max)

    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        log.do({tl.func("max"): _identity_hook}, intervention=_CONFIRM)

    assert calls["count"] == 1


def test_replay_warns_on_saved_edge_divergence() -> None:
    """Replay emits a control-flow divergence warning for edge mismatch."""

    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    relu_site.parents = ()

    with pytest.warns(ControlFlowDivergenceWarning):
        log.do({tl.func("relu"): _identity_hook}, intervention=_CONFIRM)


def test_replay_rejects_non_intervention_ready_logs() -> None:
    """Replay requires per-op argument templates; a predicate recording keeps none."""

    log = tl.record(ResidualRelu(), torch.randn(2, 3), save=tl.func("relu")).to_trace()

    with pytest.raises(ReplayPreconditionError, match="argument templates"):
        log.do({tl.func("relu"): _identity_hook}, intervention=_CONFIRM)


def _preview_log(backend: str) -> SimpleNamespace:
    """A finished preview-backend trace stand-in that records no argument templates."""

    return SimpleNamespace(
        _tracing_finished=True, is_replayable=False, options=None, _provenance_backend=backend
    )


@pytest.mark.parametrize("backend", ["jax", "tf", "mlx", "paddle", "tinygrad"])
def test_replay_refusal_names_the_preview_backend(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no replay adapter built, the refusal names the backend, not tl.record."""

    monkeypatch.setattr(_replay_sites, "backend_adapter", lambda log: None)

    message = _unmet_log_precondition(cast(Any, _preview_log(backend)))

    assert message is not None
    assert f"the {backend} backend cannot edit or patch a finished trace" in message
    assert "every other" not in message


@pytest.mark.parametrize("backend", ["jax", "tf", "mlx", "paddle", "tinygrad"])
def test_preview_backend_with_a_replay_adapter_passes_the_log_precondition(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend whose replay adapter is built replays op by op, so the log check passes."""

    monkeypatch.setattr(_replay_sites, "backend_adapter", lambda log: object())

    assert _unmet_log_precondition(cast(Any, _preview_log(backend))) is None


def test_replay_runs_on_default_capture() -> None:
    """A default capture records argument templates, so replay needs no replayable=True."""

    log = tl.trace(ResidualRelu(), torch.randn(2, 3))

    log.do({tl.func("relu"): _identity_hook}, intervention=_CONFIRM)

    assert log.last_run["engine"] == "replay"


class _GeluStages(torch.nn.Module):
    """Two stages of linear + GELU, so a selector can fan out to several sites."""

    def __init__(self) -> None:
        super().__init__()
        self.stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.GELU()),
                torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.GELU()),
            ]
        )
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


def test_default_trace_do_matches_replayable_replay_and_rerun() -> None:
    """The README shape trace.do(site, edit) works on a default trace and agrees exactly."""

    torch.manual_seed(0)
    model = _GeluStages().eval()
    x = torch.randn(3, 4)
    site = tl.func("gelu") & tl.in_module("stages.1")

    default = tl.trace(model, x)
    assert default.is_replayable is False
    replayable = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    rerun = tl.trace(model, x).fork()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", tl.errors.InterventionWarning)
        default.do(site, tl.zero_ablate())
        replayable.do(site, tl.zero_ablate())
        rerun.do(site, tl.zero_ablate(), model=model, inputs=x)
    assert default.last_run["engine"] == "replay"

    def final(trace: Any) -> torch.Tensor:
        return cast(torch.Tensor, trace[trace.output_layers[0]].out)

    with torch.no_grad():
        hooked_in = model.stages[0](x)
        expected = model.head(torch.zeros_like(model.stages[1][0](hooked_in)))
    assert torch.equal(final(default), final(replayable))
    assert torch.equal(final(default), final(rerun))
    assert torch.allclose(final(default), expected)


def test_default_trace_pickle_drops_argument_templates() -> None:
    """Templates hold live parameters; a pickled default trace must not carry the weights."""

    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.ReLU())
    log = tl.trace(model, torch.randn(1, 1024))
    assert any(op.call.args_template is not None for op in log.ops)

    data = pickle.dumps(log)
    restored = pickle.loads(data)

    weight_bytes = model[0].weight.numel() * model[0].weight.element_size()
    assert len(data) < weight_bytes
    assert all(op.call.args_template is None for op in restored.ops)
