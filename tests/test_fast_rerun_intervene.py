"""Steered legacy reruns route through the guarded fast engine, exactly.

Contract pinned here: after ``tl.trace(model, ids, save=..., intervene=spec)``
the legacy rerun ``trace.run(model, new_ids)`` runs a native forward with the
staged spec applied through real module hooks, saving only the saved sites and
the model input/output boundary ops (``last_run["engine"] == "guarded_fast"``).
It accepts a changed input length when the op structure is unchanged
(``last_run["shape_varied"]``), refreshes saved sites to the new run, and
reports unsaved ops' shape metadata as unavailable (``None``) rather than stale.
A real control-flow divergence, or an ineligible trace, falls back to the full
capture rerun (``engine == "rerun"`` with a typed ``fast_refused`` code); the
explicit ``run(inputs=..., fast=True)`` door raises ``PathDivergenceError``.
Every result is exact against a plain ``register_forward_hook`` reference that
reproduces ``tl.steer``'s arithmetic (``out + direction.reshape(1, 1, d) * m``).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import RunResult
from torchlens.errors import PathDivergenceError, RunCapabilityUnavailableError
from torchlens.intervention.errors import ControlFlowDivergenceWarning, EngineDispatchError

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword

_VOCAB = 64
_DIM = 16
_N_BLOCKS = 3
_SITE = "blocks.1"
_HEAD = "head"
_MAGNITUDE = 4.0
_CAPTURE_LEN = 5
_RERUNS = 5
_GREEDY_STEPS = 8


class _Block(nn.Module):
    """Pre-norm MLP block with a residual connection."""

    def __init__(self, dim: int) -> None:
        """Build the norm and the two projections.

        Parameters
        ----------
        dim:
            Residual stream width.
        """

        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2 * dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the residual MLP.

        Parameters
        ----------
        h:
            Residual stream, shape ``(batch, length, dim)``.

        Returns
        -------
        torch.Tensor
            Updated residual stream.
        """

        return h + self.fc2(self.act(self.fc1(self.ln(h))))


class _TinyDecoder(nn.Module):
    """Embedding, three residual blocks, and a vocabulary head."""

    def __init__(self) -> None:
        """Build the embedding, blocks, and head."""

        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _DIM)
        self.blocks = nn.ModuleList(_Block(_DIM) for _ in range(_N_BLOCKS))
        self.head = nn.Linear(_DIM, _VOCAB)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the final residual stream.

        Parameters
        ----------
        ids:
            Token ids, shape ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Residual stream after every block.
        """

        h = self.embed(ids)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return last-position logits, so the output shape is length-invariant.

        Parameters
        ----------
        ids:
            Token ids, shape ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Logits at the final position, shape ``(batch, vocab)``.
        """

        return self.head(self.hidden(ids))[:, -1, :]


class _FullLogitsDecoder(_TinyDecoder):
    """Same decoder, returning logits at every position."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return every position's logits.

        Parameters
        ----------
        ids:
            Token ids, shape ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(batch, length, vocab)``.
        """

        return self.head(self.hidden(ids))


class _LengthBranchDecoder(_TinyDecoder):
    """Takes a different torch function once the sequence is longer than four."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Branch on the input length before the head.

        Parameters
        ----------
        ids:
            Token ids, shape ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Last-position logits.
        """

        h = self.hidden(ids)
        h = torch.tanh(h) if ids.shape[1] > 4 else torch.relu(h)
        return self.head(h)[:, -1, :]


class _StochasticModel(nn.Module):
    """Train-mode dropout plus a ``torch.rand`` draw in forward."""

    def __init__(self) -> None:
        """Build the linear and the dropout."""

        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout, then add uniform noise.

        Parameters
        ----------
        x:
            Input, shape ``(batch, 8)``.

        Returns
        -------
        torch.Tensor
            Noisy output.
        """

        h = self.drop(self.fc(x))
        return h + torch.rand(h.shape) * 0.1


def _build(model_cls: type[_TinyDecoder] = _TinyDecoder) -> tuple[nn.Module, torch.Tensor]:
    """Return a seeded eval-mode model and its steering direction.

    Parameters
    ----------
    model_cls:
        Decoder class to instantiate.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        The model and a direction of size ``_DIM``.
    """

    torch.manual_seed(0)
    model = model_cls().eval()
    direction = torch.randn(_DIM)
    return model, direction


def _ids(length: int, seed: int) -> torch.Tensor:
    """Return deterministic token ids of shape ``(1, length)``.

    Parameters
    ----------
    length:
        Sequence length.
    seed:
        Generator seed.

    Returns
    -------
    torch.Tensor
        Token ids.
    """

    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, length), generator=generator)


def _spec(direction: torch.Tensor) -> Any:
    """Return the one-clause steer spec at the site.

    Parameters
    ----------
    direction:
        Steering direction of size ``_DIM``.

    Returns
    -------
    Any
        Intervention spec.
    """

    return tl.when(tl.module(_SITE), tl.steer(direction, strength=_MAGNITUDE, dim=-1))


def _steered_trace(model: nn.Module, direction: torch.Tensor, ids: torch.Tensor) -> Any:
    """Capture a steered trace saving the site and the head.

    Parameters
    ----------
    model:
        Decoder.
    direction:
        Steering direction.
    ids:
        Capture input.

    Returns
    -------
    Any
        The steered Trace.
    """

    return tl.trace(
        model,
        ids,
        save=tl.module(_SITE) | tl.module(_HEAD),
        intervene=_spec(direction),
    )


def _hooked(model: nn.Module, direction: torch.Tensor, ids: torch.Tensor) -> dict[str, Any]:
    """Run the model with a plain forward hook reproducing ``tl.steer``.

    ``tl.steer(direction, magnitude=m, feature_axis=-1)`` returns
    ``out + direction.reshape(1, 1, d) * m``; the hook spells exactly that.

    Parameters
    ----------
    model:
        Decoder.
    direction:
        Steering direction.
    ids:
        Input ids.

    Returns
    -------
    dict[str, Any]
        ``output`` (model output), ``site`` (post-steer site activation),
        and ``head`` (head module output).
    """

    seen: dict[str, Any] = {}

    def steer_hook(_module: nn.Module, _args: Any, out: torch.Tensor) -> torch.Tensor:
        steered = out + direction.reshape(1, 1, -1) * _MAGNITUDE
        seen["site"] = steered.detach().clone()
        return steered

    def head_hook(_module: nn.Module, _args: Any, out: torch.Tensor) -> None:
        seen["head"] = out.detach().clone()

    handles = [
        model.get_submodule(_SITE).register_forward_hook(steer_hook),
        model.get_submodule(_HEAD).register_forward_hook(head_hook),
    ]
    try:
        with torch.no_grad():
            seen["output"] = model(ids).detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return seen


def _site_op(trace: Any, address: str) -> Any:
    """Return the saved output op of a module site.

    Parameters
    ----------
    trace:
        Trace to search.
    address:
        Module address.

    Returns
    -------
    Any
        The site's Op.
    """

    return trace.find_sites(tl.module(address)).first()


def _output(trace: Any) -> torch.Tensor:
    """Return the model output op's value.

    Parameters
    ----------
    trace:
        Trace to read.

    Returns
    -------
    torch.Tensor
        Output activation.
    """

    return trace[trace.output_layers[0]].out


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the max absolute difference, asserting equal shapes first.

    Parameters
    ----------
    left, right:
        Tensors to compare.

    Returns
    -------
    float
        ``max |left - right|``.
    """

    assert tuple(left.shape) == tuple(right.shape)
    return float((left.detach() - right.detach()).abs().max())


def _module_hook_count(model: nn.Module) -> int:
    """Count forward and forward-pre hooks across every submodule.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    int
        Hook count.
    """

    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def _assert_matches_hook(trace: Any, reference: dict[str, Any], *, fast: bool) -> None:
    """Assert the trace's saved site and head equal the hook reference.

    An explicit ``save=`` capture leaves the model output op unsaved, and both
    rerun engines preserve that save scope (the output lives in the saved head
    site here), so only the saved sites are compared.

    Parameters
    ----------
    trace:
        Rerun Trace.
    reference:
        Output of :func:`_hooked` on the same input.
    fast:
        Whether the run took the guarded fast engine (kept for call-site
        readability; both engines are held to the same exactness).
    """

    del fast
    assert _max_abs_diff(_site_op(trace, _SITE).out, reference["site"]) == 0.0
    assert _max_abs_diff(_site_op(trace, _HEAD).out, reference["head"]) == 0.0


# Kept: sole branch coverage of _fast_live_steer.py, _fast_run.py, intervention/_trace_run.py; a sole mutation kill.
def test_control_flow_divergence_refuses_fast_and_falls_back_exactly() -> None:
    """A different branch at the new length is refused fast and reruns through capture."""

    model, direction = _build(_LengthBranchDecoder)
    short_ids = _ids(4, seed=400)
    long_ids = _ids(6, seed=401)
    reference = _hooked(model, direction, long_ids)

    legacy = _steered_trace(model, direction, short_ids)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy.run(model, long_ids)
    # The capture fallback may disclose the divergence; nothing else may warn.
    assert all(issubclass(w.category, ControlFlowDivergenceWarning) for w in caught)

    assert legacy.last_run["engine"] == "rerun"
    refused = legacy.last_run["fast_refused"]
    assert isinstance(refused, str) and refused
    _assert_matches_hook(legacy, reference, fast=False)

    explicit = _steered_trace(model, direction, short_ids)
    with pytest.raises(PathDivergenceError):
        explicit.run(inputs=long_ids, fast=True)


# Kept: sole branch coverage of intervention/_trace_run.py.
def test_fast_door_applies_staged_spec_while_plain_run_inputs_still_refuses() -> None:
    """``run(inputs=, fast=True)`` applies the steer; plain ``run(inputs=)`` refuses."""

    model, direction = _build()
    ids = _ids(_CAPTURE_LEN, seed=700)
    trace = _steered_trace(model, direction, ids)

    with pytest.raises(EngineDispatchError) as refused:
        trace.run(inputs=ids)
    assert refused.value.fields["code"] == "run_staged_spec_unapplied"

    for seed in (700, 701):
        run_ids = _ids(_CAPTURE_LEN, seed=seed)
        result = trace.run(inputs=run_ids, fast=True)
        assert isinstance(result, RunResult)
        assert _max_abs_diff(result.output, _hooked(model, direction, run_ids)["output"]) == 0.0

    with pytest.raises(EngineDispatchError) as refused_again:
        trace.run(inputs=ids)
    assert refused_again.value.fields["code"] == "run_staged_spec_unapplied"


# Kept: sole branch coverage of _fast_live_steer.py, _fast_run.py; a sole mutation kill.
def test_full_sequence_output_reruns_shape_varied_exactly() -> None:
    """A ``(1, L, V)`` output keeps its rank and follows the new length exactly."""

    model, direction = _build(_FullLogitsDecoder)
    trace = _steered_trace(model, direction, _ids(_CAPTURE_LEN, seed=800))
    longer = _ids(_CAPTURE_LEN + 2, seed=801)
    reference = _hooked(model, direction, longer)

    trace.run(model, longer)

    assert trace.last_run["engine"] == "guarded_fast"
    assert trace.last_run["shape_varied"] is True
    # The head module's output IS the model output for this decoder.
    output = _site_op(trace, _HEAD).out
    assert output.ndim == 3
    assert tuple(output.shape) == (1, _CAPTURE_LEN + 2, _VOCAB)
    assert _max_abs_diff(output, reference["output"]) == 0.0
    _assert_matches_hook(trace, reference, fast=True)


# Kept: sole branch coverage of _data_classes/trace.py, _fast_live_steer.py, intervention/_trace_edits.py and more; a sole mutation kill.
def test_fast_door_session_follows_a_restaged_spec() -> None:
    """A cached ``fast=True`` session applies the spec staged NOW, never a stale plan.

    The explicit door keeps its session between calls. Clearing the staged
    steer and attaching another at the same site must apply the new steer on
    the next call. Clearing it entirely leaves replacement ops in the recorded
    graph that no staged entry accounts for, so the explicit door refuses
    ``fast_rerun_graph_unsteered``; the legacy door, with nothing staged, takes
    the capture rerun and records the plain graph.
    """

    model, direction = _build()
    ids = _ids(_CAPTURE_LEN, seed=11)
    trace = _steered_trace(model, direction, ids)
    first = trace.run(inputs=ids, fast=True)
    assert _max_abs_diff(first.output, _hooked(model, direction, ids)["output"]) == 0.0

    other = torch.randn(_DIM, generator=torch.Generator().manual_seed(12))
    trace._clear_hooks()
    trace._attach_hooks(
        tl.module(_SITE),
        tl.steer(other, strength=_MAGNITUDE, dim=-1),
    )
    restaged = trace.run(inputs=ids, fast=True)
    assert restaged.trace.last_run["engine"] == "guarded_fast"
    assert _max_abs_diff(restaged.output, _hooked(model, other, ids)["output"]) == 0.0

    trace._clear_hooks()
    with pytest.raises(RunCapabilityUnavailableError) as refused:
        trace.run(inputs=ids, fast=True)
    assert refused.value.fields["detection_stage"] == "fast_rerun_graph_unsteered"
    trace.run(model, ids)
    assert trace.last_run["engine"] == "rerun"
    assert trace.last_run["fast_refused"] is None
    assert not trace.injected_ops
    with torch.no_grad():
        plain = model(ids)
    assert _max_abs_diff(_site_op(trace, _HEAD).out[:, -1, :], plain) == 0.0


# Kept: sole branch coverage of _fast_live_steer.py, intervention/_zero_fire.py.
def test_hooks_attached_after_capture_rerun_through_capture_once() -> None:
    """Hooks staged on a plain capture send the first rerun through the capture engine.

    The fast engine leaves the recorded graph alone, so a trace whose graph has
    never seen the staged hooks (``attach_hooks`` after a plain capture) is
    refused with ``fast_rerun_graph_unsteered`` and recaptured once, after which
    the graph records the fire at the staged module. The recapture replays the
    capture's save request, so it saves the same sites, exactly, and a correct
    staged rerun is silent (no divergence warning); the next rerun is then
    eligible for the fast engine.
    """

    from torchlens._fast_live_steer import graph_fired_addresses

    model, direction = _build()
    ids = _ids(_CAPTURE_LEN, seed=3)
    baseline = _module_hook_count(model)
    trace = tl.trace(model, ids, save=tl.module(_SITE) | tl.module(_HEAD))
    trace._attach_hooks(
        tl.module(_SITE),
        tl.steer(direction, strength=_MAGNITUDE, dim=-1),
    )
    assert _SITE not in graph_fired_addresses(trace)

    reference = _hooked(model, direction, ids)
    trace.run(model, ids)

    assert trace.last_run["engine"] == "rerun"
    refused = trace.last_run["fast_refused"]
    assert isinstance(refused, str) and refused.endswith(":fast_rerun_graph_unsteered")
    assert _SITE in graph_fired_addresses(trace)
    assert _module_hook_count(model) == baseline
    _assert_matches_hook(trace, reference, fast=False)

    trace.run(model, ids)
    assert trace.last_run["engine"] == "guarded_fast"
    _assert_matches_hook(trace, reference, fast=True)
    assert _module_hook_count(model) == baseline
