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
from torchlens._fast_run import close_fast_run_session
from torchlens.errors import PathDivergenceError
from torchlens.intervention.errors import ControlFlowDivergenceWarning, EngineDispatchError
from torchlens.runnable import RunResult

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

    return tl.when(tl.module(_SITE), tl.steer(direction, magnitude=_MAGNITUDE, feature_axis=-1))


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


@pytest.mark.smoke
def test_five_legacy_reruns_are_exact_against_hook_and_fresh_trace() -> None:
    """Five consecutive same-length steered reruns match the hook and a fresh capture."""

    model, direction = _build()
    capture_ids = _ids(_CAPTURE_LEN, seed=100)
    with torch.no_grad():
        plain = model(capture_ids)
    reference = _hooked(model, direction, capture_ids)
    # The steer must move the output, or exactness below proves nothing.
    assert not torch.equal(reference["output"], plain)

    trace = _steered_trace(model, direction, capture_ids)
    _assert_matches_hook(trace, reference, fast=False)
    for step in range(_RERUNS):
        ids = _ids(_CAPTURE_LEN, seed=200 + step)
        reference = _hooked(model, direction, ids)
        fresh = _steered_trace(model, direction, ids)
        # The two references agree before the rerun is judged against them.
        assert _max_abs_diff(_site_op(fresh, _HEAD).out, reference["head"]) == 0.0
        assert _max_abs_diff(_site_op(fresh, _SITE).out, reference["site"]) == 0.0

        returned = trace.run(model, ids)

        assert returned is trace
        assert trace.last_run["engine"] == "guarded_fast"
        assert trace.last_run["shape_varied"] is False
        _assert_matches_hook(trace, reference, fast=True)
        assert _max_abs_diff(_site_op(trace, _HEAD).out, _site_op(fresh, _HEAD).out) == 0.0


@pytest.mark.smoke
def test_greedy_growing_length_reruns_stay_exact_and_shape_varied() -> None:
    """Eight greedy steps of growing length rerun fast, exactly, with honest shapes."""

    model, direction = _build()
    ids = _ids(_CAPTURE_LEN, seed=300)
    trace = _steered_trace(model, direction, ids)
    unsaved = next(op for op in trace.layer_list if op.func_name == "gelu")
    unsaved_label = unsaved.label
    assert not unsaved.has_saved_activation
    assert unsaved.shape is not None  # capture-time metadata exists, so None is a change

    reference_ids = ids.clone()
    for step in range(_GREEDY_STEPS):
        reference = _hooked(model, direction, reference_ids)
        trace.run(model, ids)

        assert trace.last_run["engine"] == "guarded_fast"
        assert trace.last_run["shape_varied"] is (step > 0)
        _assert_matches_hook(trace, reference, fast=True)
        length = ids.shape[1]
        assert tuple(_site_op(trace, _SITE).shape) == (1, length, _DIM)
        assert tuple(_site_op(trace, _SITE).out.shape) == (1, length, _DIM)
        if step > 0:
            stale = trace[unsaved_label]
            assert stale.shape is None
            assert stale.transformed_out_shape is None
            assert stale.activation_memory is None

        token = _site_op(trace, _HEAD).out[:, -1, :].argmax(dim=-1, keepdim=True)
        reference_token = reference["output"].argmax(dim=-1, keepdim=True)
        assert torch.equal(token, reference_token)
        ids = torch.cat([ids, token], dim=1)
        reference_ids = torch.cat([reference_ids, reference_token], dim=1)
    assert torch.equal(ids, reference_ids)


@pytest.mark.smoke
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


def test_fast_session_hooks_are_installed_once_and_closed() -> None:
    """Module hook count is constant across reruns and restored by closing the session."""

    model, direction = _build()
    before = _module_hook_count(model)
    trace = _steered_trace(model, direction, _ids(_CAPTURE_LEN, seed=500))

    counts = []
    for step in range(_RERUNS):
        trace.run(model, _ids(_CAPTURE_LEN, seed=510 + step))
        assert trace.last_run["engine"] == "guarded_fast"
        counts.append(_module_hook_count(model))

    assert counts[0] == counts[-1]
    assert len(set(counts)) == 1
    close_fast_run_session(trace)
    assert _module_hook_count(model) == before


def test_staged_spec_object_is_stable_across_fast_reruns() -> None:
    """The staged spec stays the same object with the same hook count after each rerun."""

    model, direction = _build()
    trace = _steered_trace(model, direction, _ids(_CAPTURE_LEN, seed=600))
    spec = trace._intervention_spec
    staged = len(spec.hook_specs)
    assert staged >= 1

    for step in range(_RERUNS):
        trace.run(model, _ids(_CAPTURE_LEN, seed=610 + step))
        assert trace.last_run["engine"] == "guarded_fast"
        assert trace._intervention_spec is spec
        assert len(spec.hook_specs) == staged


@pytest.mark.smoke
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


@pytest.mark.smoke
@pytest.mark.parametrize("intervention_ready", [False, True], ids=["default", "ready"])
def test_default_save_fallback_replays_stochastic_forward_and_keeps_readiness(
    intervention_ready: bool,
) -> None:
    """A default-save trace reruns through capture, RNG-exact, inheriting readiness.

    The rerun replays the capture's recorded ``random_seed`` (the behavior main
    ships), so the fallback's random draws reproduce the CAPTURE's output, not
    the caller's ambient seed; a stochastic model pins exactly that, and that
    the readiness setting is inherited rather than forced on.
    """

    torch.manual_seed(0)
    model = _StochasticModel().train()
    x = torch.randn(2, 8)
    torch.manual_seed(7)
    with torch.no_grad():
        plain = model(x)

    torch.manual_seed(123)
    if intervention_ready:
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    else:
        trace = tl.trace(model, x)
    source_ready = trace.intervention_ready
    assert source_ready is intervention_ready
    captured = _output(trace).detach().clone()
    # The model is genuinely stochastic: an unseeded-equivalent forward differs.
    assert not torch.equal(captured, plain)

    torch.manual_seed(7)
    trace.run(model, x)

    assert trace.last_run["engine"] == "rerun"
    assert torch.equal(_output(trace), captured)
    assert trace.intervention_ready == source_ready


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
