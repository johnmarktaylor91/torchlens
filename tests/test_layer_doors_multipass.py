"""``Layer.do`` (edit or value, propagated or ``set_only``) addresses every pass of a multi-pass layer.

A bare layer label addresses only a single-pass layer, so the Layer doors
must not forward it for a layer that ran several passes: a module reused in a
loop would otherwise refuse with ``SiteAmbiguityError`` on every door. A Layer
spans all its passes, so each door discloses the two-site fan-out. Each edit
here is checked against the same edit written in plain torch.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword


class _SharedThenFresh(nn.Module):
    """One Linear reused for two passes, then a fresh single-pass Linear."""

    def __init__(self) -> None:
        """Initialize the shared and the fresh projection."""

        super().__init__()
        self.shared = nn.Linear(3, 3)
        self.fresh = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared projection twice, then the fresh one."""

        for _call in range(2):
            x = torch.relu(self.shared(x))
        return self.fresh(x)


def _oracle(
    model: _SharedThenFresh, x: torch.Tensor, *, shared_edit=None, fresh_edit=None
) -> torch.Tensor:
    """Run the model in plain torch with an edit on each shared pass and the fresh layer."""

    with torch.no_grad():
        for _call in range(2):
            y = model.shared(x)
            x = torch.relu(y if shared_edit is None else shared_edit(y))
        y = model.fresh(x)
        return y if fresh_edit is None else fresh_edit(y)


@pytest.fixture
def captured() -> Iterator[tuple[_SharedThenFresh, torch.Tensor, tl.Trace]]:
    """Capture the model ready for interventions."""

    torch.manual_seed(0)
    model = _SharedThenFresh().eval()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _linear_labels(trace: tl.Trace) -> tuple[str, str]:
    """Return the multi-pass shared and the single-pass fresh linear layer labels."""

    linears = [layer for layer in trace._layer_logs.values() if layer.layer_type == "linear"]
    shared = [layer.layer_label for layer in linears if layer.num_passes == 2]
    fresh = [layer.layer_label for layer in linears if layer.num_passes == 1]
    assert len(shared) == 1 and len(fresh) == 1
    return shared[0], fresh[0]


def _double(t: torch.Tensor) -> torch.Tensor:
    """Plain-torch twin of ``tl.scale(2.0)``."""

    return t * 2.0


def _zero(t: torch.Tensor) -> torch.Tensor:
    """Plain-torch twin of setting the site to zeros."""

    return torch.zeros_like(t)


def _double_hook(out: torch.Tensor, *, ctx: object = None) -> torch.Tensor:
    """A plain callable hook that doubles the activation (``ctx`` is the hook context)."""

    del ctx
    return out * 2.0


@pytest.mark.parametrize("door", ("do", "do_value", "set_only"))
def test_layer_door_on_multi_pass_layer_edits_every_pass(captured, door: str) -> None:
    """Each Layer door on a two-pass layer edits both passes, matching plain torch.

    ``do`` propagates at once (the fold of set or attach plus push); the ``set_only``
    engine stages the value on both passes and leaves the outputs as captured.
    """

    model, x, trace = captured
    shared, _fresh = _linear_labels(trace)
    fork = trace.fork()
    layer = fork.layers[shared]
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        if door == "do":
            layer.do(tl.scale(2.0))
        elif door == "do_value":
            layer.do(torch.zeros(2, 3))
        else:
            layer.do(
                torch.zeros(2, 3),
                intervention=tl.options.InterventionOptions(engine="set_only"),
            )
    if door == "set_only":
        # One staged value entry covers both passes; nothing propagates yet.
        assert len(fork._intervention_spec.target_value_specs) == 1
        assert fork._intervention_spec.hook_specs == []
        want = _oracle(model, x)
        assert torch.allclose(fork[fork.output_layers[0]].out, want, atol=1e-6)
        return
    want = _oracle(model, x, shared_edit=_double if door == "do" else _zero)
    assert torch.allclose(fork[fork.output_layers[0]].out, want, atol=1e-6)
    for pass_index in (1, 2):
        assert fork[f"{shared}:{pass_index}"].edits


def test_set_only_callable_on_multi_pass_layer_is_sticky_on_both_passes(captured) -> None:
    """A callable staged by ``do(..., engine="set_only")`` is a sticky hook on both passes.

    Walk ruling names.3: a callable passed to the set-only engine always becomes a sticky
    hook, never a one-shot value. It changes no output when staged, fires on each pass of
    the two-pass layer when a later ``do`` propagates, matching the same edit written in
    plain torch, stays in the recipe after firing, and fires on both passes of a later
    ``run`` on new inputs.
    """

    model, x, trace = captured
    shared, fresh = _linear_labels(trace)
    fork = trace.fork()
    baseline = fork[fork.output_layers[0]].out.clone()
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        fork.layers[shared].do(
            _double_hook,
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    # Staged as a hook, not a value, and nothing propagates yet.
    assert fork._intervention_spec.target_value_specs == []
    assert fork._intervention_spec.hook_specs
    assert torch.equal(fork[fork.output_layers[0]].out, baseline)

    # The next propagating do fires the staged hook on both passes. The hook keeps its
    # two-site target, so re-resolving it may disclose the fan-out again; that disclosure
    # is the only warning allowed here.
    staged_handles = {hook.handle for hook in fork._intervention_spec.hook_specs}
    assert None not in staged_handles
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", MultiMatchWarning)
        fork[fresh].do(tl.scale(1.0))
    fan_out = [w for w in seen if issubclass(w.category, MultiMatchWarning)]
    assert all("matched 2 sites" in str(w.message) for w in fan_out)
    want = _oracle(model, x, shared_edit=_double)
    assert torch.allclose(fork[fork.output_layers[0]].out, want, atol=1e-6)
    for pass_index in (1, 2):
        assert fork[f"{shared}:{pass_index}"].edits
    # Sticky: firing did not consume it. The staged hook is still in the recipe and no
    # one-shot value took its place.
    assert staged_handles <= {hook.handle for hook in fork._intervention_spec.hook_specs}
    assert fork._intervention_spec.target_value_specs == []
    # It stays attached: a fresh forward on new inputs fires it on both passes again.
    x2 = torch.randn(2, 3)
    rerun = fork.run(model, x2)
    want2 = _oracle(model, x2, shared_edit=_double)
    assert torch.allclose(rerun[rerun.output_layers[0]].out, want2, atol=1e-6)
    for pass_index in (1, 2):
        assert rerun[f"{shared}:{pass_index}"].edits


@pytest.mark.parametrize("edit", ("hook", "helper"))
def test_layer_door_hook_on_multi_pass_layer_refires_on_run(captured, edit: str) -> None:
    """A hook or helper put on a two-pass layer by ``Layer.do`` refires on both passes of ``run``.

    The Layer door stages one sticky entry addressed to every pass of the layer; a later
    ``run`` on new inputs fires it on each pass, matching the same edit in plain torch.
    """

    model, x, trace = captured
    shared, _fresh = _linear_labels(trace)
    fork = trace.fork()
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        fork.layers[shared].do(_double_hook if edit == "hook" else tl.scale(2.0))
    assert torch.allclose(
        fork[fork.output_layers[0]].out, _oracle(model, x, shared_edit=_double), atol=1e-6
    )
    x2 = torch.randn(2, 3)
    rerun = fork.run(model, x2)
    want2 = _oracle(model, x2, shared_edit=_double)
    assert torch.allclose(rerun[rerun.output_layers[0]].out, want2, atol=1e-6)
    for pass_index in (1, 2):
        assert rerun[f"{shared}:{pass_index}"].edits


@pytest.mark.parametrize("engine", ("replay", "set_only"))
def test_union_of_pass_labels_hook_refires_on_run(captured, engine: str) -> None:
    """A hook staged through ``trace.do`` on the ``|`` of two pass labels refires on ``run``.

    The union is the spelling the Layer door lowers a multi-pass layer to; written by hand
    at the trace level it reaches the next ``run`` the same way and fires on both passes.
    """

    model, _x, trace = captured
    shared, _fresh = _linear_labels(trace)
    fork = trace.fork()
    site = tl.label(f"{shared}:1") | tl.label(f"{shared}:2")
    options = tl.options.InterventionOptions(engine="set_only") if engine == "set_only" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MultiMatchWarning)
        fork.do(site, _double_hook, intervention=options)
    x2 = torch.randn(2, 3)
    rerun = fork.run(model, x2)
    want2 = _oracle(model, x2, shared_edit=_double)
    assert torch.allclose(rerun[rerun.output_layers[0]].out, want2, atol=1e-6)


def test_layer_door_on_single_pass_layer_edits_only_that_layer(captured) -> None:
    """A single-pass Layer door keeps addressing just its own layer."""

    model, x, trace = captured
    shared, fresh = _linear_labels(trace)
    fork = trace.fork()
    fork[fresh].do(tl.scale(2.0))
    assert torch.allclose(
        fork[fork.output_layers[0]].out, _oracle(model, x, fresh_edit=_double), atol=1e-6
    )
    assert not any(fork[f"{shared}:{pass_index}"].edits for pass_index in (1, 2))
