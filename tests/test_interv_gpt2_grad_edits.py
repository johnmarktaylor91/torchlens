"""Gradient-side edits on GPT-2 over every live-backward spelling (U14.5).

Every spelling that edits a gradient during a live backward is checked against
an independent oracle built from plain PyTorch hooks on a deep copy of the same
model: a tensor hook on the module output for forward selectors, and a
``Node.register_hook`` on every matching autograd node for ``tl.grad_fn``
selectors. A spelling passes only when every parameter gradient equals the
oracle's, differs from the clean backward (the edit had an effect), and left at
least one backward fire record. Spellings that cannot apply a gradient edit must
refuse typed, never run as a silent no-op.

The default rows use a small randomly initialised GPT-2 whose two blocks are
ONE shared block instance called twice (ALBERT-style layer sharing), so every op
inside it is a 2-pass layer: the multi-pass shape that crashed live-backward
selector matching on real gpt2. (Under call-site grouping distinct blocks give
single-pass layers, grouping rule row 29; one module called twice gives one
layer with 2 passes, rows 26-27.) The ``TestRealGPT2`` rows run the same checks
on the pretrained ``gpt2`` checkpoint in the offline venue.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from support.r1_venue import IN_OFFLINE_VENUE

import torchlens as tl
from torchlens.intervention.errors import (
    BindingPreflightError,
    LiveModeLabelError,
    MultiMatchWarning,
)

transformers = pytest.importorskip("transformers")

GradFn = Callable[[torch.Tensor], torch.Tensor]
Grads = dict[str, torch.Tensor | None]

_MLP = "transformer.h.0.mlp"
# A single-pass layer norm for the bare-label rows: inside the shared block every
# layer norm is a 2-pass layer, and a bare label addresses only single-pass layers.
_LN_SINGLE = "transformer.ln_f"
_LN_NODE = "NativeLayerNormBackward0"
_RUNNERS = (
    "trace.backward",
    "trace.log_backward",
    "loss.backward",
    "autograd.backward",
    "autograd.grad",
)
_OPT_INS: dict[str, dict[str, Any]] = {
    "backward_ready": {"backward_ready": True},
    "save_grads": {"save_grads": True},
}


def _zero(grad: torch.Tensor) -> torch.Tensor:
    """Return a zero gradient of the same shape."""

    return torch.zeros_like(grad)


def _double(grad: torch.Tensor) -> torch.Tensor:
    """Return the gradient scaled by two."""

    return grad * 2.0


def _zero_hook(grad: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """User ``bwd_hook`` callable zeroing the gradient."""

    del hook
    return torch.zeros_like(grad)


_HELPERS: dict[str, tuple[Callable[[], Any], GradFn]] = {
    "grad_zero": (tl.zero_grad, _zero),
    "grad_scale": (lambda: tl.scale_grad(2.0), _double),
    "bwd_hook": (lambda: tl.hook_grad(_zero_hook), _zero),
}


class _Fixture:
    """One base model, its input ids and the clean parameter gradients."""

    def __init__(self, model: torch.nn.Module, ids: torch.Tensor) -> None:
        """Store the base model and compute its clean gradients."""

        self.base = model.eval()
        self.ids = ids
        self._oracles: dict[tuple[str, str, GradFn], Grads] = {}
        clean = self.fresh()
        _loss(clean(ids)).backward()
        self.clean = _grads(clean)
        probe = tl.trace(self.fresh(), ids)
        self.ln_label = probe.find_sites(tl.module(_LN_SINGLE)).first().layer_label
        self.multi_pass = [layer.layer_label for layer in probe.layers if layer.num_passes > 1]

    def fresh(self) -> torch.nn.Module:
        """Return a deep copy of the base model with cleared gradients."""

        model = copy.deepcopy(self.base)
        model.zero_grad(set_to_none=True)
        return model

    def oracle(self, kind: str, target: str, fn: GradFn) -> Grads:
        """Return parameter gradients with ``fn`` applied by plain torch hooks (memoized)."""

        key = (kind, target, fn)
        if key not in self._oracles:
            self._oracles[key] = self._oracle(kind, target, fn)
        return self._oracles[key]

    def _oracle(self, kind: str, target: str, fn: GradFn) -> Grads:
        """Compute one oracle: a deep copy of the model edited by plain torch hooks."""

        model = self.fresh()
        handles: list[Any] = []

        def _on_output(module: Any, args: Any, out: Any) -> None:
            """Edit the gradient flowing back into the module output."""

            del module, args
            (out[0] if isinstance(out, tuple) else out).register_hook(fn)

        if kind == "module":
            handles.append(model.get_submodule(target).register_forward_hook(_on_output))
        elif kind == "every_layernorm":
            for module in model.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    handles.append(module.register_forward_hook(_on_output))
        loss = _loss(model(self.ids))
        if kind == "grad_fn":
            _hook_autograd_nodes(loss, target, fn)
        loss.backward()
        for handle in handles:
            handle.remove()
        return _grads(model)


def _hook_autograd_nodes(loss: torch.Tensor, node_type: str, fn: GradFn) -> None:
    """Register ``fn`` on the grad inputs of every autograd node of one type."""

    def _edit(grad_inputs: tuple[Any, ...], grad_outputs: tuple[Any, ...]) -> tuple[Any, ...]:
        """Apply ``fn`` to each present grad input."""

        del grad_outputs
        return tuple(fn(grad) if grad is not None else None for grad in grad_inputs)

    # Visited nodes are held, not just their ids: ``next_functions`` can mint a
    # fresh wrapper per access (torch 2.7), and a freed wrapper's address can be
    # reused by the next one, which an id-only set would skip as visited.
    seen: dict[int, Any] = {}
    stack = [loss.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = node
        if type(node).__name__ == node_type:
            node.register_hook(_edit)
        stack.extend(child for child, _ in node.next_functions)


def _loss(out: Any) -> torch.Tensor:
    """Return a scalar loss over the logits."""

    logits = out.logits if hasattr(out, "logits") else out
    return logits.float().pow(2).mean()


def _grads(model: torch.nn.Module) -> Grads:
    """Return a snapshot of every parameter gradient."""

    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in model.named_parameters()
    }


def _run_backward(trace: Any, model: torch.nn.Module, runner: str) -> Grads:
    """Run one live-backward spelling on the trace output and return the grads."""

    loss = _loss(trace[trace.output_layers[0]].out)
    if runner == "trace.backward":
        trace.backward(loss)
    elif runner == "trace.log_backward":
        trace.log_backward(loss)
    elif runner == "loss.backward":
        loss.backward()
    elif runner == "autograd.backward":
        torch.autograd.backward(loss)
    else:
        named = [(name, param) for name, param in model.named_parameters()]
        found = torch.autograd.grad(loss, [param for _, param in named], allow_unused=True)
        return {name: grad for (name, _), grad in zip(named, found, strict=True)}
    return _grads(model)


def _backward_fires(trace: Any) -> int:
    """Count the trace's backward fire records."""

    spec = trace._intervention_spec
    records = getattr(spec, "records", ()) if spec is not None else ()
    return sum(1 for record in records if record.direction == "backward")


def _assert_matches_oracle(got: Grads, want: Grads, clean: Grads) -> None:
    """Every gradient equals the oracle's and at least one differs from clean."""

    changed = False
    for name, expected in want.items():
        actual = got[name]
        if expected is None or actual is None:
            assert actual is None or not torch.count_nonzero(actual), name
            assert expected is None or not torch.count_nonzero(expected), name
            changed = changed or (clean[name] is not None)
            continue
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6, msg=name)
        changed = changed or not torch.equal(actual, clean[name])
    assert changed, "the gradient edit had no effect on any parameter gradient"


def _apply_door(trace: Any, door: str, selector: Any, helper: Any) -> None:
    """Attach one gradient edit to a finished trace through a post-capture door."""

    if door == "attach_hooks":
        trace.attach_hooks(selector, helper, confirm_mutation=True)
    elif door == "layer.attach_hooks":
        layer_label = trace.find_sites(selector).first().layer_label
        trace[layer_label].attach_hooks(helper, confirm_mutation=True)
    elif door == "set":
        trace.set(selector, _zero, direction="backward", confirm_mutation=True)
    else:
        raise AssertionError(door)


def _check_spelling(
    fixture: _Fixture,
    *,
    door: str,
    runner: str,
    opt_in: str,
    selector: Any,
    helper: Any,
    want: Grads,
) -> None:
    """Run one spelling end to end and assert it matches its oracle."""

    model = fixture.fresh()
    capture = tl.options.CaptureOptions(**_OPT_INS[opt_in])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if door == "intervene":
            trace = tl.trace(
                model, fixture.ids, capture=capture, intervene=tl.when(selector, helper)
            )
        else:
            trace = tl.trace(model, fixture.ids, capture=capture)
            _apply_door(trace, door, selector, helper)
        got = _run_backward(trace, model, runner)
    _assert_matches_oracle(got, want, fixture.clean)
    assert _backward_fires(trace) > 0


@pytest.fixture(scope="module")
def tiny() -> _Fixture:
    """Random GPT-2 whose two blocks share one instance: multi-pass layers."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=16, vocab_size=64)
    model = transformers.GPT2LMHeadModel(config)
    # grouping rule rows 26-27: one block instance called twice makes every op
    # inside it one layer with 2 passes (the oracle's module hooks fire on both
    # calls too). deepcopy in _Fixture.fresh keeps the sharing.
    model.transformer.h[1] = model.transformer.h[0]
    fixture = _Fixture(model, torch.tensor([[5, 17, 3, 42, 9]]))
    assert fixture.multi_pass, "fixture must carry multi-pass layers (the crash shape)"
    return fixture


_DOORS = ("intervene", "attach_hooks", "layer.attach_hooks", "set")


# 40 cells of about 0.4 s each: heavy-class aggregate compute (tests/AGENTS.md, families).
@pytest.mark.heavy
@pytest.mark.parametrize("opt_in", sorted(_OPT_INS))
@pytest.mark.parametrize("runner", _RUNNERS)
@pytest.mark.parametrize("door", _DOORS)
def test_module_selector_grad_zero_every_spelling(
    tiny: _Fixture, door: str, runner: str, opt_in: str
) -> None:
    """``tl.module`` + ``grad_zero`` matches the oracle on every door x runner x opt-in."""

    selector = tl.module(_MLP)
    _check_spelling(
        tiny,
        door=door,
        runner=runner,
        opt_in=opt_in,
        selector=selector,
        helper=tl.zero_grad(),
        want=tiny.oracle("module", _MLP, _zero),
    )


@pytest.mark.parametrize("helper_name", sorted(_HELPERS))
@pytest.mark.parametrize("door", ("intervene", "attach_hooks"))
def test_func_selector_every_helper(tiny: _Fixture, door: str, helper_name: str) -> None:
    """``tl.func`` reaches every layer norm, including the multi-pass graph's."""

    build, fn = _HELPERS[helper_name]
    _check_spelling(
        tiny,
        door=door,
        runner="trace.backward",
        opt_in="backward_ready",
        selector=tl.func("layer_norm"),
        helper=build(),
        want=tiny.oracle("every_layernorm", "", fn),
    )


@pytest.mark.parametrize("helper_name", sorted(_HELPERS))
@pytest.mark.parametrize("door", ("intervene", "attach_hooks"))
def test_grad_fn_selector_every_helper(tiny: _Fixture, door: str, helper_name: str) -> None:
    """``tl.grad_fn`` edits the grad inputs of every matching autograd node."""

    build, fn = _HELPERS[helper_name]
    _check_spelling(
        tiny,
        door=door,
        runner="trace.backward",
        opt_in="backward_ready",
        selector=tl.grad_fn(name=_LN_NODE),
        helper=build(),
        want=tiny.oracle("grad_fn", _LN_NODE, fn),
    )


def test_grad_fn_selector_grad_input_family(tiny: _Fixture) -> None:
    """The grad_fn-mounted ``grad_clamp`` family edits grad inputs like the oracle."""

    _check_spelling(
        tiny,
        door="intervene",
        runner="loss.backward",
        opt_in="backward_ready",
        selector=tl.grad_fn(name=_LN_NODE),
        helper=tl.clamp_grad(0.0, 0.0),
        want=tiny.oracle("grad_fn", _LN_NODE, _zero),
    )


@pytest.mark.parametrize("door", ("attach_hooks", "layer.attach_hooks", "set"))
def test_label_selector_post_capture_doors(tiny: _Fixture, door: str) -> None:
    """A finalized label addresses one layer norm on every post-capture door."""

    _check_spelling(
        tiny,
        door=door,
        runner="trace.backward",
        opt_in="backward_ready",
        selector=tl.label(tiny.ln_label),
        helper=tl.zero_grad(),
        want=tiny.oracle("module", _LN_SINGLE, _zero),
    )


def test_in_module_composite_selector(tiny: _Fixture) -> None:
    """``in_module & func`` selects the MLP's projections inside the multi-pass graph."""

    model = tiny.fresh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            tiny.ids,
            capture=tl.options.CaptureOptions(backward_ready=True),
            intervene=tl.when(tl.in_module(_MLP) & tl.func("addmm"), tl.zero_grad()),
        )
        got = _run_backward(trace, model, "trace.backward")
    assert _backward_fires(trace) > 0
    # c_proj's addmm output gradient is zeroed, so nothing flows back into c_fc.
    assert not torch.count_nonzero(got[f"{_MLP}.c_fc.weight"])
    assert not torch.equal(got["transformer.wpe.weight"], tiny.clean["transformer.wpe.weight"])


def test_label_selector_on_capture_door_refuses_typed(tiny: _Fixture) -> None:
    """A finalized label at ``intervene=`` refuses typed: live selectors run pre-postprocess."""

    with pytest.raises(LiveModeLabelError):
        trace = tl.trace(
            tiny.fresh(),
            tiny.ids,
            capture=tl.options.CaptureOptions(backward_ready=True),
            intervene=tl.when(tl.label(tiny.ln_label), tl.zero_grad()),
        )
        trace.backward(_loss(trace[trace.output_layers[0]].out))


def test_zero_match_forward_selector_gradient_edit_warns(tiny: _Fixture) -> None:
    """A gradient edit whose forward selector matches nothing is disclosed, not silent."""

    with pytest.warns(UserWarning, match="matched zero sites; no intervention fired"):
        tl.trace(
            tiny.fresh(),
            tiny.ids,
            capture=tl.options.CaptureOptions(backward_ready=True),
            intervene=tl.when(tl.in_module(_MLP) & tl.func("linear"), tl.zero_grad()),
        )


def test_rerun_refuses_gradient_edit_with_its_real_cause(tiny: _Fixture) -> None:
    """A gradient edit on the forward-only rerun engine refuses naming that cause."""

    model = tiny.fresh()
    trace = tl.trace(model, tiny.ids, capture=tl.options.CaptureOptions(backward_ready=True))
    # grouping rule rows 26-27: the shared block's MLP runs once per block slot,
    # so the module selector matches its 2 calls and discloses the fan-out.
    with (
        pytest.warns(MultiMatchWarning, match="matched 2 sites"),
        pytest.raises(LiveModeLabelError) as info,
    ):
        trace.do(
            tl.module(_MLP),
            tl.zero_grad(),
            model=model,
            inputs=tiny.ids,
            intervention=tl.options.InterventionOptions(engine="rerun", confirm_mutation=True),
        )
    assert info.value.fields["code"] == "rerun_gradient_edit_unsupported"


@pytest.mark.parametrize("helper_name", sorted(_HELPERS))
def test_bind_refuses_gradient_edits_typed(tiny: _Fixture, helper_name: str) -> None:
    """The capture-free bind lane refuses a gradient edit instead of editing activations."""

    build, _ = _HELPERS[helper_name]
    with pytest.raises(BindingPreflightError, match="edits gradients"):
        bound = tl.when(tl.module(_MLP), build()).bind(tiny.fresh())
        bound(tiny.ids)


def _scaled_layernorm_oracle(fixture: _Fixture, addresses: tuple[str, ...]) -> Grads:
    """Return gradients with the output gradient of each named module doubled."""

    model = fixture.fresh()

    def _on_output(module: Any, args: Any, out: Any) -> None:
        """Double the gradient flowing back into the module output."""

        del module, args
        out.register_hook(_double)

    handles = [model.get_submodule(name).register_forward_hook(_on_output) for name in addresses]
    _loss(model(fixture.ids)).backward()
    for handle in handles:
        handle.remove()
    return _grads(model)


requires_offline_venue = pytest.mark.skipif(
    not IN_OFFLINE_VENUE,
    reason=(
        "GATE R1_OFFLINE_VENUE: not in the offline preflighted venue; export "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 with a warmed cache. In venue "
        "this suite can NEVER skip."
    ),
)


@pytest.fixture(scope="module")
def real() -> _Fixture:
    """Pretrained gpt2 from the local cache."""

    model = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    return _Fixture(model, torch.tensor([[464, 3290, 318, 257, 922, 3290]]))


@requires_offline_venue
@pytest.mark.heavy
@pytest.mark.real_model
class TestRealGPT2:
    """The same oracle checks on the pretrained checkpoint (the August crash model)."""

    @pytest.mark.parametrize(
        ("door", "runner"),
        [
            ("intervene", "trace.backward"),
            ("attach_hooks", "loss.backward"),
            ("layer.attach_hooks", "autograd.grad"),
            ("set", "trace.log_backward"),
        ],
    )
    def test_module_selector_doors(self, real: _Fixture, door: str, runner: str) -> None:
        """``tl.module`` + ``grad_zero`` on a mid-stack MLP matches the oracle."""

        target = "transformer.h.5.mlp"
        _check_spelling(
            real,
            door=door,
            runner=runner,
            opt_in="backward_ready",
            selector=tl.module(target),
            helper=tl.zero_grad(),
            want=real.oracle("module", target, _zero),
        )

    def test_func_in_block_selector(self, real: _Fixture) -> None:
        """``tl.func`` scoped to one block scales both of its layer-norm gradients."""

        model = real.fresh()
        selector = tl.func("layer_norm") & tl.in_module("transformer.h.3")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = tl.trace(
                model,
                real.ids,
                capture=tl.options.CaptureOptions(backward_ready=True),
                intervene=tl.when(selector, tl.scale_grad(2.0)),
            )
            got = _run_backward(trace, model, "trace.backward")
        want = _scaled_layernorm_oracle(real, ("transformer.h.3.ln_1", "transformer.h.3.ln_2"))
        _assert_matches_oracle(got, want, real.clean)
        assert _backward_fires(trace) > 0
