"""Deferred activation-payload clones (clone-on-write) behavior contract.

Eligible plain captures save payload ALIASES instead of eager clones; the
torch wrapper materializes a pending alias onto exclusive fresh storage
before any wrapped call writes its storage. These tests pin the contract:

  * saved payload bytes/metadata are identical to eager-clone behavior,
    including across in-place ops, ``out=`` destinations, ``__setitem__``,
    ``.data`` writes, ``inplace=True`` conveniences, and train-mode
    batch-norm buffer side effects;
  * post-hoc in-place edits of one saved activation never bleed into
    another (eager isolation semantics);
  * a user-retained live activation mutated after capture never corrupts
    the saved value;
  * the version-counter belt refuses loudly when a storage was mutated
    through a path interception never saw.
"""

import contextlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.backends.torch.wrappers as _wrappers
from torchlens.utils import tensor_utils as _tu


@contextlib.contextmanager
def _payload_clone_mode(defer: bool, grad_defer: bool = False):
    """Force deferred or eager payload clones for the duration of the block.

    ``grad_defer`` additionally opts into deferral of GRAPH-CONNECTED payloads
    (the default grad-enabled capture regime), which ships off by default.
    """
    prev_tu, prev_w = _tu._DEFER_ENABLED, _wrappers._COW_ENABLED
    prev_grad = _tu._DEFER_GRAD_ENABLED
    _tu._DEFER_ENABLED = defer
    _wrappers._COW_ENABLED = defer
    _tu._DEFER_GRAD_ENABLED = defer and grad_defer
    try:
        yield
    finally:
        _tu._DEFER_ENABLED = prev_tu
        _wrappers._COW_ENABLED = prev_w
        _tu._DEFER_GRAD_ENABLED = prev_grad


class _InplaceZoo(nn.Module):
    """Every wrapped mutation surface the interceptor must cover."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.lin = nn.Linear(8, 8)
        self.drop = nn.Dropout(p=0.5, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.lin(x)
        y = self.bn(y)  # train mode: buffer side effects (state stays eager)
        y = self.relu(y)  # in-place relu on the bn output storage
        z = y * 2
        z += 1  # augmented-assignment dunder
        w = torch.empty_like(z)
        torch.add(z.detach(), 3, out=w)  # out= destination
        w[0, 0:2] = 5.0  # __setitem__
        v = w.clone()
        v.data.mul_(2)  # write through a .data alias
        u = self.drop(v)  # F.dropout(..., inplace=True) under the hood
        q = u.reshape(2, 4, 2)  # view chain: payloads share one storage
        q2 = q.permute(0, 2, 1)
        self.stash = z  # user-retained live activation
        return q2.contiguous().sum(dim=-1)


def _zoo_trace(defer: bool, seed: int = 7, grad: bool = False, grad_defer: bool = False):
    with _payload_clone_mode(defer, grad_defer=grad_defer):
        torch.manual_seed(seed)
        model = _InplaceZoo().train()
        torch.manual_seed(seed)
        x = torch.randn(2, 8)
        if grad:
            log = tl.trace(model, x, capture=tl.options.CaptureOptions(random_seed=99))
        else:
            with torch.no_grad():
                log = tl.trace(model, x, capture=tl.options.CaptureOptions(random_seed=99))
    return model, log


def _tensor_payload_labels(log):
    return [
        k
        for k in log.layer_labels
        if isinstance(log[k].out, torch.Tensor)
        # empty_like output is uninitialized memory: nondeterministic across
        # runs under eager cloning too, so it carries no byte contract.
        and not k.startswith("empty")
    ]


def _assert_payloads_identical(log_eager, log_defer):
    labels_e = list(log_eager.layer_labels)
    labels_d = list(log_defer.layer_labels)
    assert labels_e == labels_d
    for k in _tensor_payload_labels(log_eager):
        a, b = log_eager[k].out, log_defer[k].out
        assert a.dtype == b.dtype, k
        assert a.shape == b.shape, k
        assert a.stride() == b.stride(), k
        assert a.requires_grad == b.requires_grad, k
        assert (a.grad_fn is None) == (b.grad_fn is None), k
        if a.numel():
            ac = a.detach().contiguous()
            bc = b.detach().contiguous()
            assert ac.numpy().tobytes() == bc.numpy().tobytes(), k


def test_deferred_payloads_byte_identical_no_grad():
    _, log_eager = _zoo_trace(defer=False)
    _, log_defer = _zoo_trace(defer=True)
    _assert_payloads_identical(log_eager, log_defer)


def test_deferred_payloads_byte_identical_grad_mode():
    # Grad-enabled captures defer only no-autograd payloads; the public
    # surface (values AND grad_fn presence) must be unchanged either way.
    _, log_eager = _zoo_trace(defer=False, grad=True)
    _, log_defer = _zoo_trace(defer=True, grad=True)
    _assert_payloads_identical(log_eager, log_defer)


def test_post_hoc_inplace_edit_isolation():
    _, log = _zoo_trace(defer=True)
    labels = _tensor_payload_labels(log)
    snapshot = {k: log[k].out.clone() for k in labels}
    victim = next(k for k in labels if log[k].out.is_floating_point() and log[k].out.numel() > 0)
    log[victim].out.add_(1234.5)
    for k in labels:
        if k == victim:
            continue
        assert torch.equal(snapshot[k], log[k].out), k


def test_retained_live_activation_mutation_does_not_corrupt_saved():
    model, log = _zoo_trace(defer=True)
    labels = _tensor_payload_labels(log)
    saved = {k: log[k].out.clone() for k in labels}
    # The model kept a live handle to a mid-forward activation; a post-capture
    # in-place write through it must be intercepted before the bytes move.
    model.stash.add_(999.0)
    for k in labels:
        assert torch.equal(saved[k], log[k].out), k


def test_validation_tripwire_green_on_deferred_capture():
    with _payload_clone_mode(True):
        torch.manual_seed(7)
        model = _InplaceZoo().train()
        torch.manual_seed(7)
        x = torch.randn(2, 8)
        assert tl.validate(model, x, scope="forward", random_seed=99) is True


def test_version_belt_refuses_unintercepted_mutation():
    x = torch.randn(4)
    _tu.arm_deferred_payload_window(frozenset())
    try:
        alias = _tu.safe_copy(x, detach_tensor=True, save_mode="copy")
    finally:
        _tu.disarm_deferred_payload_window()
    assert alias.data_ptr() == x.data_ptr()  # genuinely deferred
    # Simulate a mutation path the wrapper never sees: the raw dispatcher
    # surface bypasses TorchLens wrapping entirely but still bumps the shared
    # version counter, which is exactly what the belt exists to catch.
    torch.ops.aten.relu_(x)
    key, entry = next(
        (k, e) for k, entries in _tu._DEFER_PENDING.items() for e in entries if e.ref() is alias
    )
    with pytest.raises(RuntimeError, match="deferred-clone tripwire"):
        _tu._belt_check_pending_alias(entry, alias)
    _tu._DEFER_PENDING.pop(key, None)


# ---------------------------------------------------------------------------
# Graph-connected payload deferral (the default grad-enabled capture regime).
#
# A plain ``detach()`` alias cannot stand in for ``x.clone()`` when the clone
# stays attached to the graph, so these payloads mint an identity-grafted
# alias instead. The tests below pin the two hazards that mint closes, the
# byte/gradient equivalence it must preserve, and the residual that keeps it
# opt-in.
# ---------------------------------------------------------------------------


def _grad_zoo_payloads(defer: bool, grad_defer: bool):
    """Trace the zoo grad-enabled and return (log, saved tensor payloads)."""
    _, log = _zoo_trace(defer=defer, grad=True, grad_defer=grad_defer)
    return log, _tensor_payload_labels(log)


def test_grad_connected_deferral_is_off_by_default():
    # Residual H3 (autograd's saved-tensor machinery is a second holder that
    # interception cannot rebind) keeps this opt-in; pin the shipped default so
    # it cannot be flipped on by accident.
    assert _tu._DEFER_GRAD_ENABLED is False


@pytest.mark.smoke
def test_grad_connected_deferral_actually_defers():
    # Guard against the whole feature silently degrading to eager clones: at
    # least one graph-connected payload must genuinely share its source
    # storage, and it must look exactly like a clone otherwise.
    log, labels = _grad_zoo_payloads(defer=True, grad_defer=True)
    grafted = [
        k
        for k in labels
        if log[k].out.grad_fn is not None
        and type(log[k].out.grad_fn).__name__.startswith("_DeferredPayloadCloneFn")
    ]
    assert grafted, "no graph-connected payload was deferred"
    for k in grafted:
        out = log[k].out
        assert out.requires_grad is True, k
        # H1: an autograd VIEW would let a later in-place write to the source
        # rebase this grad_fn and re-route the gradient past the capture point.
        assert out._is_view() is False, k


def test_grad_connected_payloads_byte_identical_to_eager_clones():
    _, log_eager = _zoo_trace(defer=False, grad=True)
    _, log_defer = _zoo_trace(defer=True, grad=True, grad_defer=True)
    _assert_payloads_identical(log_eager, log_defer)


def test_grad_connected_payload_gradients_match_eager_clones():
    """Backward through every saved payload must match the eager clone exactly.

    This is the real fidelity bar for graph-connected deferral: the grafted
    identity node has to reproduce ``CloneBackward0``'s gradient, and
    materialization (the zoo mutates several payload storages in place) must
    not poison the graph on the way.
    """

    def grads(defer: bool, grad_defer: bool):
        model, log = _zoo_trace(defer=defer, grad=True, grad_defer=grad_defer)
        out = {}
        for k in _tensor_payload_labels(log):
            payload = log[k].out
            if not (payload.requires_grad and payload.is_floating_point()):
                continue
            got = torch.autograd.grad(
                payload.sum(),
                [p for p in model.parameters() if p.requires_grad],
                retain_graph=True,
                allow_unused=True,
            )
            out[k] = [None if g is None else g.clone() for g in got]
        return out

    eager = grads(defer=False, grad_defer=False)
    deferred = grads(defer=True, grad_defer=True)
    assert set(eager) == set(deferred)
    assert eager, "no graph-connected payload was reachable by backward"
    for k in eager:
        for ge, gd in zip(eager[k], deferred[k]):
            assert (ge is None) == (gd is None), k
            if ge is not None:
                assert torch.equal(ge, gd), k


def test_grad_connected_alias_materialization_preserves_backward():
    """H2 regression: ``set_`` has no derivative and would poison the graph.

    Rebinding a graph-connected alias with ``Tensor.set_`` leaves a payload
    whose backward dies with "derivative for set_ is not implemented" — a
    corrupted saved activation that only shows up when someone differentiates.
    """
    x = torch.randn(4, 4, requires_grad=True)
    src = x * 2
    alias = _tu._mint_graph_connected_alias(src)
    before = alias.detach().clone()
    _tu._rebind_alias_to_fresh_clone(alias)
    assert alias.data_ptr() != src.data_ptr()  # exclusive storage now
    assert torch.equal(alias.detach(), before)  # capture-time bytes preserved
    assert alias.requires_grad is True
    # The gradient path survived the rebind and still matches a plain clone.
    (grad,) = torch.autograd.grad(alias.sum(), [x])
    (expected,) = torch.autograd.grad((x * 2).clone().sum(), [x])
    assert torch.equal(grad, expected)


def test_grad_connected_alias_survives_source_mutation_without_rebasing():
    """H1 regression: an autograd view would re-route the gradient.

    ``aten.alias`` preserves ``requires_grad`` but registers a differentiable
    view, so an in-place write to the SOURCE rebases the payload's ``grad_fn``
    and the gradient starts flowing through ops that ran after the capture
    point (in-place ``ReLU`` makes this the common path). The grafted mint is
    not a view, so it stays pinned to the capture point.
    """
    x = torch.randn(4, 4, requires_grad=True)
    src = x * 2
    alias = _tu._mint_graph_connected_alias(src)
    _tu._rebind_alias_to_fresh_clone(alias)  # interception, before the write
    snapshot = alias.detach().clone()
    torch.ops.aten.relu_(src)  # source mutated in place afterwards
    assert torch.equal(alias.detach(), snapshot)
    assert type(alias.grad_fn).__name__.startswith("_DeferredPayloadCloneFn")
    (grad,) = torch.autograd.grad(alias.sum(), [x])
    assert torch.equal(grad, torch.full_like(x, 2.0))  # identity through mul, not relu


def test_grad_connected_alias_keeps_sharing_the_source_version_counter():
    """Residual H3 stays LOUD rather than silently wrong.

    The grafted alias deliberately keeps sharing the source's autograd version
    counter. An unintercepted write therefore still trips the belt, and an
    autograd graph built on a still-pending payload fails on autograd's own
    version guard instead of quietly differentiating stale bytes. Giving the
    alias a private counter would turn that loud failure into a silent wrong
    gradient, so this sharing is load-bearing, not an oversight.
    """
    x = torch.randn(4, 4, requires_grad=True)
    src = x * 2
    alias = _tu._mint_graph_connected_alias(src)
    version = int(alias._version)
    torch.ops.aten.relu_(src)
    assert int(alias._version) != version


def test_kill_switch_restores_eager_clones():
    import gc

    gc.collect()  # earlier tests' (cyclic) traces may still pin live aliases
    baseline_live = sum(
        1 for entries in _tu._DEFER_PENDING.values() for e in entries if e.ref() is not None
    )
    with _payload_clone_mode(False):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True))
        with torch.no_grad():
            log = tl.trace(model, torch.randn(2, 4))
        pending_live = sum(
            1 for entries in _tu._DEFER_PENDING.values() for e in entries if e.ref() is not None
        )
        assert pending_live == baseline_live
        assert isinstance(log[log.layer_labels[-1]].out, torch.Tensor)


# ---------------------------------------------------------------------------
# Positionally-passed ``inplace`` (the mobilenet_v3_small regression).
#
# torch's ``Hardswish.forward`` runs ``F.hardswish(input, self.inplace)`` —
# POSITIONAL — and the underscore mutation below it (``torch._C._nn.
# hardswish_``) is not a wrapped surface, so the ``F.hardswish`` wrapper is
# the only interception point. A kwargs-only ``inplace`` probe missed the
# mutation, left the upstream payload's pending alias stale, and the version
# belt refused at the next wrapped mutating call (2026-08-16, found by the
# L1 churn census on mobilenet_v3_small).
# ---------------------------------------------------------------------------


class _PositionalInplaceNet(nn.Module):
    """The mobilenet_v3 classifier pattern, minus torchvision."""

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 8)
        self.hs = nn.Hardswish(inplace=True)  # F.hardswish(input, self.inplace)
        self.hsig = nn.Hardsigmoid(inplace=True)  # F.hardsigmoid(input, self.inplace)
        self.drop = nn.Dropout(p=0.5, inplace=True)
        self.lin2 = nn.Linear(8, 4)

    def forward(self, x):
        y = self.lin1(x)
        y = self.hs(y)  # positional inplace mutates lin1's saved storage
        y = self.hsig(y)
        y = self.drop(y)  # wrapped _VF.dropout_ ran the belt check here
        return self.lin2(y)


def _positional_inplace_trace(defer: bool, seed: int = 3):
    with _payload_clone_mode(defer):
        torch.manual_seed(seed)
        model = _PositionalInplaceNet().train()
        torch.manual_seed(seed)
        x = torch.randn(2, 8)
        with torch.no_grad():
            log = tl.trace(model, x, capture=tl.options.CaptureOptions(random_seed=99))
    return log


def test_positional_inplace_index_located_at_decoration_time():
    import torch.nn.functional as F

    assert _wrappers._positional_inplace_index(F.hardswish) == 1
    assert _wrappers._positional_inplace_index(F.hardsigmoid) == 1
    assert _wrappers._positional_inplace_index(F.dropout) == 3
    # C builtins expose no signature and cannot carry the parameter.
    assert _wrappers._positional_inplace_index(torch._C._nn.hardswish_) is None
    assert _wrappers._positional_inplace_index(torch.add) is None


def test_positional_inplace_capture_does_not_trip_belt():
    log = _positional_inplace_trace(defer=True)
    assert isinstance(log[log.layer_labels[-1]].out, torch.Tensor)


def test_positional_inplace_payloads_byte_identical_to_eager():
    log_eager = _positional_inplace_trace(defer=False)
    log_defer = _positional_inplace_trace(defer=True)
    _assert_payloads_identical(log_eager, log_defer)
    # The saved upstream payload must hold PRE-mutation bytes: applying
    # hardswish to lin1's saved output must reproduce the hardswish payload.
    lin1 = next(k for k in log_defer.layer_labels if k.startswith("linear"))
    hs = next(k for k in log_defer.layer_labels if k.startswith("hardswish"))
    recomputed = torch.nn.functional.hardswish(log_defer[lin1].out)
    assert torch.equal(recomputed, log_defer[hs].out)
    assert not torch.equal(log_defer[lin1].out, log_defer[hs].out)


@pytest.mark.heavy
def test_mobilenet_v3_small_captures_and_validates():
    torchvision = pytest.importorskip("torchvision")

    torch.manual_seed(0)
    model = torchvision.models.mobilenet_v3_small().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        log = tl.trace(model, x)
    assert len(log.layer_labels) > 0
    torch.manual_seed(0)
    model2 = torchvision.models.mobilenet_v3_small().eval()
    assert tl.validate(model2, torch.randn(1, 3, 224, 224), scope="forward") is True
