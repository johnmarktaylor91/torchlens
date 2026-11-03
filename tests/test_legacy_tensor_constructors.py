"""Legacy tensor constructors capture as logged ops and validate.

2017-2019 model code builds tensors with ``torch.<dtype>Tensor(...)`` (directly,
or through a module attribute bound at init such as ``self.FloatTensor =
torch.FloatTensor``) and wraps tensors in ``torch.autograd.Variable``. Their C
constructors used to dispatch ``aten.empty`` / ``aten.detach`` with no owning
logged op, so ``tl.validate`` failed ``bfs_completeness`` (the real Sylvester
flow VAE). These tests pin the capture, the eager-equal values, the class
identity and type behavior, and the exact unwrap restore.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from torch.autograd import Variable

import torchlens as tl
from torchlens.errors import TraceCaptureError


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Run seeded forward validation."""

    torch.manual_seed(0)
    return bool(tl.validate(model, x, scope="forward"))


def _func_names(model: nn.Module, x: torch.Tensor) -> list[str]:
    """Return the captured op func names in execution order."""

    torch.manual_seed(0)
    trace = tl.trace(model, x)
    return [op.func_name for op in trace._layer_list]


def _outcome(fn) -> tuple[object, str]:
    """Return ("ok", shape/dtype) for a successful call, else (type, message)."""

    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - the outcome comparison records any eager error
        return type(exc), str(exc)
    return "ok", f"{tuple(out.shape)} {out.dtype}"


def _error_of(fn) -> tuple[type, str]:
    """Return the (type, message) a call raises."""

    with pytest.raises(Exception) as info:
        fn()
    return type(info.value), str(info.value)


class SylvesterReparam(nn.Module):
    """The Sylvester VAE idiom: an init-bound class alias, ``normal_``, ``Variable``."""

    def __init__(self) -> None:
        super().__init__()
        self.FloatTensor = torch.FloatTensor
        self.mu = nn.Linear(4, 4)
        self.logvar = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.mu(x), self.logvar(x)
        std = logvar.mul(0.5).exp_()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class Plain(nn.Module):
    """No legacy constructor anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)) + torch.empty(2, 4).normal_()


def test_variable_on_non_tensor_raises_inside_capture() -> None:
    """``Variable(3)`` inside a captured forward still raises eager's TypeError."""

    class BadVariable(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + Variable(3)

    expected_type, expected_message = _error_of(lambda: Variable(3))
    with pytest.raises(TraceCaptureError, match="Variable data has to be a tensor") as excinfo:
        tl.trace(BadVariable(), torch.randn(2, 4))
    assert type(excinfo.value.__cause__) is expected_type
    assert "Variable data has to be a tensor" in expected_message


def test_unwrap_restores_legacy_classes_exactly_and_rewrap_reinstalls() -> None:
    """``unwrap_torch`` restores ``__new__``, ``tp_new`` and immutability; rewrap patches again."""

    from torchlens._utils._type_new_slot import _type_view
    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch

    tl.trace(Plain(), torch.randn(2, 4))
    installed = legacy_ctors.installed_legacy_constructor_classes()
    assert installed["FloatTensor"] is torch.FloatTensor
    assert installed["Variable"] is Variable
    assert installed["cuda_FloatTensor"] is torch.cuda.FloatTensor
    records = {name: rec.patch for name, rec in _installed_records(legacy_ctors).items()}

    unwrap_torch()
    try:
        assert legacy_ctors.installed_legacy_constructor_classes() == {}
        for name, patch in records.items():
            cls = patch.cls
            assert _type_view(cls).tp_new == patch.original_tp_new, name
            if name == "Variable":
                assert "__new__" not in cls.__dict__
            else:
                assert cls.__dict__["__new__"] is patch.original_dict_new, name
                assert cls.__flags__ & (1 << 8), name  # still immutable
                with pytest.raises(TypeError, match="immutable"):
                    cls.probe_attr = 1  # type: ignore[attr-defined]
        assert torch.FloatTensor(2).shape == (2,)
        assert torch.equal(Variable(torch.ones(2)), torch.ones(2))
    finally:
        tl.trace(Plain(), torch.randn(2, 4))
    assert legacy_ctors.installed_legacy_constructor_classes()["FloatTensor"] is torch.FloatTensor
    assert _validate(SylvesterReparam(), torch.randn(2, 4))


@pytest.mark.smoke
def test_legacy_constructor_ops_never_become_runnable(tmp_path) -> None:
    """The resolver never resolves the legacy-constructor ops; a runnable run refuses typed.

    The legacy dtype constructors carry the same hidden ``cdata=`` raw-pointer
    overload as ``torch.Tensor.__new__`` (the documented bounded disposition), so
    capture support must not make them replayable from an untrusted bundle: the
    dtype classes stay ``nonforward_callable_denied`` and ``Variable`` stays
    unresolved, and running a saved runnable artifact refuses at reattachment.
    """

    from torchlens._io.runnable import (
        build_sparse_run_descriptor,
        preflight_sparse_run_descriptor,
    )
    from torchlens._runnable import ResolverStatus
    from torchlens.errors import ReattachError

    x = torch.randn(2, 4)
    trace = tl.trace(
        SylvesterReparam().eval(),
        x,
        capture=tl.options.CaptureOptions(
            replayable=True, capture_container_structure=True, cache=False
        ),
    )
    report, attachments = preflight_sparse_run_descriptor(build_sparse_run_descriptor(trace))
    assert attachments is None
    statuses = {
        record.recorded_key.qualname: (record.status, record.provenance)
        for record in report.resolver_records
        if record.recorded_key.qualname in {"FloatTensor", "Variable"}
    }
    assert statuses["FloatTensor"] == (ResolverStatus.UNAVAILABLE, "nonforward_callable_denied")
    assert statuses["Variable"][0] is ResolverStatus.UNAVAILABLE
    path = tmp_path / "legacy.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    with pytest.raises(ReattachError):
        tl.load(path).run(inputs=x, seed=0)


def _installed_records(legacy_ctors):
    """Return the installed records keyed by func name (test-only peek)."""

    return {rec.func_name: rec for rec in legacy_ctors._INSTALLED.values()}


class VariableOnly(nn.Module):
    """``Variable`` inside a forward, nothing else legacy."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Variable(x * 2) + 1


def _set_tp_new(cls: type, address: int) -> None:
    """Write a saved ``tp_new`` pointer back (test cleanup of a foreign patch)."""

    import ctypes

    from torchlens._utils._type_new_slot import _type_view

    _type_view(cls).tp_new = address
    ctypes.pythonapi.PyType_Modified(ctypes.py_object(cls))


def test_foreign_python_new_on_variable_is_skipped_not_recursed() -> None:
    """A ``Variable.__new__`` installed by another tool before wrap stays eager and is disclosed.

    Wrapping over it used to snapshot ``slot_tp_new`` as the "original", which
    re-entered the TorchLens override forever (RecursionError on every call,
    inside or outside capture). The dtype classes are still patched.
    """

    from torchlens._errors import TorchLensWarning
    from torchlens._utils._type_new_slot import _type_view
    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    base_new = torch._C._LegacyVariableBase.__new__
    foreign_calls: list[int] = []

    def foreign_new(subtype, *args, **kwargs):
        foreign_calls.append(1)
        return base_new(subtype, *args, **kwargs)

    unwrap_torch()
    c_tp_new = _type_view(Variable).tp_new
    Variable.__new__ = staticmethod(foreign_new)  # type: ignore[method-assign]
    try:
        getattr(legacy_ctors, "_WARNED_FOREIGN_NEW", set()).discard("Variable")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrap_torch()
        t = torch.ones(2)
        out = Variable(t)  # recursed forever before the fix
        assert torch.equal(out, t) and out is not t and foreign_calls
        assert any(
            issubclass(w.category, TorchLensWarning) and "Variable uncaptured" in str(w.message)
            for w in caught
        ), [str(w.message) for w in caught]
        leaf = Variable(t, requires_grad=True)
        assert leaf.requires_grad and leaf.grad_fn is None
        installed = legacy_ctors.installed_legacy_constructor_classes()
        assert "Variable" not in installed
        assert installed["FloatTensor"] is torch.FloatTensor
        assert legacy_ctors.skipped_legacy_constructor_classes() == {"Variable": Variable}
        assert Variable.__dict__["__new__"].__func__ is foreign_new
        with warnings.catch_warnings():
            # The skipped Variable is uncaptured, so its output has no provenance
            # (the disclosed gap); the call itself must still run eagerly.
            warnings.simplefilter("ignore")
            trace = tl.trace(VariableOnly(), torch.randn(2, 4))
        assert "Variable" not in [op.func_name for op in trace.ops]
    finally:
        unwrap_torch()
        del Variable.__new__
        _set_tp_new(Variable, c_tp_new)
        tl.trace(Plain(), torch.randn(2, 4))
    assert legacy_ctors.installed_legacy_constructor_classes()["Variable"] is Variable
    assert legacy_ctors.skipped_legacy_constructor_classes() == {}
    assert _validate(VariableOnly(), torch.randn(2, 4))


def _drifted_mirror(insert_before: str):
    """Return a ``PyTypeObject`` mirror with one extra pointer field before ``insert_before``."""

    import ctypes

    from torchlens._utils._type_new_slot import _PyTypeObjectToNew

    fields = list(_PyTypeObjectToNew._fields_)
    index = [name for name, _ctype in fields].index(insert_before)
    fields.insert(index, ("_drift", ctypes.c_void_p))
    return type("_DriftedTypeObject", (ctypes.Structure,), {"_fields_": fields})


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "insert_before",
    [
        "tp_traverse",
    ],
)
def test_layout_probe_rejects_drift_before_tp_new(monkeypatch, insert_before: str) -> None:
    """The probe catches a struct drift between ``tp_doc`` and ``tp_new``, or at ``tp_new`` itself."""

    from torchlens._utils import _type_new_slot

    roster = [torch.FloatTensor, torch.LongTensor, torch.cuda.FloatTensor, Variable]
    assert all(_type_new_slot.probe_type_new_slot_patch(cls) for cls in roster)
    mirror = _drifted_mirror(insert_before)
    monkeypatch.setattr(_type_new_slot, "_type_view", lambda cls: mirror.from_address(id(cls)))
    assert not any(_type_new_slot.probe_type_new_slot_patch(cls) for cls in roster)


def test_failed_layout_probe_leaves_everything_unwrapped_and_flag_false(monkeypatch) -> None:
    """A layout mismatch at install flips the flag and writes nothing (no crash, no patch)."""

    import types

    from torchlens._utils import _torch_compat, _type_new_slot
    from torchlens._utils._torch_compat import TorchCapabilityWarning
    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    real_view = _type_new_slot._type_view
    unwrap_torch()
    c_tp_new = {cls: real_view(cls).tp_new for cls in (torch.FloatTensor, Variable)}
    try:
        with monkeypatch.context() as patch:
            patch.setattr(_torch_compat, "_warned_missing_capabilities", set())
            patch.setattr(_torch_compat, "HAS_LEGACY_CONSTRUCTOR_NEW_PATCH", True)
            mirror = _drifted_mirror("tp_new")
            patch.setattr(_type_new_slot, "_type_view", lambda cls: mirror.from_address(id(cls)))
            with pytest.warns(TorchCapabilityWarning, match="HAS_LEGACY_CONSTRUCTOR_NEW_PATCH"):
                wrap_torch()
            assert _torch_compat.HAS_LEGACY_CONSTRUCTOR_NEW_PATCH is False
            assert legacy_ctors.installed_legacy_constructor_classes() == {}
            assert type(torch.FloatTensor.__dict__["__new__"]) is types.BuiltinFunctionType
            assert "__new__" not in Variable.__dict__
            assert {cls: real_view(cls).tp_new for cls in c_tp_new} == c_tp_new
            assert torch.FloatTensor(3).shape == (3,)
            assert torch.equal(Variable(torch.ones(2)), torch.ones(2))
            unwrap_torch()
    finally:
        tl.trace(Plain(), torch.randn(2, 4))
    assert legacy_ctors.installed_legacy_constructor_classes()["FloatTensor"] is torch.FloatTensor


def test_foreign_new_set_after_wrap_then_removed_is_repatched_and_captured() -> None:
    """A foreign ``Variable.__new__`` set over ours and later deleted never leaves a stale record.

    CPython keeps ``Variable`` on the ``slot_tp_new`` trampoline after the
    foreign ``__new__`` is deleted, with no Python ``__new__`` left in the MRO.
    The old slot-pointer heuristic read that as "foreign patch still present",
    kept the stale record, skipped the class and still listed it as installed,
    so ``Variable`` went silently uncaptured and later foreign patches went
    undisclosed. The next wrap must restore the recorded C constructor and
    re-patch; a foreign patch still in effect must be left untouched.
    """

    from torchlens._errors import TorchLensWarning
    from torchlens._utils._type_new_slot import _type_view
    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    c_tp_new = _type_view(Variable).tp_new
    tl.trace(Plain(), torch.randn(2, 4))
    ours = Variable.__dict__["__new__"].__func__

    def foreign_new(subtype, *args, **kwargs):
        return ours(subtype, *args, **kwargs)

    Variable.__new__ = staticmethod(foreign_new)  # type: ignore[method-assign]
    try:
        unwrap_torch()  # leaves the foreign patch as found
        wrap_torch()  # foreign patch still in effect: untouched, not re-patched over
        assert Variable.__dict__["__new__"].__func__ is foreign_new
        assert "Variable" not in legacy_ctors.skipped_legacy_constructor_classes()
        unwrap_torch()
        del Variable.__new__
        assert _type_view(Variable).tp_new != c_tp_new  # CPython left the trampoline
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrap_torch()
        assert not [w for w in caught if issubclass(w.category, TorchLensWarning)]
        assert legacy_ctors.installed_legacy_constructor_classes()["Variable"] is Variable
        assert legacy_ctors.skipped_legacy_constructor_classes() == {}
        record = _installed_records(legacy_ctors)["Variable"]
        assert Variable.__dict__["__new__"] is record.patch.installed_new
        assert record.patch.original_tp_new == c_tp_new
        assert "Variable" in _func_names(VariableOnly(), torch.randn(2, 4))
        assert _validate(VariableOnly(), torch.randn(2, 4))
        assert torch.equal(Variable(torch.ones(2)), torch.ones(2))
        unwrap_torch()
        assert _type_view(Variable).tp_new == c_tp_new
        assert "__new__" not in Variable.__dict__

        # The stale record no longer silences the disclosure of a later foreign patch.
        Variable.__new__ = staticmethod(foreign_new)  # type: ignore[method-assign]
        legacy_ctors._WARNED_FOREIGN_NEW.discard("Variable")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrap_torch()
        disclosures = [w for w in caught if issubclass(w.category, TorchLensWarning)]
        assert len(disclosures) == 1 and "Variable uncaptured" in str(disclosures[0].message)
        assert disclosures[0].message.fields["code"] == "legacy_constructor_uncaptured"
        assert legacy_ctors.skipped_legacy_constructor_classes() == {"Variable": Variable}
        assert "Variable" not in legacy_ctors.installed_legacy_constructor_classes()
    finally:
        unwrap_torch()
        if "__new__" in Variable.__dict__:
            del Variable.__new__
        _set_tp_new(Variable, c_tp_new)
        tl.trace(Plain(), torch.randn(2, 4))
    assert legacy_ctors.installed_legacy_constructor_classes()["Variable"] is Variable


def test_foreign_new_removed_before_unwrap_leaves_no_stale_trampoline() -> None:
    """Unwrap restores the C constructor when a foreign patch over ours is already gone.

    A foreign ``Variable.__new__`` set over ours and deleted while wrapped leaves
    CPython's ``slot_tp_new`` trampoline with no Python ``__new__`` in
    ``Variable``'s dict. An unwrap-and-stop process must still end exactly as
    found (C ``tp_new``, no dict entry, no record); a foreign patch still in
    effect at unwrap is left untouched and keeps its record.
    """

    from torchlens._utils._type_new_slot import _type_view
    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch

    unwrap_torch()
    c_tp_new = _type_view(Variable).tp_new
    tl.trace(Plain(), torch.randn(2, 4))
    ours = Variable.__dict__["__new__"].__func__

    def foreign_new(subtype, *args, **kwargs):
        return ours(subtype, *args, **kwargs)

    try:
        # Negative: a foreign patch still in effect at unwrap is never clobbered.
        Variable.__new__ = staticmethod(foreign_new)  # type: ignore[method-assign]
        unwrap_torch()
        assert Variable.__dict__["__new__"].__func__ is foreign_new
        assert "Variable" in _installed_records(legacy_ctors)
        del Variable.__new__
        tl.trace(Plain(), torch.randn(2, 4))  # re-patches from the recorded original

        # Positive: deleted before the unwrap, the class goes back to its C constructor.
        Variable.__new__ = staticmethod(foreign_new)  # type: ignore[method-assign]
        del Variable.__new__
        assert _type_view(Variable).tp_new != c_tp_new  # CPython left the trampoline
        unwrap_torch()
        assert _type_view(Variable).tp_new == c_tp_new
        assert "__new__" not in Variable.__dict__
        assert "Variable" not in _installed_records(legacy_ctors)
        assert torch.equal(Variable(torch.ones(2)), torch.ones(2))
    finally:
        unwrap_torch()
        if "__new__" in Variable.__dict__:
            del Variable.__new__
        _set_tp_new(Variable, c_tp_new)
        tl.trace(Plain(), torch.randn(2, 4))
    assert legacy_ctors.installed_legacy_constructor_classes()["Variable"] is Variable


def test_python_new_set_and_removed_before_wrap_is_captured_not_skipped() -> None:
    """A ``Variable.__new__`` set and deleted before the first wrap is no foreign patch.

    The class is left on the trampoline with nothing Python-level in its MRO,
    so it is captured (no skip, no warning) through the C constructor its base
    owns, and unwrap restores exactly the slot TorchLens found.
    """

    from torchlens._errors import TorchLensWarning
    from torchlens._utils._type_new_slot import _type_view, recover_c_new_from_bases
    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    c_tp_new = _type_view(Variable).tp_new
    Variable.__new__ = staticmethod(  # type: ignore[method-assign]
        lambda subtype, *a, **k: torch._C._LegacyVariableBase.__new__(subtype, *a, **k)
    )
    del Variable.__new__
    stale_slot = _type_view(Variable).tp_new
    try:
        assert stale_slot != c_tp_new
        assert recover_c_new_from_bases(Variable) == c_tp_new
        legacy_ctors._WARNED_FOREIGN_NEW.discard("Variable")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrap_torch()
        assert not [w for w in caught if issubclass(w.category, TorchLensWarning)]
        assert legacy_ctors.installed_legacy_constructor_classes()["Variable"] is Variable
        assert legacy_ctors.skipped_legacy_constructor_classes() == {}
        t = torch.ones(2)
        out = Variable(t)
        assert torch.equal(out, t) and out is not t
        leaf = Variable(t, requires_grad=True)
        assert leaf.requires_grad and leaf.grad_fn is None
        assert "Variable" in _func_names(VariableOnly(), torch.randn(2, 4))
        assert _validate(VariableOnly(), torch.randn(2, 4))
        unwrap_torch()
        assert _type_view(Variable).tp_new == stale_slot  # restored as found
        assert "__new__" not in Variable.__dict__
        assert torch.equal(Variable(t), t)
    finally:
        unwrap_torch()
        _set_tp_new(Variable, c_tp_new)
        tl.trace(Plain(), torch.randn(2, 4))
    assert legacy_ctors.installed_legacy_constructor_classes()["Variable"] is Variable


def test_c_constructor_recovery_refuses_what_it_cannot_prove() -> None:
    """``recover_c_new_from_bases`` returns ``None`` unless a base's C builtin is in effect."""

    from torchlens._utils._type_new_slot import (
        effective_new_entry,
        is_c_new_entry,
        recover_c_new_from_bases,
    )

    owner, entry = effective_new_entry(torch.FloatTensor)
    assert owner is torch.FloatTensor  # its own builtin (or our override while wrapped)
    assert recover_c_new_from_bases(torch.FloatTensor) is None

    class PyNew:
        def __new__(cls):
            return object.__new__(cls)

    class Child(PyNew):
        pass

    owner, entry = effective_new_entry(Child)
    assert owner is PyNew and not is_c_new_entry(entry)
    assert recover_c_new_from_bases(Child) is None
    assert is_c_new_entry(object.__dict__["__new__"])
