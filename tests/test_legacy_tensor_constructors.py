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

DTYPE_CLASS_NAMES = (
    "FloatTensor",
    "DoubleTensor",
    "HalfTensor",
    "BFloat16Tensor",
    "LongTensor",
    "IntTensor",
    "ShortTensor",
    "CharTensor",
    "ByteTensor",
    "BoolTensor",
)


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Run seeded forward validation."""

    torch.manual_seed(0)
    return bool(tl.validate(model, x, scope="forward"))


def _func_names(model: nn.Module, x: torch.Tensor) -> list[str]:
    """Return the captured op func names in execution order."""

    torch.manual_seed(0)
    trace = tl.trace(model, x)
    return [op.func_name for op in trace.ops]


def _outcome(fn) -> tuple[object, str]:
    """Return ("ok", shape/dtype) for a successful call, else (type, message)."""

    try:
        out = fn()
    except Exception as exc:
        return type(exc), str(exc)
    return "ok", f"{tuple(out.shape)} {out.dtype}"


def _error_of(fn) -> tuple[type, str]:
    """Return the (type, message) a call raises."""

    with pytest.raises(Exception) as info:
        fn()
    return type(info.value), str(info.value)


class SizeForm(nn.Module):
    """``torch.<dtype>Tensor(*sizes)`` filled in place, then consumed."""

    def __init__(self, class_name: str) -> None:
        super().__init__()
        self.class_name = class_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = getattr(torch, self.class_name)(x.shape[0], 4).fill_(1)
        return x * 2 + e.to(x.dtype)


class DataForm(nn.Module):
    """``torch.<dtype>Tensor(data)``, then consumed."""

    def __init__(self, class_name: str) -> None:
        super().__init__()
        self.class_name = class_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = getattr(torch, self.class_name)([1, 0, 1, 1])
        return x * 2 + e.to(x.dtype)


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


class ModernReparam(SylvesterReparam):
    """The same reparameterization spelled with ``torch.empty`` and no ``Variable``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.mu(x), self.logvar(x)
        std = logvar.mul(0.5).exp_()
        eps = torch.empty(std.size()).normal_()
        return eps.mul(std).add_(mu)


class VariableForms(nn.Module):
    """``Variable(t)`` and ``Variable(t, requires_grad=True)`` inside a forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = Variable(x * 2)
        b = Variable(torch.ones(4), requires_grad=True)
        return a + b


class TensorBaseForms(nn.Module):
    """``torch.Tensor(size)`` and ``torch.Tensor(data)`` (already wrapped via ``__new__``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.Tensor(x.size()).normal_()
        f = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        return x + e + f


class TypeChecks(nn.Module):
    """``isinstance`` / ``x.type(...)`` against the legacy classes during capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.FloatTensor)
        assert not isinstance(x, torch.LongTensor)
        assert x.type() == "torch.FloatTensor"
        y = x.type(torch.DoubleTensor)
        assert isinstance(y, torch.DoubleTensor)
        assert y.type() == "torch.DoubleTensor"
        return y * 2


class Plain(nn.Module):
    """No legacy constructor anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)) + torch.empty(2, 4).normal_()


@pytest.mark.parametrize("class_name", DTYPE_CLASS_NAMES)
def test_dtype_class_size_form_captures_and_validates(class_name: str) -> None:
    """Every dtype class's size form is a logged op and validates."""

    model, x = SizeForm(class_name), torch.randn(2, 4)
    assert class_name in _func_names(model, x)
    assert _validate(model, x)


@pytest.mark.parametrize("class_name", DTYPE_CLASS_NAMES)
def test_dtype_class_data_form_captures_and_validates(class_name: str) -> None:
    """Every dtype class's data form is a logged op and validates."""

    model, x = DataForm(class_name), torch.randn(2, 4)
    assert class_name in _func_names(model, x)
    assert _validate(model, x)


def test_sylvester_reparam_idiom_captures_validates_and_matches_modern_spelling() -> None:
    """``self.FloatTensor(size).normal_()`` + ``Variable`` validates and draws RNG as eager does.

    With the capture seeded (``random_seed=7``) the legacy spelling's captured
    output equals its eager output under ``torch.manual_seed(7)``, and both equal
    the modern ``torch.empty(size).normal_()`` spelling: the legacy op consumes
    the RNG exactly as its modern equivalent.
    """

    model, x = SylvesterReparam().eval(), torch.randn(2, 4)
    modern = ModernReparam().eval()
    modern.load_state_dict(model.state_dict())
    assert model.FloatTensor is torch.FloatTensor
    names = _func_names(model, x)
    assert names.index("FloatTensor") < names.index("normal_") < names.index("Variable")
    assert _validate(model, x)

    outputs = []
    for candidate in (model, modern):
        trace = tl.trace(candidate, x, capture=tl.options.CaptureOptions(random_seed=7))
        (out_label,) = trace.output_layers
        outputs.append(trace[out_label].out)
    torch.manual_seed(7)
    eager_legacy = model(x)
    torch.manual_seed(7)
    eager_modern = modern(x)
    assert torch.equal(eager_legacy, eager_modern)
    assert torch.equal(outputs[0], eager_legacy)
    assert torch.equal(outputs[1], eager_modern)


def test_variable_forms_capture_and_validate() -> None:
    """``Variable(t)`` and ``Variable(t, requires_grad=True)`` are logged and validate."""

    model, x = VariableForms(), torch.randn(2, 4)
    assert _func_names(model, x).count("Variable") == 2
    assert _validate(model, x)


def test_tensor_base_constructor_forms_capture_and_validate() -> None:
    """``torch.Tensor(size)`` / ``torch.Tensor(data)`` keep capturing and validating."""

    model, x = TensorBaseForms(), torch.randn(2, 4)
    assert _func_names(model, x).count("__new__") == 2
    assert _validate(model, x)


def test_legacy_class_type_behavior_survives_capture() -> None:
    """Class identity, ``isinstance`` and ``x.type(cls)`` work during and after capture."""

    original_float_tensor = torch.FloatTensor
    model, x = TypeChecks(), torch.randn(2, 4)
    assert _validate(model, x)
    assert torch.FloatTensor is original_float_tensor
    assert type(torch.FloatTensor).__name__ == "tensortype"
    made = torch.FloatTensor(2, 3)
    assert isinstance(made, torch.FloatTensor) and made.dtype == torch.float32
    assert made.type() == "torch.FloatTensor"
    assert isinstance(made.type(torch.LongTensor), torch.LongTensor)


def test_plain_model_graph_has_no_legacy_ops() -> None:
    """A model without legacy constructors captures exactly the same op names."""

    model, x = Plain(), torch.randn(2, 4)
    names = _func_names(model, x)
    assert not set(names) & {*DTYPE_CLASS_NAMES, "Variable"}
    # Pinned from the pre-change base (87c393a7e): the graph is unchanged.
    assert names == ["none", "linear", "relu", "empty", "normal_", "__add__", "none"]
    assert _validate(model, x)


def test_legacy_constructors_outside_capture_behave_as_eager() -> None:
    """With logging disabled the patched constructors return eager results and errors."""

    tl.trace(Plain(), torch.randn(2, 4))  # make sure the wrappers are installed
    assert torch.equal(torch.LongTensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    sized = torch.DoubleTensor(2, 3)
    assert sized.shape == (2, 3) and sized.dtype == torch.float64
    assert torch.FloatTensor().shape == (0,)
    assert torch.FloatTensor(torch.Size([3])).shape == (3,)
    base = torch.ones(2)
    assert torch.FloatTensor(base).data_ptr() == base.data_ptr()
    torch.manual_seed(3)
    a = torch.FloatTensor(5).normal_()
    torch.manual_seed(3)
    b = torch.empty(5).normal_()
    assert torch.equal(a, b)
    t = torch.ones(3)
    v = Variable(t, requires_grad=True)
    assert v.requires_grad and v.is_leaf and torch.equal(v, t)
    assert _error_of(lambda: torch.FloatTensor("bad"))[0] is TypeError


def test_errors_and_warnings_match_unwrapped_eager() -> None:
    """Error types/messages and the ``volatile=`` warning are identical wrapped vs unwrapped."""

    from torchlens.backends.torch.wrappers import unwrap_torch

    def observe() -> list[object]:
        seen: list[object] = [
            _outcome(lambda: Variable(3)),
            _outcome(lambda: torch.FloatTensor("bad")),
            _outcome(lambda: torch.LongTensor(torch.ones(2))),
            _outcome(lambda: torch.FloatTensor(torch.ones(2), device="cpu")),
            _outcome(lambda: torch.FloatTensor(2, 3, device="meta")),
            # CPU build: the "not compiled with CUDA" error, unchanged by the patch.
            _outcome(lambda: torch.cuda.FloatTensor(3)),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Variable(torch.ones(2), volatile=True)
        seen.append([(w.category, str(w.message)) for w in caught])
        return seen

    tl.trace(Plain(), torch.randn(2, 4))
    wrapped = observe()
    unwrap_torch()
    try:
        unwrapped = observe()
    finally:
        tl.trace(Plain(), torch.randn(2, 4))  # re-install for later tests
    assert wrapped == unwrapped


def test_variable_on_non_tensor_raises_inside_capture() -> None:
    """``Variable(3)`` inside a captured forward still raises eager's TypeError."""

    class BadVariable(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + Variable(3)

    expected_type, expected_message = _error_of(lambda: Variable(3))
    with pytest.raises(expected_type, match="Variable data has to be a tensor"):
        tl.trace(BadVariable(), torch.randn(2, 4))
    assert "Variable data has to be a tensor" in expected_message


def test_unwrap_restores_legacy_classes_exactly_and_rewrap_reinstalls() -> None:
    """``unwrap_torch`` restores ``__new__``, ``tp_new`` and immutability; rewrap patches again."""

    from torchlens.backends.torch import legacy_ctors
    from torchlens.backends.torch.wrappers import unwrap_torch
    from torchlens.utils._type_new_slot import _type_view

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
    from torchlens.errors import ReattachError
    from torchlens.runnable import ResolverStatus

    x = torch.randn(2, 4)
    trace = tl.trace(
        SylvesterReparam().eval(),
        x,
        capture=tl.options.CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
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
    tl.save(trace, path, level="runnable", include_weights=True)
    with pytest.raises(ReattachError):
        tl.load(path).run(inputs=x, seed=0)


def _installed_records(legacy_ctors):
    """Return the installed records keyed by func name (test-only peek)."""

    return {rec.func_name: rec for rec in legacy_ctors._INSTALLED.values()}


def test_legacy_constructor_roster_rows_resolve_to_live_classes() -> None:
    """Every roster row names a live class on this torch (no row goes dead silently)."""

    from torchlens.constants import LEGACY_TENSOR_CONSTRUCTOR_SITES
    from torchlens.utils._torch_compat import get_optional_torch_namespace

    for namespace_name, class_name in LEGACY_TENSOR_CONSTRUCTOR_SITES:
        namespace = get_optional_torch_namespace(namespace_name)
        assert isinstance(getattr(namespace, class_name, None), type), (
            namespace_name,
            class_name,
        )
    names = {name for _ns, name in LEGACY_TENSOR_CONSTRUCTOR_SITES}
    assert names == {*DTYPE_CLASS_NAMES, "Variable"}
