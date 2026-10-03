"""Dynamo / fake-mode boundary: graceful typed degradation, never a raw crash.

Tensors inside a ``torch.compile`` region are data-free ``FakeTensor``s. Every
TorchLens step that reads a value -- ``safe_copy``, ``torch.equal``, ``.item()``,
``data_ptr()``, memory accounting -- is meaningless or fatal on them, so before
this guard:

* a compiled *callable* reached during capture died with a raw
  ``torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'FakeTensor'
  object has no attribute 'fake_mode'``, and
* a ``FakeTensor`` passed as a model input died mid-forward with a bare
  ``AssertionError: Please convert all Tensors to FakeTensors first`` from
  torch's own fake machinery, after TorchLens had already tripped torch's
  "almost definitely a bug in your code" warning by reading a FakeTensor's
  ``data_ptr()``.

TorchLens already unwraps compiled child ``nn.Module``s to their eager source
before capture, so these tests cover the two cases that unwrap cannot reach: a
compiled callable held as a plain attribute, and a compiled free function.

On torch >= 2.6 the rung-2 stance integration
(``torch.compiler.set_stance("force_eager")``, see
``test_compile_set_stance.py``) supersedes this bypass-and-ceiling behavior:
compiled interiors run their original eager Python and ARE logged. The
compiled-region tests below therefore pin the torch < 2.6 / no-stance FALLBACK
path through the ``_no_stance`` fixture -- the compile verdict's tamper
contract that the fallback stays unchanged.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._robustness import UnsupportedTensorVariantError, _tracing_tensor_kind
from torchlens.backends.torch import wrappers as torch_wrappers
from torchlens.utils import _torch_compat
from torchlens.utils._torch_compat import (
    dynamo_is_compiling,
    get_torch_capability_snapshot,
    get_tracing_tensor_types,
)


@pytest.fixture()
def _no_stance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the pre-2.6 fallback: simulate a torch without ``set_stance``.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    monkeypatch.setattr(_torch_compat, "HAS_SET_STANCE", False)


def _fake_mode() -> object:
    """Return torch's ``FakeTensorMode``, skipping the test when unavailable.

    Returns
    -------
    object
        ``FakeTensorMode`` class.
    """

    fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
    return fake_tensor.FakeTensorMode


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def test_dynamo_is_compiling_is_false_outside_a_compiled_region() -> None:
    """The probe must be quiet on the ordinary path."""

    assert dynamo_is_compiling() is False


def test_tracing_tensor_types_are_probed_not_version_parsed() -> None:
    """The fake/functional classes resolve through the capability probe."""

    types = get_tracing_tensor_types(force_probe=True)
    assert types, "torch 2.x is expected to expose at least FakeTensor"
    assert any(candidate.__name__ == "FakeTensor" for candidate in types)


def test_new_capability_flags_are_in_the_snapshot() -> None:
    """Every degradation point is visible as a named flag."""

    snapshot = get_torch_capability_snapshot()
    for flag in ("HAS_DYNAMO_IS_COMPILING", "HAS_TRACING_TENSOR_TYPES"):
        assert flag in snapshot
        assert isinstance(snapshot[flag], bool)


# ---------------------------------------------------------------------------
# Fake / functional tensors refuse at capture entry
# ---------------------------------------------------------------------------


def test_ordinary_tensor_is_not_classified_as_a_tracing_tensor() -> None:
    """The classifier must not false-positive on dense tensors or Parameters."""

    assert _tracing_tensor_kind(torch.randn(2, 4)) is None
    assert _tracing_tensor_kind(nn.Parameter(torch.randn(2, 4))) is None


def test_user_tensor_subclass_is_not_classified_as_a_tracing_tensor() -> None:
    """Name-based fallback must not catch unrelated user subclasses."""

    class MyTensor(torch.Tensor):
        """Plain user tensor subclass."""

    assert _tracing_tensor_kind(torch.randn(2, 4).as_subclass(MyTensor)) is None


def test_fake_tensor_is_classified() -> None:
    """A real FakeTensor is recognized by the exact probed class."""

    fake_mode = _fake_mode()
    with fake_mode():
        fake = torch.randn(2, 4)
    assert _tracing_tensor_kind(fake) == "FakeTensor"


def test_exact_torch_functional_tensor_is_classified_and_refused() -> None:
    """PyTorch's ordinary functional wrapper retains exact ``torch.Tensor`` type."""

    to_functional = getattr(torch, "_to_functional_tensor", None)
    if not callable(to_functional):
        pytest.skip("torch._to_functional_tensor is unavailable")
    functional = to_functional(torch.randn(2, 4))
    assert type(functional) is torch.Tensor
    assert _tracing_tensor_kind(functional) == "FunctionalTensor"

    with pytest.raises(UnsupportedTensorVariantError, match="FunctionalTensor in input"):
        tl.trace(nn.Identity(), functional)

    compat_row = tl.compat.report(nn.Identity(), torch.randn(2, 4)).row("vmap_functorch")
    assert "created inside forward" in compat_row.details


def test_fake_tensor_input_refuses_with_a_typed_explanatory_error() -> None:
    """A FakeTensor input must be refused up front, not crash mid-forward."""

    fake_mode = _fake_mode()
    with fake_mode():
        fake_x = torch.randn(2, 4)

    with pytest.raises(UnsupportedTensorVariantError) as excinfo:
        tl.trace(nn.Linear(4, 4), fake_x)
    message = str(excinfo.value)
    assert "FakeTensor in input" in message
    assert "no data" in message
    # The docs pointer must cite a location that actually exists (B8-30: the
    # old pointer named a phantom README/LIMITATIONS.md section).
    assert "docs/reference/limitations.md" in message


def test_fake_tensor_in_a_nested_input_container_refuses() -> None:
    """The input walk reaches fake tensors inside containers."""

    fake_mode = _fake_mode()
    with fake_mode():
        fake_x = torch.randn(2, 4)

    class TakesList(nn.Module):
        """Model whose forward takes a list of tensors."""

        def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
            """Sum a list of tensors.

            Parameters
            ----------
            xs:
                Input tensors.

            Returns
            -------
            torch.Tensor
                Element-wise sum.
            """

            return xs[0] + xs[1]

    with pytest.raises(UnsupportedTensorVariantError):
        tl.trace(TakesList(), [torch.randn(2, 4), fake_x])


def test_fake_tensor_keyword_input_refuses() -> None:
    """Keyword inputs are walked too."""

    fake_mode = _fake_mode()
    with fake_mode():
        fake_x = torch.randn(2, 4)

    class TakesKwarg(nn.Module):
        """Model with a keyword-only tensor argument."""

        def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            """Add an optional bias.

            Parameters
            ----------
            x:
                Input batch.
            bias:
                Optional bias tensor.

            Returns
            -------
            torch.Tensor
                Sum of the inputs.
            """

            return x if bias is None else x + bias

    with pytest.raises(UnsupportedTensorVariantError) as excinfo:
        tl.trace(TakesKwarg(), torch.randn(2, 4), {"bias": fake_x})
    assert "keyword input" in str(excinfo.value)


def test_fake_model_parameters_refuse() -> None:
    """A model built under fake mode holds no weights, so capture refuses."""

    fake_mode = _fake_mode()
    with fake_mode():
        fake_model = nn.Linear(4, 4)
        fake_x = torch.randn(2, 4)

    with pytest.raises(UnsupportedTensorVariantError) as excinfo:
        tl.trace(fake_model, fake_x)
    assert "FakeTensor" in str(excinfo.value)


def test_dense_capture_is_unaffected_after_fake_mode_was_used() -> None:
    """Importing/using fake mode must not disturb ordinary capture."""

    fake_mode = _fake_mode()
    with fake_mode():
        _unused = torch.randn(2, 4)

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x)
    assert trace.num_params > 0
    assert any("linear" in label for label in trace.layer_labels)


# ---------------------------------------------------------------------------
# Compiled regions degrade gracefully mid-capture
# ---------------------------------------------------------------------------


class _CompiledAttributeModel(nn.Module):
    """Model holding a compiled callable as a plain attribute.

    ``unwrap_compiled_submodules`` swaps compiled ``nn.Module`` children only, so
    a compiled callable in ``__dict__`` still reaches the wrappers and is exactly
    the case the in-wrapper boundary exists for.
    """

    def __init__(self) -> None:
        """Build an eager linear layer plus a compiled activation."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.compiled_activation = torch.compile(lambda t: torch.relu(t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the eager layer then the compiled activation.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Activated output.
        """

        return self.compiled_activation(self.fc(x))


@pytest.mark.heavy
def test_compiled_attribute_callable_degrades_instead_of_crashing(_no_stance: None) -> None:
    """Capture completes with an explanatory warning instead of a Dynamo crash."""

    torch.compiler.reset()
    model = _CompiledAttributeModel()
    compiled_callable = model.compiled_activation
    x = torch.randn(2, 4)

    with pytest.warns(UserWarning, match="torch.compile"):
        trace = tl.trace(model, x)

    # The eager part is captured; the compiled interior honestly is not.
    assert any("linear" in label for label in trace.layer_labels)
    assert not any("relu" in label for label in trace.layer_labels)
    assert model.compiled_activation is compiled_callable


@pytest.mark.heavy
def test_compiled_region_warning_names_the_gap_and_the_remedy(_no_stance: None) -> None:
    """The warning must explain what is missing and what to do about it."""

    torch.compiler.reset()
    model = _CompiledAttributeModel()
    x = torch.randn(2, 4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, x)

    messages = [str(item.message) for item in caught]
    dynamo_messages = [text for text in messages if "torch.compile (Dynamo) region" in text]
    assert dynamo_messages, f"expected a Dynamo warning, saw: {messages}"
    text = dynamo_messages[0]
    assert "are not logged" in text
    assert "FakeTensor" in text
    assert "OUTSIDE" in text
    assert "eager" in text


@pytest.mark.heavy
def test_compiled_region_marks_the_trace_as_having_an_unlogged_escape(_no_stance: None) -> None:
    """The gap is recorded on the Trace, not only shouted in a warning."""

    torch.compiler.reset()
    model = _CompiledAttributeModel()
    with pytest.warns(UserWarning, match="torch.compile"):
        trace = tl.trace(model, torch.randn(2, 4))
    assert trace._raw_dynamo_region_detected is True
    assert trace._raw_transform_escape_detected is True


@pytest.mark.heavy
def test_compiled_region_verdict_names_dynamo_not_an_incidental_symptom(_no_stance: None) -> None:
    """``capture_verified`` must be False for the RIGHT reason.

    Dynamo spawns compile threads and leaves unaccounted aten dispatches, so before
    the dedicated reason the Trace blamed ``owner_thread_tripwire_changed`` -- true,
    but pointing the user at threading rather than at the unlogged compiled region.
    """

    torch.compiler.reset()
    with pytest.warns(UserWarning, match="torch.compile"):
        trace = tl.trace(_CompiledAttributeModel(), torch.randn(2, 4))

    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "dynamo_region_not_logged"


@pytest.mark.heavy
def test_compiled_region_verdict_survives_a_warm_compile_cache(_no_stance: None) -> None:
    """The verdict must not depend on compilation happening during this capture.

    A second capture of an already-compiled callable spawns no new threads, so a
    verdict riding only on the thread tripwire would read ``capture_verified=True``
    on an equally incomplete Trace.
    """

    torch.compiler.reset()
    model = _CompiledAttributeModel()
    x = torch.randn(2, 4)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tl.trace(model, x)  # warm the compile cache

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        warm = tl.trace(model, x)

    assert warm._raw_dynamo_region_detected is True
    assert warm.capture_verified is False
    assert warm.capture_verification_reason == "dynamo_region_not_logged"


@pytest.mark.heavy
def test_compiled_attribute_boundary_does_not_depend_on_is_compiling_timing(
    monkeypatch: pytest.MonkeyPatch,
    _no_stance: None,
) -> None:
    """Plain-attribute inventory arms the gap even when Dynamo's timing probe stays false."""

    torch.compiler.reset()
    model = _CompiledAttributeModel()
    x = torch.randn(2, 4)
    model(x)  # populate the compiled callable's execution cache before capture
    monkeypatch.setattr(torch_wrappers, "_is_inside_dynamo_compilation", lambda: False)

    with pytest.warns(UserWarning, match="torch.compile"):
        trace = tl.trace(model, x)

    assert trace._raw_dynamo_region_detected is True
    assert trace._raw_transform_escape_detected is True
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "dynamo_region_not_logged"


@pytest.mark.heavy
def test_compile_compat_row_reports_direct_attribute_and_free_function_residual(
    _no_stance: None,
) -> None:
    """Compatibility reporting matches the preflight inventory and its residual."""

    compiled_row = tl.compat.report(_CompiledAttributeModel(), torch.randn(2, 4)).row(
        "torch_compile"
    )
    assert compiled_row.detected is True
    assert "plain module attribute" in compiled_row.details
    assert "warm-cache" in compiled_row.details

    clear_row = tl.compat.report(nn.Identity(), torch.randn(2, 4)).row("torch_compile")
    assert clear_row.detected is False
    assert "globals/free-function references" in clear_row.details


@pytest.mark.heavy
def test_ordinary_capture_verdict_is_unchanged_by_the_dynamo_branch() -> None:
    """The new precedence must not touch a capture with no compiled region."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    assert trace._raw_dynamo_region_detected is False
    assert trace.capture_verification_reason != "dynamo_region_not_logged"


@pytest.mark.heavy
def test_compiled_region_warns_at_most_once_per_forward(_no_stance: None) -> None:
    """A compiled region hit many times must not spam one warning per op."""

    torch.compiler.reset()

    class ManyCompiledCalls(nn.Module):
        """Model that calls the same compiled callable repeatedly."""

        def __init__(self) -> None:
            """Build the eager layer and the compiled activation."""

            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.compiled_activation = torch.compile(lambda t: torch.relu(t) + 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Call the compiled activation several times.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Result after repeated compiled calls.
            """

            out = self.fc(x)
            for _ in range(3):
                out = self.compiled_activation(out)
            return out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(ManyCompiledCalls(), torch.randn(2, 4))

    dynamo_warnings = [
        item for item in caught if "torch.compile (Dynamo) region" in str(item.message)
    ]
    assert len(dynamo_warnings) == 1, f"expected exactly one warning, got {len(dynamo_warnings)}"


@pytest.mark.heavy
def test_model_is_reusable_after_a_compiled_region_capture(_no_stance: None) -> None:
    """Degrading must not leak capture state onto the model or torch."""

    torch.compiler.reset()
    model = _CompiledAttributeModel()
    x = torch.randn(2, 4)
    with pytest.warns(UserWarning, match="torch.compile"):
        tl.trace(model, x)

    assert model(x).shape == (2, 4)
    plain = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(plain, x)
    assert any("relu" in label for label in trace.layer_labels)


@pytest.mark.heavy
def test_compiled_submodule_is_still_unwrapped_not_merely_skipped() -> None:
    """The pre-existing unwrap path must keep winning over the new bypass.

    A compiled child ``nn.Module`` is swapped for its eager source before capture,
    so its interior IS logged. The in-wrapper bypass must not regress that into a
    silent gap.
    """

    torch.compiler.reset()

    class CompiledChild(nn.Module):
        """Model whose child module is compiled."""

        def __init__(self) -> None:
            """Build an eager layer and a compiled child module."""

            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.act = torch.compile(nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the layer then the compiled child.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Activated output.
            """

            return self.act(self.fc(x))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        trace = tl.trace(CompiledChild(), torch.randn(2, 4))

    labels = trace.layer_labels
    assert any("linear" in label for label in labels)
    assert any("relu" in label for label in labels), (
        "a compiled child nn.Module is unwrapped to eager, so its interior must be logged"
    )
