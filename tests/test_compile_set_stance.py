"""torch.compile rung-2: ``force_eager`` stance capture (torch >= 2.6).

The design-review compile verdict (2026-08-12) verified that the public
``torch.compiler.set_stance("force_eager")`` API makes every compiled callable
run its ORIGINAL Python for the duration of a capture:

* interiors of formerly-opaque compiled regions are fully logged with FULL
  verified semantics (no ``dynamo_region_not_logged`` ceiling);
* outputs are bitwise-identical to eager;
* ZERO new compiles happen during the stance, including on fresh input shapes;
* the warm compiled artifact is bitwise-reproduced after the stance exits (the
  user's compile caches are untouched);
* interior interventions work inside formerly-opaque regions.

One honest qualification (review): the stance invalidates nothing, but TorchLens's
own wrapper install/uninstall can cost ONE bounded recompile on the next
compiled call after capture. The coexistence contract is "one bounded
recompile", not "zero cost".

Each verified scenario is pinned here as a regression contract, plus the
torch < 2.6 tamper leg proving the no-stance fallback path is unchanged. The
broad fallback contract lives in ``test_dynamo_fake_guard.py`` under the
``_no_stance`` fixture.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _capture_state_helpers
from torchlens.utils import _torch_compat

pytestmark = [pytest.mark.heavy]

# MODULE-LEVEL skip, not a skipif mark: this module calls torch.compile at import
# time (the compiled free function below), and a `pytestmark` skipif is evaluated
# only AFTER the module body has executed. On the floor row torch 2.2.2 + Python
# 3.12 that import-time compile raises `RuntimeError: Dynamo is not supported on
# Python 3.12+`, which is a COLLECTION error -- it aborted the entire smoke run
# for that row (2026-08-19), so 9k tests never ran because of one unguarded
# module-level call. Skipping before the body runs is what makes the guard real.
#
# HAS_SET_STANCE (torch >= 2.6) is the right predicate for the Dynamo question
# too: every torch new enough to expose set_stance also supports Dynamo on 3.12.
if not _torch_compat.HAS_SET_STANCE:
    pytest.skip(
        "torch.compiler.set_stance requires torch >= 2.6",
        allow_module_level=True,
    )


class _CompiledAttrModel(nn.Module):
    """Model holding a compiled multi-op callable as a plain attribute.

    Without the stance this is the canonical opaque case: the attribute cannot
    be unwrapped like a compiled child ``nn.Module``, so its interior
    (``sin -> relu -> add``) is honestly not logged and the capture ceilings.
    """

    def __init__(self) -> None:
        """Build an eager linear layer plus a compiled three-op activation."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.compiled_act = torch.compile(lambda t: torch.relu(torch.sin(t)) + 1)

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

        return self.compiled_act(self.fc(x))


def _frames_compiled() -> int:
    """Return Dynamo's cumulative frame-compilation count.

    Returns
    -------
    int
        ``counters["frames"]["total"]``, which increments on every frame
        compilation including recompiles, and stays flat on warm cache hits
        and under an active ``force_eager`` stance.
    """

    counters = _torch_compat.get_dynamo_compile_counters(force_probe=True)
    assert counters is not None, "torch >= 2.6 is expected to expose Dynamo counters"
    return int(counters["frames"]["total"])


def test_stance_capability_flags_are_in_the_snapshot() -> None:
    """Every new degradation point is visible as a named flag."""

    snapshot = _torch_compat.get_torch_capability_snapshot()
    for flag in ("HAS_SET_STANCE", "HAS_DYNAMO_COMPILE_COUNTERS"):
        assert flag in snapshot
        assert isinstance(snapshot[flag], bool)
    assert snapshot["HAS_SET_STANCE"] is True


def test_compiled_attribute_interior_is_logged_with_full_verified_semantics() -> None:
    """The formerly-opaque plain-attribute interior is real eager capture.

    Verdict parity is with a plain never-compiled capture, not a special
    stance vocabulary: no ``_raw_dynamo_region_detected``, no
    ``dynamo_region_not_logged`` reason, and the recorded values are
    bitwise-identical to the original eager Python.
    """

    torch.compiler.reset()
    model = _CompiledAttrModel()
    x = torch.randn(2, 4)
    model(x)  # warm the compile cache: the stance must not depend on cold state

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        trace = tl.trace(model, x)

    labels = trace.layer_labels
    for interior in ("sin", "relu", "add"):
        assert any(interior in label for label in labels), (
            f"compiled-attribute interior op '{interior}' must be logged under the stance"
        )
    assert trace._raw_dynamo_region_detected is False
    assert trace.capture_verification_reason != "dynamo_region_not_logged"

    control = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    assert trace.capture_verified == control.capture_verified

    with torch.no_grad():
        expected = torch.relu(torch.sin(model.fc(x))) + 1
    recorded = next(trace[label].out for label in labels if "add" in label)
    assert torch.equal(recorded, expected), "captured values must be eager-path values"


def test_forward_validation_passes_under_the_stance() -> None:
    """Stance capture satisfies the ordinary forward-replay tripwire."""

    torch.compiler.reset()
    model = _CompiledAttrModel()
    x = torch.randn(2, 4)
    model(x)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert bool(tl.validate(model, x, scope="forward")) is True


def test_forced_eager_disclosure_note_fires_once_per_process() -> None:
    """Inventoried compiled attributes get a one-time forced-eager note."""

    torch.compiler.reset()
    _capture_state_helpers.reset_compiled_forced_eager_warning_state()
    model = _CompiledAttrModel()
    x = torch.randn(2, 4)

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        tl.trace(model, x)
    notes = [w for w in first if "force_eager" in str(w.message)]
    assert len(notes) == 1
    text = str(notes[0].message)
    assert "interiors ARE logged" in text
    assert "one bounded recompile" in text

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        tl.trace(model, x)
    assert not [w for w in second if "force_eager" in str(w.message)]


_compiled_free_fn = torch.compile(lambda t: torch.tanh(t) + torch.sigmoid(t), fullgraph=True)


class _UsesCompiledFreeFunction(nn.Module):
    """Model calling a module-level compiled free function.

    This is the case no inventory can reach (the callable lives in globals,
    not on the module), which without the stance ceilinged the trace at 2
    opaque ops -- and with ``fullgraph=True`` crashed outright when Dynamo
    tried to trace through TorchLens's installed wrappers.
    """

    def __init__(self) -> None:
        """Build the eager entry layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the eager layer then the compiled free function.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Free-function output.
        """

        return _compiled_free_fn(self.fc(x))


def test_compiled_free_function_interior_is_logged_without_any_warning() -> None:
    """The free-function trace matches a never-compiled control: no gap, no noise."""

    torch.compiler.reset()
    model = _UsesCompiledFreeFunction()
    x = torch.randn(2, 4)
    model(x)  # warm compile; also proves fullgraph compilation itself succeeds

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, x)

    labels = trace.layer_labels
    for interior in ("tanh", "sigmoid", "add"):
        assert any(interior in label for label in labels), (
            f"compiled free-function interior op '{interior}' must be logged"
        )
    dynamo_warnings = [w for w in caught if "Dynamo" in str(w.message)]
    assert not dynamo_warnings, f"expected no Dynamo warning, saw: {dynamo_warnings}"
    assert trace._raw_dynamo_region_detected is False
    assert trace.capture_verification_reason != "dynamo_region_not_logged"


def test_nested_compiled_modules_run_original_python() -> None:
    """Compiled-inside-compiled modules are fully logged and restored."""

    torch.compiler.reset()

    class Inner(nn.Module):
        """Child module with a distinctive interior op."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the distinctive op.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Transformed batch.
            """

            return torch.sin(x) * 2

    class Outer(nn.Module):
        """Parent module holding a compiled child."""

        def __init__(self) -> None:
            """Build the eager layer and the compiled child."""

            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.inner = torch.compile(Inner())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the layer then the compiled child.

            Parameters
            ----------
            x:
                Input batch.

            Returns
            -------
            torch.Tensor
                Child output.
            """

            return self.inner(self.fc(x))

    model = torch.compile(Outer())
    x = torch.randn(2, 4)
    model(x)  # warm both compiled levels
    compiled_child = model._orig_mod.inner

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        trace = tl.trace(model, x)

    labels = trace.layer_labels
    assert any("sin" in label for label in labels)
    assert any("mul" in label for label in labels)
    assert trace._raw_dynamo_region_detected is False
    assert trace.capture_verification_reason != "dynamo_region_not_logged"
    assert model._orig_mod.inner is compiled_child, (
        "the compiled child slot must be restored after capture"
    )


def test_interior_intervention_in_formerly_opaque_region() -> None:
    """An ablation inside a compiled attribute fires and changes the output."""

    torch.compiler.reset()
    # Seeded: with unseeded weights/input, sin(fc(x)) is all-non-positive for
    # ~0.7% of draws, making the CLEAN relu output exactly zero and the final
    # assert vacuously equal (observed once on the cluster's full-suite RNG
    # stream). Seed 0 gives a non-degenerate relu input.
    torch.manual_seed(0)
    model = _CompiledAttrModel()
    x = torch.randn(2, 4)
    model(x)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        clean = tl.trace(model, x)
        ablated = tl.trace(
            model,
            x,
            intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        )

    ablated_out = next(ablated[label].out for label in ablated.layer_labels if "add" in label)
    clean_out = next(clean[label].out for label in clean.layer_labels if "add" in label)
    assert torch.equal(ablated_out, torch.ones_like(ablated_out)), (
        "zero-ablating the interior relu must zero its output (leaving the +1)"
    )
    assert not torch.equal(ablated_out, clean_out)


def test_fresh_input_shape_triggers_zero_compiles_during_capture() -> None:
    """A never-seen shape runs eagerly under the stance: no new compile."""

    torch.compiler.reset()
    model = _CompiledAttrModel()
    model(torch.randn(2, 4))  # warm the (2, 4) variant

    before = _frames_compiled()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.randn(7, 4))
    assert _frames_compiled() - before == 0
    assert any("relu" in label for label in trace.layer_labels)


def test_failed_capture_restores_force_eager_stance() -> None:
    """An escaped forward error cannot leave future compiled calls forced eager."""

    torch.compiler.reset()
    compile_events: list[object] = []

    def counting_backend(graph_module: object, _example_inputs: list[torch.Tensor]) -> object:
        """Record a real backend compile and return the eager graph callable.

        Parameters
        ----------
        graph_module:
            Dynamo-produced graph module.
        _example_inputs:
            Example tensor inputs supplied to the backend.

        Returns
        -------
        object
            Graph forward callable used as the compiled artifact.
        """

        compile_events.append(graph_module)
        return graph_module.forward  # type: ignore[attr-defined, no-any-return]

    compiled_free = torch.compile(
        lambda tensor: torch.sin(tensor), backend=counting_backend, fullgraph=True
    )

    class _CompiledThenRaise(nn.Module):
        """Call an initially-cold compiled function and then fail."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Execute under the capture stance before raising.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                This path never returns.

            Raises
            ------
            RuntimeError
                Always raised after the compiled call returns eagerly.
            """

            _ = compiled_free(x)
            raise RuntimeError("injected stance-scope failure")

    x = torch.randn(2, 4)
    with pytest.raises(RuntimeError, match="injected stance-scope failure"):
        tl.trace(_CompiledThenRaise(), x)
    assert compile_events == [], "the stance must compile zero graphs"

    compiled_free(x)
    assert len(compile_events) == 1, (
        "the first post-failure compiled call must compile; force_eager leaked"
    )


def test_warm_artifact_is_reproduced_and_the_recompile_is_bounded() -> None:
    """The coexistence contract: caches intact, at most ONE recompile after.

    ``force_eager`` itself invalidates nothing; TorchLens's wrapper
    install/uninstall may cost one bounded recompile on the next compiled call
    after capture. The reproduced output must be bitwise-identical to the
    original warm artifact's.
    """

    torch.compiler.reset()
    model = _CompiledAttrModel()
    x = torch.randn(2, 4)
    warm = model(x)
    assert torch.equal(warm, model(x)), "the warm artifact must be deterministic"

    during_start = _frames_compiled()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tl.trace(model, x)
    assert _frames_compiled() - during_start == 0, "zero compiles during the stance"

    after_start = _frames_compiled()
    post = model(x)
    recompiles = _frames_compiled() - after_start
    assert recompiles <= 1, f"the post-capture recompile must be bounded at one, got {recompiles}"
    assert torch.equal(post, warm), "the warm compiled artifact must be bitwise-reproduced"


def test_count_compiles_verifies_the_coexistence_contract() -> None:
    """``tl.debug.count_compiles`` measures both contract promises directly."""

    torch.compiler.reset()
    model = _CompiledAttrModel()
    x = torch.randn(2, 4)
    model(x)  # warm

    with tl.debug.count_compiles() as during, warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tl.trace(model, x)
    assert during.frames_compiled == 0

    with tl.debug.count_compiles() as after:
        model(x)
    assert after.frames_compiled <= 1
    # The count freezes at block exit: later compile events (a fresh shape
    # compiling normally) must not leak into an already-measured block.
    frozen = during.frames_compiled
    model(torch.randn(9, 4))
    assert during.frames_compiled == frozen


def test_count_compiles_missing_counters_refuses_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torch runtime without Dynamo counters refuses with the stable code."""

    from torchlens.debug import CompileCountsUnavailableError

    monkeypatch.setattr(
        _torch_compat, "get_dynamo_compile_counters", lambda force_probe=False: None
    )

    with pytest.raises(CompileCountsUnavailableError, match="unavailable") as excinfo:
        with tl.debug.count_compiles():
            pass
    assert excinfo.value.fields["code"] == "compile_counts_unavailable"


def test_compat_row_states_the_coexistence_contract_under_stance() -> None:
    """The torch.compile row documents the contract instead of a scope refusal."""

    torch.compiler.reset()
    row = tl.compat.report(_CompiledAttrModel(), torch.randn(2, 4)).row("torch_compile")
    assert row.detected is True
    assert row.status == "pass"
    assert "set_stance" in row.details
    assert "one bounded recompile" in row.details
    assert "unwrap_torch()" in row.details
    assert "tl.debug.graph_breaks" in row.suggestion
    assert "tl.debug.count_compiles" in row.suggestion

    clear_row = tl.compat.report(nn.Identity(), torch.randn(2, 4)).row("torch_compile")
    assert clear_row.detected is False
    assert "original eager Python" in clear_row.details


def test_no_stance_tamper_fallback_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the capability flag off, the exact pre-stance path re-engages.

    This is the torch < 2.6 tamper leg: bypass wrapper installed and restored,
    interior not logged, one Dynamo warning, and the honest
    ``dynamo_region_not_logged`` ceiling. The broad fallback contract lives in
    ``test_dynamo_fake_guard.py``.
    """

    torch.compiler.reset()
    monkeypatch.setattr(_torch_compat, "HAS_SET_STANCE", False)
    model = _CompiledAttrModel()
    compiled_callable = model.compiled_act
    x = torch.randn(2, 4)

    with pytest.warns(UserWarning, match="torch.compile"):
        trace = tl.trace(model, x)

    assert not any("relu" in label for label in trace.layer_labels)
    assert trace._raw_dynamo_region_detected is True
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "dynamo_region_not_logged"
    assert model.compiled_act is compiled_callable
