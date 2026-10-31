"""Phase 4c live hook execution tests."""

from __future__ import annotations

from collections import namedtuple

import pytest
import torch

import torchlens as tl
from torchlens._trace_state import TraceState
from torchlens.intervention.errors import LiveModeLabelError, SiteResolutionError
from torchlens.options import CaptureOptions


class _ReluReturnModel(torch.nn.Module):
    """Model that exposes the returned ReLU out for comparison."""

    def __init__(self) -> None:
        """Initialize the model with no captured output."""

        super().__init__()
        self.latest: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single ReLU out.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU out after live hooks.
        """

        self.latest = torch.relu(x)
        return self.latest


class _LinearModel(torch.nn.Module):
    """Single-module model for capture-time module selectors."""

    def __init__(self) -> None:
        """Initialize a linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.linear(x)


class _TwoModuleModel(torch.nn.Module):
    """Model with two module boundaries for selector specificity tests."""

    def __init__(self) -> None:
        """Initialize submodules."""

        super().__init__()
        self.a = torch.nn.ReLU()
        self.b = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two modules in sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sigmoid of the relu output.
        """

        return self.b(self.a(x))


class _ChunkModel(torch.nn.Module):
    """Model with a tuple-output operation."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return two chunks.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Chunk outputs.
        """

        return torch.chunk(torch.relu(x), 2, dim=1)


class _NamedTupleModule(torch.nn.Module):
    """Module returning a namedtuple of tensor leaves."""

    def __init__(self) -> None:
        """Initialize the stable namedtuple type."""

        super().__init__()
        self._pair_type = namedtuple("Pair", ["left", "right"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a namedtuple carrying two tensor leaves.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Namedtuple pair of transformed tensors.
        """

        return self._pair_type(x + 1, x + 2)


class _NamedTupleBoundaryModel(torch.nn.Module):
    """Model whose submodule boundary returns a namedtuple."""

    def __init__(self) -> None:
        """Initialize the namedtuple-returning submodule."""

        super().__init__()
        self.sub = _NamedTupleModule()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the submodule's namedtuple payload.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Namedtuple payload.
        """

        return self.sub(x)


class _MultiOpBlock(torch.nn.Module):
    """Block whose containment and output boundary are observably different."""

    def __init__(self) -> None:
        """Initialize the block's learned operation."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three captured operations before returning.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled rectified projection.
        """

        return torch.relu(self.linear(x)) * 2.0


class _MultiOpModuleModel(torch.nn.Module):
    """Model exposing a multi-operation block followed by another module."""

    def __init__(self) -> None:
        """Initialize the selected block and unselected tail."""

        super().__init__()
        self.block = _MultiOpBlock()
        self.tail = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selected block and unselected tail.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tail projection of the block output.
        """

        return self.tail(self.block(x))


def _zero_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return a zeroed out.

    Parameters
    ----------
    out:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Zeroed out with matching metadata.
    """

    return out * 0


def _identity_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
    """Return an out unchanged.

    Parameters
    ----------
    out:
        Activation at the hook site.
    hook:
        Hook context.

    Returns
    -------
    torch.Tensor
        Original out.
    """

    return out


def test_live_label_error_for_finalized_style_label() -> None:
    """Finalized postprocess labels fail loudly in live capture."""

    with pytest.raises(LiveModeLabelError, match="tl.where"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            capture=CaptureOptions(
                intervention_ready=True,
                hooks={tl.label("relu_4_27:2"): _identity_hook},
            ),
        )


def test_module_selector_matches_capture_time_module_context() -> None:
    """Module selectors can match live capture-time module context."""

    log = tl.trace(
        _LinearModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.module("linear"): _zero_hook},
        ),
    )

    hooked_layers = [layer for layer in log.layer_list if layer.interventions]

    assert hooked_layers
    assert hooked_layers[0].out is not None
    assert torch.count_nonzero(hooked_layers[0].out) == 0


def test_module_selector_does_not_overmatch_other_live_modules() -> None:
    """Live module selectors only match the requested module boundary."""

    log = tl.trace(
        _TwoModuleModel(),
        torch.tensor([-1.0, 2.0]),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.module("b"): _zero_hook},
        ),
    )

    hooked_labels = {layer.layer_label for layer in log.layer_list if layer.interventions}

    assert hooked_labels == {"interventionreplacement_1_3"}


def test_module_selector_matches_exact_multi_op_boundary() -> None:
    """A module selector intervenes only on a multi-op module's output boundary."""

    log = tl.trace(
        _MultiOpModuleModel(),
        torch.randn(2, 3),
        intervene=tl.when(tl.module("block"), tl.zero_ablate()),
    )

    intervened = {op.layer_label for op in log.ops if op.interventions}
    resolved_boundary = {
        op.layer_label for op in log.resolve_sites(tl.module("block"), max_fanout=len(log.ops))
    }

    assert len(intervened) == 1
    assert intervened == resolved_boundary


def test_module_selector_namedtuple_output_rebuilds_without_crashing() -> None:
    """Module-boundary hooks rebuild namedtuple outputs after tensor replacement."""

    x = torch.randn(2, 3)
    trace = tl.trace(
        _NamedTupleBoundaryModel(),
        x,
        intervene=tl.when(tl.module("sub"), tl.zero_ablate()),
    )
    outputs = [trace[label].out for label in trace.output_layers]

    assert len(outputs) == 2
    assert all(torch.equal(out, torch.zeros_like(x)) for out in outputs)


def test_module_selector_save_is_exact_for_trace_and_record() -> None:
    """Trace and sparse save retain only the selected module output op."""

    model = _MultiOpModuleModel()
    inputs = torch.randn(2, 3)

    log = tl.trace(model, inputs, save=tl.module("block"))
    recording = tl.record(model, inputs, save=tl.module("block"))
    recording_trace = recording.to_trace()

    saved = {op.layer_label for op in log.ops if op.has_saved_activation}
    resolved_boundary = {
        op.layer_label for op in log.resolve_sites(tl.module("block"), max_fanout=len(log.ops))
    }
    assert saved == resolved_boundary
    assert len(recording.records) == 1
    assert recording.records[0].ctx.output_of_module_calls == ("block:1",)
    recording_saved = {op.layer_label for op in recording_trace.ops if op.has_saved_activation}
    recording_boundary = {
        op.layer_label
        for op in recording_trace.resolve_sites(
            tl.module("block"), max_fanout=len(recording_trace.ops)
        )
    }
    assert recording_saved == recording_boundary


@pytest.mark.parametrize(
    "selector",
    [
        tl.label("relu_1_2"),
        tl.contains("relu_1_2"),
        tl.regex("relu_1_2"),
    ],
)
def test_predicate_intervention_rejects_finalized_label_styles(selector: object) -> None:
    """All label-oriented predicate selectors share the live-label guard."""

    with pytest.raises(LiveModeLabelError, match="tl.where"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            intervene=tl.when(selector, tl.zero_ablate()),  # type: ignore[arg-type]
        )


def test_zero_match_capture_selectors_warn() -> None:
    """Successful capture warns when save or intervention selectors match nothing."""

    with pytest.warns(UserWarning, match="save selector .* matched zero sites"):
        tl.trace(_ReluReturnModel(), torch.randn(2, 3), save=tl.func("missing"))
    with pytest.warns(UserWarning, match="intervention selector .* matched zero sites"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            intervene=tl.when(tl.func("missing"), tl.zero_ablate()),
        )
    with pytest.warns(UserWarning, match="save selector .* matched zero sites"):
        tl.record(_ReluReturnModel(), torch.randn(2, 3), save=tl.func("missing"))


def test_unsupported_capture_selector_kind_fails_loudly() -> None:
    """Unsupported selector kinds raise instead of silently matching nothing."""

    with pytest.raises(SiteResolutionError, match="resolve through intervention mutators"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            intervene=tl.when(tl.facet("resid"), tl.zero_ablate()),
        )


def test_hook_body_type_error_is_not_reclassified() -> None:
    """A TypeError raised by user hook code propagates unchanged."""

    def broken_hook(out: torch.Tensor, *, hook: tl.HookContext) -> torch.Tensor:
        """Raise a user-authored TypeError after successful argument binding."""

        del out, hook
        raise TypeError("body bug")

    with pytest.raises(TypeError, match="body bug"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            intervene=tl.when(tl.func("relu"), broken_hook),
        )


def test_live_regex_and_output_at_selectors_execute() -> None:
    """Live regex and output-path selectors are executable hook selectors."""

    regex_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.regex("relu"): _zero_hook},
        ),
    )
    chunk_log = tl.trace(
        _ChunkModel(),
        torch.randn(2, 4),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.output_at(1): _zero_hook},
        ),
    )

    assert any(layer.interventions for layer in regex_log.layer_list if layer.func_name == "relu")
    hooked_chunk_paths = {
        layer.container_path for layer in chunk_log.layer_list if layer.interventions
    }
    assert len(hooked_chunk_paths) == 1


def test_input_at_live_hook_target_rejects_with_honest_error() -> None:
    """Input-path selectors resolve placeholders but are not live hook sites."""

    with pytest.raises(SiteResolutionError, match="not live hook application sites"):
        tl.trace(
            _ReluReturnModel(),
            torch.randn(2, 3),
            capture=CaptureOptions(
                intervention_ready=True,
                hooks={tl.input_at(0): _zero_hook},
            ),
        )


def test_raw_label_where_and_in_module_selectors_work_at_capture_time() -> None:
    """Raw labels, predicates, and module containment selectors resolve live."""

    raw_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(intervention_ready=True),
    )
    raw_label = next(
        layer._layer_label_raw for layer in raw_log.layer_list if layer.func_name == "relu"
    )

    label_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.label(raw_label): _zero_hook},
        ),
    )
    where_log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.where(lambda p: p.func_name == "relu"): _zero_hook},
        ),
    )
    in_module_log = tl.trace(
        _LinearModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.in_module("linear"): _zero_hook},
        ),
    )

    assert any(layer.interventions for layer in label_log.layer_list)
    assert any(layer.interventions for layer in where_log.layer_list)
    assert any(layer.interventions for layer in in_module_log.layer_list)


def test_no_hooks_preserves_pristine_run_state() -> None:
    """Intervention-ready capture without hooks stays pristine."""

    log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(intervention_ready=True),
    )

    assert log.state is TraceState.PRISTINE


def test_live_replacement_metadata_matches_saved_out() -> None:
    """Hook replacement refreshes tensor metadata and saved-out flags."""

    log = tl.trace(
        _ReluReturnModel(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.func("relu"): _zero_hook},
        ),
    )
    relu_layer = next(layer for layer in log.layer_list if layer.func_name == "relu")

    assert relu_layer.out is not None
    assert relu_layer.has_saved_activation is True
    assert relu_layer.shape == tuple(relu_layer.out.shape)
    assert relu_layer.dtype == relu_layer.out.dtype
    assert relu_layer.activation_memory == relu_layer.out.nelement() * relu_layer.out.element_size()
