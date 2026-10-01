"""The postprocess recording/enforcement axes matrix (design-ppdag-v3 §8.1).

Every axis is one FRESH capture reaching ``postprocess()`` (refresh included
— it is a fresh capture projected onto the source trace afterwards). The
matrix is the shared input of the one-time declaration seeding sweep
(``tools/record_postprocess_matrix.py``), the read-before-write findings
classification, and the env-gated read-enforcement CI leg. A configuration
that gates postprocess code paths belongs here: the honest residual of read
enforcement is exactly the axes NOT in this matrix.
"""

from __future__ import annotations

import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import torchlens as tl

_SEED = 20260812

# Fallback tmp dirs for callers that omit ``tmp_dir``. Each ``iter_axes()`` call
# gets a FRESH directory (axis callables embed bundle paths, so sharing one dir
# across calls collides), and the handles are held here so their finalizers
# remove the directories at interpreter exit -- a bare ``mkdtemp`` leaked one
# directory per call, created as early as collection time.
_FALLBACK_TMP_DIRS: list[tempfile.TemporaryDirectory] = []


def _seed_everything() -> None:
    """Seed every RNG the fixtures consume."""

    torch.manual_seed(_SEED)
    random.seed(_SEED)
    np.random.seed(_SEED % (2**32 - 1))


def _oracle_case(model_axis: str) -> tuple[nn.Module, torch.Tensor]:
    """Build one deterministic capture-oracle model case."""

    from capture_oracle._models import build_model_case

    return build_model_case(model_axis)


class OrphanEquivalenceModel(nn.Module):
    """Disconnected island sharing an equivalence class with survivors."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        z = torch.ones(5, 5)
        z = z + 1
        _dead = z**2
        return x**2


class InternalSourceModel(nn.Module):
    """Internally-initialized tensor feeding the output path."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = torch.ones(4) * 2.0
        return self.lin(x) * gain


class FactorySourceOutputModel(nn.Module):
    """Parentless factory tensor returned directly as the final output.

    The b9 R71-1 shape: the synthetic output node's DIRECT parent is a
    parentless internal source, so step 1's ``root_ancestors`` re-derivation
    is content-effective here (the wholesale clone would otherwise inherit
    the factory row's empty set) and the axis keeps that write out of the
    permanent no-op ledger.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.arange(x.shape[-1], device=x.device)


class DoubleBufferModel(nn.Module):
    """Registered buffer read twice (buffer dedup / step-6 pressure)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(4))
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x * self.scale
        b = x + self.scale
        return self.bn(a + b)


class DuplicateBufferModel(nn.Module):
    """Genuinely triggers step 6's ``_merge_buffer_entries`` (B2 axis).

    The merge needs two buffer nodes with the same module stack, the same
    ``buffer_source``, the same address, and ``torch.equal`` values. A
    module that reassigns its buffer to an equal-valued plain-attribute
    tensor mid-forward (a cached mask, the speechbrain CRDNN pattern)
    produces a second source-``None`` node that collides with the initial
    read. The extra reassign-of-the-stash after an intervening version adds
    a surviving node whose ``buffer_source`` names the REMOVED node, so the
    merge's scalar ``buffer_source`` repoint (and its ``parent_arg_positions``
    arg-0 mirror) fires too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", torch.ones(4))
        self._stash = torch.ones(4)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x * self.mask
        self.mask = self._stash
        b = a + self.mask  # source-None duplicate: merges into the initial node
        self.mask = self.mask + 0.0
        c = b * self.mask  # distinct source: survives as its own version
        self.mask = self._stash
        d = c + self.mask  # buffer_source names the merged-away node: repoint fires
        return self.lin(d)


class DivergentReachBufferModel(nn.Module):
    """Step-6 merge where the SURVIVOR dead-ends and the duplicate reaches output.

    r3 b1-opus R04-F1: e12aa996's child-direction reach repair (merge-time
    ``output_descendants`` union + ancestor-cone re-derivation) shipped with
    no axis able to observe it — every recorded merge already shared the
    survivor's reach, so its writes were pinned content no-ops and an
    effect-neutralizing revert left the whole gate green. Here the initial
    read (the merge survivor) feeds only an input-connected DEAD END while
    the equal-valued duplicate read feeds the output, so the merge must flip
    the survivor's ``has_output_descendant`` False→True and grow
    ``output_descendants`` — the axis that retired both step-6 pinned rows.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", torch.ones(4))
        self._stash = torch.ones(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _dead = x * self.mask  # survivor's only pre-merge child: dead end
        self.mask = self._stash  # equal-valued reassign -> duplicate node
        return x + self.mask  # duplicate's child reaches the output


class BufferFromInputModel(nn.Module):
    """Buffer reassigned from an input-derived tensor (B2 residual closure).

    The journaled write's version node carries ``buffer_source`` naming the
    input-derived producer op, so step 6's buffer-source ancestry fallback
    (``control_flow.py``) has real content to copy. Step 4 pre-propagates
    transitive input ancestry whenever it runs — and it runs on default
    captures — so the fallback is only content-effective with layer depths
    OFF. The axis therefore captures with ``mark_layer_depths=False``: it
    retired the former ``("6", "has_input_ancestor")`` permanent no-op row
    and is the axis that observes step 6's ``input_ancestors`` write family.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("acc", torch.zeros(4))
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.acc = self.acc + x.mean(dim=0)
        return self.lin(x) * self.acc


class ElifElseBranchModel(nn.Module):
    """Two if/elif/else chains, one taking the elif arm and one the else arm.

    Closes the matrix gap behind the ``conditional_elif_children`` /
    ``conditional_else_children`` permanent no-op rows: without an
    elif/else-bearing axis, steps 5 and 9 never write those views
    effectively, and guard 2 would flag every later read of them as a
    finding for the wrong reason (matrix gap, not laundering).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        if total > 1e6:
            y = x * 2.0
        elif total > -1e6:  # taken
            y = x - 1.0
        else:
            y = x + 3.0
        pivot = y.mean()
        if pivot > 1e6:
            z = y * 2.0
        elif pivot > 1e5:
            z = y - 1.0
        else:  # taken
            z = y + 3.0
        return z * 1.5


class AssigningVarNamesModel(nn.Module):
    """Direct torch-call assignments so step 11.5 resolves real var_names.

    Closes the matrix gap behind the ``("11.5", "var_names")`` permanent
    no-op row (module-internal calls never name the wrapped function in
    their assignment line, so the oracle axes resolve to the empty
    default). Runs under ``save_code_context=True``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(x)
        activated = torch.relu(hidden)
        return activated


def _axis_oracle(model_axis: str) -> Callable[[], Any]:
    def run() -> Any:
        _seed_everything()
        model, model_input = _oracle_case(model_axis)
        return tl.trace(model, model_input)

    return run


def _axis_backward_armed() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(
        model,
        model_input.requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def _axis_save_code_context() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(model, model_input, capture=tl.options.CaptureOptions(save_code_context=True))


def _axis_streaming(tmp_dir: str) -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(model, model_input, storage=tl.to_disk(Path(tmp_dir) / "axis.tlspec"))


def _axis_layer_depths() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(
        model,
        model_input,
        capture=tl.options.CaptureOptions(compute_input_output_distances=True),
    )


def _axis_recurrence_off() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("recurrent")
    return tl.trace(
        model, model_input, capture=tl.options.CaptureOptions(recurrence_detection=False)
    )


def _axis_intervention() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(
        model,
        model_input,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )


def _axis_orphan_keep() -> Any:
    _seed_everything()
    return tl.trace(
        OrphanEquivalenceModel(),
        torch.ones(5, 5),
        capture=tl.options.CaptureOptions(keep_orphans=True),
    )


def _axis_orphan_remove() -> Any:
    _seed_everything()
    return tl.trace(OrphanEquivalenceModel(), torch.ones(5, 5))


def _axis_transform() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(
        model,
        model_input,
        save=tl.options.SaveOptions(activation_transform=lambda t: t.detach().float() * 1.0),
    )


def _axis_container_structure() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(
        model,
        model_input,
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )


def _axis_internal_source() -> Any:
    _seed_everything()
    return tl.trace(InternalSourceModel(), torch.randn(2, 4))


def _axis_factory_source_output() -> Any:
    _seed_everything()
    return tl.trace(FactorySourceOutputModel(), torch.randn(4, 4))


class MultiOutputModuleModel(nn.Module):
    """A module CALL with more than one output entry (r7 R04-2 axis).

    ``_assign_output_roles`` early-returns at one output entry, so the whole
    enforcement matrix ran green while step 16's undeclared
    ``multi_output_name``/``_source_trace_ref`` reads sat on the
    LSTM/GRU-cell and tuple-returning-submodule family. Both sub-shapes ride
    this one axis: the LSTMCell (real multi-output cell) feeds a submodule
    that returns a plain 2-tuple.
    """

    class _TupleHead(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x + 1

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(4, 8)
        self.head = self._TupleHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], 8)
        context = torch.zeros(x.shape[0], 8)
        hidden, context = self.cell(x, (hidden, context))
        doubled, shifted = self.head(hidden)
        return doubled + shifted + context


def _axis_multi_output_module() -> Any:
    _seed_everything()
    return tl.trace(MultiOutputModuleModel(), torch.randn(2, 4))


def _axis_buffer_pressure() -> Any:
    _seed_everything()
    return tl.trace(DoubleBufferModel().train(), torch.randn(3, 4))


def _axis_buffer_duplicate() -> Any:
    _seed_everything()
    return tl.trace(DuplicateBufferModel(), torch.randn(2, 4))


def _axis_buffer_divergent_reach() -> Any:
    _seed_everything()
    return tl.trace(DivergentReachBufferModel(), torch.randn(2, 4))


def _axis_buffer_from_input() -> Any:
    _seed_everything()
    return tl.trace(
        BufferFromInputModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(compute_input_output_distances=False),
    )


def _axis_lookback(tmp_dir: str, *, streaming: bool, transform: bool) -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    kwargs: dict[str, Any] = {}
    if streaming:
        kwargs["storage"] = tl.to_disk(Path(tmp_dir) / "lookback.tlspec")
    if transform:
        kwargs["activation_transform"] = lambda t: t.detach() * 1.0
    return tl.trace(
        model,
        model_input,
        save=tl.func("relu"),
        lookback=2,
        lookback_payload_policy="detached_raw",
        **kwargs,
    )


def _axis_deferred_retention() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    # A final-label-only selector (negative index) defers the retention
    # decision to step 11.75 (resolve_deferred_retention), the axis that
    # populates its save_activation write family.
    return tl.trace(model, model_input, capture=tl.options.CaptureOptions(layers_to_save=[-2, -1]))


def _axis_transform_streaming(tmp_dir: str) -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    # Transform + disk streaming: step 19's transformed_out eviction path.
    return tl.trace(
        model,
        model_input,
        storage=tl.to_disk(Path(tmp_dir) / "transform.tlspec"),
        save=tl.options.SaveOptions(activation_transform=lambda t: t.detach() * 1.0),
    )


def _axis_halted() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    return tl.trace(model, model_input, halt=tl.func("relu"))


def _axis_cooked_recording() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    recording = tl.record(model, model_input, save=tl.func("relu"))
    return recording.to_trace()


def _axis_cooked_recording_halted() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    recording = tl.record(model, model_input, save=tl.func("relu"), halt=tl.func("relu"))
    return recording.to_trace()


def _axis_conditional_alternate() -> Any:
    _seed_everything()
    model, _ = _oracle_case("conditional")
    # Drive the branch the default oracle input does not take.
    return tl.trace(model, torch.tensor([[-1.0, 0.25], [-0.5, -0.25]]))


def _axis_conditional_elif_else() -> Any:
    _seed_everything()
    return tl.trace(ElifElseBranchModel(), torch.randn(2, 4))


def _axis_var_names() -> Any:
    _seed_everything()
    return tl.trace(
        AssigningVarNamesModel().eval(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_code_context=True),
    )


def _axis_refresh() -> Any:
    _seed_everything()
    model, model_input = _oracle_case("plain_cnn")
    trace = tl.trace(model, model_input)
    try:
        result = trace.run(inputs=model_input)
        result.trace.cleanup()
    finally:
        trace.cleanup()
    return None


def iter_axes(tmp_dir: str | None = None) -> list[tuple[str, Callable[[], Any]]]:
    """Return the named axes matrix.

    Each callable runs ONE fresh capture and returns the resulting trace
    (or ``None`` when the axis cleans up internally, e.g. refresh). The
    caller owns cleanup of returned traces. ``tmp_dir`` hosts streaming
    bundles; a temporary directory is created when omitted.
    """

    if tmp_dir is None:
        handle = tempfile.TemporaryDirectory(prefix="tl-ppdag-axes-")
        _FALLBACK_TMP_DIRS.append(handle)
        tmp_dir = handle.name
    axes: list[tuple[str, Callable[[], Any]]] = [
        (f"oracle:{model_axis}", _axis_oracle(model_axis))
        for model_axis in (
            "plain_cnn",
            "train_batchnorm",
            "recurrent",
            "conditional",
            "in_place",
            "tiny_transformer",
        )
    ]
    axes += [
        ("backward_armed", _axis_backward_armed),
        ("save_code_context", _axis_save_code_context),
        ("streaming", lambda: _axis_streaming(tmp_dir)),
        ("layer_depths", _axis_layer_depths),
        ("recurrence_off", _axis_recurrence_off),
        ("intervention", _axis_intervention),
        ("orphan_keep", _axis_orphan_keep),
        ("orphan_remove", _axis_orphan_remove),
        ("transform", _axis_transform),
        ("container_structure", _axis_container_structure),
        ("multi_output_module", _axis_multi_output_module),
        ("internal_source", _axis_internal_source),
        ("factory_source_output", _axis_factory_source_output),
        ("buffer_pressure", _axis_buffer_pressure),
        ("buffer_duplicate", _axis_buffer_duplicate),
        ("buffer_divergent_reach", _axis_buffer_divergent_reach),
        ("buffer_from_input", _axis_buffer_from_input),
        ("lookback", lambda: _axis_lookback(tmp_dir, streaming=False, transform=False)),
        ("lookback_transform", lambda: _axis_lookback(tmp_dir, streaming=False, transform=True)),
        ("lookback_streaming", lambda: _axis_lookback(tmp_dir, streaming=True, transform=False)),
        ("deferred_retention", _axis_deferred_retention),
        ("transform_streaming", lambda: _axis_transform_streaming(tmp_dir)),
        ("halted", _axis_halted),
        ("cooked_recording", _axis_cooked_recording),
        ("cooked_recording_halted", _axis_cooked_recording_halted),
        ("conditional_alternate", _axis_conditional_alternate),
        ("conditional_elif_else", _axis_conditional_elif_else),
        ("var_names", _axis_var_names),
        ("refresh", _axis_refresh),
    ]
    return axes
