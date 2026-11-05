"""Validation's reproducibility re-trace saves only what the structure hash reads.

The re-trace compares graph-shape hashes, so it saves the buffer nodes and buffer
sources and drops every other output (``torchlens/_validation_retrace_saves.py``).
These tests prove the hash depends on nothing the slim re-trace drops: every value
the hash (and the divergence hint) reads is identical between a slim and a
full-save capture of the same forward. A save set that misses a buffer the hash
needs fails closed as a re-trace mismatch.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens._user_public_impls as user_public_impls
from torchlens import _user_funcs
from torchlens._record_type import record_type
from torchlens._utils._record_group_table import record_member
from torchlens._utils.hashing import compute_graph_shape_hash
from torchlens._validation_divergence import _first_reproducibility_divergence
from torchlens._validation_retrace_saves import (
    compare_retrace_graphs,
    retrace_save_coverage_gap,
    retrace_save_indices,
)
from torchlens.errors import TraceNotReproducibleWarning
from torchlens.validation import validate_forward_pass

_MISSING = object()


class _Mlp(nn.Module):
    """Two linear layers and no buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _TrainBatchNorm(nn.Module):
    """Train-mode BatchNorm: running statistics rewritten by the kernel every forward."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x)) * 2


class _Reassign(nn.Module):
    """Reassigns a buffer to an input-dependent tensor and reads it back."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.b
        self.b = self.b * 1.5 + x.mean()
        return x + self.b


class _RewriteInPlace(nn.Module):
    """Reads a buffer, rewrites it in place, reads it again."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.b
        self.b.mul_(1.5)
        return x + self.b


class _SometimesRewrites(nn.Module):
    """Rewrites its buffer only when asked."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor, rewrite: bool) -> torch.Tensor:
        x = x + self.b
        if rewrite:
            self.b = x.detach() * 0.5
        return x


class _RewriteOnSomePasses(nn.Module):
    """One buffer-holding module called four times (re-logged reads merge); two rewrite."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _SometimesRewrites()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inner(x, True)
        x = self.inner(x, False)
        x = self.inner(x, True)
        x = self.inner(x, False)
        return x * 2


class _DataSwap(nn.Module):
    """Swaps a buffer's storage: a recorded write that keeps its own version node."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + self.b
        self.b.data = torch.zeros(4)
        return a + self.b


_MODELS: dict[str, tuple[Callable[[], nn.Module], Callable[[], torch.Tensor]]] = {
    "mlp": (_Mlp, lambda: torch.rand(2, 4)),
    "train_batch_norm": (_TrainBatchNorm, lambda: torch.rand(5, 4)),
    "reassign": (_Reassign, lambda: torch.rand(4)),
    "rewrite_in_place": (_RewriteInPlace, lambda: torch.rand(4)),
    "rewrite_on_some_passes": (_RewriteOnSomePasses, lambda: torch.rand(4)),
    "data_swap": (_DataSwap, lambda: torch.rand(4)),
}


def _capture(model: nn.Module, x: torch.Tensor, saved: tuple[int, ...] | None) -> Any:
    """Capture like the re-trace does: everything saved when ``saved`` is None, else slim."""

    return _user_funcs._run_model_and_save_specified_outs(
        model=model,
        input_args=[x],
        input_kwargs={},
        layers_to_save="all" if saved is None else "none",
        save_args=False,
        seed=0,
        save_rng=False,
        _resolved_layer_nums_to_save=saved,
    )


class _ReadRecorder:
    """Forwards attribute reads to a record and remembers each name read."""

    def __init__(self, record: Any, reads: set[str]) -> None:
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_reads", reads)

    def __getattr__(self, name: str) -> Any:
        self._reads.add(name)
        return getattr(self._record, name)


def _names_the_hash_reads(trace: Any) -> set[str]:
    """Return every layer attribute the re-trace hash and divergence hint read."""

    reads: set[str] = set()
    proxy = SimpleNamespace(_layer_list=[_ReadRecorder(r, reads) for r in trace._layer_list])
    compute_graph_shape_hash(proxy, include_module_address=False)
    _first_reproducibility_divergence(proxy, proxy)  # type: ignore[arg-type]
    return reads


def _comparable(value: Any) -> Any:
    """Return a form of ``value`` that compares by content (tensors by bytes)."""

    if isinstance(value, torch.Tensor):
        return ("tensor", value.dtype, tuple(value.shape), value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(item) for item in value)
    return value


def _read(record: Any, name: str) -> tuple[Any, Any]:
    """Read ``name`` both as plain ``getattr`` and through its record group."""

    try:
        plain = getattr(record, name, _MISSING)
    except Exception as exc:  # a property that refuses is itself a comparable outcome
        plain = ("raised", type(exc).__name__)
    return _comparable(plain), _comparable(record_member(record, name, _MISSING))


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_retrace_hash_reads_only_values_the_slim_retrace_keeps(name: str) -> None:
    """Every value the hash reads is identical between a slim and a full-save capture."""

    factory, make_input = _MODELS[name]
    torch.manual_seed(0)
    model, x = factory(), make_input()
    full = _capture(copy.deepcopy(model), x, None)
    saved = retrace_save_indices(full)
    slim = _capture(copy.deepcopy(model), x, saved)
    try:
        reads = _names_the_hash_reads(full)
        assert {"func_name", "parents"} <= reads
        assert len(full._layer_list) == len(slim._layer_list)
        for full_record, slim_record in zip(full._layer_list, slim._layer_list, strict=True):
            assert record_type(full_record, None) == record_type(slim_record, None)
            for read_name in sorted(reads):
                assert _read(full_record, read_name) == _read(slim_record, read_name), (
                    f"{name}: hash reads {read_name!r} of {full_record.layer_label!r}, "
                    "which the slim re-trace does not keep"
                )
        assert compute_graph_shape_hash(
            full, include_module_address=False
        ) == compute_graph_shape_hash(slim, include_module_address=False)
        assert retrace_save_coverage_gap(slim, frozenset(saved)) is None
        assert slim._num_saved_ops < full._num_saved_ops
    finally:
        full.cleanup()
        slim.cleanup()


def _retrace_saved_op_counts(monkeypatch: pytest.MonkeyPatch, model: nn.Module, x: Any) -> Any:
    """Validate ``model`` and return (validation capture, re-trace) saved-op counts."""

    real_run = _user_funcs._run_model_and_save_specified_outs
    counts: dict[str, int] = {}

    def spy(*args: Any, **kwargs: Any) -> Any:
        trace = real_run(*args, **kwargs)
        counts["retrace" if not kwargs.get("save_args") else "capture"] = trace._num_saved_ops
        return trace

    monkeypatch.setattr(_user_funcs, "_run_model_and_save_specified_outs", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TraceNotReproducibleWarning)
        assert validate_forward_pass(model, x) is True
    return counts["capture"], counts["retrace"]


def test_reproducibility_retrace_saves_no_outputs_for_a_model_without_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no buffers the hash needs no saved output, so the re-trace saves none."""

    torch.manual_seed(0)
    captured, retraced = _retrace_saved_op_counts(monkeypatch, _Mlp(), torch.rand(2, 4))
    assert captured > 0
    assert retraced == 0


def test_reproducibility_retrace_saves_only_buffer_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buffer model's re-trace saves its buffer nodes and sources, not the whole graph."""

    torch.manual_seed(0)
    captured, retraced = _retrace_saved_op_counts(monkeypatch, _Reassign(), torch.rand(4))
    assert 0 < retraced < captured


@pytest.mark.parametrize("name", ["train_batch_norm", "reassign", "rewrite_on_some_passes"])
def test_retrace_save_set_that_misses_a_buffer_still_fails(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planted bug: a save set dropping every buffer output fails closed as a mismatch."""

    monkeypatch.setattr(user_public_impls, "retrace_save_indices", lambda first_trace: ())
    factory, make_input = _MODELS[name]
    torch.manual_seed(0)
    with pytest.warns(TraceNotReproducibleWarning, match="stateful/non-reproducible"):
        assert validate_forward_pass(factory(), make_input()) is False


@pytest.mark.parametrize("name", ["reassign", "rewrite_on_some_passes"])
def test_retrace_save_set_that_misses_a_buffer_source_fails_closed(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planted bug: a save set that drops only the buffer sources fails closed on coverage.

    The re-trace saves every output (so its hash still equals the first trace's) but
    the save set handed to the comparison omits the buffer sources: the coverage
    check, not a hash difference, must report the miss.
    """

    factory, make_input = _MODELS[name]
    torch.manual_seed(0)
    model, x = factory(), make_input()
    full = _capture(copy.deepcopy(model), x, None)
    retrace = _capture(copy.deepcopy(model), x, None)
    try:
        sources = {
            source.raw_index
            for record in full._layer_list
            if getattr(record, "_graph_buffer_source", None) is not None
            for source in [full._layer_dict_all_keys[record._graph_buffer_source]]
        }
        assert sources, f"{name}: no buffer source to drop -- inconclusive"
        saved = tuple(i for i in retrace_save_indices(full) if i not in sources)
        verdict = compare_retrace_graphs(full, retrace, saved)
        assert verdict is not None
        first_hash, second_hash, hint = verdict
        assert first_hash == second_hash
        assert hint is not None and "outside the re-trace save set" in hint
        assert compare_retrace_graphs(full, retrace, retrace_save_indices(full)) is None
    finally:
        full.cleanup()
        retrace.cleanup()
