"""A model changed between two traces is traced as it is, and is freed afterwards.

The per-trace cost work keeps a few things between traces (sampled checkpoint
tensors for the torchvision weights match, a process handle for the memory
reading, the set of modules the stale-reference sweep has already visited).
None of them may make a second trace of the same model object answer from the
first: these tests change the model between two traces (a replaced submodule, a
swapped parameter, an added hook) and assert the second trace sees the change,
and that the model object is released once its traces are gone.
"""

from __future__ import annotations

import gc
import os
import sys
import types
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._autoroute._builtin_output import semantic_output_cache_key
from torchlens._capture import peak_memory
from torchlens._utils import torchvision_weights as tvw
from torchlens.backends.torch import _prepared_tree as prepared_tree, belt


class _Small(nn.Module):
    """Linear, activation, linear."""

    def __init__(self) -> None:
        """Build the three submodules."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act: nn.Module = nn.ReLU()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three submodules in order."""

        return self.head(self.act(self.fc(x)))


class _FakeVision(nn.Module):
    """Stand-in for a torchvision class (module name spoofed below)."""

    def __init__(self) -> None:
        """Build one linear layer and one buffer."""

        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.register_buffer("scale", torch.arange(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the linear output by the buffer."""

        return self.fc(x) * self.scale


_FakeVision.__module__ = "torchvision.models._per_trace_cost_test"


class _Member:
    """A weights-member double with ``url``, ``meta`` and ``transforms``."""

    def __init__(self, name: str, url: str, preset: Any = None) -> None:
        """Record the member's name, checkpoint address and preprocessing preset."""

        self.name = name
        self.url = url
        self.meta = {"num_params": 15, "categories": ["a", "b", "c"]}
        self._preset = preset

    def transforms(self) -> Any:
        """Return the preprocessing preset."""

        return self._preset

    def __str__(self) -> str:
        """Return the member name."""

        return self.name


@pytest.fixture(autouse=True)
def _empty_sample_cache() -> Any:
    """Start and end every test with no checkpoint samples kept."""

    tvw._sample_cache.clear()
    yield
    tvw._sample_cache.clear()


@pytest.fixture
def hub_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the torch hub cache at an empty temporary directory."""

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    (tmp_path / "checkpoints").mkdir()
    return tmp_path / "checkpoints"


def _count_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every checkpoint file the weights match loads."""

    loads: list[str] = []
    real_load = tvw._load_checkpoint

    def counting_load(path: str) -> Any:
        """Note the path, then load as usual."""

        loads.append(os.path.basename(path))
        return real_load(path)

    monkeypatch.setattr(tvw, "_load_checkpoint", counting_load)
    return loads


def _traced_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Trace ``model`` on ``x`` and return the saved value of its one output."""

    trace = tl.trace(model, x)
    (output_op,) = trace.output_ops
    return output_op.out.detach().clone()


def test_replaced_submodule_is_traced_as_replaced() -> None:
    """A submodule replaced between two traces is the one the second trace records."""

    torch.manual_seed(0)
    model = _Small().eval()
    x = torch.randn(3, 4)

    first = tl.trace(model, x)
    assert first.modules["act"].class_name == "ReLU"
    assert "relu" in {op.func_name for op in first.ops}

    model.act = nn.Tanh()
    second = tl.trace(model, x)
    assert second.modules["act"].class_name == "Tanh"
    second_funcs = {op.func_name for op in second.ops}
    assert "tanh" in second_funcs
    assert "relu" not in second_funcs
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))

    model.act = nn.Sequential(nn.ReLU(), nn.Linear(4, 4))
    third = tl.trace(model, x)
    assert third.modules["act"].class_name == "Sequential"
    assert third.modules["act.1"].class_name == "Linear"
    (output_op,) = third.output_ops
    assert torch.equal(output_op.out, model(x))


class _Chain(nn.Module):
    """Runs its registered children in registration order."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every child in order."""

        for layer in self.children():
            x = layer(x)
        return x


def test_submodule_replaced_at_the_same_name_after_deletion() -> None:
    """A submodule deleted and re-created under its old name is prepared again."""

    torch.manual_seed(0)
    model = _Small().eval()
    x = torch.randn(3, 4)
    tl.trace(model, x)

    del model.act
    model.act = nn.Tanh()
    second = tl.trace(model, x)
    assert second.modules["act"].class_name == "Tanh"
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))


def test_renamed_and_added_submodules_get_their_current_addresses() -> None:
    """A submodule moved to a new name, and one added later, are recorded where they are now."""

    torch.manual_seed(0)
    model = _Chain().eval()
    model.a = nn.Linear(4, 4)
    model.b = nn.ReLU()
    x = torch.randn(3, 4)
    first = tl.trace(model, x)
    assert first.modules["a"].class_name == "Linear"
    assert first.modules["b"].class_name == "ReLU"

    model.c = model.a
    del model.a
    model.d = nn.Tanh()
    second = tl.trace(model, x)
    assert second.modules["b"].class_name == "ReLU"
    assert second.modules["c"].class_name == "Linear"
    assert second.modules["d"].class_name == "Tanh"
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))


def test_module_tree_key_tells_changed_trees_apart() -> None:
    """The tree key is equal for an unchanged tree and differs for every kind of change."""

    model = _Small()
    key = prepared_tree.module_tree_key(model)
    assert key is not None
    assert prepared_tree.module_tree_key(model) == key

    relu = model.act
    model.act = nn.ReLU()
    replaced = prepared_tree.module_tree_key(model)
    assert replaced != key
    model.act = relu
    assert prepared_tree.module_tree_key(model) == key

    model.extra = None
    with_none = prepared_tree.module_tree_key(model)
    assert with_none != key
    model.extra = nn.Identity()
    assert prepared_tree.module_tree_key(model) not in (key, with_none)
    del model.extra
    assert prepared_tree.module_tree_key(model) == key

    model.act = nn.Sequential(relu)
    nested = prepared_tree.module_tree_key(model)
    assert nested != key
    model.act.append(nn.Identity())
    assert prepared_tree.module_tree_key(model) not in (key, nested)

    assert prepared_tree.module_tree_key(object()) is None


def test_unchanged_model_keeps_the_prepared_fast_path() -> None:
    """A model traced twice without a change is not prepared in full again."""

    model = _Small().eval()
    x = torch.randn(3, 4)
    tl.trace(model, x)
    assert model in _state._prepared_models
    assert prepared_tree.changed_module_tree(model) is prepared_tree.UNCHANGED
    assert model in _state._prepared_models

    model.act = nn.Tanh()
    changed = prepared_tree.changed_module_tree(model)
    assert changed is not prepared_tree.UNCHANGED
    assert model not in _state._prepared_models
    tl.trace(model, x)
    assert model in _state._prepared_models
    assert prepared_tree.changed_module_tree(model) is prepared_tree.UNCHANGED


def test_swapped_parameter_is_traced_as_swapped() -> None:
    """A parameter swapped or rewritten between two traces changes the second trace."""

    torch.manual_seed(0)
    model = _Small().eval()
    x = torch.randn(3, 4)
    before = _traced_output(model, x)
    assert torch.equal(before, model(x))

    model.fc.weight = nn.Parameter(torch.full_like(model.fc.weight, 0.5))
    swapped = _traced_output(model, x)
    assert torch.equal(swapped, model(x))
    assert not torch.equal(swapped, before)

    with torch.no_grad():
        model.fc.weight.data.mul_(-1.0)
    rewritten = _traced_output(model, x)
    assert torch.equal(rewritten, model(x))
    assert not torch.equal(rewritten, swapped)


def test_added_hook_runs_in_the_next_trace() -> None:
    """A forward hook added between two traces runs in the second and changes its output."""

    torch.manual_seed(0)
    model = _Small().eval()
    x = torch.randn(3, 4)
    plain = _traced_output(model, x)
    calls: list[int] = []

    def double(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        """Count the call and double the module output."""

        calls.append(1)
        return output * 2

    handle = model.fc.register_forward_hook(double)
    hooked = _traced_output(model, x)
    assert calls, "the hook added after the first trace did not run in the second"
    assert torch.equal(hooked, model(x))
    assert not torch.equal(hooked, plain)

    handle.remove()
    calls.clear()
    unhooked = _traced_output(model, x)
    assert not calls
    assert torch.equal(unhooked, plain)


def test_weights_match_follows_the_model_between_traces(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights match is recomputed from the model's values, without reloading the file."""

    torchvision_models = pytest.importorskip("torchvision.models")
    preset = torchvision_models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    model = _FakeVision().eval()
    member = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth", preset)
    torch.save(model.state_dict(), hub_dir / "fake_v1.pth")
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [member])
    loads = _count_loads(monkeypatch)
    x = torch.randn(2, 4)

    tl.trace(model, x)
    matched = semantic_output_cache_key(model)
    assert tvw.match_torchvision_weights(model) is member
    assert loads == ["fake_v1.pth"], "the checkpoint is loaded once, not once per match"

    built_with = model.fc.weight.detach().clone()
    with torch.no_grad():
        model.fc.weight.data.add_(1.0)
    tl.trace(model, x)
    assert tvw.match_torchvision_weights(model) is None
    assert semantic_output_cache_key(model) != matched

    model.fc.weight = nn.Parameter(built_with)
    tl.trace(model, x)
    assert tvw.match_torchvision_weights(model) is member
    assert semantic_output_cache_key(model) == matched
    assert loads == ["fake_v1.pth"]


def test_rewritten_checkpoint_file_is_read_again(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint file replaced on disk under the same name is not answered from memory."""

    model = _FakeVision().eval()
    other = _FakeVision().eval()
    member = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth")
    path = hub_dir / "fake_v1.pth"
    torch.save(model.state_dict(), path)
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [member])
    loads = _count_loads(monkeypatch)

    assert tvw.match_torchvision_weights(model) is member
    assert tvw.match_torchvision_weights(other) is None
    assert len(loads) == 1

    stamp = path.stat().st_mtime_ns + 5_000_000_000
    torch.save(other.state_dict(), path)
    os.utime(path, ns=(stamp, stamp))
    assert tvw.match_torchvision_weights(model) is None
    assert tvw.match_torchvision_weights(other) is member
    assert len(loads) == 2

    path.unlink()
    assert tvw.match_torchvision_weights(other) is None


def test_sample_over_the_budget_is_not_kept(hub_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sample larger than the byte budget is read from the file on every match."""

    model = _FakeVision().eval()
    member = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth")
    torch.save(model.state_dict(), hub_dir / "fake_v1.pth")
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [member])
    monkeypatch.setattr(tvw, "_SAMPLE_CACHE_BYTES", 8)
    loads = _count_loads(monkeypatch)

    assert tvw.match_torchvision_weights(model) is member
    assert tvw.match_torchvision_weights(model) is member
    assert len(loads) == 2
    assert not tvw._sample_cache


def test_sample_cache_stays_inside_its_byte_budget(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keeping a new sample drops the oldest ones until the total fits the budget."""

    model = _FakeVision().eval()
    members = [
        _Member(f"FAKE_V{index}", f"https://example.invalid/w/fake_v{index}.pth")
        for index in range(3)
    ]
    for index in range(3):
        torch.save(_FakeVision().state_dict(), hub_dir / f"fake_v{index}.pth")
    torch.save(model.state_dict(), hub_dir / "fake_v2.pth")
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: members)
    one_sample = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    monkeypatch.setattr(tvw, "_SAMPLE_CACHE_BYTES", 2 * one_sample)

    assert tvw.match_torchvision_weights(model) is members[2]
    kept = [os.path.basename(key[0]) for key in tvw._sample_cache]
    assert kept == ["fake_v1.pth", "fake_v2.pth"]
    assert sum(entry[1] for entry in tvw._sample_cache.values()) <= 2 * one_sample


def test_checkpoint_that_is_not_a_state_mapping_never_matches(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached file holding something other than a state mapping declines every time."""

    model = _FakeVision().eval()
    member = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth")
    torch.save([1, 2, 3], hub_dir / "fake_v1.pth")
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [member])

    assert tvw.match_torchvision_weights(model) is None
    assert tvw.match_torchvision_weights(model) is None


def test_model_is_freed_after_its_traces(hub_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing kept between traces holds the model once its traces are dropped."""

    member = _Member("FAKE_V1", "https://example.invalid/w/fake_v1.pth")
    monkeypatch.setattr(tvw, "_candidate_weights", lambda _model: [member])
    for build in (_Small, _FakeVision):
        model = build().eval()
        if build is _FakeVision:
            torch.save(model.state_dict(), hub_dir / "fake_v1.pth")
        model_ref = weakref.ref(model)
        parameter_ref = weakref.ref(next(model.parameters()))
        x = torch.randn(2, 4)
        first = tl.trace(model, x)
        second = tl.trace(model, x)
        del first, second
        gc.collect()
        del model
        gc.collect()
        assert model_ref() is None, build.__name__
        assert parameter_ref() is None, build.__name__


def _unswept_module_names() -> list[str]:
    """Return the names of the real modules the next sweep would visit."""

    return [
        name
        for name, module in belt._unswept_module_items()
        if isinstance(module, types.ModuleType)
    ]


def test_sweep_visits_only_modules_it_has_not_swept() -> None:
    """After a full sweep, a new module is the only one visited, and it is still patched."""

    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    wrapper = torch.from_numpy
    original = _state._decorated_to_orig[id(wrapper)]
    belt.sweep_stale_belt_references()
    assert _unswept_module_names() == []

    late = types.ModuleType("_tl_per_trace_cost_late")
    late.op = original
    sys.modules[late.__name__] = late
    try:
        assert _unswept_module_names() == [late.__name__]
        assert belt.sweep_stale_belt_references() >= 1
        assert late.op is wrapper
        assert _unswept_module_names() == []
        assert belt.sweep_stale_belt_references() == 0
    finally:
        sys.modules.pop(late.__name__, None)
        belt.restore_belt_references()
    assert late.op is original


def test_memory_reading_keeps_one_process_handle() -> None:
    """The resident-size reading reuses its handle and rebuilds it for another process id."""

    pytest.importorskip("psutil")
    assert peak_memory.process_rss_bytes() > 0
    ((pid, handle),) = peak_memory._PSUTIL_PROCESS
    assert pid == os.getpid()
    assert peak_memory.process_rss_bytes() > 0
    assert peak_memory._PSUTIL_PROCESS[0][1] is handle

    peak_memory._PSUTIL_PROCESS[:] = [(-1, object())]
    assert peak_memory.process_rss_bytes() > 0
    ((pid, rebuilt),) = peak_memory._PSUTIL_PROCESS
    assert pid == os.getpid()
    assert rebuilt.pid == os.getpid()
