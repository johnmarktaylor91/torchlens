"""Tests for Phase 2 onramp and discovery APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import utils
from torchlens.experimental import attribute_walk


class TinyMLP(nn.Module):
    """Small feedforward model for onramp tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.fc2(self.relu(self.fc1(x)))


class TinyCNN(nn.Module):
    """Small convolutional model for onramp tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TinyRecurrent(nn.Module):
    """Small recurrent-parameter model for onramp tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        for _index in range(2):
            x = torch.tanh(self.fc(x))
        return x


class SignatureModel(nn.Module):
    """Model with an inferable forward input annotation."""

    def forward(self, x: Annotated[torch.Tensor, (2, 4)]) -> torch.Tensor:
        """Return the input."""

        return x


@pytest.fixture(
    params=[
        (TinyMLP, lambda: torch.randn(2, 4)),
        (TinyCNN, lambda: torch.randn(2, 1, 4, 4)),
        (TinyRecurrent, lambda: torch.randn(2, 3)),
    ],
    ids=["mlp", "cnn", "recurrent"],
)
def architecture(request: pytest.FixtureRequest) -> tuple[nn.Module, torch.Tensor]:
    """Return a model/input pair for one small architecture."""

    model_cls, input_factory = request.param
    torch.manual_seed(0)
    return model_cls(), input_factory()


def _first_saved_label(model: nn.Module, x: torch.Tensor) -> str:
    """Return a non-input, non-output layer label for a model."""

    log = _metadata_log(model, x)
    for label in log.layer_labels_no_pass:
        if not label.startswith(("input", "output")):
            if log.layer_num_passes.get(label, 1) > 1:
                return f"{label}:1"
            return label
    raise AssertionError("No internal layer found.")


def _metadata_log(model: nn.Module, x: torch.Tensor) -> tl.ModelLog:
    """Return a metadata-only ModelLog."""

    return tl.log_forward_pass(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )


def test_peek_on_three_architectures(architecture: tuple[nn.Module, torch.Tensor]) -> None:
    """``tl.peek`` returns one activation across representative architectures."""

    model, x = architecture
    label = _first_saved_label(model, x)
    activation = tl.peek(model, x, label)
    assert isinstance(activation, torch.Tensor)
    assert activation.shape[0] == x.shape[0]


def test_extract_list_and_dict_on_three_architectures(
    architecture: tuple[nn.Module, torch.Tensor],
) -> None:
    """``tl.extract`` supports list and mapping layer specs."""

    model, x = architecture
    label = _first_saved_label(model, x)
    listed = tl.extract(model, x, [label])
    mapped = tl.extract(model, x, {"feature": label})
    assert list(listed) == [label]
    assert set(mapped) == {"feature"}
    assert torch.equal(listed[label], mapped["feature"])


def test_batched_extract_memory_and_disk_on_three_architectures(
    architecture: tuple[nn.Module, torch.Tensor],
    tmp_path: Path,
) -> None:
    """``tl.batched_extract`` supports in-memory and per-batch disk outputs."""

    model, x = architecture
    label = _first_saved_label(model, x)
    stimuli = torch.cat([x, x], dim=0)
    in_memory = tl.batched_extract(model, stimuli, [label], batch_size=x.shape[0], progress=False)
    assert in_memory[label].shape[0] == stimuli.shape[0]

    output_paths = tl.batched_extract(
        model,
        stimuli,
        [label],
        batch_size=x.shape[0],
        output_dir=tmp_path,
        progress=False,
    )
    assert len(output_paths) == 2
    assert all(path.exists() for path in output_paths)
    assert label in torch.load(output_paths[0])


def test_list_modules_on_three_architectures(architecture: tuple[nn.Module, torch.Tensor]) -> None:
    """``utils.list_modules`` includes root and child modules."""

    model, _x = architecture
    modules = utils.list_modules(model)
    assert modules[0][0] == "self"
    assert all(isinstance(address, str) for address, _module_type in modules)


def test_list_ops_on_three_architectures(architecture: tuple[nn.Module, torch.Tensor]) -> None:
    """``utils.list_ops`` returns op counts and train/eval exports."""

    model, x = architecture
    ops = utils.list_ops(model, x)
    both = utils.list_ops(model, x, mode="both")
    assert isinstance(ops, list)
    assert ops
    assert set(both) == {"eval", "train"}


def test_find_layers_and_suggest_on_three_architectures(
    architecture: tuple[nn.Module, torch.Tensor],
) -> None:
    """``ModelLog.find_layers`` and ``suggest`` return useful labels."""

    model, x = architecture
    log = _metadata_log(model, x)
    label = _first_saved_label(model, x)
    base_label = label.split(":", 1)[0]
    prefix = base_label.split("_", 1)[0]
    assert base_label in log.find_layers(prefix)
    assert log.suggest(prefix)
    with pytest.raises(ValueError, match="Did you mean"):
        tl.peek(model, x, "definitely_missing_layer")


def test_layer_accessor_queries_and_completion(
    architecture: tuple[nn.Module, torch.Tensor],
) -> None:
    """Layer accessors expose query helpers and key completions."""

    model, x = architecture
    log = _metadata_log(model, x)
    label = _first_saved_label(model, x)
    base_label = label.split(":", 1)[0]
    operator = log[label].func_name or log[label].layer_type
    assert log.layers.total() == len(log.layer_logs)
    assert base_label in log.layers._ipython_key_completions_()
    assert base_label in dir(log.layers)
    assert isinstance(log.layers.by_operator(), dict)
    assert base_label in log.layers.by_operator(operator)
    assert isinstance(log.layers.by_module(), dict)
    assert isinstance(log.layers.by_module_and_operator(), dict)


def test_model_log_site_reports(architecture: tuple[nn.Module, torch.Tensor]) -> None:
    """ModelLog site reports return lists."""

    model, x = architecture
    log = _metadata_log(model, x)
    assert isinstance(log.unsupported_ops(), list)
    assert isinstance(log.uncalled_modules(), list)


def test_peek_graph_writes_file(tmp_path: Path) -> None:
    """``utils.peek_graph`` renders to disk."""

    model = TinyMLP()
    output_path = tmp_path / "quick_graph"
    utils.peek_graph(model, torch.randn(1, 4), output_path=output_path, file_format="pdf")
    assert list(tmp_path.iterdir())


def test_synthetic_input_success_and_error() -> None:
    """``utils.synthetic_input`` builds annotated inputs and errors when ambiguous."""

    synthetic = utils.synthetic_input(SignatureModel())
    assert synthetic.shape == (2, 4)

    with pytest.raises(ValueError, match="Cannot infer"):
        utils.synthetic_input(TinyMLP())


def test_flop_count_and_save_set() -> None:
    """Utility FLOP and executable-save heuristics return basic values."""

    model = TinyMLP()
    x = torch.randn(2, 4)
    log = _metadata_log(model, x)
    label = _first_saved_label(model, x)
    assert utils.flop_count(model, x) >= 0
    assert utils.find_executable_save_set(log, [label], "1 GB") == [label]


def test_attribute_walk() -> None:
    """Experimental attribute walking resolves dotted and indexed addresses."""

    class Wrapped(nn.Module):
        """Wrapper with indexed module address."""

        def __init__(self) -> None:
            """Initialize wrapped layers."""

            super().__init__()
            self.layers = nn.ModuleList([TinyMLP()])

    model = Wrapped()
    assert attribute_walk(model, "layers[0].fc1") is model.layers[0].fc1
