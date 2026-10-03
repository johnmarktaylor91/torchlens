"""W051-EXTRACT: default collation of (input, label)-shaped items teaches (audit 3.13).

Tuple items, ``TensorDataset`` rows, and ``DataLoader`` batches default-collate
to one positional input per element and used to die inside the model with a
raw ``TypeError``; when ``forward`` cannot bind that many positionals the
engine now refuses ``extraction_collate_ambiguous`` naming the ``collate=``
door. Forwards that CAN bind them (or take ``*args``) are untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._errors import InvalidArgumentError
from torchlens.dataset_extraction import extract_dataset


class _OneInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        return h, self.fc2(h)


class _TwoInputs(_OneInput):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        h = torch.relu(self.fc1(x)) + y.to(x.dtype).unsqueeze(-1)
        return h, self.fc2(h)


class _VarArgs(_OneInput):
    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        h = torch.relu(self.fc1(inputs[0]))
        return h, self.fc2(h)


def _pairs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(7, 3), torch.randint(0, 2, (7,))


def _stimuli_variants() -> dict[str, object]:
    x, y = _pairs()
    dataset = torch.utils.data.TensorDataset(x, y)
    return {
        "tuple_items": [(x[i], y[i]) for i in range(7)],
        "tensor_dataset": dataset,
        "dataloader_bs1": torch.utils.data.DataLoader(dataset, batch_size=1),
    }


@pytest.mark.parametrize("variant", sorted(_stimuli_variants()))
def test_pair_items_refuse_typed_when_forward_binds_one_positional(
    tmp_path: Path, variant: str
) -> None:
    stimuli = _stimuli_variants()[variant]
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _OneInput().eval(),
            stimuli,
            {"h": "fc1"},
            batch_size=3,
            output_dir=tmp_path,
            progress=False,
        )
    fields = excinfo.value.fields
    assert fields["code"] == "extraction_collate_ambiguous"
    assert fields["n_positional"] == 2
    assert fields["n_bindable"] == 1
    assert fields["batch_index"] == 0
    assert "collate=" in str(excinfo.value)
    assert not (tmp_path / "ledger.jsonl").exists(), "refused before any shard committed"


def test_pair_items_bind_when_forward_declares_two_positionals(tmp_path: Path) -> None:
    x, y = _pairs()
    paths = extract_dataset(
        _TwoInputs().eval(),
        [(x[i], y[i]) for i in range(7)],
        {"h": "fc1"},
        batch_size=3,
        output_dir=tmp_path,
        progress=False,
    )
    assert len(paths) == 3


def test_var_positional_forward_is_left_alone(tmp_path: Path) -> None:
    x, y = _pairs()
    paths = extract_dataset(
        _VarArgs().eval(),
        [(x[i], y[i]) for i in range(7)],
        {"h": "fc1"},
        batch_size=3,
        output_dir=tmp_path,
        progress=False,
    )
    assert len(paths) == 3


def test_user_collate_returning_the_input_is_the_taught_door(tmp_path: Path) -> None:
    x, y = _pairs()
    paths = extract_dataset(
        _OneInput().eval(),
        [(x[i], y[i]) for i in range(7)],
        {"h": "fc1"},
        batch_size=3,
        output_dir=tmp_path,
        progress=False,
        collate=lambda items: torch.stack([a for a, _ in items]),
    )
    assert len(paths) == 3
