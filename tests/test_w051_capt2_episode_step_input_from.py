"""``EpisodeSpec.step_input_from`` (W051-EPISODE out-of-fence item 3).

The options owner gains the declared carried-input argument selector. The
consumer (``resolve_episode_declaration`` -> ``EpisodeJoinSession``, the
``entry_basis`` envelope slot) ships with lane W051-EPISODE and reads the
field via ``getattr(spec, "step_input_from", None)``; the behavioural pin
below is skip-guarded on that consumer's presence so it turns green the
moment the lane merges, and never passes vacuously before.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._episode_spec import EpisodeSpec

pytestmark = pytest.mark.smoke


def test_step_input_from_field_defaults_to_none_and_accepts_str_or_int() -> None:
    names = [f.name for f in dataclasses.fields(EpisodeSpec)]
    assert "step_input_from" in names
    assert names.index("step_input_from") == names.index("crossings") + 1
    assert EpisodeSpec().step_input_from is None
    assert EpisodeSpec(step_input_from="input_ids").step_input_from == "input_ids"
    assert EpisodeSpec(step_input_from=0).step_input_from == 0
    assert tl.options.EpisodeSpec is EpisodeSpec


def _join_consumer_wired() -> bool:
    from torchlens.capture._episode_join import EpisodeJoinSession

    return "step_input_from" in inspect.signature(EpisodeJoinSession.__init__).parameters


class _Cell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(input_ids))


class _Root(nn.Module):
    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self.cell = _Cell()
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for _ in range(self.n_steps):
            hidden = self.cell(input_ids=hidden)
        return hidden


def test_unknown_step_input_name_records_step_input_not_found() -> None:
    if not _join_consumer_wired():
        pytest.skip(
            "EpisodeJoinSession does not take step_input_from on this tree "
            "(lane W051-EPISODE not merged); the field is inert until it lands"
        )
    torch.manual_seed(0)
    model = _Root(n_steps=3).eval()
    x = torch.randn(1, 4)
    trace = tl.trace(
        model,
        x,
        episode=EpisodeSpec(
            stepped_module=model.cell,
            n_steps=3,
            step_output_kind="none",
            step_input_from="no_such_kwarg",
        ),
    )
    ledger = trace.annotations["episode"]
    header = ledger["header"] if isinstance(ledger, dict) else ledger.header
    envelope = header["step_join"] if isinstance(header, dict) else header.step_join
    # The miss is disclosed in the measured envelope (the join's unchecked
    # reason), never guessed around by falling back to the preference rule.
    assert "step_input_not_found" in repr(envelope)
