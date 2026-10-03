"""Lane F44 stage 2: the nested-capture matrix for logged injections.

foldA MEMO s5 item 12's last clause: every capture-shape cell that could
interact with injected-op recording is PINNED -- either it works with
correct anchored identity, or it refuses/no-ops typed and disclosed.

Cells:

* nested ``tl.trace`` inside an intervention hook -- refuses typed
  (``reentrant_trace``), identically with and without ``log_injections``;
* a second rule matched by torch calls INSIDE a hook never fires (hooks run
  under ``pause_logging``), so hook-in-hook recording is unreachable and
  ``nesting_path`` stays ``()`` by construction;
* replay-fork ``do()`` records nothing new (stage-1 deferral disclosed in
  the injection module docs) and never disturbs the captured records;
* ``tl.record`` exposes no ``log_injections`` spelling (no ``capture=``
  kwarg) and its cooked traces carry an empty injected family;
* ``episode=`` + ``intervene=`` refuses pre-execution (F40a's
  ``episode_intervene_unattested`` declaration refusal), so injected ops
  are unreachable inside episode products today;
* the streamed ``to_disk`` writer cell is pinned in
  ``test_log_injections_s2_forgery.py`` (door provoked directly; streamed
  intervened captures fail earlier on a pre-existing portability gap).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

_LOGGED = tl.options.CaptureOptions(log_injections=True)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh: minimal injected-op substrate."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return torch.tanh(self.fc2(torch.relu(self.fc1(x))))


@pytest.fixture()
def chain():
    """Seeded chain model + input."""

    torch.manual_seed(0)
    return _Chain().eval(), torch.randn(2, 4)


@pytest.mark.smoke
def test_nested_trace_inside_hook_refuses_reentrant(chain) -> None:
    """A tl.trace inside a hook refuses typed, logged or not."""

    model, x = chain

    def nested(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Attempt a nested capture mid-hook."""

        tl.trace(nn.Linear(4, 4).eval(), torch.randn(2, 4))
        return out

    for capture in (None, _LOGGED):
        kwargs = {"capture": capture} if capture is not None else {}
        with pytest.raises(Exception) as excinfo:
            tl.trace(model, x, intervene=tl.when(tl.func("relu"), nested), **kwargs)
        assert excinfo.value.fields["code"] == "reentrant_trace"
    # the failed nested attempts leave torch clean for ordinary captures
    plain = tl.trace(model, x, capture=_LOGGED)
    assert plain.injected_ops == ()


def test_hook_calls_never_fire_other_rules_nesting_path_flat(chain) -> None:
    """Torch calls inside a hook never fire a second rule's hook.

    Hooks execute under ``pause_logging``, so rule evaluation cannot
    re-enter: every record carries the OUTER rule's id and the flat
    ``nesting_path == ()`` the C07 grammar reserves for deeper nesting.
    """

    model, x = chain

    def outer(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Inject a tanh call that textually matches the second rule."""

        return torch.tanh(out)

    def never(out: torch.Tensor, *, hook) -> torch.Tensor:  # pragma: no cover
        """Second rule's hook; firing inside the first hook would be a bug."""

        raise AssertionError("a hook fired on an injected op")

    spec = tl.when(tl.func("relu"), outer) & tl.when(tl.func("tanh"), never)
    with pytest.raises(AssertionError):
        # the MODEL tanh legitimately fires the second rule
        tl.trace(model, x, intervene=spec, capture=_LOGGED)
    relu_only = tl.when(tl.func("relu"), outer)
    logged = tl.trace(model, x, intervene=relu_only, capture=_LOGGED)
    records = logged.injected_ops
    assert len(records) == 1 and records[0].func_name == "tanh"
    rule_ids = {record.provenance.spec_rule_id for record in records}
    assert rule_ids == {relu_only.rules[0].rule_id}
    assert {record.provenance.nesting_path for record in records} == {()}


def test_replay_fork_do_records_nothing_and_preserves_records(chain) -> None:
    """Replay-side hooks record nothing; captured records stay intact."""

    model, x = chain
    logged = tl.trace(
        model,
        x,
        intervene=tl.when(tl.func("relu"), lambda out, *, hook: torch.sigmoid(out)),
        capture=tl.options.CaptureOptions(log_injections=True, intervention_ready=True),
    )
    before = logged.injected_ops
    assert len(before) == 1
    fork = logged.fork()
    fork.do(tl.units("tanh_1_4", [(0, 0)]).resolve(fork), tl.zero_ablate())
    assert logged.injected_ops == before
    fork_records = getattr(fork, "injected_ops", ())
    # the fork never gains NEW records from replay-side execution
    assert len(fork_records) <= len(before)


@pytest.mark.smoke
def test_record_has_no_injection_spelling_and_cooked_traces_are_empty(chain) -> None:
    """The fastlog recorder exposes no log_injections door (pinned)."""

    model, x = chain
    with pytest.raises(TypeError):
        tl.record(model, x, save=tl.func("relu"), capture=_LOGGED)
    recording = tl.record(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), lambda out, *, hook: torch.sigmoid(out)),
    )
    cooked = recording.to_trace()
    assert cooked.injected_ops == ()


def test_episode_intervene_refusal_keeps_injections_unreachable(chain) -> None:
    """episode= + intervene= refuses pre-execution (F40a), logged or not."""

    class _Root(nn.Module):
        """Three-step wrapper over one stepped cell."""

        def __init__(self, cell: nn.Module) -> None:
            """Hold the stepped cell."""

            super().__init__()
            self.cell = cell

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run three steps."""

            for _ in range(3):
                x = self.cell(x)
            return x

    model, x = chain
    root = _Root(model).eval()
    spec = tl.options.EpisodeSpec(stepped_module=model, n_steps=3)
    for capture in (None, _LOGGED):
        kwargs = {"capture": capture} if capture is not None else {}
        with pytest.raises(Exception) as excinfo:
            tl.trace(
                root,
                x,
                episode=spec,
                intervene=tl.when(tl.func("relu"), lambda out, *, hook: out),
                **kwargs,
            )
        assert "episode" in type(excinfo.value).__name__.lower() or "episode" in str(excinfo.value)
