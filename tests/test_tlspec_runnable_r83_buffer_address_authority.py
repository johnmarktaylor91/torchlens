"""Recorded buffer address is authoritative over the shape heuristic (r83 C2, r82 free).

A shape heuristic in postprocess OVERRODE the address the capture had correctly
recorded. ``_buffer_addresses_by_label`` discarded the recorded address whenever
it did not name an unassigned REGISTERED buffer and re-derived one from
``trace._buffer_initial_values`` (registered buffers ONLY) through an
equivalence-class -> value -> **shape** ladder. A non-registered tensor source
(a plain ``__dict__`` attribute or a list element) consumed BEFORE a same-shaped
registered buffer therefore CLAIMED that buffer's address, passed the runnable
preflight -- whose ``address not in buffer_names`` refusal the stolen address
walks straight past -- and was bound to the WRONG state at replay. Reported
``VERIFIED`` with ``WitnessCompleteness.COMPLETE``, ``random_filled_slot_ids=()``
and no diagnostic.

Observed pre-fix, all silent wrong results:

* a 4-line ImageNet normalizer with ``mean`` a plain attribute and ``std``
  registered -> op addresses ``[('buffer_1','std'), ('buffer_2','std')]``,
  ``VERIFIED``, max-diff 2.02;
* the minimal deterministic form -> ``VERIFIED``, max-diff 2.0;
* a nested model with a registered ``child.cb``, a plain-attribute ``child.q``
  and a registered ``top`` -> ``child.q`` was given ``'top'``, ``VERIFIED``,
  max-diff 4.37. (This one was inside the r82 over-trigger battery and had been
  scored a PASS -- the bug was hiding in the zero-collateral evidence.)

The corruption also reaches plain, save-free public metadata:
``trace["buffer_1"].address`` reported a name that was not that tensor's.

PRE-EXISTING (byte-identical at ``10975d98``) and DISTINCT from the documented
multi-positional-arg residual, which is cross-capture, needs a prior save, and
manifests as a REFUSAL. This is a single fresh capture in a fresh process
manifesting as a silent wrong bind.

THE r83 CLOSURE. The recorded address is recovered from the source event itself
(``equivalence_class`` is built as ``f"buffer_{address}"`` plus a module-stack
suffix reconstructible exactly from ``event.modules``) and is AUTHORITATIVE; the
value/shape ladder may only fill a genuinely MISSING address. A recorded address
that is not a registered buffer now reaches the honest
``UNSUPPORTED_TENSOR_CONSTANT`` refusal instead of stealing a registered one.

The live tensor's own ``TensorMeta.address`` stamp is NOT usable at this point:
session cleanup strips it before postprocess runs, which is why the recorded
address has to come off the event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _buffer_addresses(trace: Any) -> list[tuple[str, str | None]]:
    """Return ``(layer_label, address)`` for every materialized buffer op."""

    return [
        (op.layer_label, op.address) for op in trace if getattr(op, "layer_type", None) == "buffer"
    ]


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture one runnable-ready trace."""

    return tl.trace(model, x, capture=_CAPTURE)


def _replay_or_refuse(model: nn.Module, x: torch.Tensor, path: Path) -> Any | None:
    """Save/load/run an artifact, returning ``None`` on an honest save refusal."""

    trace = _capture(model, x)
    try:
        trace.save(path, level="runnable", include_weights=True)
    except RunnablePreflightError:
        return None
    return tl.load(path).run(inputs=x.clone())


# --------------------------------------------------------------------------- #
# The three observed wrong-bind models.
# --------------------------------------------------------------------------- #


class _Normalizer(nn.Module):
    """Ordinary ImageNet normalizer: one constant registered, one plain."""

    def __init__(self) -> None:
        """Build the normalizer with a deliberately mixed constant style."""

        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with a plain-attribute mean and a registered std."""

        return (x - self.mean) / self.std


class _Minimal(nn.Module):
    """Minimal deterministic form: plain attr consumed before a registered buffer."""

    def __init__(self) -> None:
        """Build two same-shaped constants of different registration style."""

        super().__init__()
        self.q = torch.full((2, 2), 3.0)
        self.register_buffer("top", torch.full((2, 2), 5.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Consume the plain attribute first, then the registered buffer."""

        return (z - self.q) + self.top


class _ListElement(nn.Module):
    """List-element spelling of the same reach (r82 matrix case F)."""

    def __init__(self) -> None:
        """Hold the non-registered constant as a list element."""

        super().__init__()
        # No underscore prefix: model preparation stamps list-element buffers,
        # so this reaches the address ladder exactly like r82's matrix case F.
        self.h = [torch.full((2, 2), 3.0)]
        self.register_buffer("top", torch.full((2, 2), 5.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Consume the list element first, then the registered buffer."""

        return (z - self.h[0]) + self.top


class _NestedChild(nn.Module):
    """Submodule mixing a registered buffer and a plain attribute."""

    def __init__(self) -> None:
        """Register ``cb`` and keep ``q`` as a plain attribute."""

        super().__init__()
        torch.manual_seed(2)
        self.register_buffer("cb", torch.randn(2, 4, 8, 8))
        self.q = torch.randn(2, 4, 8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume both constants."""

        return x + self.cb - self.q


class _Nested(nn.Module):
    """Nested model whose plain-attribute constant stole ``top``'s address."""

    def __init__(self) -> None:
        """Build the nested stack."""

        super().__init__()
        self.child = _NestedChild()
        torch.manual_seed(3)
        self.register_buffer("top", torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child and add the top-level buffer."""

        return self.child(x) + self.top


@pytest.mark.smoke
def test_plain_attribute_does_not_steal_a_registered_buffer_address() -> None:
    """The recorded address must survive postprocess, registered or not.

    Pre-fix: ``[('buffer_1','top'), ('buffer_2','top')]`` -- ``q``'s event was
    given ``top``'s address by the shape rung, so TWO ops claimed ``top`` and
    neither reported ``q``.
    """

    trace = _capture(_Minimal(), torch.zeros(2, 2))
    assert _buffer_addresses(trace) == [("buffer_1", "q"), ("buffer_2", "top")]


@pytest.mark.smoke
def test_list_element_does_not_steal_a_registered_buffer_address() -> None:
    """Matrix case F: the list-element spelling must resolve honestly too."""

    trace = _capture(_ListElement(), torch.zeros(2, 2))
    addresses = dict(_buffer_addresses(trace))
    assert "top" in addresses.values()
    assert sorted(addresses.values()) != ["top", "top"], "two ops claimed one address"
    assert "h.0" in addresses.values(), f"list element lost its address: {addresses}"


@pytest.mark.smoke
def test_nested_plain_attribute_keeps_its_own_address() -> None:
    """The nested case that hid inside the r82 over-trigger battery.

    Pre-fix ``child.q`` was reported as ``'top'`` and the artifact replayed
    ``VERIFIED`` at max-diff 4.37 against a fresh oracle-1 instance.
    """

    trace = _capture(_Nested(), torch.randn(2, 4, 8, 8))
    addresses = dict(_buffer_addresses(trace))
    assert "child.q" in addresses.values(), f"plain attribute lost its address: {addresses}"
    assert "child.cb" in addresses.values()
    assert "top" in addresses.values()


@pytest.mark.parametrize(
    "factory",
    [_Normalizer, _Minimal, _ListElement, _Nested],
    ids=["realistic_normalizer", "minimal", "list_element", "nested"],
)
def test_non_registered_constant_never_replays_a_silent_wrong_verified(
    tmp_path: Path, factory: Any
) -> None:
    """Either refuse honestly or replay correctly -- never a silent wrong VERIFIED.

    A non-registered constant is not part of the declared state model
    (``state_dict`` = named parameters plus persistent buffers), so the honest
    outcome is the save-side ``UNSUPPORTED_TENSOR_CONSTANT`` refusal. What must
    never happen again is the third option the shape heuristic created: binding
    it to a DIFFERENT buffer's state and calling the result faithful.
    """

    model = factory()
    x = (
        torch.randn(2, 3, 4, 4)
        if factory is _Normalizer
        else torch.zeros(2, 2)
        if factory in (_Minimal, _ListElement)
        else torch.randn(2, 4, 8, 8)
    )
    with torch.no_grad():
        oracle = factory()(x.clone())

    result = _replay_or_refuse(model, x, tmp_path / "addr.tlspec")
    if result is None:
        return  # honest save-side refusal

    max_diff = (result.output - oracle).abs().max().item()
    if result.report.path_faithfulness is PathFaithfulness.VERIFIED:
        assert not result.report.poisoned
        assert max_diff < 1e-6, f"FALSE VERIFIED: replay disagrees with oracle 1 by {max_diff}"


# --------------------------------------------------------------------------- #
# ZERO COLLATERAL -- the honest cases the ladder was carrying must keep working.
# --------------------------------------------------------------------------- #


class _ConvBN(nn.Module):
    """BatchNorm running stats: the ladder's most important honest client."""

    def __init__(self) -> None:
        """Build a conv/bn stack."""

        super().__init__()
        torch.manual_seed(4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        return self.bn(self.conv(x))


class _NestedRegistered(nn.Module):
    """Two blocks of registered buffers at differing module depths."""

    def __init__(self) -> None:
        """Build nested registered buffers only."""

        super().__init__()
        torch.manual_seed(5)
        self.bn = nn.BatchNorm2d(4)
        self.register_buffer("scale", torch.ones(1, 4, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume a nested BN buffer set and a top-level buffer."""

        return self.bn(x) * self.scale


@pytest.mark.smoke
@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_registered_buffers_resolve_to_their_own_addresses(train: bool) -> None:
    """Every registered buffer must still resolve to its OWN dotted address.

    The ladder's honest job. ``num_batches_tracked`` and the BatchNorm
    running-stat pairing are resolved ahead of the generic rung and must be
    untouched by making the recorded address authoritative.
    """

    model = _NestedRegistered()
    model.train(train)
    trace = _capture(model, torch.randn(2, 4, 8, 8))
    addresses = {address for _label, address in _buffer_addresses(trace)}
    assert "bn.running_mean" in addresses
    assert "bn.running_var" in addresses
    assert "scale" in addresses


@pytest.mark.smoke
@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_convbn_still_replays_verified(tmp_path: Path, train: bool) -> None:
    """ConvBN must stay VERIFIED in both modes -- no new ceiling, no refusal."""

    model = _ConvBN()
    model.train(train)
    x = torch.randn(2, 4, 8, 8)
    result = _replay_or_refuse(model, x, tmp_path / f"convbn_{train}.tlspec")
    assert result is not None, "ConvBN must not start refusing at save"
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
