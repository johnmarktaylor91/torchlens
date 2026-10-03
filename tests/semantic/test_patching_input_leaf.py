"""Input-facet patcher fixes (grind-p3-fixplan T3, fix-patching lane).

Four coupled defects in the input-facet patching path of
``torchlens/semantic/patching.py``:

* [HIGH] the patcher indexed BFS-ordered ``input_ops`` ordinals into a DFS
  tree walk, silently patching the WRONG leaf on mixed-nesting inputs. The
  leaf is now bound by the op's recorded ``io_role`` ADDRESS.
* [MED] the ``seen``/``id()`` dedupe returned the UNPATCHED original tensor at
  repeated sites, so aliased input trees reran HALF-PATCHED. Every site of an
  aliased leaf is now replaced.
* [MED] the Mapping branch rebuilt inputs as a bare ``dict``, dropping
  subclass container types the model's forward depends on. Container types
  are now preserved.
* [MED] every activation-patching entry point tore down as unguarded
  sequential statements, so a raising ``Trace.cleanup()`` skipped
  ``guard.close()`` and stranded the caller's model state and global RNG.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import FacetSpec


class MixedLeafReader(nn.Module):
    """Combine three input leaves with distinguishable weights.

    Tracks a BatchNorm-style running counter buffer so tests can observe
    whether the state guard restored the model after a run.
    """

    def __init__(self) -> None:
        """Register the forward-call counter buffer."""

        super().__init__()
        self.register_buffer("calls", torch.zeros(1))

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Weight each leaf so patched-leaf identity is visible in the output."""

        self.calls.add_(1.0)
        return a + 2.0 * b + 4.0 * c


class MixedNestingModel(nn.Module):
    """Model whose single input arg mixes nesting depths (BFS != DFS order)."""

    def __init__(self) -> None:
        """Initialize the leaf reader."""

        super().__init__()
        self.reader = MixedLeafReader()

    def forward(self, x: Any) -> torch.Tensor:
        """Unpack ((a, [b]), c) and read the three leaves."""

        (a, b_list), c = x
        return self.reader(a, b_list[0], c)


class AliasLeafReader(nn.Module):
    """Read the same aliased input tensor from two sites."""

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Weight the two sites so a half-patched rerun is visible."""

        return first + 3.0 * second


class AliasedInputModel(nn.Module):
    """Model receiving one tensor object at two input sites."""

    def __init__(self) -> None:
        """Initialize the alias reader."""

        super().__init__()
        self.reader = AliasLeafReader()

    def forward(self, x: Any) -> torch.Tensor:
        """Feed both sites of the aliased pair to the reader."""

        return self.reader(x[0], x[1])


class BatchDict(dict):
    """dict subclass whose accessor the model's forward depends on."""

    def query(self) -> torch.Tensor:
        """Return the query leaf through subclass-only behavior."""

        return self["q"]


class MappingLeafReader(nn.Module):
    """Combine the mapping leaves with distinguishable weights."""

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Weight the mapping leaves."""

        return q + 2.0 * k


class MappingInputModel(nn.Module):
    """Model whose forward requires the Mapping subclass container type."""

    def __init__(self) -> None:
        """Initialize the mapping reader."""

        super().__init__()
        self.reader = MappingLeafReader()

    def forward(self, x: BatchDict) -> torch.Tensor:
        """Read one leaf through the subclass accessor and one by key."""

        return self.reader(x.query(), x["k"])


def _input_facet(trace: Any, io_role: str) -> FacetSpec:
    """Return a facet spec homed on the input op recorded at ``io_role``."""

    home = next(op for op in trace.input_ops if getattr(op, "io_role", None) == io_role)
    return FacetSpec.from_home(home, recipe_id="test_input_leaf_patching")


def mixed_leaf_reader(module: Any) -> dict[str, Any]:
    """Expose one facet per mixed-nesting input leaf."""

    trace = module.trace
    return {
        "leaf_a": _input_facet(trace, "input.x.0.0"),
        "leaf_b": _input_facet(trace, "input.x.0.1.0"),
        "leaf_c": _input_facet(trace, "input.x.1"),
    }


def alias_leaf_reader(module: Any) -> dict[str, Any]:
    """Expose the single deduped aliased input leaf as a facet."""

    return {"shared_leaf": _input_facet(module.trace, "input.x.0")}


def mapping_leaf_reader(module: Any) -> dict[str, Any]:
    """Expose the mapping's key leaf as a facet."""

    return {"key_leaf": _input_facet(module.trace, "input.x.k")}


@pytest.fixture(autouse=True, scope="module")
def _register_module_facet_recipes() -> Iterator[None]:
    """Register this module's recipes at RUN time, restoring the registry after.

    A module-level ``@tl.facets.register`` fires at pytest COLLECTION and
    polluted the process-global registry outside any fixture's reach
    (hunt-b2-sol R76/R77).
    """

    from torchlens.semantic import facets as _facets

    saved = list(_facets._REGISTRY)
    tl.facets.register(
        class_name="MixedLeafReader",
        target_scope="module",
        facets=("leaf_a", "leaf_b", "leaf_c"),
    )(mixed_leaf_reader)
    tl.facets.register(
        class_name="AliasLeafReader",
        target_scope="module",
        facets=("shared_leaf",),
    )(alias_leaf_reader)
    tl.facets.register(
        class_name="MappingLeafReader",
        target_scope="module",
        facets=("key_leaf",),
    )(mapping_leaf_reader)
    try:
        yield
    finally:
        _facets._REGISTRY[:] = saved
        _facets._REGISTRY_VERSION += 1


def _metric(log: Any) -> torch.Tensor:
    """Return the summed model output as the patching metric."""

    return log[log.output_layers[0]].out.sum()


def _mixed_tree(value: float) -> tuple[tuple[torch.Tensor, list[torch.Tensor]], torch.Tensor]:
    """Return a ((a, [b]), c) input tree filled with ``value``."""

    def leaf() -> torch.Tensor:
        return torch.full((1, 2), value)

    return ((leaf(), [leaf()]), leaf())


def test_input_facet_patch_binds_recorded_address_not_ordinal() -> None:
    """Patching the facet homed on leaf ``c`` must change ``c``, not leaf ``a``.

    Capture flattens ``((a, [b]), c)`` in BFS order (c, a, b) while the old
    patcher counted DFS ordinals (a, b, c), so the facet homed on ``c``
    (input-op ordinal 0) silently patched ``a``.
    """

    model = MixedNestingModel()
    clean, corrupted = _mixed_tree(10.0), _mixed_tree(1.0)

    scores = tl.facets.patching.activation_patch_residual_stream(
        model, clean, corrupted, _metric, facet_name="leaf_c", patch_positions=False
    )

    assert scores.shape == (1,)
    # corrupted a=1, b=1 with clean c=10: 2 * (1 + 2*1 + 4*10) = 86.
    assert torch.isclose(scores[0], torch.tensor(86.0)), scores


@pytest.mark.parametrize(
    ("facet_name", "expected"),
    [
        ("leaf_a", 2.0 * (10.0 + 2.0 + 4.0)),
        ("leaf_b", 2.0 * (1.0 + 20.0 + 4.0)),
        ("leaf_c", 2.0 * (1.0 + 2.0 + 40.0)),
    ],
)
def test_every_input_leaf_patches_its_own_recorded_address(
    facet_name: str, expected: float
) -> None:
    """Each input op's facet patch lands on exactly the leaf its io_role names."""

    model = MixedNestingModel()
    clean, corrupted = _mixed_tree(10.0), _mixed_tree(1.0)

    scores = tl.facets.patching.activation_patch_residual_stream(
        model, clean, corrupted, _metric, facet_name=facet_name, patch_positions=False
    )

    assert torch.isclose(scores[0], torch.tensor(expected)), (facet_name, scores)


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore::torchlens._errors.TorchLensCaptureGapWarning")
def test_aliased_input_leaf_patches_every_site() -> None:
    """One tensor object at two input sites is replaced at BOTH sites.

    Capture dedupes the repeated object into ONE input op, so patching that op
    must patch every site; the old walk returned the unpatched original at the
    second site and reran half-patched. The intervention-ready rerun discloses
    the aliased sites as a capture gap (it cannot encode shared identity in a
    sparse descriptor) -- that honest ceiling is expected here, the values are
    what the assertion pins.
    """

    model = AliasedInputModel()
    t_clean = torch.full((1, 2), 5.0)
    t_corrupted = torch.full((1, 2), 1.0)

    scores = tl.facets.patching.activation_patch_residual_stream(
        model,
        (t_clean, t_clean),
        (t_corrupted, t_corrupted),
        _metric,
        facet_name="shared_leaf",
        patch_positions=False,
    )

    # Both sites patched to clean 5.0: 2 * (5 + 3*5) = 40 (half-patched: 16).
    assert torch.isclose(scores[0], torch.tensor(40.0)), scores


@pytest.mark.smoke
def test_mapping_input_container_type_is_preserved() -> None:
    """A Mapping-subclass input keeps its type through the patched rerun.

    The old Mapping branch rebuilt the tree as a bare ``dict``, so a forward
    depending on subclass behavior (``x.query()``) crashed in the rerun.
    """

    model = MappingInputModel()
    clean = BatchDict(q=torch.full((1, 2), 7.0), k=torch.full((1, 2), 7.0))
    corrupted = BatchDict(q=torch.full((1, 2), 1.0), k=torch.full((1, 2), 1.0))

    scores = tl.facets.patching.activation_patch_residual_stream(
        model, clean, corrupted, _metric, facet_name="key_leaf", patch_positions=False
    )

    # corrupted q=1 with clean k=7: 2 * (1 + 2*7) = 30.
    assert torch.isclose(scores[0], torch.tensor(30.0)), scores


def test_teardown_restores_state_when_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising ``Trace.cleanup()`` must not strand the caller's model state.

    The forward mutates a running-counter buffer (the BatchNorm drift class);
    if a cleanup failure skips ``guard.close()``, the last run's drift is
    never restored and the USER's model is returned mutated. The global-RNG
    fork must be released too.
    """

    from torchlens.data_classes.trace import Trace

    model = MixedNestingModel()
    clean, corrupted = _mixed_tree(10.0), _mixed_tree(1.0)

    def _boom(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("cleanup boom")

    rng_before = torch.get_rng_state()
    monkeypatch.setattr(Trace, "cleanup", _boom)
    with pytest.raises(RuntimeError, match="cleanup boom"):
        tl.facets.patching.activation_patch_residual_stream(
            model, clean, corrupted, _metric, facet_name="leaf_c", patch_positions=False
        )

    assert torch.equal(model.reader.calls, torch.zeros(1))
    assert torch.equal(torch.get_rng_state(), rng_before)
