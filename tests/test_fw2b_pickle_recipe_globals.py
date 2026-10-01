"""Plain pickles of a Trace must not embed builtin facet recipe function identities.

FW2-B item 12-A. ``Trace.facet_registry_snapshot`` is session-time state
(``FieldPolicy.DROP``) whose ``_RegisteredRecipe.func`` entries pickle the
builtin recipe FUNCTIONS by import identity. Carrying it through
``Trace.__getstate__`` coupled every earlier plain pickle to the current
private recipe names: renaming ``gqa_sdpa_attention`` -> ``gqa_attention``
(2b552c886) made ``tests/fixtures/legacy_trace_85c8c60f.pkl`` unloadable.
``__getstate__`` now serializes the snapshot as ``None`` (the loaded-artifact
form) and facet reads fall back to the live registry, exactly as they do for
every tlspec-loaded Trace.
"""

from __future__ import annotations

import copy
import pickle
import pickletools

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic.facets import FacetRegistrySnapshot

_RECIPE_PREFIX = "torchlens.semantic.recipes."


def _global_modules(payload: bytes) -> set[str]:
    """Return every module named by a GLOBAL / STACK_GLOBAL opcode in ``payload``."""

    modules: set[str] = set()
    recent: list[str] = []
    for opcode, arg, _pos in pickletools.genops(payload):
        if opcode.name == "GLOBAL":
            modules.add(str(arg).split(" ", 1)[0])
        elif opcode.name == "STACK_GLOBAL" and len(recent) >= 2:
            modules.add(recent[-2])
        if isinstance(arg, str):
            recent.append(arg)
            del recent[:-2]
    return modules


def _tiny_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    return tl.trace(model, torch.randn(2, 3))


@pytest.mark.smoke
def test_plain_pickle_carries_no_recipe_function_globals() -> None:
    """The live snapshot is dropped at the pickle boundary; no recipe module is named."""

    trace = _tiny_trace()
    # The pin is meaningful only if the live trace really carries the snapshot.
    assert isinstance(trace.facet_registry_snapshot, FacetRegistrySnapshot)

    payload = pickle.dumps(trace, protocol=pickle.HIGHEST_PROTOCOL)
    recipe_modules = sorted(m for m in _global_modules(payload) if m.startswith(_RECIPE_PREFIX))
    assert recipe_modules == [], recipe_modules

    loaded = pickle.loads(payload)  # noqa: S301 - bytes produced in-process just above
    assert loaded.facet_registry_snapshot is None
    # Facet reads still serve after the round-trip (live-registry fallback).
    for label in trace.layer_labels:
        assert set(loaded[label].facets) == set(trace[label].facets)


@pytest.mark.smoke
def test_deepcopy_follows_the_same_pickle_boundary() -> None:
    """``copy.deepcopy`` routes through the pickle semantics and drops the snapshot too."""

    trace = _tiny_trace()
    cloned = copy.deepcopy(trace)
    assert cloned.facet_registry_snapshot is None
    assert len(cloned) == len(trace)
    assert set(cloned[trace.layer_labels[-1]].facets) == set(trace[trace.layer_labels[-1]].facets)
