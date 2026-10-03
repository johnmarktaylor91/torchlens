"""Pickle / ``tl.save`` parity for intervention-spec-carrying traces.

Pre-fix, plain ``pickle`` of a Trace captured with ``intervene=`` failed
(``zero_ablate.<locals>.factory`` and ``when.<locals>._predicate`` are local
closures) while ``tl.save`` of the same trace succeeded -- a serialization
split on one object. Two aligned fixes:

* ``HelperSpec`` pickle hooks drop the builtin factory closure and rebuild it
  at restore through the SAME builtin registry ``tl.load`` uses
  (``rebuild_builtin_helper``); ``opaque_audit`` factories drop to the
  canonical factory-less audit-only form.
* ``Trace.__getstate__`` serializes the capture-session predicate carriers
  (``_stop_directive``, ``_capture_config``, ``_predicate_save_options`` --
  all ``FieldPolicy.DROP``) to their loaded-artifact form (absent / None),
  which every post-capture consumer already tolerates.
"""

import copy
import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.types import HelperSpec


class _Net(nn.Module):
    """Linear + relu net."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer and relu."""

        return torch.relu(self.fc(x))


@pytest.fixture(scope="module")
def intervened_trace() -> Iterator[tl.Trace]:
    """Return a trace captured with a builtin-helper intervention."""

    torch.manual_seed(0)
    trace = tl.trace(
        _Net(),
        torch.randn(4, 8),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


def test_builtin_helper_spec_pickle_rebuilds_factory() -> None:
    """A builtin ``HelperSpec`` pickles; the restored factory works."""

    spec = tl.zero_ablate()
    restored = pickle.loads(pickle.dumps(spec))
    assert isinstance(restored, HelperSpec)
    assert restored.helper_name == spec.helper_name
    assert restored.factory is not None
    hook = restored()
    out = hook(torch.ones(3), hook=None)
    assert torch.equal(out, torch.zeros(3))


def test_builtin_helper_spec_pickle_preserves_args() -> None:
    """Constructor arguments survive the rebuild (identity source)."""

    spec = tl.scale(0.5)
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.args == spec.args
    hook = restored()
    out = hook(torch.ones(3), hook=None)
    assert torch.allclose(out, torch.full((3,), 0.5))


def test_helper_spec_deepcopy_keeps_factory_identity() -> None:
    """``copy.deepcopy`` keeps its pre-pickle-hook factory-sharing semantics."""

    spec = tl.zero_ablate()
    duplicate = copy.deepcopy(spec)
    assert duplicate.factory is spec.factory


def test_intervened_trace_pickles_like_it_saves(intervened_trace: tl.Trace, tmp_path) -> None:
    """The RED-capable parity case: pickle succeeds where tl.save succeeds."""

    blob = pickle.dumps(intervened_trace)
    restored = pickle.loads(blob)
    assert len(restored.layer_list) == len(intervened_trace.layer_list)
    # The saved (intervened) activation payload survives the round-trip.
    assert torch.equal(restored["relu_1_2"].ops[0].out, intervened_trace["relu_1_2"].ops[0].out)
    # tl.save of the very same object keeps working (the parity claim).
    path = str(tmp_path / "iv.tlspec")
    tl.save(intervened_trace, path)
    assert tl.load(path)["relu_1_2"] is not None


@pytest.mark.smoke
def test_intervened_fork_pickles(intervened_trace: tl.Trace) -> None:
    """A fork carrying the deep-copied intervention spec also pickles."""

    fork = intervened_trace.fork()
    restored = pickle.loads(pickle.dumps(fork))
    assert len(restored.layer_list) == len(fork.layer_list)


def test_pickled_trace_matches_loaded_predicate_carrier_form(
    intervened_trace: tl.Trace,
) -> None:
    """Pickle serializes the predicate carriers to the loaded-artifact form."""

    restored = pickle.loads(pickle.dumps(intervened_trace))
    assert "_stop_directive" not in restored.__dict__
    assert "_capture_config" not in restored.__dict__
    assert restored.__dict__.get("_predicate_save_options") is None


@pytest.mark.smoke
def test_add_and_replace_with_pickle_round_trip() -> None:
    """R10-1: ``add``/``replace_with`` mint portability="builtin" specs but
    were missing from the rebuild registry -- pickling (and therefore
    tl.save/tl.load of a trace intervened with them) minted a dead artifact
    that raised ``intervention_helper_unknown`` at restore."""

    added = pickle.loads(pickle.dumps(tl.add(1.5)))
    assert added.helper_name == "add"
    assert added.args == (1.5,)
    assert callable(added.factory)

    replacement = torch.ones(2)
    replaced = pickle.loads(pickle.dumps(tl.replace_with(replacement)))
    assert replaced.helper_name == "replace_with"
    assert torch.equal(replaced.args[0], replacement)
    assert callable(replaced.factory)


def test_every_minted_builtin_helper_name_is_rebuildable() -> None:
    """Registry-completeness gate: every helper name the public constructors
    mint with builtin portability must resolve through the ONE rebuild
    registry, so a future helper cannot re-open the dead-artifact class."""

    import ast
    import pathlib

    from torchlens._errors import InvalidArgumentError
    from torchlens.intervention import helpers as helpers_module, predicates as predicates_module
    from torchlens.intervention.helpers import rebuild_builtin_helper

    minted: set[str] = set()
    for module in (helpers_module, predicates_module):
        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            portability = next((kw.value for kw in node.keywords if kw.arg == "portability"), None)
            if portability is not None and not (
                isinstance(portability, ast.Constant) and portability.value == "builtin"
            ):
                continue
            func_name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if func_name == "_helper_spec" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    minted.add(first.value)
            if func_name == "HelperSpec":
                name_kw = next((kw.value for kw in node.keywords if kw.arg == "helper_name"), None)
                if isinstance(name_kw, ast.Constant) and isinstance(name_kw.value, str):
                    minted.add(name_kw.value)

    assert {"zero_ablate", "add", "replace_with", "grad_scale"} <= minted, (
        f"AST sweep lost its anchors -- helper minting moved: {sorted(minted)}"
    )
    unknown: list[str] = []
    for name in sorted(minted):
        try:
            rebuild_builtin_helper(name, (), {})
        except InvalidArgumentError as exc:
            if exc.fields.get("code") == "intervention_helper_unknown":
                unknown.append(name)
        except Exception:
            # Known constructor rejecting the canned empty args is fine; the
            # gate only proves the NAME resolves.
            pass
    assert not unknown, (
        f"builtin helper constructors mint names the rebuild registry cannot "
        f"restore (dead saved artifacts): {unknown}"
    )
