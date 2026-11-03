"""Pickle / ``Trace.save`` parity for intervention-spec-carrying traces.

Pre-fix, plain ``pickle`` of a Trace captured with ``intervene=`` failed
(``zero_ablate.<locals>.factory`` and ``when.<locals>._predicate`` are local
closures) while ``Trace.save`` of the same trace succeeded -- a serialization
split on one object. Two aligned fixes:

* ``Edit`` pickle hooks drop the builtin factory closure and rebuild it
  at restore through the SAME builtin registry ``tl.load`` uses
  (``rebuild_builtin_helper``); ``opaque_audit`` factories drop to the
  canonical factory-less audit-only form.
* ``Trace.__getstate__`` serializes the capture-session predicate carriers
  (``_stop_directive``, ``_capture_config``, ``_predicate_save_options`` --
  all ``FieldPolicy.DROP``) to their loaded-artifact form (absent / None),
  which every post-capture consumer already tolerates.
"""

import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl


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


def test_intervened_fork_pickles(intervened_trace: tl.Trace) -> None:
    """A fork carrying the deep-copied intervention spec also pickles."""

    fork = intervened_trace.fork()
    restored = pickle.loads(pickle.dumps(fork))
    assert len(restored._layer_list) == len(fork._layer_list)


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
            if func_name == "Edit":
                name_kw = next((kw.value for kw in node.keywords if kw.arg == "name"), None)
                if isinstance(name_kw, ast.Constant) and isinstance(name_kw.value, str):
                    minted.add(name_kw.value)

    assert {"zero_ablate", "add", "replace_with", "clip_norm"} <= minted, (
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
