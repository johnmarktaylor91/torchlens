"""Meta-test for the split-rebind facade trap: patch the HUB, never the child.

b9 R43-1 (fable+opus+sol converged): ``_split_rebind.rebind_function`` gives
every split-child function the HUB module's globals, and the hub then pushes
its whole namespace back into the children. Two consequences every test and
fault-injection harness must know:

* ``monkeypatch.setattr(child_module, helper, ...)`` SILENTLY NO-OPS for any
  behavior reached through a hub-rebound function -- the rebound bodies
  resolve names in the hub's globals, never the child's ``__dict__``.
* The one effective patch point is the HUB module.

This file pins that contract both structurally (every rebound callable on
the hub resolves through the hub's globals) and empirically (a child patch
is proven ineffective while the identical hub patch takes effect). If the
r3 split ever grows real seams -- children owning their own namespaces --
these tests go red and the patch-the-parent rule gets retired WITH the
mechanism, instead of silently inverting.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.validation._invariants_equivalence as _equivalence_child
import torchlens.validation.invariants as _invariants_hub

#: (hub module, child modules) families produced by the r3 god-file split.
_SPLIT_FAMILIES: tuple[tuple[types.ModuleType, tuple[str, ...]], ...] = (
    (
        _invariants_hub,
        (
            "_invariants_backward_domain",
            "_invariants_backward_flow",
            "_invariants_backward_graph",
            "_invariants_buffers",
            "_invariants_conditional_base",
            "_invariants_conditional_modules",
            "_invariants_conditionals",
            "_invariants_connectivity",
            "_invariants_entry",
            "_invariants_equivalence",
            "_invariants_modules_params",
            "_invariants_payloads",
            "_invariants_topology",
        ),
    ),
)


def test_every_hub_function_resolves_through_hub_globals() -> None:
    """Every plain function reachable on the hub reads the HUB's namespace.

    This is the structural fact behind the patch-the-parent rule: if any
    rebound function kept child globals (or a child kept an unrebound
    original that the hub also re-exports), the effective patch point would
    silently depend on the call path.
    """

    for hub, child_names in _SPLIT_FAMILIES:
        hub_vars = vars(hub)
        family_files = {f"{name}.py" for name in child_names}
        family_files.add(hub.__file__.rsplit("/", 1)[-1])

        def _is_family_function(
            value: object, bound_family_files: frozenset[str] | set[str] = frozenset(family_files)
        ) -> bool:
            """Return whether a value is a plain function DEFINED in this family.

            Imported helpers (``rebind_function``, ``status`` predicates, ...)
            legitimately keep their own modules' globals and are out of scope.

            Parameters
            ----------
            value:
                Candidate namespace member.
            bound_family_files:
                Family filenames bound at definition time (B023 idiom).

            Returns
            -------
            bool
                True for functions whose code lives in a family file.
            """

            return (
                isinstance(value, types.FunctionType)
                and value.__code__.co_filename.rsplit("/", 1)[-1] in bound_family_files
            )

        for name, value in sorted(hub_vars.items()):
            if _is_family_function(value) and not name.startswith("__"):
                assert value.__globals__ is hub_vars, (
                    f"{hub.__name__}.{name} resolves through "
                    f"{value.__globals__.get('__name__')!r}; the split facade "
                    "contract expects hub globals"
                )
        for child_name in child_names:
            child = getattr(
                __import__(f"torchlens.validation.{child_name}", fromlist=[child_name]),
                "__dict__",
            )
            for name, value in child.items():
                if _is_family_function(value) and not name.startswith("__"):
                    assert value.__globals__ is hub_vars, (
                        f"{child_name}.{name} escaped the namespace push -- a "
                        "second patchable copy now exists and child patches "
                        "would take effect for SOME call paths only"
                    )


class _TwiceLinear(nn.Module):
    """Apply one linear layer twice to mint an equivalence group."""

    def __init__(self) -> None:
        """Build the shared linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer twice.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Twice-transformed batch.
        """

        return self.fc(self.fc(x))


class _Boom(RuntimeError):
    """Sentinel raised by the planted helper replacement."""


def test_child_patch_is_inert_and_hub_patch_takes_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demonstrate the trap live: child setattr no-ops, hub setattr works.

    A fault-injection harness that patches the child gets a silent green --
    the exact failure mode that would let a disarmed tripwire look armed
    (both b9 mutation batteries had to mutate SOURCE to dodge this).
    """

    def _boom(owner: object) -> object:
        """Raise the sentinel to prove the patch point is live.

        Parameters
        ----------
        owner:
            Op or Layer handed to the real helper.

        Returns
        -------
        object
            Never returns.
        """

        raise _Boom("patched helper reached")

    log = tl.trace(_TwiceLinear(), torch.randn(2, 4))
    try:
        # Patch the CHILD: the hub-rebound check must NOT see it.
        monkeypatch.setattr(_equivalence_child, "_canonical_equivalent_ops", _boom)
        _invariants_hub._check_equivalence_symmetry(log)  # no _Boom: patch inert

        # Patch the HUB: the same check must hit the sentinel immediately.
        monkeypatch.setattr(_invariants_hub, "_canonical_equivalent_ops", _boom)
        with pytest.raises(_Boom):
            _invariants_hub._check_equivalence_symmetry(log)
    finally:
        log.cleanup()
