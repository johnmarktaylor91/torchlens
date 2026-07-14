"""Round-12 security regression: ungated intervention ``import_ref`` resolver RCE.

r3/r5/r6/r7/r9/r10/r11 hardened the container and function-key resolvers, but a THIRD
ungated sink survived in ``torchlens/intervention/save.py``: ``_resolve_import_ref`` did a
bare ``importlib.import_module(module_name)`` + attribute walk with NO trust gate, NO
denylist, and NO purity check. It was reached from ``LazyImportRef.__call__`` and the
import-ref helper factory, both materialized UNGATED from the bundle JSON during
``load_intervention_spec`` / public ``tl.load``. Applying a loaded intervention spec whose
hook was serialized as an ``import_ref`` therefore imported the attacker-named module,
executing its top-level code = arbitrary code execution under the DEFAULT
``trust_custom_callables=False`` load.

The fix routes every bundle-reachable import ref through the SAME trust gate as the
function-key resolver (``torchlens.intervention.resolver.resolve_import_ref`` ->
``resolve_function_registry_key``): a genuinely foreign module default-denies with a typed
``UntrustedCallableError`` and is NEVER imported unless the caller opts into
``trust_custom_callables=True`` or a matching ``allowed_custom_callable_modules`` entry, the
fixed torch/operator namespaces stay purity-gated, and TorchLens-owned / torch refs always
resolve. The trust context is threaded from ``load_intervention_spec`` down into every
materialized ``LazyImportRef`` so the deferred resolution enforces the gate.
"""

from __future__ import annotations

from collections.abc import Iterator
import importlib
from pathlib import Path
import sys
import textwrap

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import UntrustedCallableError
from torchlens.intervention.save import (
    LazyImportRef,
    _resolve_import_ref,
    load_intervention_spec,
)


class _ReluModel(nn.Module):
    """Tiny model with a single ``relu`` op to attach a hook to."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(x) + 1``."""

        return torch.relu(x) + 1


def _write_evil_module(tmp_path: Path, sentinel: Path) -> str:
    """Write a module whose IMPORT writes ``sentinel`` (the RCE payload)."""

    mod_name = "r12_evil_mod_test"
    src = textwrap.dedent(
        f"""
        from pathlib import Path
        Path({str(sentinel)!r}).write_text("pwned")

        def hook(out, *, hook):
            return out * 0
        """
    )
    (tmp_path / f"{mod_name}.py").write_text(src)
    return mod_name


def _save_import_ref_spec(tmp_path: Path, mod_name: str) -> Path:
    """Build + save an intervention spec whose hook is an ``import_ref`` to ``mod_name``."""

    mod = importlib.import_module(mod_name)
    log = tl.trace(_ReluModel(), torch.ones(1, 3))
    log.attach_hooks(tl.func("relu"), mod.hook, confirm_mutation=True)
    spec_path = tmp_path / "evil.tlspec"
    log.save_intervention(str(spec_path), level="executable_with_callables")
    return spec_path


def _forget_module_and_sentinel(mod_name: str, sentinel: Path) -> None:
    """Simulate the victim: drop the evil module and clear the sentinel."""

    sys.modules.pop(mod_name, None)
    if sentinel.exists():
        sentinel.unlink()


@pytest.fixture
def _evil_spec(tmp_path: Path) -> Iterator[tuple[Path, str, Path]]:
    """Yield a saved malicious import-ref spec plus its module name and sentinel path."""

    sentinel = tmp_path / "r12_pwned"
    mod_name = _write_evil_module(tmp_path, sentinel)
    sys.path.insert(0, str(tmp_path))
    spec_path = _save_import_ref_spec(tmp_path, mod_name)
    _forget_module_and_sentinel(mod_name, sentinel)
    yield spec_path, mod_name, sentinel
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    sys.modules.pop(mod_name, None)


def test_default_trust_tl_load_denies_import_ref_rce(_evil_spec: tuple[Path, str, Path]) -> None:
    """``tl.load`` (default trust) then applying the spec imports NOTHING + executes NO code."""

    spec_path, mod_name, sentinel = _evil_spec

    spec = tl.load(str(spec_path))
    # Load is lazy: the foreign module is not imported and the payload never runs.
    assert not sentinel.exists()
    assert mod_name not in sys.modules

    fresh = tl.trace(_ReluModel(), torch.ones(1, 3))
    fresh._intervention_spec = spec
    with pytest.raises(UntrustedCallableError):
        fresh.run(_ReluModel(), torch.ones(1, 3))

    # The attacker code never ran and the module was never imported.
    assert not sentinel.exists()
    assert mod_name not in sys.modules


def test_load_intervention_spec_default_trust_denies(_evil_spec: tuple[Path, str, Path]) -> None:
    """The lower-level loader defaults to deny-by-default just like ``tl.load``."""

    spec_path, mod_name, sentinel = _evil_spec

    spec = load_intervention_spec(str(spec_path))
    assert not sentinel.exists()
    assert mod_name not in sys.modules

    fresh = tl.trace(_ReluModel(), torch.ones(1, 3))
    fresh._intervention_spec = spec
    with pytest.raises(UntrustedCallableError):
        fresh.run(_ReluModel(), torch.ones(1, 3))
    assert not sentinel.exists()
    assert mod_name not in sys.modules


@pytest.mark.parametrize("import_path", ["os:system", "subprocess:Popen", "builtins:eval"])
def test_direct_resolver_denies_foreign_without_import(import_path: str) -> None:
    """The resolver denies dangerous foreign callables by default WITHOUT importing."""

    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref(import_path)


def test_torch_ref_always_resolves_without_trust() -> None:
    """A fixed-namespace torch op resolves without any trust opt-in."""

    assert _resolve_import_ref("torch:relu") is torch.relu


def test_torch_save_denied_by_purity_even_under_trust() -> None:
    """A fixed-namespace side-effecting op stays denied by the purity gate under trust."""

    with pytest.raises(UntrustedCallableError):
        _resolve_import_ref("torch:save", trust_custom_callables=True)


def test_torchlens_owned_ref_always_resolves() -> None:
    """A TorchLens-owned custom helper resolves without trust (our own code)."""

    from torchlens.intervention.helpers import zero_ablate

    assert _resolve_import_ref("torchlens.intervention.helpers:zero_ablate") is zero_ablate


def test_trusted_and_allowlisted_import_ref_resolves(tmp_path: Path) -> None:
    """A legit foreign import ref resolves under explicit trust / allowlist opt-in."""

    mod_name = "r12_trusted_mod_test"
    (tmp_path / f"{mod_name}.py").write_text(
        textwrap.dedent(
            """
            def hook(out, *, hook):
                return out * 2
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(mod_name)
        import_path = f"{mod_name}:hook"

        # Default: denied.
        with pytest.raises(UntrustedCallableError):
            _resolve_import_ref(import_path)

        # Broad trust: resolves to the real callable.
        assert _resolve_import_ref(import_path, trust_custom_callables=True) is mod.hook

        # Narrow allowlist: resolves when listed.
        assert (
            _resolve_import_ref(import_path, allowed_custom_callable_modules={mod_name}) is mod.hook
        )

        # Narrow allowlist stays enforced even under broad trust: mismatch denies.
        with pytest.raises(UntrustedCallableError):
            _resolve_import_ref(
                import_path,
                trust_custom_callables=True,
                allowed_custom_callable_modules={"some_other_module"},
            )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod_name, None)


def test_lazy_import_ref_carries_trust_context() -> None:
    """``LazyImportRef`` stores the trust context so deferred resolution stays gated."""

    ref = LazyImportRef("os:system")
    # Frozen defaults are fail-closed.
    assert ref.trust_custom_callables is False
    assert ref.allowed_custom_callable_modules is None
    with pytest.raises(UntrustedCallableError):
        ref("echo hi")


def test_trusted_import_ref_round_trips_through_tl_load(tmp_path: Path) -> None:
    """A trusted spec load resolves the import ref and applies the real hook."""

    mod_name = "r12_roundtrip_mod_test"
    (tmp_path / f"{mod_name}.py").write_text(
        textwrap.dedent(
            """
            def hook(out, *, hook):
                return out * 0
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module(mod_name)
        log = tl.trace(_ReluModel(), torch.ones(1, 3))
        log.attach_hooks(tl.func("relu"), mod.hook, confirm_mutation=True)
        spec_path = tmp_path / "trusted.tlspec"
        log.save_intervention(str(spec_path), level="executable_with_callables")

        spec = load_intervention_spec(str(spec_path), allowed_custom_callable_modules={mod_name})
        fresh = tl.trace(_ReluModel(), torch.ones(1, 3))
        fresh._intervention_spec = spec
        result = fresh.run(_ReluModel(), torch.ones(1, 3))
        assert result is not None
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(mod_name, None)


def test_no_ungated_import_module_remains_in_bundle_resolvers() -> None:
    """Grep-proof: no bundle-reachable ``importlib.import_module`` outside the gated resolver.

    Every remaining ``importlib.import_module`` in the intervention resolver surface must live
    inside ``resolver.resolve_function_registry_key`` (already trust-gated). ``save.py``'s
    ``_resolve_import_ref`` must no longer import directly -- it delegates to the shared gate.
    """

    import torchlens.intervention.save as save_mod

    save_src = Path(save_mod.__file__).read_text()
    assert "importlib.import_module" not in save_src
