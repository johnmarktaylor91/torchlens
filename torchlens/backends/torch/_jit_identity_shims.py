"""TorchScript compilation-entry shims that hand jit a wrapper's original function.

TorchScript compiles a called Python function from its source and resolves that
source's free names through a resolution callback built from the function
object's globals and closure. A torchlens wrapper's ``__prepare_scriptable__``
returns the original torch function, so entries that honor the hook compile a
self-consistent (source, globals) pair. Two entries the C++ sugared-value layer
calls do not honor it on every supported torch:

* ``torch.jit._script._get_overloads`` never calls the hook, so an overloaded
  pure-Python functional (``F.interpolate``, ``F.adaptive_avg_pool2d/3d``)
  compiled its original source against the wrapper's globals
  (``undefined value math``).
* ``torch.jit._recursive.try_compile_fn`` on torch 2.1 builds the resolution
  callback from the wrapper BEFORE ``torch.jit.script`` applies the hook, so a
  functional reached from another scripted functional (``F.interpolate``
  calling ``adaptive_avg_pool2d``) failed with ``undefined value
  _list_with_default``. Newer torch applies the hook first.

Each shim resolves a torchlens wrapper to its ledger original
(``_state._decorated_to_orig``, keyed on identity, never on copied ``__tl_*``
attributes) and delegates to the upstream entry, so torch's own logic decides
everything else. Installed and removed with the identity-shim family by
:mod:`torchlens.backends.torch.identity_shims`.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import Any

from ... import _state
from ..._utils import _torch_compat


def _jit_entries() -> tuple[tuple[Any, str], ...]:
    """Return the (module, attribute) pairs of the wrapper-blind jit entries present.

    Returns
    -------
    tuple[tuple[Any, str], ...]
        ``(torch.jit._script, "_get_overloads")`` when the overload resolver is
        available, and ``(torch.jit._recursive, "try_compile_fn")`` when that
        module is loaded (``torch.jit`` imports it).
    """

    entries: list[tuple[Any, str]] = []
    script_module = _torch_compat.get_jit_overload_resolver_module()
    if script_module is not None:
        entries.append((script_module, "_get_overloads"))
    recursive_module = sys.modules.get("torch.jit._recursive")
    if recursive_module is not None:
        entries.append((recursive_module, "try_compile_fn"))
    return tuple(entries)


def _make_jit_entry_shim(original_entry: Callable[..., Any]) -> Callable[..., Any]:
    """Build a shim that resolves a torchlens wrapper argument to its original.

    Parameters
    ----------
    original_entry:
        The upstream TorchScript entry, called with the function to compile first.

    Returns
    -------
    Callable[..., Any]
        The shim; every argument after the first passes through unchanged.
    """

    @functools.wraps(original_entry)
    def jit_entry_shim(obj: Any, *args: Any, **kwargs: Any) -> Any:
        """Resolve a torchlens wrapper to its original, then delegate.

        Keyed on LEDGER IDENTITY, never on ``__tl_*`` attribute presence: a
        foreign ``@functools.wraps(F.relu)`` wrapper inherits the torchlens
        ``__dict__`` markers, and an attribute-keyed check would swap it for the
        pristine original and drop the foreign behavior (b8-fable R56).
        """

        original = _state._decorated_to_orig.get(id(obj))
        return original_entry(obj if original is None else original, *args, **kwargs)

    return jit_entry_shim


def install_jit_entry_shims(
    records: list[tuple[Any, str, Any]],
    *,
    register_shim: Callable[[Any], Any],
    is_shimmed: Callable[[Any], bool],
) -> None:
    """Install the wrapper-resolving shim on every wrapper-blind jit entry present.

    Parameters
    ----------
    records:
        The identity-shim install records; one ``(module, name, original)`` row
        is appended per installed shim so teardown restores it.
    register_shim:
        The identity-shim registry's enrolment function (teardown authority).
    is_shimmed:
        The identity-shim registry's marker check; an entry already shimmed is
        left alone.
    """

    for module, name in _jit_entries():
        original_entry = vars(module).get(name)
        if original_entry is None or is_shimmed(original_entry):
            continue
        shim = register_shim(_make_jit_entry_shim(original_entry))
        setattr(module, name, shim)
        records.append((module, name, original_entry))
