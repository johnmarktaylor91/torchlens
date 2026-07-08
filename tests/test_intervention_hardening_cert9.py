"""Regression tests for the cert9 (round-9-prep) intervention hardening fix.

MAJOR-1 (cert8 intervention bucket) -- ``helper_from_serialized``
(``torchlens/intervention/helpers.py``) is exported in ``helpers.__all__`` as the
module's public reconstruction entry point, but its own ``value_decoder`` keyword
argument DEFAULTED to the narrow ``_decode_jsonish`` decoder (only understands the
``__tensor_ref__`` wrapper tag) whenever a caller omitted it. That default
re-triggered the exact BLOCKER-2 corruption class a prior round fixed at the sole
production call site: silently constructing a non-opaque ``HelperSpec`` whose
``module=``/args carry a raw ``{"__callable__": ...}`` wrapper dict instead of the
resolved callable, with no exception, no ``NonExecutableSpecError``, and no
``opaque_audit`` flag -- corruption invisible until the hook actually fires.

The fix makes ``value_decoder`` a required keyword-only parameter (no default), so
a caller can no longer silently opt into the narrow, out-of-lockstep decoder just
by omitting a kwarg -- omission is now a loud ``TypeError`` at the call site,
before any decoding happens.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from torchlens.intervention.helpers import helper_from_serialized
from torchlens.intervention.save import _resolve_import_ref


def test_value_decoder_has_no_default() -> None:
    """MAJOR-1: value_decoder is required -- no default to silently fall back to."""

    signature = inspect.signature(helper_from_serialized)
    parameter = signature.parameters["value_decoder"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_omitting_value_decoder_raises_typeerror_not_silent_corruption() -> None:
    """MAJOR-1: omitting value_decoder fails loudly instead of building a corrupt spec.

    Before the fix this silently built a non-opaque ``HelperSpec`` whose arg was a
    raw ``{"__callable__": ...}`` dict -- exactly the BLOCKER-2 corruption class.
    """

    payload = {
        "portability": "builtin",
        "name": "splice_module",
        "kind": "forward",
        "direction": "forward",
        "args": [{"__callable__": {"portability": "import_ref", "import_path": "x:y"}}],
        "kwargs": {},
    }
    with pytest.raises(TypeError, match="value_decoder"):
        helper_from_serialized(  # type: ignore[call-arg]
            payload,
            tensor_loader=lambda tid: torch.zeros(1),
            import_resolver=_resolve_import_ref,
        )


def test_decode_jsonish_still_pinned_but_unreachable_from_helper_from_serialized() -> None:
    """MINOR-1: _decode_jsonish is quarantined -- no longer reachable as a fallback.

    It is retained only so the characterization test can keep pinning the narrow
    decoder's wrapper-leaking behavior; nothing in ``helper_from_serialized``
    references it anymore.
    """

    import ast
    import inspect as _inspect

    from torchlens.intervention import helpers as helpers_mod

    source = _inspect.getsource(helpers_mod.helper_from_serialized)
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_decode_jsonish" not in called_names
