"""R10-4: the save-side "is this type/factory load-reconstructible?" predicate
was a namespace prefix test (``root in {"torch", "torchlens"}`` / ``builtins``),
while the safe unpickler admits torchlens types ONLY on the vetted-inert
``_SAFE_TORCHLENS_TYPES`` allowlist (not appliance modules), torchlens callables
only via ``is_inert_first_party_callable``, and builtins globals only on the
``_SAFE_EXPLICIT_GLOBALS`` pure-data allowlist. The prefix test preserved
off-allowlist types/callables that the loader then refused -- a save-succeeds /
load-refuses trap. The predicate now consults the loader's real authorities.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from torchlens._io.scrub import (
    _factory_is_load_reconstructible,
    _type_is_load_reconstructible,
)


class _OffAllowlistTorchlensType(tuple):
    """A torchlens-namespace tuple subclass that is NOT on the allowlist."""


_OffAllowlistTorchlensType.__module__ = "torchlens._io._probe_offallowlist"


class _ApplianceType(tuple):
    """A tuple subclass masquerading as an extras-gated appliance type."""


_ApplianceType.__module__ = "torchlens.neuro._probe"


def test_off_allowlist_torchlens_type_is_not_reconstructible() -> None:
    assert not _type_is_load_reconstructible(_OffAllowlistTorchlensType)


def test_appliance_torchlens_type_is_not_reconstructible() -> None:
    assert not _type_is_load_reconstructible(_ApplianceType)


def test_allowlisted_torchlens_type_is_reconstructible() -> None:
    from torchlens._io import BlobRef  # ("torchlens._io", "BlobRef") is allowlisted

    assert _type_is_load_reconstructible(BlobRef)


def test_torch_data_type_is_reconstructible() -> None:
    import torch

    assert _type_is_load_reconstructible(torch.Size)


def test_pure_data_builtin_factory_is_reconstructible() -> None:
    for factory in (list, dict, set, int, tuple):
        assert _factory_is_load_reconstructible(factory), factory


def test_code_exec_builtin_factory_is_not_reconstructible() -> None:
    # A prefix test on root=="builtins" preserved these; the loader hard-denies
    # them, so preserving them was a load-refuses trap.
    for factory in (eval, getattr):
        assert not _factory_is_load_reconstructible(factory), factory


def test_collections_factory_is_reconstructible() -> None:
    from collections import OrderedDict

    assert _factory_is_load_reconstructible(OrderedDict)


def test_defaultdict_with_code_exec_factory_survives_round_trip(tmp_path) -> None:
    """A defaultdict whose factory the loader would refuse must save+load clean.

    The factory is dropped (disclosed) rather than preserved-and-unloadable.
    """

    import torch
    from torch import nn

    import torchlens as tl

    model = nn.Linear(4, 3)
    trace = tl.trace(model, torch.randn(2, 4))
    # Plant a defaultdict with a code-exec factory on a portable-scrubbed field.
    trace.ops[0].annotations = defaultdict(eval, {"k": 1})  # type: ignore[assignment]
    bundle = tmp_path / "b.tlspec"
    with pytest.warns(UserWarning):
        tl.save(trace, bundle, overwrite=True)
    loaded = tl.load(bundle)  # must not raise UnpicklingError
    assert loaded is not None
