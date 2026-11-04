"""grind-p3 T5.3/T5.4: a bundle that SAVES always LOADS (foreign container types).

Two save-ok/load-fail traps in the portable scrub:

* A USER tuple subclass (e.g. a namedtuple from the caller's module) in KEEP
  metadata was preserved by TYPE into ``metadata.pkl``. The default-deny safe
  unpickler then refused the foreign class at load, making the WHOLE bundle
  unloadable. The scrub now preserves only load-reconstructible tuple types
  (torch/torchlens-owned); every foreign type is flattened to a plain tuple at
  SAVE time with a disclosure warning -- and a preserved type whose constructor
  rejects the rebuild is disclosed too, never silently downgraded.
* A ``defaultdict`` whose ``default_factory`` came from a user module loaded as
  a booby-trap: the factory rehydrated as an inert foreign-callable placeholder
  that raised ``UnpicklingError`` on the first missing-key read mid-analysis.
  The scrub now drops a non-portable factory at SAVE time with disclosure; the
  loaded mapping behaves as a plain dict (``KeyError`` on a missing key).
"""

from __future__ import annotations

import collections

import pytest
import torch
from _format3_envelope import read_manifest
from torch import nn

import torchlens as tl

_Point = collections.namedtuple("_Point", ["x", "y"])


class _WeirdTuple(tuple):
    """Tuple subclass WITHOUT a one-iterable constructor."""

    def __new__(cls, a, b):  # noqa: ANN001, ANN206 - test shim
        return super().__new__(cls, (a, b))


def _user_factory() -> str:
    return "USER-DEFAULT"


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _traced():
    return tl.trace(_TinyModel(), torch.randn(1, 4))


def test_user_namedtuple_saves_with_disclosure_and_loads(tmp_path) -> None:
    """A foreign namedtuple flattens at save (disclosed) and the bundle loads."""

    log = _traced()
    log.annotations["user_point"] = _Point(1, 2)
    bundle = tmp_path / "point.tlspec"
    with pytest.warns(UserWarning, match="_Point"):
        log.save(bundle)
    loaded = tl.load(bundle)
    restored = loaded.annotations["user_point"]
    assert type(restored) is tuple
    assert restored == (1, 2)


def test_torch_size_round_trips_unflattened(tmp_path) -> None:
    """Load-reconstructible tuple types keep their exact type, no disclosure."""

    log = _traced()
    log.annotations["size"] = torch.Size([2, 3])
    bundle = tmp_path / "size.tlspec"
    with warnings_as_errors_for_flattening():
        log.save(bundle)
    loaded = tl.load(bundle)
    restored = loaded.annotations["size"]
    assert type(restored) is torch.Size
    assert tuple(restored) == (2, 3)


class warnings_as_errors_for_flattening:
    """Fail the enclosed block if any container-flattening disclosure fires."""

    def __enter__(self):
        import warnings as warnings_module

        self._catcher = warnings_module.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        warnings_module.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._catcher.__exit__(exc_type, exc, tb)
        flattening = [
            str(record.message)
            for record in self._records
            if "Flatten" in str(record.message) or "flatten" in str(record.message)
        ]
        assert flattening == [], flattening
        return False


def test_defaultdict_foreign_factory_is_dropped_with_disclosure(tmp_path) -> None:
    """A user-module default_factory never becomes a load-time booby-trap.

    Format 3 (docs/maintainers/saved_format_3.md, the ``dict`` tag): the foreign
    factory is written as an ``unencodable`` disclosure that the manifest counts,
    the mapping reloads as a plain dict, and the load carries the
    ``unencodable_values_stringified`` advisory. No factory is executed on load.
    """

    log = _traced()
    trapped = collections.defaultdict(_user_factory)
    trapped["present"] = 1
    log.annotations["user_dd"] = trapped
    bundle = tmp_path / "dd.tlspec"
    log.save(bundle)
    disclosures = read_manifest(bundle)["disclosures"]
    assert disclosures["count"] >= 1
    assert any(str(at).endswith("default_factory") for at in disclosures["first"])
    loaded = tl.load(bundle)
    restored = loaded.annotations["user_dd"]
    assert type(restored) is dict
    assert restored["present"] == 1
    with pytest.raises(KeyError):
        restored["missing_key"]
    advisories = loaded.annotations["capture_advisories"]
    assert any(row["kind"] == "unencodable_values_stringified" for row in advisories)
