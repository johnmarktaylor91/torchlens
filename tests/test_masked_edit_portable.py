"""An element-masked builtin edit saves portably and reloads as the same edit.

``do(tl.units(site, mask), verb)`` with a builtin verb keeps the verb's name and
``builtin`` portability and stores the dense bool ``mask`` and its ``mask_sha256``
among the verb's keywords (the mask travels in the tensor sidecar), so the default
portable save of a trace or spec carrying it succeeds and a reload applies
bit-identically. A malformed or tampered stored mask refuses typed at load, a reader
that does not know the two keywords refuses typed, and an edit that needs user code
still refuses a portable save.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import redigest
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.errors import OpaqueCallableError
from torchlens.intervention._subspace_projection import _mask_digest
from torchlens.intervention.helpers import rebuild_builtin_helper

_SITE = "linear_1_1"
_D = 6


class _SeqNet(nn.Module):
    """``lin2(relu(lin1(x)))`` over ``[batch, seq, 6]`` in float64."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(_D, _D).double()
        self.lin2 = nn.Linear(_D, 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))


_MODEL = _SeqNet().eval()
_X = torch.randn(2, 5, _D, generator=torch.Generator().manual_seed(1), dtype=torch.float64)


def _ready() -> Any:
    return tl.trace(_MODEL, _X, capture=tl.options.CaptureOptions(replayable=True))


def _mask(position: int = 3) -> torch.Tensor:
    mask = torch.zeros(2, 5, _D, dtype=torch.bool)
    mask[:, position, : _D // 2] = True
    return mask


def _session(edit: Any, mask: torch.Tensor | None = None) -> Any:
    fork = _ready().fork()
    fork.do(tl.units(_SITE, _mask() if mask is None else mask), edit)
    return fork


def _output(trace: Any) -> torch.Tensor:
    return trace["output_1"].out


def _apply(spec: Any) -> Any:
    fork = _ready().fork()
    return fork.do([(hook_spec.site_target, hook_spec.helper) for hook_spec in spec.hook_specs])


def _coded(error: BaseException) -> Any:
    current: BaseException | None = error
    while current is not None:
        if getattr(current, "fields", {}).get("code"):
            return current
        current = current.__cause__ or current.__context__
    raise AssertionError(f"no coded error in the chain of {error!r}")


_VERBS = {
    "zero_ablate": lambda: tl.zero_ablate(),
    "scale": lambda: tl.scale(2.0),
    "clamp": lambda: tl.clamp(min=-0.1, max=0.1),
}


@pytest.mark.parametrize("verb", sorted(_VERBS))
def test_masked_builtin_edit_round_trips_bit_identically(tmp_path: Path, verb: str) -> None:
    """Save the spec portably, load it, re-apply it: the outputs are bit-identical."""

    session = _session(_VERBS[verb]())
    staged = session._intervention_spec.hook_specs[0].helper
    assert staged.portability == "builtin" and staged.name == verb
    stored = dict(staged.kwargs)
    assert torch.equal(stored["mask"], _mask())
    assert stored["mask_sha256"] == _mask_digest(_mask())
    path = tmp_path / "masked.tlspec"
    tl.io.save_intervention(session, path, level="portable")
    assert _mask_digest(_mask()) in (path / "intervention.json").read_text(encoding="utf-8")
    reapplied = _apply(tl.io.load_intervention_spec(path))
    assert torch.equal(reapplied[_SITE].out, session[_SITE].out)
    assert torch.equal(_output(reapplied), _output(session))
    assert not torch.equal(session[_SITE].out, _ready()[_SITE].out)


def test_masked_trace_saves_at_the_default_level_and_loads(tmp_path: Path) -> None:
    """A trace carrying a masked builtin edit saves at the default (portable) level."""

    session = _session(tl.zero_ablate())
    path = tmp_path / "masked_trace.tlspec"
    session.save(path)
    loaded = tl.load(path)
    assert torch.equal(loaded[_SITE].out, session[_SITE].out)
    assert torch.equal(_output(loaded), _output(session))


def test_reloaded_masked_edit_under_another_mask_stores_the_intersection(tmp_path: Path) -> None:
    """The session scatters through both masks; the re-saved edit stores their intersection."""

    session = _session(tl.scale(3.0))
    tl.io.save_intervention(session, tmp_path / "first.tlspec", level="portable")
    loaded = tl.io.load_intervention_spec(tmp_path / "first.tlspec").hook_specs[0].helper
    fork = _ready().fork()
    fork.do(tl.units(_SITE, _mask(3) | _mask(4)), loaded)
    assert torch.equal(fork[_SITE].out, session[_SITE].out)
    tl.io.save_intervention(fork, tmp_path / "second.tlspec", level="portable")
    again = tl.io.load_intervention_spec(tmp_path / "second.tlspec")
    keys = [key for key, _ in again.hook_specs[0].helper.kwargs]
    assert keys.count("mask") == 1 and keys.count("mask_sha256") == 1
    assert torch.equal(dict(again.hook_specs[0].helper.kwargs)["mask"], _mask(3))
    assert torch.equal(_apply(again)[_SITE].out, session[_SITE].out)


_MALFORMED = {
    "not-bool": ("mask", lambda: _mask().to(torch.uint8), lambda m: _mask_digest(_mask())),
    "digest-missing": ("mask_sha256", _mask, lambda m: None),
    "digest-without-mask": ("mask", lambda: None, lambda m: "0" * 64),
    "digest-mismatch": ("mask", _mask, lambda m: _mask_digest(_mask(1))),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_or_tampered_stored_mask_refuses_typed(case: str) -> None:
    field, mask, digest = _MALFORMED[case]
    value = mask()
    pairs = (("mask", value), ("mask_sha256", digest(value)))
    stored = {key: item for key, item in pairs if item is not None}
    with pytest.raises(InvalidArgumentError) as excinfo:
        rebuild_builtin_helper("scale", (2.0,), stored)
    assert excinfo.value.fields["code"] == "intervention_helper_arguments_invalid"
    assert excinfo.value.fields["argument"] == field
    assert "level='portable'" in excinfo.value.fields["remedy"]


def test_mask_tampered_on_disk_refuses_at_load(tmp_path: Path) -> None:
    session = _session(tl.zero_ablate())
    path = tmp_path / "tampered.tlspec"
    tl.io.save_intervention(session, path, level="portable")
    spec_json = path / "intervention.json"
    text = spec_json.read_text(encoding="utf-8")
    digest = _mask_digest(_mask())
    assert digest in text
    spec_json.write_text(text.replace(digest, _mask_digest(_mask(1))), encoding="utf-8")
    redigest(path, "intervention.json")
    with pytest.raises(Exception) as excinfo:
        tl.io.load_intervention_spec(path)
    error = _coded(excinfo.value)
    assert error.fields["code"] == "intervention_helper_arguments_invalid"
    assert error.fields["argument"] == "mask"


def test_a_reader_without_the_mask_keywords_refuses_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torchlens that does not know the two keywords refuses the spec, never drops the mask."""

    session = _session(tl.zero_ablate())
    path = tmp_path / "masked.tlspec"
    tl.io.save_intervention(session, path, level="portable")
    masked_spec = importlib.import_module("torchlens.intervention._masked_spec")
    monkeypatch.setattr(masked_spec, "split_stored_mask", lambda name, kwargs: (kwargs, None))
    with pytest.raises(Exception) as excinfo:
        tl.io.load_intervention_spec(path)
    assert _coded(excinfo.value).fields["code"] == "intervention_helper_arguments_invalid"


def test_a_masked_user_callable_still_refuses_a_portable_save(tmp_path: Path) -> None:
    """An edit that needs user code to rebuild keeps refusing the default save."""

    def _halve(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        return out * 0.5

    session = _session(_halve)
    with pytest.raises(OpaqueCallableError) as caught:
        session.save(tmp_path / "callable.tlspec")
    assert caught.value.fields["code"] == "intervention_spec_not_portable"
    assert "level='audit'" in caught.value.fields["remedy"]


def test_masked_edit_repr_never_prints_the_mask_as_a_constructor_keyword() -> None:
    """``tl.zero_ablate`` takes no ``mask``: the repr is not a call that would refuse."""

    staged = _session(tl.zero_ablate())._intervention_spec.hook_specs[0].helper
    reloaded = rebuild_builtin_helper("zero_ablate", (), dict(staged.kwargs))
    assert "tl.zero_ablate(mask" not in repr(reloaded)
    assert torch.equal(dict(reloaded.kwargs)["mask"], _mask())
