"""Captum site-probe plumbing that needs neither captum nor a model forward."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
from torch import nn

from torchlens._errors import InvalidArgumentError
from torchlens.bridge import captum


def _probe(site_key: str = "s1|block|add||1", ordinal: int = 1) -> captum._SiteProbe:
    return captum._SiteProbe(nn.Identity(), site_key, "add_1_4", ordinal)


def test_probe_is_an_identity_module_that_names_its_site() -> None:
    """The probe returns its input unchanged and reports label, key and ordinal."""

    probe = _probe(ordinal=2)
    value = object()
    assert probe(value) is value
    assert (
        "add_1_4" in repr(probe) and "s1|block|add||1" in repr(probe) and "ordinal=2" in repr(probe)
    )


def test_probe_refuses_once_its_model_is_collected() -> None:
    """A probe outliving its model refuses to run instead of resurrecting it."""

    model = nn.Identity()
    probe = captum._SiteProbe(model, "s1|x|relu||1", "relu_1_1")
    assert probe.model() is model
    del model
    gc.collect()
    with pytest.raises(ValueError, match="outlived"):
        probe.model()


def test_single_probe_finds_one_and_refuses_two() -> None:
    """``_single_probe`` reads a module or list layer; two probes refuse typed."""

    probe = _probe()
    assert captum._single_probe(SimpleNamespace(layer=probe)) is probe
    assert captum._single_probe(SimpleNamespace(layer=[nn.ReLU(), probe])) is probe
    assert captum._single_probe(SimpleNamespace(layer=nn.ReLU())) is None
    assert captum._single_probe(SimpleNamespace()) is None
    with pytest.raises(InvalidArgumentError) as info:
        captum._single_probe(SimpleNamespace(layer=[probe, _probe(ordinal=2)]))
    assert info.value.fields["code"] == "bridge_captum_multi_probe_unsupported"


def test_site_ordinal_counts_earlier_ops_sharing_the_key() -> None:
    """The ordinal is the 1-based execution-order rank among ops with the same site key."""

    key = "s1|block.relu|relu||1"
    ops = [
        SimpleNamespace(raw_index=2, site_key=key),
        SimpleNamespace(raw_index=5, site_key="s1|block|add||1"),
        SimpleNamespace(raw_index=6, site_key=key),
        SimpleNamespace(raw_index=9, site_key=key),
    ]
    layers = [SimpleNamespace(ops=(ops[0], ops[2])), ops[1], SimpleNamespace(ops=(ops[3],))]
    log = SimpleNamespace(layer_list=layers)
    assert captum._site_ordinal(log, ops[0], key) == 1
    assert captum._site_ordinal(log, ops[2], key) == 2
    assert captum._site_ordinal(log, ops[3], key) == 3
    assert captum._site_ordinal(log, SimpleNamespace(raw_index=None), key) == 1
