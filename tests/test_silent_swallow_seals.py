"""grind-r5 R22: fail-open swallows sealed (b7 fable/opus/sol + b1 opus).

Four sites shared one shape -- an exception swallowed into a value that then
participated in an equality key, a provenance fact, a budget charge, or a
lost-forever warning:

* the capture-fingerprint ndarray branch fell through to a STABLE
  content-blind fragment (false cache HIT under ``cache=True``);
* a raising ``model.config`` getter silently dropped the whole config axis
  from the semantic-output cache key (two models differing only in labels
  collided);
* provenance digests conflated could-not-compute with does-not-apply;
* an unreadable projected allocation was charged as ZERO bytes against the
  fail-closed save budget;
* the never-abort output-decode belt left no durable record of a failed
  decode (warning-only disclosure).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_fingerprint import _attribute_state_fragment
from torchlens.autoroute import _builtin_output
from torchlens.autoroute._builtin_output import semantic_output_cache_key


def test_object_dtype_ndarray_fragment_never_matches() -> None:
    """Object-dtype buffers serialize raw pointers: content-blind AND
    address-churning, so the fragment must be an always-miss token."""

    array = np.array([object(), object()], dtype=object)
    first = _attribute_state_fragment(array)
    second = _attribute_state_fragment(array)
    assert first != second, "object-dtype ndarray produced a stable fragment -- false cache HITs"


@pytest.mark.smoke
def test_plain_ndarray_fragment_stays_content_keyed() -> None:
    """The sealed branch keeps honest content hashing for readable arrays."""

    first = _attribute_state_fragment(np.arange(6, dtype=np.float32))
    second = _attribute_state_fragment(np.arange(6, dtype=np.float32))
    assert first == second
    third = _attribute_state_fragment(np.arange(1, 7, dtype=np.float32))
    assert first != third


class _RaisingConfigModel(nn.Module):
    """Model whose ``config`` property raises (the observed real-world case)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)

    @property
    def config(self):  # noqa: ANN201
        raise RuntimeError("partially implemented delegation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_raising_config_getter_never_collides_in_cache_key() -> None:
    """A raising config getter mints a never-matching key component instead
    of silently dropping the config axis (false cache HIT vector)."""

    model = _RaisingConfigModel()
    first = semantic_output_cache_key(model, output_style=None, output_head=None)
    second = semantic_output_cache_key(model, output_style=None, output_head=None)
    assert first != second, (
        "two cache keys for a raising-config model compared equal -- the "
        "config axis was silently dropped"
    )


def test_readable_config_cache_key_is_stable() -> None:
    """The never-match token fires only on raising getters."""

    model = nn.Linear(2, 2)
    first = semantic_output_cache_key(model, output_style=None, output_head=None)
    second = semantic_output_cache_key(model, output_style=None, output_head=None)
    assert first == second


@pytest.mark.smoke
def test_failed_decode_leaves_a_durable_annotation(monkeypatch) -> None:
    """A tripped decode belt must leave trace.annotations['decode_skipped'],
    not just a losable warning, and must roll back partial writes."""

    def _raising_decode(trace, outputs, *, output_style, output_head):
        trace.decoded_output = {"kind": "partial"}  # simulate the partial-write window
        raise RuntimeError("synthetic decoder fault")

    monkeypatch.setattr(_builtin_output, "_decode_outputs_for_trace_unguarded", _raising_decode)
    with pytest.warns(match="decode_skipped"):
        trace = tl.trace(
            nn.Linear(4, 2),
            torch.randn(1, 4),
            capture=tl.options.CaptureOptions(output_style="imagenet"),
        )

    assert trace.decoded_output is None, "partial decode write survived the belt"
    assert trace.output_postprocessor is None
    rows = trace.annotations.get("decode_skipped")
    assert rows and rows[0]["error"].startswith("RuntimeError"), (
        f"no durable decode-skip record: {trace.annotations!r}"
    )


def test_unreadable_projected_allocation_refuses_instead_of_charging_zero() -> None:
    """grind-r5 b7 R22 (sol HIGH): a tensor whose size reads raise must raise
    the typed sentinel, never contribute an empty charge."""

    from torchlens._runnable_execution import (
        _new_allocation_bytes,
        _UnreadableProjectedOutput,
    )

    class _UnreadableTensor(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return super().__new__(cls)

        def numel(self):  # noqa: ANN201
            raise RuntimeError("size unreadable")

    with pytest.raises(_UnreadableProjectedOutput):
        _new_allocation_bytes(_UnreadableTensor(), frozenset())


def test_provenance_digest_failures_record_unavailable_sentinels(monkeypatch) -> None:
    """Could-not-compute must stay distinguishable from does-not-apply."""

    from torchlens import hash as trace_hash_module
    from torchlens._io import bundle as bundle_module

    trace = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))

    def _raising_digest(value):  # noqa: ANN001, ANN202
        raise ValueError("digest machinery broken")

    monkeypatch.setattr(trace_hash_module, "content", _raising_digest)
    monkeypatch.setattr(trace_hash_module, "trace", _raising_digest)
    provenance = bundle_module._collect_provenance(trace, include_source=False)

    assert provenance.input_hash == "unavailable:ValueError"
    assert provenance.model_structure_hash == "unavailable:ValueError"
    rng_digests = provenance.rng_state_digests
    if rng_digests:
        assert all(value == "unavailable:ValueError" for value in rng_digests.values()), (
            f"a failed engine digest was silently omitted or mis-recorded: {rng_digests!r}"
        )
