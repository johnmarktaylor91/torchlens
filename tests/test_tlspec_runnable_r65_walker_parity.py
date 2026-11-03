"""r65 Cluster Y: input-boundary walker container parity (single-sourced dispatch).

The three input-boundary walkers -- W1 (capture literal-leaf), W2 (capture
metadata-read-site), W3 (runtime literal-path) -- witness the SAME physical input
tree for different fact families, so a container kind handled by one walker but
missed by another silently drops a whole fact family for every leaf beneath it.
That is the r64 Finding-1 false-VERIFIED: W2 did not descend dataclass containers,
so ``box.x.is_contiguous()`` on a dataclass input field recorded no metadata
witness and a same-value non-contiguous twin replayed the captured branch as
VERIFIED. These tests pin:

1. the single-sourced dispatch STRUCTURALLY (source-scan meta-test: no private
   container dispatch inside any walker body -- a future container kind must land
   in ``torchlens/_input_walk.py`` and reaches every walker in lockstep),
2. container-kind parity across all three walkers BEHAVIORALLY (identical
   supported set incl. dataclass/namedtuple/dict/list/tuple/empty/opaque),
3. the declared DUAL mapping-key vocabulary (tagged literal vs raw fact-site,
   residual R6) exactly as declared in the one module,
4. the r64 repro end-to-end (dataclass metadata read witnessed; layout twin
   diverges under the default policy, never VERIFIED) and zero over-trigger
   (tensor-only dataclass with no metadata read stays VERIFIED + ATTESTED),
5. behavioral round-trip parity of the NON-rerouted W4/W5/W6 walkers against the
   shared dispatch on a torture container.
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.trace import (
    _OPAQUE_INPUT_LEAF,
    _record_runnable_input_literal_leaves,
    _record_runnable_input_tensor_sites,
)
from torchlens._runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
)
from torchlens._runnable.input_aliases import _runtime_nontensor_leaf_paths
from torchlens._runnable.seam import runnable_trace_state
from torchlens._runnable.witness_contracts import (
    _tensor_leaf_paths,
)
from torchlens.errors import PathDivergenceError, RunnablePreflightError
from torchlens.options import CaptureOptions

# ======================================================================================
# Shared fixtures: probe containers spanning the closed container-kind vocabulary
# ======================================================================================


class _Key(enum.Enum):
    """A mapping key outside the frozen literal grammar (non-representable)."""

    K = "k"


class _EmptyNT(NamedTuple):
    """Zero-field namedtuple (classified as an EMPTY container by kind)."""


class _FakeTrace:
    """Minimal weak-referenceable Trace stand-in for standalone walker runs."""

    is_replayable = True


def _w1_leaves(root: Any) -> tuple[tuple[object, tuple[Any, ...], Any], ...]:
    """Run the capture literal-leaf walker (W1) standalone on one positional root."""

    trace = _FakeTrace()
    _record_runnable_input_literal_leaves(trace, [root], {})
    return runnable_trace_state(trace).input_nontensor_leaves or ()


def _w1_paths(root: Any) -> set[tuple[Any, ...]]:
    """Return W1's recorded non-tensor leaf-path set for one positional root."""

    return {path for _position, path, _leaf in _w1_leaves(root)}


def _w2_paths(root: Any) -> set[tuple[Any, ...]]:
    """Run the capture metadata-site walker (W2) standalone; return its site paths."""

    trace = _FakeTrace()
    _record_runnable_input_tensor_sites(trace, [root], {})
    sites = runnable_trace_state(trace).input_tensor_sites or {}
    return {path for _position, path in sites.values()}


# ======================================================================================
# 1. Source-scan meta-test: no private container dispatch inside any walker body
# ======================================================================================


# ======================================================================================
# 2. Closed container-kind vocabulary + classification order pins
# ======================================================================================


class _RegisteredProbe:
    """r67 C2: registered-container probe for the closed kind vocabulary."""

    def __init__(self, t):
        self.t = t


from torchlens._ir.container import register_container as _register_container  # noqa: E402

_register_container(
    _RegisteredProbe,
    lambda box: ([box.t], None),
    lambda aux, children: _RegisteredProbe(children[0]),
    state_complete=True,
)


# ======================================================================================
# 3. Three-walker container-kind parity (behavioral) + dual key vocabulary pins
# ======================================================================================


def test_opaque_mapping_key_subtree_parity() -> None:
    """A non-representable key collapses the literal subtree to ONE opaque parent leaf.

    W2 still RAW-indexes the tensor beneath it (declared vocabulary): the fact site
    fails literal encoding at witness time while W1's opaque leaf independently
    ceilings the run to UNVERIFIABLE -- never a false VERIFIED.
    """

    tensor = torch.zeros(2)
    root = {_Key.K: {"deep": tensor}}
    assert _w1_paths(root) == {()}
    leaves = _w1_leaves(root)
    assert len(leaves) == 1
    assert leaves[0][2] is _OPAQUE_INPUT_LEAF
    assert _runtime_nontensor_leaf_paths(root) == {()}
    assert _w2_paths(root) == {(_Key.K, "deep")}


# ======================================================================================
# 4. r64 Finding-1 end-to-end behavioral pins (the money bug) + no-over-trigger
# ======================================================================================


@dataclasses.dataclass
class _LayoutBox:
    """Dataclass model input with a single tensor field (r64 Finding-1 shape)."""

    x: torch.Tensor


class _DataclassLayoutBranch(nn.Module):
    """Branch on a dataclass tensor FIELD's contiguity (r64 Finding-1 verbatim)."""

    def forward(self, box: _LayoutBox) -> torch.Tensor:
        """Add or subtract according to the field's memory layout."""

        if box.x.is_contiguous():
            return box.x + 10
        return box.x - 10


def _capture_runnable(model: nn.Module, box: _LayoutBox, path: Path) -> Any:
    """Capture one dataclass-input model and save a runnable artifact."""

    trace = tl.trace(
        model,
        box,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return trace


def _layout_twin(x: torch.Tensor) -> torch.Tensor:
    """Return a byte-identical, same-shape, NON-contiguous twin of a square tensor."""

    twin = x.t().contiguous().t()
    assert torch.equal(twin, x)
    assert not twin.is_contiguous()
    return twin


def test_r64_dataclass_metadata_read_witnessed_and_layout_twin_diverges(
    tmp_path: Path,
) -> None:
    """r64 Finding-1 verbatim: the dataclass-field metadata read is now witnessed.

    Pre-fix, W2 never indexed the dataclass field, no ``model_input_metadata`` fact
    was recorded, and the non-contiguous twin replayed the captured ``+10`` arm as
    VERIFIED. Post-fix the read is witnessed and the twin diverges (default policy
    raises), never VERIFIED, never ATTESTED.
    """

    x = torch.arange(16.0).reshape(4, 4).contiguous()
    path = tmp_path / "r64_dc_layout.tlspec"
    trace = _capture_runnable(_DataclassLayoutBranch(), _LayoutBox(x), path)
    reads = trace._runnable.input_metadata_reads
    assert reads, (
        "the is_contiguous() read on the dataclass input field recorded no metadata "
        "witness -- the r64 Finding-1 walker gap has reopened"
    )

    twin = _layout_twin(x)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=_LayoutBox(twin))
    diverged = tl.load(path).run(inputs=_LayoutBox(twin), on_divergence="return")
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.is_poisoned
    # A fresh model on the twin takes the OTHER arm; the replayed arm is wrong.
    fresh = _DataclassLayoutBranch()(_LayoutBox(twin))
    assert torch.equal(fresh, twin - 10)


def test_r64_dataclass_original_input_still_verifies_and_attests(tmp_path: Path) -> None:
    """Honest original-input dataclass metadata run stays VERIFIED + ATTESTED."""

    x = torch.arange(16.0).reshape(4, 4).contiguous()
    path = tmp_path / "r64_dc_layout_ok.tlspec"
    _capture_runnable(_DataclassLayoutBranch(), _LayoutBox(x), path)

    result = tl.load(path).run(inputs=_LayoutBox(x))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(result.output, x + 10)


# ======================================================================================
# 5. W4/W5/W6 behavioral round-trip parity on a torture container (not rerouted in r65)
# ======================================================================================


@dataclasses.dataclass
class _OpaqueKeyBox:
    """Dataclass input whose mapping field carries a non-representable key."""

    d: dict[Any, Any]


class _EnumKeyedModel(nn.Module):
    """Read a tensor stored under a non-representable mapping key."""

    def forward(self, box: _OpaqueKeyBox) -> torch.Tensor:
        """Return ``box.d[_Key.K] * 2``."""

        return box.d[_Key.K] * 2.0


class _StrKeyedModel(nn.Module):
    """Read a str-keyed tensor while an opaque-keyed sibling entry sits unread."""

    def forward(self, box: _OpaqueKeyBox) -> torch.Tensor:
        """Return ``box.d['k'] * 2``."""

        return box.d["k"] * 2.0


def test_opaque_key_subtree_is_never_silently_witnessable(tmp_path: Path) -> None:
    """A subtree under a non-representable key can never yield VERIFIED.

    W2 RAW-indexes a tensor under such a key (declared vocabulary) while W4's key
    filter skips it, so the dual vocabulary is only honest because every path that
    could bless such an artifact fails closed: a CONSUMED opaque-keyed tensor refuses
    at save (binding container paths require str/int keys), and an opaque-keyed
    NON-tensor sibling saves but the literal walker's OPAQUE parent leaf ceilings the
    run -- never VERIFIED, never ATTESTED.
    """

    tensor = torch.randn(2)
    consumed = _OpaqueKeyBox({_Key.K: tensor})
    assert _w2_paths(consumed) == {("d", _Key.K)}
    assert set(_tensor_leaf_paths(consumed)) == set()  # W4's (str, int) key filter
    assert _w1_paths(consumed) == {("d",)}  # one opaque leaf at the mapping's path

    # Prong A (r67 C2 disposition matrix): a consumed tensor under a non-grammar key
    # refuses TYPED at runnable save through the positive structure proof (the existing
    # input_container_contract_missing) -- no artifact exists that could ever replay it
    # as VERIFIED.
    trace = tl.trace(
        _EnumKeyedModel(),
        consumed,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(tmp_path / "enum_key.tlspec", level="runnable", include_activations=True)
    assert "input_container_contract_missing" in str(excinfo.value.fields.get("diagnostics"))
    assert "opaque_mapping_key" in str(excinfo.value.fields.get("diagnostics"))

    # Prong B (r67 C2): an opaque-keyed NON-tensor sibling refuses at save the SAME
    # typed way -- the r66 downgrade lane (save-then-UNVERIFIABLE) is retired for the
    # uniform refusal the agreed disposition matrix ratified.
    sibling = _OpaqueKeyBox({"k": tensor, b"kk": 5})
    assert _w1_paths(sibling) == {("d",)}
    trace = tl.trace(
        _StrKeyedModel(),
        sibling,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(tmp_path / "bytes_key.tlspec", level="runnable", include_activations=True)
    assert "input_container_contract_missing" in str(excinfo.value.fields.get("diagnostics"))
