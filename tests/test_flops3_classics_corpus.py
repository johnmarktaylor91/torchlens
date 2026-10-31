"""Classics-corpus FLOPs acceptance: forward and executed backward totals per entry.

The pre-registered oracle (``tests/fixtures/flops3_corpus_oracle.json``,
counts only) holds torch FlopCounterMode's forward and backward totals for
every corpus entry, built by ``build_entry`` in eval mode, at fma2 with only
the matmul and convolution families counted. The seven entries that dispatch
CPU fused attention were re-counted under the MATH attention backend, and
that total is the bar for them; TorchLens counts the default backend.

Per entry, TorchLens's forward (2 x the true MACs of the matmul and
convolution family ops) and its executed backward (``backward_status``)
equal the oracle exactly, apart from two explained differences, each
computed independently here from a plain run of the same entry:

* FlopCounterMode counts a grouped convolution's weight gradient ``groups``
  times over; TorchLens counts it once. The test expects the oracle total
  minus exactly that excess.
* TorchLens counts kernels FlopCounterMode has no rule for
  (``COUNTED_BEYOND_FLOP_COUNTER_MODE``: ``mv``, ``dot``, the oneDNN LSTM
  layer kernels); the test expects the oracle total plus exactly that work.

Both are pinned per entry in ``GROUPED_DW_EXCESS`` and ``BEYOND_FCM`` so a
change in either shows up by name.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import torchlens as tl
from tests.classics_corpus._loader import CorpusEntry, build_entry, corpus_entries
from torchlens._capture.compute_record import MAC_FAMILY_NAMES, derive_compute_record_for_op
from torchlens.report import backward_status

pytestmark = pytest.mark.slow

ORACLE = json.loads(
    (Path(__file__).parent / "fixtures" / "flops3_corpus_oracle.json").read_text(encoding="utf-8")
)
ROWS = {row["id"]: row for row in ORACLE["classics_corpus"]}

#: FlopCounterMode's grouped-convolution weight-gradient excess (FLOPs),
#: ``(groups - 1) x`` the true weight gradient, per entry.
GROUPED_DW_EXCESS = {
    "SpatialConvTasNet": 717_120,
    "paddledet_picodet": 10_570_752,
    "SINet Portrait": 538_570_368,
    "MoViNet-A0 PyTorch compact": 104_067_072,
    "Coordinate Attention (efficient mobile attention)": 1_105_920,
    "MediaPipe Selfie Segmentation": 1_118_044_160,
}

#: Work TorchLens counts that FlopCounterMode has no rule for, per entry:
#: ``(forward, backward)`` FLOPs.
BEYOND_FCM: dict[str, tuple[int, int]] = {
    "denoiser.Demucs (speech enhancement)": (260_096, 520_192),
    "studiogan_sngan_generator": (110_214, 0),
    "LogAnomaly": (196_608, 393_216),
    "CoKT": (18_432, 36_864),
    "PyanNet": (1_966_080, 3_932_160),
    "ATDKT": (65_536, 131_072),
    "IMPALA_AtariNet": (17_239_104, 34_478_208),
    "pointer_network_tsp": (131_072, 262_144),
    "Real-Time Demucs (denoiser)": (8_159_232, 16_318_464),
    "DPTNet": (2_162_688, 4_325_376),
    "Pointer Sentinel-LSTM mixture model": (196_608, 393_216),
    "DeepConvLSTM-HARModel": (917_504, 1_835_008),
    "Soft committee machine": (20, 20),
    "CLDNN (Automatic Modulation Classification)": (20_240_000, 40_480_000),
    "DeepLog": (4_546_560, 9_093_120),
    "SketchLattice": (187_392, 374_784),
}

#: The entry whose own backward fails in eval mode (an in-place write the
#: gradient needs); its forward is still checked.
BACKWARD_RAISES = {"KUIELAB-MDX-Net (ConvTDFNet)"}


def _prod(shape: Any) -> int:
    return math.prod(int(size) for size in shape)


class _Census(TorchDispatchMode):
    """Measure, on a plain run, the two explained differences (never TorchLens code)."""

    def __init__(self) -> None:
        super().__init__()
        self.beyond = 0
        self.grouped_dw_excess = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in ("mv", "addmv"):
            matrix = args[1] if name == "addmv" else args[0]
            self.beyond += 2 * _prod(matrix.shape)
        elif name in ("dot", "vdot"):
            self.beyond += 2 * int(args[0].shape[0])
        elif name in ("mkldnn_rnn_layer", "mkldnn_rnn_layer_backward"):
            weights = [arg for arg in args[1:5] if isinstance(arg, torch.Tensor) and arg.dim() == 2]
            rows = args[0].numel() // args[0].shape[-1]
            once = 2 * rows * sum(weight.numel() for weight in weights)
            self.beyond += 2 * once if name.endswith("_backward") else once
        elif name == "convolution_backward":
            groups, mask, transposed = args[9], args[10], args[7]
            if groups > 1 and mask[1]:
                small = args[1].shape if transposed else args[0].shape
                unit = small[0] * _prod(args[2].shape) * _prod(small[2:])
                self.grouped_dw_excess += (groups - 1) * 2 * unit
        return out


def _float_leaves(obj: Any) -> list[torch.Tensor]:
    """The oracle's loss leaves: floating tensors in the output structure."""

    if isinstance(obj, torch.Tensor):
        return [obj] if obj.is_floating_point() or obj.is_complex() else []
    if isinstance(obj, dict):
        return [tensor for value in obj.values() for tensor in _float_leaves(value)]
    if isinstance(obj, (list, tuple)):
        return [tensor for value in obj for tensor in _float_leaves(value)]
    for attr in ("last_hidden_state", "logits"):
        if hasattr(obj, attr):
            return _float_leaves(getattr(obj, attr))
    return []


def _loss(leaves: list[torch.Tensor]) -> torch.Tensor:
    return sum(tensor.real.sum() if tensor.is_complex() else tensor.sum() for tensor in leaves)


def _call_args(inputs: Any, whole: bool) -> tuple[list[Any], dict[str, Any] | None]:
    """The oracle's calling convention: spread containers, or pass them whole."""

    if whole:
        return [inputs], None
    if isinstance(inputs, dict):
        return [], inputs
    if type(inputs) in (tuple, list):
        return list(inputs), None
    return [inputs], None


def _same(output: Any) -> Any:
    """Keep the live model output on ``trace.raw_output`` (the loss leaves)."""

    return output


def _plain_call(model: torch.nn.Module, inputs: Any, whole: bool) -> Any:
    args, kwargs = _call_args(inputs, whole)
    return model(*args, **(kwargs or {}))


def _census(entry: CorpusEntry, whole: bool) -> tuple[_Census, _Census]:
    """Forward and backward census of a fresh, plain build of the entry."""

    model, inputs = build_entry(entry)
    forward, backward = _Census(), _Census()
    with forward:
        out = _plain_call(model, inputs, whole)
    leaves = [tensor for tensor in _float_leaves(out) if tensor.requires_grad]
    if leaves and entry.id not in BACKWARD_RAISES:
        with backward:
            _loss(leaves).backward()
    return forward, backward


def _forward_family_flops(trace: Any) -> int | None:
    """2 x true MACs over the matmul and convolution family ops (None if any is unknown)."""

    total = 0
    for op in trace.layer_list:
        if getattr(op, "func_name", None) not in MAC_FAMILY_NAMES:
            continue
        if getattr(op, "is_input", False) or getattr(op, "is_output", False):
            continue
        record = derive_compute_record_for_op(op)
        if record is None or record.fma_macs is None:
            return None
        total += 2 * record.fma_macs
    return total


def _entry_param(entry: CorpusEntry) -> Any:
    return pytest.param(entry, id=entry.id)


@pytest.mark.parametrize("entry", [_entry_param(entry) for entry in corpus_entries()])
def test_corpus_forward_and_executed_backward_equal_the_oracle(entry: CorpusEntry) -> None:
    """Forward and executed backward FLOPs equal the oracle up to the two explained differences.

    The models' own warnings (deprecations, unsupported output containers)
    are not what this test asserts, as in ``test_classics_corpus.py``.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        _check_entry(entry)


def _check_entry(entry: CorpusEntry) -> None:
    row = ROWS[entry.id]
    whole = "call_convention" in row
    forward_census, backward_census = _census(entry, whole)
    model, inputs = build_entry(entry)
    args, kwargs = _call_args(inputs, whole)
    capture = tl.options.CaptureOptions(backward_ready=True, output_transform=_same)
    trace = tl.trace(model, args, kwargs, capture=capture)
    try:
        beyond_fwd, beyond_bwd = BEYOND_FCM.get(entry.id, (0, 0))
        assert (forward_census.beyond, backward_census.beyond) == (beyond_fwd, beyond_bwd)
        assert backward_census.grouped_dw_excess == GROUPED_DW_EXCESS.get(entry.id, 0)
        assert _forward_family_flops(trace) == row["fcm_fwd"] + beyond_fwd, entry.id

        leaves = [tensor for tensor in _float_leaves(trace.raw_output) if tensor.requires_grad]
        if row["fcm_bwd"] is None and entry.id not in BACKWARD_RAISES:
            # No loss leaf needs a gradient: no backward runs, and none is counted.
            assert not leaves
            status = backward_status(trace)
            assert status.state in ("grad_disabled", "grad_enabled_unobserved")
            assert status.num_backward_passes == 0
            return
        assert leaves
        if entry.id in BACKWARD_RAISES:
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                trace.log_backward(_loss(leaves))
            # The pass raised part way: its backward did not run, and no number is printed.
            status = backward_status(trace)
            assert status.executed_flops is None, status.status_line
            assert "UNKNOWN" in status.status_line
            return
        trace.log_backward(_loss(leaves))
        status = backward_status(trace)
        assert status.state == "backward_observed"
        expected = row["fcm_bwd"] - GROUPED_DW_EXCESS.get(entry.id, 0) + beyond_bwd
        assert status.executed_flops == expected, (entry.id, status.reason)
    finally:
        trace.cleanup()


def test_registered_corpus_accounts_for_every_entry() -> None:
    """The acceptance covers all 207 entries and the oracle's own gaps by name."""

    ids = [entry.id for entry in corpus_entries()]
    assert len(ids) == 207 == len(ROWS) and set(ids) == set(ROWS)
    no_backward = sorted(i for i, row in ROWS.items() if row["fcm_bwd"] is None)
    assert len(no_backward) == 13
    assert set(no_backward) >= BACKWARD_RAISES
    assert {i for i, row in ROWS.items() if row.get("default_backend")} == {
        "Exphormer-GPS-VirtualNode",
        "Paint-Transformer",
        "StyleTTS2",
        "ATDKT",
        "paddledet_detr",
        "ProDESIGN-LE",
        "21cmBERT",
    }
    assert len([row for row in ROWS.values() if "call_convention" in row]) == 8
    assert set(GROUPED_DW_EXCESS) <= set(ROWS) and set(BEYOND_FCM) <= set(ROWS)
