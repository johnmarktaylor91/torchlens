"""R23-3: ParamRef construction pairs each RAW parent-param with ITS barcode.

``_param_barcodes`` is the DEDUPED key list of ``parent_param_ops`` (a dict)
while ``parent_params`` is the raw occurrence list, so the old
``zip(barcodes, params)`` misaligned every pairing after a repeated param
(weight-tied einsum / hypernetwork class): ``[A, B, A, C]`` persisted
``ParamRef(barcode=C, geometry-of-A)`` and silently dropped a ref while
``num_params`` said four.
"""

import pytest
import torch
from torch import nn

import torchlens  # noqa: F401 -- full package init before backend submodule imports
import torchlens.backends.torch.ops  # noqa: F401 -- ops.py must initialize before _ops_capture_records


def _fields_dict_for(params: list[nn.Parameter]) -> dict:
    from torchlens.backends.torch.tensor_tracking import _process_parent_param_ops

    parent_param_ops = _process_parent_param_ops(params)
    return {
        "parent_params": params,
        "_param_barcodes": list(parent_param_ops.keys()),
    }


def test_tied_param_occurrences_each_get_their_own_ref() -> None:
    from torchlens.backends.torch._ops_capture_records import _param_refs_from_fields
    from torchlens.backends.torch._tl import get_param_meta

    param_a = nn.Parameter(torch.randn(2, 3))
    param_b = nn.Parameter(torch.randn(4))
    param_c = nn.Parameter(torch.randn(5, 1))
    raw = [param_a, param_b, param_a, param_c]

    refs = _param_refs_from_fields(_fields_dict_for(raw))

    assert len(refs) == 4, "every raw occurrence gets a ref"
    expected_barcodes = [get_param_meta(p).param_barcode for p in raw]
    assert [ref.barcode for ref in refs] == expected_barcodes
    assert [ref.shape for ref in refs] == [tuple(p.shape) for p in raw]
    # The tied occurrences share one barcode and their own (identical) geometry.
    assert refs[0].barcode == refs[2].barcode
    assert refs[0].shape == refs[2].shape == (2, 3)


def test_unbarcoded_param_refuses_loudly() -> None:
    from torchlens.backends.torch._ops_capture_records import _param_refs_from_fields

    stray = nn.Parameter(torch.randn(2))
    with pytest.raises(RuntimeError, match="barcode"):
        _param_refs_from_fields({"parent_params": [stray], "_param_barcodes": []})
