"""Safetensors export of a Trace (R2-08, lane EXTRACT).

``tl.export.safetensors(trace, path)`` writes one pickle-free safetensors
file and ``tl.io.read_safetensors(path)`` reads it back: every saved payload
round-trips bit-equal (complex included), every writer-contract KEEP field
of the trace and op records is either decoded to its original value or
disclosed as omitted, and the file is pure safetensors structure.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._state_adapter import state_items
from torchlens._errors import InvalidArgumentError
from torchlens._io.writer_contract import writer_contract, writer_contract_digest


class _Net(nn.Module):
    """Small net with a complex-valued intermediate."""

    def __init__(self) -> None:
        """Build the layers deterministically."""

        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear, relu, an FFT round trip, linear."""

        hidden = torch.relu(self.fc1(x))
        spectrum = torch.fft.fft(hidden)
        return self.fc2(torch.fft.ifft(spectrum).real) * 1.0


def _trace() -> Any:
    """Capture the net with every activation saved."""

    x = torch.randn(3, 6, generator=torch.Generator().manual_seed(2))
    return tl.trace(_Net().eval(), x, capture=tl.options.CaptureOptions(layers_to_save="all"))


def _same(left: Any, right: Any) -> bool:
    """Structural equality that treats NaN as equal to NaN."""

    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    return bool(left == right)


#: Trace fields ``tl.save``'s source-privacy policy rewrites (compared separately).
_SOURCE_FILE_FIELDS = {
    "class_source_file",
    "init_source_file",
    "forward_source_file",
    "backward_source_file",
}
_PRIVACY_FIELDS = _SOURCE_FILE_FIELDS | {
    "init_signature",
    "forward_signature",
    "backward_signature",
    "annotations",
    "_source_code_blob",
}


def _keep(record: str) -> set[str]:
    """KEEP-policy field names of one writer-contract record."""

    policies = writer_contract()["record_contracts"][record]["field_policies"]
    return {name for name, policy in policies.items() if policy == "keep"}


def test_trace_round_trips_through_safetensors(tmp_path: Path) -> None:
    """Payloads bit-equal; KEEP fields decoded equal or disclosed omitted."""

    trace = _trace()
    path = tl.export.safetensors(trace, tmp_path / "trace.safetensors")
    loaded = tl.io.read_safetensors(path)

    assert loaded.header["schema"] == "tl_trace_safetensors_v1"
    assert loaded.header["writer_contract_digest"] == writer_contract_digest()
    assert loaded.header["capture_honesty"]
    n_payloads = 0
    saw_complex = False
    for label in trace.op_labels:
        state = dict(state_items(trace.ops[label]))
        out = state.get("out")
        if isinstance(out, torch.Tensor):
            stored = loaded.tensors[label]["out"]
            assert stored.dtype == out.dtype and stored.shape == out.shape
            assert torch.equal(stored, out.detach().cpu())
            n_payloads += 1
            saw_complex = saw_complex or out.is_complex()
        exported = loaded.op_fields[label]
        omitted = loaded.omitted["op"].get(label, {})
        for name in _keep("op") & set(state):
            assert (name in exported) != (name in omitted), name
            if name in exported:
                assert _same(exported[name], state[name]), (label, name)
    assert n_payloads >= 5
    assert saw_complex, "the FFT intermediate exercises the complex encoding"
    trace_state = dict(state_items(trace))
    for name in _keep("trace") & set(trace_state):
        in_fields = name in loaded.trace_fields
        assert in_fields != (name in loaded.omitted["trace"]), name
        if not in_fields or name in _PRIVACY_FIELDS:
            continue
        assert _same(loaded.trace_fields[name], trace_state[name]), name
    for name in _SOURCE_FILE_FIELDS & set(loaded.trace_fields):
        raw = trace_state[name]
        assert loaded.trace_fields[name] == (Path(raw).name if raw else raw), name


def test_export_embeds_no_host_path(tmp_path: Path) -> None:
    """tl.save's source-privacy policy applies: no absolute path reaches the file."""

    path = tl.export.safetensors(_trace(), tmp_path / "trace.safetensors")
    data = path.read_bytes()
    (header_len,) = struct.unpack("<Q", data[:8])
    header_text = data[8 : 8 + header_len].decode("utf-8")
    for host_path in (str(Path(__file__).resolve().parent), str(Path.home())):
        assert host_path not in header_text


def test_export_file_is_pure_safetensors_with_no_pickle(tmp_path: Path) -> None:
    """Header length + JSON header + exactly the tensor bytes, nothing else."""

    path = tl.export.safetensors(_trace(), tmp_path / "trace.safetensors")
    data = path.read_bytes()
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len])
    ends = [entry["data_offsets"][1] for name, entry in header.items() if name != "__metadata__"]
    assert len(data) == 8 + header_len + max(ends)
    assert set(header["__metadata__"]) == {"torchlens"}
    assert all(
        set(entry) == {"dtype", "shape", "data_offsets"}
        for name, entry in header.items()
        if name != "__metadata__"
    )


def test_export_registry_row_and_refusals(tmp_path: Path) -> None:
    """The door is a present-tier export target; misuse refuses typed."""

    assert tl.export.export_target_info("safetensors").capabilities["tier"] == "present"
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.export.safetensors(object(), tmp_path / "x.safetensors")
    assert excinfo.value.fields["code"] == "trace_safetensors_trace_required"
    from safetensors.torch import save_file

    foreign = tmp_path / "foreign.safetensors"
    save_file({"w": torch.zeros(2)}, str(foreign))
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.io.read_safetensors(foreign)
    assert excinfo.value.fields["code"] == "trace_safetensors_invalid"


def _rewrite_header(path: Path, **changes: Any) -> None:
    """Rewrite the export's header document in place (tamper helper)."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(str(path), framework="pt") as handle:
        document = json.loads(handle.metadata()["torchlens"])
        names = list(handle.keys())
        tensors = {name: handle.get_tensor(name) for name in names}
    document.update(changes)
    save_file(tensors, str(path), metadata={"torchlens": json.dumps(document)})


def test_reader_refuses_versions_and_digests_it_cannot_honour(tmp_path: Path) -> None:
    """A newer tlspec version or a non-digest refuses; a current file says so."""

    path = tl.export.safetensors(_trace(), tmp_path / "trace.safetensors")
    assert tl.io.read_safetensors(path).header["writer_contract_current"] is True
    for changes, code in [
        ({"tlspec_version": 99}, "trace_safetensors_version_unsupported"),
        ({"writer_contract_digest": "not-a-digest"}, "trace_safetensors_invalid"),
        ({"schema": "tl_trace_safetensors_v0"}, "trace_safetensors_invalid"),
    ]:
        target = tmp_path / f"tampered-{code}-{len(changes)}.safetensors"
        target.write_bytes(path.read_bytes())
        _rewrite_header(target, **changes)
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.io.read_safetensors(target)
        assert excinfo.value.fields["code"] == code


def test_export_writes_only_keep_fields(tmp_path: Path) -> None:
    """No DROP or session-time field ever reaches the file."""

    loaded = tl.io.read_safetensors(tl.export.safetensors(_trace(), tmp_path / "t.safetensors"))
    assert set(loaded.trace_fields) <= _keep("trace")
    for fields in loaded.op_fields.values():
        assert set(fields) <= _keep("op")
