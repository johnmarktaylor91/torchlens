"""Regression tests for r20l IR docstring/consistency polish (N11, N12).

These lock the documented (and deliberately divergent) semantics of the
same-named ``backend`` field on ``DtypeRef`` vs ``DeviceRef`` (N11) and the
presence of the ``output_of_module_calls`` field plus its docstring entry on
``RecordContext`` (N12). They are behavior-preserving contract locks: the field
values are unchanged; the tests fail if a future edit repurposes a field or drops
the clarifying documentation.
"""

from __future__ import annotations

import dataclasses

import torch

from torchlens._ir.predicate import RecordContext
from torchlens._ir.refs import DeviceRef, DtypeRef


class TestBackendFieldDivergence:
    """N11: ``backend`` means framework on DtypeRef but hardware class on DeviceRef."""

    def test_dtyperef_backend_is_framework_namespace(self) -> None:
        ref = DtypeRef.from_value(torch.float32)
        assert ref is not None
        assert ref.backend == "torch"
        assert ref.name == "torch.float32"

    def test_dtyperef_backend_unknown_without_namespace(self) -> None:
        ref = DtypeRef.from_value("float32")
        assert ref is not None
        assert ref.backend == "unknown"

    def test_deviceref_backend_is_hardware_class_not_framework(self) -> None:
        cpu = DeviceRef.from_value(torch.device("cpu"))
        assert cpu is not None
        # Hardware device class, NOT the framework ("torch").
        assert cpu.backend == "cpu"

        cuda = DeviceRef.from_value("cuda:0")
        assert cuda is not None
        assert cuda.backend == "cuda"
        assert cuda.name == "cuda:0"

    def test_backend_semantics_diverge_between_refs(self) -> None:
        # The whole point of N11: identical field name, different meaning.
        dtype_backend = DtypeRef.from_value(torch.float32).backend  # type: ignore[union-attr]
        device_backend = DeviceRef.from_value("cuda:0").backend  # type: ignore[union-attr]
        assert dtype_backend == "torch"
        assert device_backend == "cuda"
        assert dtype_backend != device_backend

    def test_divergence_is_documented_on_both_classes(self) -> None:
        # Docstring clarification must survive future edits.
        assert DtypeRef.__doc__ is not None
        assert DeviceRef.__doc__ is not None
        assert "DeviceRef.backend" in DtypeRef.__doc__
        assert "DtypeRef.backend" in DeviceRef.__doc__


class TestRecordContextOutputOfModuleCalls:
    """N12: ``output_of_module_calls`` is a real field and must be documented."""

    def test_field_exists(self) -> None:
        field_names = {f.name for f in dataclasses.fields(RecordContext)}
        assert "output_of_module_calls" in field_names

    def test_field_is_documented(self) -> None:
        assert RecordContext.__doc__ is not None
        assert "output_of_module_calls" in RecordContext.__doc__
