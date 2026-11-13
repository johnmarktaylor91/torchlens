"""TensorFlow replay adapter contract checks that need no TensorFlow runtime."""

from __future__ import annotations

import subprocess
import sys

from torchlens.backends._replay_adapters import get_adapter, tf as adapter
from torchlens.backends.tf._raw_op_replay import _raw_op_dispatch
from torchlens.backends.tf.validation import (
    _ANNOTATION_OPS,
    _PURE_REPLAY_ALLOWLIST,
    _STRUCTURAL_VALUE_OPS,
    _VALUE_OPS,
)


def test_validator_kernel_set_matches_the_validator_dispatch_and_allowlist() -> None:
    """The adapter defers to the validator for exactly the validator's own kernels."""

    assert frozenset(_raw_op_dispatch()) == adapter._VALIDATOR_KERNEL_OPS
    assert adapter._VALIDATOR_KERNEL_OPS == _PURE_REPLAY_ALLOWLIST


def test_extra_kernels_cover_only_types_the_validator_classifies() -> None:
    """Extra kernels replay classified value, structural or annotation types, nothing new."""

    classified = _VALUE_OPS | _STRUCTURAL_VALUE_OPS | _ANNOTATION_OPS
    assert classified >= adapter._EXTRA_KERNEL_OPS
    assert not adapter._EXTRA_KERNEL_OPS & adapter._VALIDATOR_KERNEL_OPS
    assert adapter.KERNEL_OPS | {adapter.INERT_IDENTITY} == adapter.ALLOWLIST


def test_get_adapter_returns_a_complete_module() -> None:
    """The package lookup finds every contract function on the TensorFlow module."""

    assert get_adapter("tf") is adapter


def test_importing_the_adapter_does_not_import_tensorflow() -> None:
    """Framework imports stay inside the adapter functions."""

    code = (
        "import sys, torchlens; before = 'tensorflow' in sys.modules; "
        "import torchlens.backends._replay_adapters.tf; "
        "print(before, 'tensorflow' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    before, after = result.stdout.strip().splitlines()[-1].split()
    assert after == before


class _FakeDType:
    as_datatype_enum = 3


def test_attr_encoding_is_data_only() -> None:
    """Bytes, dtypes, nested lists and tuples encode to plain data; unknowns stay opaque."""

    assert adapter._encode_attr(b"SAME") == {"tf": "bytes", "hex": "53414d45"}
    assert adapter._encode_attr(_FakeDType()) == {"tf": "dtype", "enum": 3}
    assert adapter._encode_attr([1, (2, 3)]) == [1, (2, 3)]
    assert adapter._encode_attr(object()) == {"tf": "opaque", "type": "object"}
    assert adapter._decode_attr(None, {"tf": "bytes", "hex": "53414d45"}) == b"SAME"
    opaque = adapter._decode_attr(None, {"tf": "opaque", "type": "object"})
    assert isinstance(opaque, adapter._OpaqueAttr)
