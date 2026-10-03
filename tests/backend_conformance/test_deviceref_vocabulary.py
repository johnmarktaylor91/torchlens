"""DeviceRef vocabulary conformance (R17-2).

``ir/refs.py`` documents ``DeviceRef.backend`` as the HARDWARE device class
(``"cpu"``, ``"cuda"``), never the framework namespace ``DtypeRef.backend``
carries. tf and paddle used to write ``DeviceRef(backend="tf"/"paddle", ...)``
directly; every backend must now route through ``DeviceRef.from_value`` (or a
normalizer that does), and this gate keeps the next backend from regressing
the vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from torchlens.backends.paddle.backend import _device_ref_from_paddle_place
from torchlens.backends.tf.modules import device_ref_from_tf_device

BACKENDS_ROOT = Path(__file__).resolve().parents[2] / "torchlens" / "backends"

FRAMEWORK_NAMES = {"torch", "tf", "tensorflow", "keras", "jax", "mlx", "tinygrad", "paddle"}

# Hardware device classes are lowercase identifiers ("cpu", "cuda", "gpu",
# "mps", "tfrt_cpu", ...), never framework namespaces.
_HARDWARE_CLASS = re.compile(r"^[a-z_][a-z0-9_]*$")


def test_no_direct_deviceref_construction_in_backends() -> None:
    """Backends must route device metadata through ``DeviceRef.from_value``.

    A direct ``DeviceRef(...)`` construction is how the framework-name
    vocabulary violation shipped twice; ``from_value`` derives the hardware
    class from the canonical device string and cannot spell it wrong.
    """

    offenders: list[str] = []
    for path in sorted(BACKENDS_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\bDeviceRef\(", text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(BACKENDS_ROOT)}:{line_no}")
    assert offenders == [], (
        "Direct DeviceRef(...) construction in backends/ -- route through "
        f"DeviceRef.from_value or a normalizer that does: {offenders}"
    )


@pytest.mark.smoke_cells("test_tf_device_normalizer_yields_hardware_class[/device:GPU:1-gpu-gpu:1]")
@pytest.mark.parametrize(
    ("raw", "expected_backend", "expected_name"),
    [
        ("/job:localhost/replica:0/task:0/device:CPU:0", "cpu", "cpu:0"),
        ("/device:GPU:1", "gpu", "gpu:1"),
        ("CPU:0", "cpu", "cpu:0"),
    ],
)
def test_tf_device_normalizer_yields_hardware_class(
    raw: str, expected_backend: str, expected_name: str
) -> None:
    ref = device_ref_from_tf_device(raw)
    assert ref is not None
    assert ref.backend == expected_backend
    assert ref.name == expected_name
    assert _HARDWARE_CLASS.match(ref.backend)
    assert ref.backend not in FRAMEWORK_NAMES


@pytest.mark.parametrize("raw", ["", None])
def test_tf_device_normalizer_unknown_is_none(raw: object) -> None:
    assert device_ref_from_tf_device(raw) is None


@pytest.mark.smoke_cells("test_paddle_place_normalizer_yields_hardware_class[Place(cpu)-cpu-cpu]")
@pytest.mark.parametrize(
    ("raw", "expected_backend", "expected_name"),
    [
        ("Place(cpu)", "cpu", "cpu"),
        ("Place(gpu:0)", "gpu", "gpu:0"),
        ("Place(gpu_pinned)", "gpu_pinned", "gpu_pinned"),
    ],
)
def test_paddle_place_normalizer_yields_hardware_class(
    raw: str, expected_backend: str, expected_name: str
) -> None:
    ref = _device_ref_from_paddle_place(raw)
    assert ref is not None
    assert ref.backend == expected_backend
    assert ref.name == expected_name
    assert _HARDWARE_CLASS.match(ref.backend)
    assert ref.backend not in FRAMEWORK_NAMES


def test_paddle_place_normalizer_unknown_is_none() -> None:
    assert _device_ref_from_paddle_place(None) is None


@pytest.mark.smoke
def test_unknown_module_identity_mode_refuses_typed_on_all_five_previews() -> None:
    """An unknown module_identity_mode is a CAPABILITY refusal on every preview.

    jax historically raised a bare ValueError while the four siblings raised
    BackendUnsupportedError (R17-7); the resolvers are pure functions, so the
    parity is pinned runtime-free.
    """

    from torchlens.backends import BackendUnsupportedError
    from torchlens.backends.jax.backend import _resolve_jax_module_identity_mode
    from torchlens.backends.mlx.backend import _resolve_mlx_module_identity_mode
    from torchlens.backends.paddle.backend import _resolve_paddle_module_identity_mode
    from torchlens.backends.tf.backend import _resolve_tf_module_identity_mode
    from torchlens.backends.tinygrad.backend import _resolve_tinygrad_module_identity_mode

    for resolver in (
        _resolve_jax_module_identity_mode,
        _resolve_mlx_module_identity_mode,
        _resolve_paddle_module_identity_mode,
        _resolve_tf_module_identity_mode,
        _resolve_tinygrad_module_identity_mode,
    ):
        with pytest.raises(BackendUnsupportedError, match="module_identity_mode"):
            resolver("bogus_mode", None)
