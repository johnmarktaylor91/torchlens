"""Shared TensorFlow backend test configuration."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from os.path import join as opj
from pathlib import Path
from typing import Any

import pytest
from packaging.version import InvalidVersion, Version

TESTS_DIR = str(Path(__file__).resolve().parents[1])
TEST_OUTPUTS_DIR = opj(TESTS_DIR, "test_outputs")
REPORTS_DIR = opj(TEST_OUTPUTS_DIR, "reports")
VIS_OUTPUT_DIR = opj(TEST_OUTPUTS_DIR, "visualizations")

_MIN_TENSORFLOW_VERSION = Version("2.16")
_MIN_KERAS_MAJOR = 3
_TF_CPU_CONFIGURED = False


def _installed_version(package_name: str) -> Version | None:
    """Return an installed package version when available.

    Parameters
    ----------
    package_name
        Distribution package name.

    Returns
    -------
    Version | None
        Parsed package version, or ``None`` when the package is unavailable or unparsable.
    """

    try:
        return Version(version(package_name))
    except PackageNotFoundError:
        return None
    except InvalidVersion as exc:
        raise pytest.UsageError(f"{package_name} has an unparsable version: {exc}") from exc


def _unsupported_tensorflow_reason() -> str | None:
    """Return the TensorFlow backend test skip reason for unsupported stacks.

    Returns
    -------
    str | None
        Skip reason for unsupported stacks, or ``None`` when the stack is supported.
    """

    tensorflow_version = _installed_version("tensorflow")
    keras_version = _installed_version("keras")

    if tensorflow_version is None:
        return "tensorflow is not installed"
    if keras_version is None:
        return "keras is not installed"
    if tensorflow_version < _MIN_TENSORFLOW_VERSION:
        return f"TensorFlow backend tests require tensorflow>={_MIN_TENSORFLOW_VERSION}"
    if keras_version.major < _MIN_KERAS_MAJOR:
        return "TensorFlow backend tests require Keras 3"
    return None


def _force_tensorflow_cpu(tf: Any) -> None:
    """Hide GPUs from TensorFlow without changing process-wide CUDA environment.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    """

    global _TF_CPU_CONFIGURED

    if _TF_CPU_CONFIGURED:
        return

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        visible_gpus: list[Any] = tf.config.get_visible_devices("GPU")
        if visible_gpus:
            raise
    _TF_CPU_CONFIGURED = True


def tensorflow_backend_modules() -> tuple[Any, Any, str | None]:
    """Return TensorFlow/Keras modules plus the backend test skip reason.

    Returns
    -------
    tuple[Any, Any, str | None]
        Imported TensorFlow and Keras modules, plus a skip reason for unsupported stacks.
    """

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    skip_reason = _unsupported_tensorflow_reason()
    if skip_reason is None:
        _force_tensorflow_cpu(tf)
    return tf, keras, skip_reason
