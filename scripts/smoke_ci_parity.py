"""Run the commit-level smoke gate in the environment of CI's enforcing smoke row.

A plain ``pytest tests/ -m smoke`` on a development box does not give the
same answer as the GitHub Tests workflow, for two environment reasons:

* CI installs the ``dev``, ``tabular`` and ``viz`` extras on every row. A venv
  built from ``.[dev]`` or ``.[test]`` lacks pandas, so every smoke test that
  reaches a tabular surface (profiles, receptive fields, gradient audits,
  report families) fails with the documented "install torchlens[tabular]"
  ImportError.
* The env-fingerprinted golden families (``tests/_oracle_env.py``) enforce
  byte identity only on the environment the goldens were recorded under, and
  fail closed on any other non-CI environment (no committed ``env-*``
  baseline, by design: a dev box must never self-baseline). The CI row that
  matches that environment declares ``TORCHLENS_ORACLE_ENFORCE=1``.

This script builds (once, then reuses) a venv pinned to that enforcing row
and runs the same commands as the workflow's "Run smoke tests and
compatibility coverage" and "Fail on an under-executed smoke suite" steps.
The pins below are kept in lockstep with ``.github/workflows/tests.yml`` by
``tests/test_ci_packaging_gates.py``. Requires ``uv`` and the Graphviz
``dot`` binary.

Usage::

    python scripts/smoke_ci_parity.py              # 4 xdist workers, as CI
    python scripts/smoke_ci_parity.py -n 8         # more workers
    python scripts/smoke_ci_parity.py --reinstall  # rebuild the venv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The Tests workflow smoke row that declares ``oracle_enforce: "1"``.
PYTHON = "3.10"
TORCH = "2.13.0+cpu"
TORCHVISION = "0.28.0+cpu"
TRANSFORMERS = "5.18.0"
NUMPY_SPEC = "numpy"

#: The workflow's "Install dependencies" step: extras and byte-emitter pins.
EXTRAS = ".[dev,tabular,viz]"
EXTRA_PINS = ("pydot==4.0.1", "graphviz==0.21")
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"

#: The smoke step's environment (``CI`` is what GitHub Actions sets itself).
SMOKE_ENV = {"CI": "true", "OMP_NUM_THREADS": "1", "TORCHLENS_ORACLE_ENFORCE": "1"}

#: The executed-floor attestation step's arguments.
EXECUTED_FLOOR = "710"
SKIP_FRACTION = "0.15"

DEFAULT_VENV = PROJECT_ROOT / ".venv-ci-smoke"
_STAMP_NAME = "torchlens-ci-smoke.stamp"


def install_spec() -> list[str]:
    """Return the ``uv pip install`` package arguments CI's enforcing row uses."""

    return [
        "-e",
        EXTRAS,
        f"torch=={TORCH}",
        f"torchvision=={TORCHVISION}",
        NUMPY_SPEC,
        *EXTRA_PINS,
        f"transformers=={TRANSFORMERS}",
    ]


def _venv_python(venv: Path) -> Path:
    """Return the interpreter path inside ``venv``."""

    return venv / "bin" / "python"


def ensure_venv(venv: Path, reinstall: bool) -> Path:
    """Create or refresh the pinned venv; return its interpreter.

    Parameters
    ----------
    venv:
        Venv directory to create or reuse.
    reinstall:
        Rebuild from scratch even when the stamp matches.

    Returns
    -------
    Path
        The venv's python executable.
    """

    if shutil.which("uv") is None:
        raise SystemExit("smoke_ci_parity: uv is required (CI installs with uv)")
    stamp_text = f"python {PYTHON}\n" + "\n".join(install_spec()) + "\n"
    stamp = venv / _STAMP_NAME
    if not reinstall and stamp.exists() and stamp.read_text() == stamp_text:
        return _venv_python(venv)
    subprocess.run(["uv", "venv", "--clear", "-p", PYTHON, str(venv)], check=True)
    python = _venv_python(venv)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--index-url",
            TORCH_INDEX,
            "--extra-index-url",
            PYPI_INDEX,
            "--index-strategy",
            "unsafe-best-match",
            *install_spec(),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    stamp.write_text(stamp_text)
    return python


def serial_test_files() -> list[str]:
    """Return the test files carrying the ``serial`` marker, as CI collects them."""

    pattern = re.compile(r"mark\.serial")
    return sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "tests").rglob("test_*.py")
        if pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    )


def run_gate(python: Path, workers: int, extra_args: list[str]) -> int:
    """Run CI's smoke-step commands in order, stopping at the first failure.

    Parameters
    ----------
    python:
        Interpreter of the pinned venv.
    workers:
        xdist worker count for the main smoke selection.
    extra_args:
        Extra pytest arguments for the main smoke selection.

    Returns
    -------
    int
        Exit status of the first failing command, else 0.
    """

    env = {**os.environ, **SMOKE_ENV}
    pytest = [str(python), "-m", "pytest"]
    with tempfile.TemporaryDirectory(prefix="torchlens-ci-smoke-") as tmp:
        junit = str(Path(tmp) / "smoke.junit.xml")
        commands = [
            [*pytest, "tests/", "-m", "smoke", "-n", str(workers), "--tb=short", "-q"]
            + [f"--junitxml={junit}", *extra_args],
            [*pytest, *serial_test_files(), "-m", "serial and not slow and not rare and not heavy"]
            + ["--tb=short", "-q"],
            [*pytest, "tests/test_torch_compat.py", "--tb=short", "-q"],
            [str(python), "scripts/check_ci_executed_tests.py", junit, EXECUTED_FLOOR]
            + [SKIP_FRACTION],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
            if result.returncode != 0:
                return result.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, prepare the venv and run the gate."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--workers", type=int, default=4, help="xdist workers (CI: 4)")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="pinned venv directory")
    parser.add_argument("--reinstall", action="store_true", help="rebuild the venv")
    parser.add_argument("pytest_args", nargs="*", help="extra args for the main smoke run")
    args = parser.parse_args(argv)
    if shutil.which("dot") is None:
        raise SystemExit("smoke_ci_parity: Graphviz `dot` is required (CI apt-installs it)")
    python = ensure_venv(args.venv.resolve(), args.reinstall)
    return run_gate(python, args.workers, args.pytest_args)


if __name__ == "__main__":
    sys.exit(main())
