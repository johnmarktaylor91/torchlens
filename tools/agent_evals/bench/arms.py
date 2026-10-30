"""Per-arm environments built from one base, with an import audit.

Every arm starts from the same base requirements (interpreter, torch, numerical libraries) and
adds at most one interpretability library. Each arm is a separate virtual environment built with
uv; the audit then proves the arm holds its own library and none of the others, records the
frozen package list and its hash, and installs the import guard. Agent material (the context file
each arm's agent reads) is chosen per arm and copied into the work folder by the runner.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import guards

PYTHON = "3.10"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
#: The base every arm shares; the same pins as the repository's enforcing CI row.
BASE_REQUIREMENTS: tuple[str, ...] = (
    "torch==2.13.0+cpu",
    "torchvision==0.28.0+cpu",
    "transformers==5.18.0",
    "numpy",
    "scipy",
)
#: Import names of every interpretability library any arm may hold (and others must not).
LIBRARY_IMPORTS: tuple[str, ...] = (
    "torchlens",
    "nnsight",
    "transformer_lens",
    "captum",
    "pyvene",
    "baukit",
)


@dataclass(frozen=True)
class Arm:
    """One benchmark arm.

    Attributes
    ----------
    arm_id:
        ``A`` to ``D``.
    library:
        Import name of the arm's interpretability library, or ``None`` for plain PyTorch.
    requirements:
        Requirements added on top of the base. ``{repo}`` is replaced by the repository root.
    material:
        Path of the arm's agent material, relative to the material root (``None`` until written).
    """

    arm_id: str
    description: str
    library: str | None
    requirements: tuple[str, ...]
    material: str | None

    def forbidden_imports(self) -> list[str]:
        """Return the library imports this arm must not be able to import."""

        return [name for name in LIBRARY_IMPORTS if name != self.library]


ARMS: dict[str, Arm] = {
    "A": Arm("A", "plain PyTorch and numerical libraries", None, (), "arm-a-pytorch.md"),
    "B": Arm(
        "B", "A plus the library under test and its agent index", "torchlens", ("{repo}",), None
    ),
    "C": Arm(
        "C", "A plus nnsight, local execution only", "nnsight", ("nnsight",), "nnsight/nnsight.md"
    ),
    "D": Arm(
        "D",
        "A plus TransformerLens",
        "transformer_lens",
        ("transformer_lens",),
        "transformerlens/transformerlens.md",
    ),
}


def _python(env: Path) -> Path:
    return env / "bin" / "python"


def _run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, check=True, **kwargs)


def site_packages(python: Path) -> Path:
    """Return the environment's purelib directory."""

    out = _run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
    return Path(out.stdout.strip())


def build(arm: Arm, dest: Path, repo: Path, *, natives: Mapping[str, Any] | None = None) -> Path:
    """Create the arm's environment at ``dest`` and install the fail-closed guard.

    Returns
    -------
    Path
        The environment's interpreter.
    """

    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to build arm environments")
    _run(["uv", "venv", "--clear", "-p", PYTHON, str(dest)])
    python = _python(dest)
    extra = [req.replace("{repo}", str(repo)) for req in arm.requirements]
    _run(
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
            *BASE_REQUIREMENTS,
            *extra,
        ]
    )
    guards.install(site_packages(python), guards.union_for(arm.arm_id, natives))
    return python


AUDIT_PROBE = """
import importlib.util, json, sys
print(json.dumps({name: importlib.util.find_spec(name) is not None for name in sys.argv[1:]}))
"""


def audit(arm: Arm, python: Path) -> dict[str, Any]:
    """Audit one built environment: which libraries import, the frozen packages and their hash.

    ``ok`` is true when the arm's own library is importable (if it has one) and no other
    interpretability library is.
    """

    found = json.loads(_run([str(python), "-c", AUDIT_PROBE, *LIBRARY_IMPORTS]).stdout)
    freeze = _run(["uv", "pip", "freeze", "--python", str(python)]).stdout
    leaked = sorted(name for name in arm.forbidden_imports() if found.get(name))
    missing = bool(arm.library) and not found.get(arm.library or "")
    return {
        "arm": arm.arm_id,
        "libraries_found": found,
        "leaked": leaked,
        "own_library_missing": missing,
        "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "freeze": freeze.splitlines(),
        "ok": not leaked and not missing,
    }


def base_drift(audits: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return, per arm, base packages whose pinned version differs from arm A's.

    Arms share one base; a peer library that moves a base package (torch, transformers) shows
    here and is reported, never hidden.
    """

    def pins(lines: Sequence[str]) -> dict[str, str]:
        return dict(line.split("==", 1) for line in lines if "==" in line)

    by_arm = {a["arm"]: pins(a["freeze"]) for a in audits}
    base = by_arm.get("A", {})
    names = {req.split("==")[0].split("[")[0] for req in BASE_REQUIREMENTS}
    return {
        arm: sorted(f"{n} {base.get(n)} -> {p.get(n)}" for n in names if p.get(n) != base.get(n))
        for arm, p in by_arm.items()
        if arm != "A"
    }
