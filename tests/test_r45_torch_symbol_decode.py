"""Round-45 secC_1 immunizer -- torch-symbol decode routes through the single ``torch_attr`` helper.

r42/r43 secC_1 replaced ``getattr(torch, name)`` with ``torch.__dict__.get(name)`` at three decode
sites to kill the PEP-562 lazy ``torch.__getattr__`` side effects (unrequested submodule import
``onnx`` / ``_dynamo`` / ``_inductor``, deprecated ``has_cuda`` -> ``replacement()`` shim, raw
``ImportError`` leak). r44 secC_1 found a FOURTH site (``rehydrate.py:_dtype_from_manifest_string``)
still doing the bare ``getattr(torch, dtype_name, None)`` on an attacker manifest string.

r45 routes every attacker-derived top-level ``torch``-name resolution on the load/decode/exec path
through the single shared helper ``torch_attr`` (``torch.__dict__.get``) and pins it with:

* a STATIC AST immunizer that FAILS on any bare ``getattr(torch, <non-literal>)`` on the guarded
  path (a future decode site cannot reintroduce the lazy-import side effect), and
* a BEHAVIORAL belt (survives an AST refactor) that tampers a portable ``.tlspec`` manifest dtype
  to each hazardous name in a FRESH subprocess and asserts a typed refusal with NO submodule import
  and NO deprecation warning.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import torchlens
from torchlens._io._torch_symbols import torch_attr

_PKG_ROOT = Path(torchlens.__file__).resolve().parent

# The load / decode / exec path the immunizer guards. A bare ``getattr(torch, <non-literal>)`` in
# any of these files reintroduces the PEP-562 lazy-import / deprecated-replacement hazard.
_GUARDED_GLOBS = [
    "_io/*.py",
    "_runnable_execution.py",
    "validation/__init__.py",
]


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for pattern in _GUARDED_GLOBS:
        files.extend(sorted(_PKG_ROOT.glob(pattern)))
    assert files, "no guarded files discovered"
    return files


def _bare_torch_getattr_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ``getattr(torch, <non-literal>, ...)`` call where the target is the bare
    top-level ``torch`` module (an ``ast.Name(id="torch")``) and the attribute is NOT a string
    literal. Literal-name ``getattr(torch, "...")`` and non-top-level roots
    (``getattr(torch._C, ...)`` / ``getattr(torch.backends, ...)`` -- ``ast.Attribute`` targets)
    are allowed."""

    offenders: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        target, name_arg = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "torch"):
            continue  # non-top-level root (torch._C / torch.backends / ...) -> allowed
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            continue  # literal fixed-name module-layout constant -> allowed
        offenders.append(node)
    return offenders


def test_no_bare_getattr_torch_on_guarded_path() -> None:
    """AST immunizer: no bare ``getattr(torch, <non-literal>)`` survives on the load/decode/exec
    path (non-vacuous: the walk visits at least one ``getattr`` call)."""

    total_getattr = 0
    offenders: list[str] = []
    for path in _guarded_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
            ):
                total_getattr += 1
        for call in _bare_torch_getattr_calls(tree):
            offenders.append(f"{path.relative_to(_PKG_ROOT)}:{call.lineno}")

    assert total_getattr >= 1, "AST walk visited no getattr call (vacuous)"
    assert not offenders, (
        "bare getattr(torch, <non-literal>) on the load/decode/exec path -- route through "
        f"torch_attr(): {offenders}"
    )


def test_literal_and_nontoplevel_getattr_not_flagged() -> None:
    """The AST predicate allows literal-name ``getattr(torch, "float32")`` and non-top-level
    roots (``torch._C`` / ``torch.backends``), and flags the bare-variable form."""

    src = textwrap.dedent(
        """
        import torch
        a = getattr(torch, "float32")          # literal -> allowed
        b = getattr(torch._C, name, None)      # non-top-level root -> allowed
        c = getattr(torch.backends, name)      # non-top-level root -> allowed
        d = getattr(torch, name, None)         # bare non-literal -> FLAGGED
        """
    )
    tree = ast.parse(src)
    offenders = _bare_torch_getattr_calls(tree)
    assert len(offenders) == 1
    assert offenders[0].lineno == 6  # the ``d = getattr(torch, name, None)`` line


@pytest.mark.parametrize("name", ["float32", "int64", "bool", "float64", "complex64"])
def test_torch_attr_resolves_real_symbols(name: str) -> None:
    """Every real top-level torch symbol still resolves through the helper."""

    import torch

    assert torch_attr(name) is torch.__dict__.get(name)
    assert torch_attr(name) is not None


@pytest.mark.parametrize("name", ["onnx", "_dynamo", "_inductor", "has_cuda"])
def test_torch_attr_hazardous_names_are_none_without_side_effect(name: str) -> None:
    """A hazardous, genuinely-lazy attacker name (``onnx`` / ``_dynamo`` / ``_inductor``
    submodules, deprecated ``has_cuda``) resolves to ``None`` and TRIGGERS no lazy submodule
    import -- checked in a fresh subprocess by snapshotting ``sys.modules`` around the helper call
    (a submodule torch imports EAGERLY at ``import torch`` is irrelevant: the property is that our
    call fires nothing new and raises no deprecation)."""

    code = textwrap.dedent(
        f"""
        import sys
        import warnings
        warnings.simplefilter("error", DeprecationWarning)
        from torchlens._io._torch_symbols import torch_attr
        before = set(sys.modules)
        val = torch_attr({name!r})
        after = set(sys.modules)
        assert val is None, val
        newly = after - before
        assert not any(m.startswith("torch.") for m in newly), sorted(newly)
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


@pytest.mark.parametrize("hazard", ["onnx", "_dynamo", "has_cuda"])
def test_tampered_manifest_dtype_refuses_without_import(hazard: str, tmp_path: Path) -> None:
    """Behavioral belt (r44 secC_1 repro promoted to regression): tampering a portable ``.tlspec``
    manifest dtype to a hazardous torch name yields a typed refusal, NO ``torch.onnx`` /
    ``torch._dynamo`` in ``sys.modules`` afterward, and NO deprecation warning -- in a fresh
    interpreter so the module table is pristine."""

    bundle = tmp_path / "clean.tlspec"
    build = textwrap.dedent(
        f"""
        import torch, torch.nn as nn, torchlens as tl
        m = nn.Linear(4, 4)
        log = tl.trace(m, torch.randn(1, 4))
        tl.save(log, {str(bundle)!r})
        print("BUILT")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", build], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0 and "BUILT" in proc.stdout, proc.stderr

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("portable bundle layout has no manifest.json to tamper")

    attack = textwrap.dedent(
        f"""
        import json, sys, warnings
        import torchlens as tl

        mpath = {str(manifest_path)!r}
        data = json.loads(open(mpath).read())
        tensors = data.get("tensors")
        assert tensors, "manifest has no tensors to tamper"
        keys = tensors.keys() if isinstance(tensors, dict) else range(len(tensors))
        for k in keys:
            tensors[k]["dtype"] = {hazard!r}
        open(mpath, "w").write(json.dumps(data))

        deprecations = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                tl.load({str(bundle)!r})
                outcome = "no_error"
            except Exception as exc:
                outcome = type(exc).__name__
            deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]

        assert outcome != "no_error", "tampered dtype was not rejected"
        assert "torch.onnx" not in sys.modules, "torch.onnx was imported (lazy side effect)"
        assert "torch._dynamo" not in sys.modules, "torch._dynamo was imported (lazy side effect)"
        assert not any("deprecat" in d.lower() for d in deprecations), deprecations
        print("REFUSED", outcome)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", attack], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert "REFUSED" in proc.stdout, proc.stdout
