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


def _guarded_files() -> list[Path]:
    """r47 secD_1/secF_1: the immunizer now guards the ENTIRE ``torchlens/**/*.py`` package (zero
    per-line exemptions), not just the r45 load/decode subset. The ``.run()`` random-init dtype
    site (``_runnable_state.py:_torch_dtype``) lived at the package ROOT and was invisible to the
    r45 ``_io/`` glob -- so its tripwire was vacuous there. Any bare ``getattr(torch, <var>)`` /
    ``hasattr(torch, <var>)`` ANYWHERE in the package reintroduces the PEP-562 lazy-import /
    deprecated-replacement hazard."""

    files = sorted(_PKG_ROOT.rglob("*.py"))
    assert files, "no package files discovered"
    return files


def _name_arg_is_literal_str(name_arg: ast.expr) -> bool:
    """A fixed-name string literal carries no lazy hazard; an f-string / ``.format()`` name is
    dynamic and MUST route through ``torch_attr`` (it fires ``torch.__getattr__`` just the same)."""

    return isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)


def _flagged_torch_probe_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ``getattr(torch, <non-literal>, ...)`` OR ``hasattr(torch, <non-literal>)``
    call where the target is the bare top-level ``torch`` module (an ``ast.Name(id="torch")``) and
    the attribute is NOT a string literal (an f-string ``ast.JoinedStr`` or a ``.format()`` call is
    non-literal -> flagged). Literal-name ``getattr(torch, "...")`` and non-top-level roots
    (``getattr(torch._C, ...)`` / ``getattr(torch.backends, ...)`` -- ``ast.Attribute`` targets)
    are allowed."""

    offenders: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id in ("getattr", "hasattr")):
            continue
        if len(node.args) < 2:
            continue
        target, name_arg = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "torch"):
            continue  # non-top-level root (torch._C / torch.backends / ...) -> allowed
        if _name_arg_is_literal_str(name_arg):
            continue  # literal fixed-name module-layout constant -> allowed
        offenders.append(node)
    return offenders


def test_no_bare_getattr_torch_anywhere_in_package() -> None:
    """AST immunizer: no bare ``getattr(torch, <non-literal>)`` / ``hasattr(torch, <non-literal>)``
    survives ANYWHERE in ``torchlens/**/*.py`` (non-vacuous: >300 files, >=1 probe call visited)."""

    files = _guarded_files()
    assert len(files) > 300, f"expected a whole-package scan, saw {len(files)} files (vacuous)"

    total_probes = 0
    offenders: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
            ):
                total_probes += 1
        for call in _flagged_torch_probe_calls(tree):
            offenders.append(f"{path.relative_to(_PKG_ROOT)}:{call.lineno}")

    assert total_probes >= 1, "AST walk visited no getattr/hasattr call (vacuous)"
    assert not offenders, (
        "bare getattr(torch, <non-literal>) / hasattr(torch, <non-literal>) in the package -- "
        f"route through torch_attr(): {offenders}"
    )


def test_literal_and_nontoplevel_probes_not_flagged() -> None:
    """The AST predicate allows literal-name ``getattr(torch, "float32")`` and non-top-level
    roots (``torch._C`` / ``torch.backends``), and flags the bare-variable ``getattr`` / ``hasattr``
    and f-string / ``.format()`` name forms."""

    src = textwrap.dedent(
        """
        import torch
        a = getattr(torch, "float32")          # literal -> allowed
        b = getattr(torch._C, name, None)      # non-top-level root -> allowed
        c = getattr(torch.backends, name)      # non-top-level root -> allowed
        d = getattr(torch, name, None)         # bare non-literal -> FLAGGED
        e = hasattr(torch, name)               # bare hasattr -> FLAGGED
        f = getattr(torch, f"{name}", None)    # f-string name -> FLAGGED
        g = getattr(torch, "d{}".format(name)) # .format() name -> FLAGGED
        """
    )
    tree = ast.parse(src)
    offenders = _flagged_torch_probe_calls(tree)
    flagged_lines = sorted(call.lineno for call in offenders)
    assert flagged_lines == [6, 7, 8, 9], flagged_lines


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


@pytest.mark.parametrize("hazard", ["onnx", "_dynamo", "has_cuda"])
def test_tampered_run_slot_dtype_refuses_without_import(hazard: str, tmp_path: Path) -> None:
    """r47 secD_1/secF_1 behavioral belt: the ``.run()`` random-init path resolves a tensor-slot
    dtype via ``_runnable_state._torch_dtype`` (``_runnable_state.py:908``) -- a package-ROOT site
    the r45 ``_io/`` glob missed. Tampering ``run.tensor_slots[*].dtype`` to a hazardous torch name
    yields a typed refusal (parse-validated at load, defense-in-depth; the run-path belt is also
    ``torch_attr``) with NO ``torch.onnx`` / ``torch._dynamo`` in ``sys.modules`` and NO deprecation
    warning -- in a FRESH interpreter so the module table is pristine."""

    bundle = tmp_path / "runnable.tlspec"
    build = textwrap.dedent(
        f"""
        import torch, torch.nn as nn, torchlens as tl
        from torchlens.options import CaptureOptions
        m = nn.Linear(4, 4)
        log = tl.trace(
            m, torch.randn(2, 4),
            capture=CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False),
        )
        log.save({str(bundle)!r}, level="runnable")  # include_weights=False -> random init at run
        print("BUILT")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", build], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0 and "BUILT" in proc.stdout, proc.stderr

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("runnable bundle layout has no manifest.json to tamper")

    attack = textwrap.dedent(
        f"""
        import json, sys, warnings

        mpath = {str(manifest_path)!r}
        data = json.loads(open(mpath).read())

        def _tamper_slots(obj):
            n = 0
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("tensor_slots", "slots") and isinstance(v, list):
                        for s in v:
                            if isinstance(s, dict) and "dtype" in s:
                                s["dtype"] = {hazard!r}
                                n += 1
                    else:
                        n += _tamper_slots(v)
            elif isinstance(obj, list):
                for v in obj:
                    n += _tamper_slots(v)
            return n

        tampered = _tamper_slots(data.get("run", data))
        assert tampered >= 1, "no tensor slots found to tamper"
        open(mpath, "w").write(json.dumps(data))

        import torchlens as tl
        import torch
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                loaded = tl.load({str(bundle)!r})
                loaded.run(inputs=torch.randn(2, 4))
                outcome = "no_error"
            except Exception as exc:
                outcome = type(exc).__name__
            deprecations = [
                str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
            ]

        assert outcome != "no_error", "tampered slot dtype was not rejected"
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
