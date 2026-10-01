"""Env-flag doctrine: one parser, one registry, "true" never means OFF.

WT1 A-VI item 34: every torchlens-owned boolean environment knob parses
through ``closed_bool_env`` (or a reason-ledgered inline closed-vocabulary
parser), and every ``TORCHLENS_*`` environment READ in the package has a row
in ``ENV_FLAG_REGISTRY``. The AST lint below derives the read universe from
the source, so an unregistered knob or a fresh raw ``os.environ`` boolean
comparison is red the session it lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.errors import CaptureContextError
from torchlens.utils.env_flags import ENV_FLAG_REGISTRY, closed_bool_env

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"

#: Environment reads NOT parsed by ``closed_bool_env``, each with the audited
#: reason. A new raw read must either move to the parser or be enrolled here
#: with its reason -- silence is not an option.
_LEDGERED_RAW_READS: dict[str, str] = {
    "TORCHLENS_CACHE_DIR": "kind=path: filesystem path, no boolean semantics",
    "TORCHLENS_POSTPROCESS_ASSERTIONS": (
        "reviewed inline closed-vocabulary parser: adds the -O/-OO "
        "stripped-assertions refusal and the audit-specific refusal code"
    ),
    "TORCHLENS_POSTPROCESS_WRITE_AUDIT": "kind=enum ('' / 'record'), inline closed vocabulary",
    "TORCHLENS_POSTPROCESS_READ_AUDIT": (
        "kind=enum ('' / 'record' / 'enforce'), inline closed vocabulary"
    ),
}


def _env_read_sites() -> dict[str, set[tuple[str, str]]]:
    """Walk the package AST for TORCHLENS_* environment reads.

    Returns
    -------
    dict[str, set[tuple[str, str]]]
        ``env_var_name -> {(kind, relative file path)}`` where ``kind`` is
        ``"raw"`` for ``os.environ.get / os.getenv / os.environ[...]`` reads
        and ``"parser"`` for ``closed_bool_env(...)`` call sites. Keys are
        ``TORCHLENS_*`` string literals or module-level string constants
        assigned such a literal.
    """

    reads: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "environ" not in source and "getenv" not in source and "closed_bool_env" not in source:
            continue
        tree = ast.parse(source)
        constants: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, str) and value.startswith("TORCHLENS_"):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = value

        def _key_name(node: ast.AST, constants: dict[str, str] = constants) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value if node.value.startswith("TORCHLENS_") else None
            if isinstance(node, ast.Name):
                return constants.get(node.id)
            return None

        rel = str(path.relative_to(_REPO_ROOT))
        for node in ast.walk(tree):
            key: str | None = None
            kind = "raw"
            if isinstance(node, ast.Call):
                func = node.func
                is_env_get = (
                    isinstance(func, ast.Attribute) and func.attr in {"get", "getenv"} and node.args
                )
                is_parser_call = node.args and (
                    (isinstance(func, ast.Name) and func.id == "closed_bool_env")
                    or (isinstance(func, ast.Attribute) and func.attr == "closed_bool_env")
                )
                if is_parser_call:
                    kind = "parser"
                    key = _key_name(node.args[0])
                elif is_env_get:
                    key = _key_name(node.args[0])
            elif isinstance(node, ast.Subscript):
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr == "environ":
                    key = _key_name(node.slice)
            if key is not None:
                reads.setdefault(key, set()).add((kind, rel))
    return reads


def test_every_torchlens_env_read_is_registered() -> None:
    """Every TORCHLENS_* environment read has an ENV_FLAG_REGISTRY row."""

    reads = _env_read_sites()
    assert reads, "the AST scanner found no environment reads -- scanner broken"
    unregistered = {
        name: sorted(sites) for name, sites in reads.items() if name not in ENV_FLAG_REGISTRY
    }
    assert not unregistered, (
        "TORCHLENS_* environment reads without an ENV_FLAG_REGISTRY row "
        f"(register them in torchlens/utils/env_flags.py): {unregistered}"
    )


def test_every_registered_flag_is_actually_read() -> None:
    """No stale registry rows: every registered flag is read somewhere."""

    reads = _env_read_sites()
    stale = sorted(set(ENV_FLAG_REGISTRY) - set(reads))
    assert not stale, f"ENV_FLAG_REGISTRY rows with no surviving read site: {stale}"


def test_bool_flags_parse_through_the_one_parser() -> None:
    """Boolean knobs route through closed_bool_env unless reason-ledgered."""

    reads = _env_read_sites()
    offenders: dict[str, list[str]] = {}
    for name, sites in reads.items():
        spec = ENV_FLAG_REGISTRY.get(name)
        if spec is None or spec.kind != "bool" or name in _LEDGERED_RAW_READS:
            continue
        for kind, rel in sites:
            if kind == "parser":
                continue
            if rel.endswith("torchlens/utils/env_flags.py"):
                continue
            offenders.setdefault(name, []).append(rel)
    assert not offenders, (
        "bool-kind env knobs read RAW (route them through closed_bool_env or "
        f"enroll a ledgered reason): {offenders}"
    )


def test_ledger_rows_stay_registered_and_live() -> None:
    """Every ledgered raw read names a registered, still-read flag."""

    reads = _env_read_sites()
    for name in _LEDGERED_RAW_READS:
        assert name in ENV_FLAG_REGISTRY, f"ledgered {name} lost its registry row"
        assert name in reads, f"ledgered {name} is no longer read anywhere -- prune the row"


def test_true_never_means_off_for_torchlens_auto() -> None:
    """TORCHLENS_AUTO=true refuses exactly like =1; junk refuses typed."""

    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    for spelling in ("1", "true", "YES", "on"):
        try:
            import os

            os.environ["TORCHLENS_AUTO"] = spelling
            with pytest.raises(CaptureContextError) as excinfo:
                tl.trace(model, x)
            assert excinfo.value.fields["code"] == "auto_environment_unsupported"
        finally:
            os.environ.pop("TORCHLENS_AUTO", None)
    try:
        import os

        os.environ["TORCHLENS_AUTO"] = "treu"
        with pytest.raises(InvalidArgumentError) as excinfo2:
            tl.trace(model, x)
        assert excinfo2.value.fields["code"] == "env_flag_invalid"
    finally:
        os.environ.pop("TORCHLENS_AUTO", None)


def test_suppress_capability_warnings_false_means_off() -> None:
    """The suppress knob's false spellings mean OFF (raw truthiness killed)."""

    import os

    try:
        os.environ["TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS"] = "false"
        assert closed_bool_env("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS") is False
        os.environ["TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS"] = "0"
        assert closed_bool_env("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS") is False
        os.environ["TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS"] = "on"
        assert closed_bool_env("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS") is True
    finally:
        os.environ.pop("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS", None)
