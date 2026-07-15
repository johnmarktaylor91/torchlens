"""Regression coverage for round-3 run-award and isolation binding edges."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from menagerie.crawler.constants import RunMode
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes, stable_hash
from menagerie.crawler.proposal import ProposalValidationError, validate_author_proposal
from menagerie.crawler.recipe import RecipeError, load_declarative_recipe
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.tests.test_slice_d_proposal_author import _ground_proposal
from menagerie.crawler.worker import WorkerRequest, run_worker
from menagerie.crawler.worker_supervisor import (
    _macos_denial_audit,
    _parent_owned_audit_path,
    _parse_linux_denial_audit,
    supervise_worker,
)


def _adapter_revision(path: Path, source_identity: str = "source-test") -> str:
    """Return the revision for the exact current adapter bytes.

    Parameters
    ----------
    path:
        Adapter source path.
    source_identity:
        Source identity bound into the recipe.

    Returns
    -------
    str
        Exact typed-adapter recipe revision.
    """

    return compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=path.read_bytes(),
    )


def _tiny_adapter(constructor_body: str = "return Tiny()") -> str:
    """Return a complete tiny typed adapter with a configurable constructor body.

    Parameters
    ----------
    constructor_body:
        Indented-body text placed in ``build_model``.

    Returns
    -------
    str
        Complete adapter source.
    """

    body = "\n".join(f"    {line}" for line in constructor_body.splitlines())
    return f"""from __future__ import annotations
import torch

class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1

def build_model() -> object:
{body}

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {{}})
"""


def _worker_request(
    adapter: Path,
    tmp_path: Path,
    expected_revision: str,
    *,
    expected_adapter_sha256: str | None = None,
) -> WorkerRequest:
    """Build one eval-only typed worker request.

    Parameters
    ----------
    adapter:
        Typed adapter source path.
    tmp_path:
        Per-test writable root.
    expected_revision:
        Parent-authorized adapter revision.
    expected_adapter_sha256:
        Parent-authorized adapter digest, defaulting to the current bytes.

    Returns
    -------
    WorkerRequest
        Closed worker request.
    """

    return WorkerRequest(
        stable_id="m_round3",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": expected_adapter_sha256 or hash_bytes(adapter.read_bytes()),
        },
        modality="unknown",
        input_spec=InputSpec((1, 2), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
        source_identity="source-test",
        recipe_revision=expected_revision,
    )


def test_worker_refuses_changed_adapter_and_reports_matching_observation(tmp_path: Path) -> None:
    """Only byte-identical expected adapter source reaches constructor execution."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(_tiny_adapter(), encoding="utf-8")
    expected_revision = _adapter_revision(adapter)
    expected_digest = hash_bytes(adapter.read_bytes())
    adapter.write_text(_tiny_adapter() + "\n# changed after validation\n", encoding="utf-8")

    mismatch = run_worker(
        _worker_request(
            adapter,
            tmp_path,
            expected_revision,
            expected_adapter_sha256=expected_digest,
        )
    )

    assert mismatch["constructor_started"] is False
    assert mismatch["per_mode"] == {}
    assert mismatch["error"]["exception_type"] == "menagerie.crawler.recipe.RecipeError"
    assert "digest mismatch" in mismatch["error"]["message"]

    matching_root = tmp_path / "matching"
    matching_root.mkdir()
    matching_adapter = matching_root / "adapter.py"
    matching_adapter.write_text(_tiny_adapter(), encoding="utf-8")
    matching_revision = _adapter_revision(matching_adapter)
    receipt = run_worker(_worker_request(matching_adapter, matching_root, matching_revision))

    assert receipt["error"] is None
    assert receipt["per_mode"]["eval"]["forward_completed"] is True
    assert receipt["recipe_revision"] == matching_revision
    assert receipt["observed_recipe_revision"] == matching_revision
    assert receipt["observed_adapter_sha256"] == hash_bytes(matching_adapter.read_bytes())


def test_worker_refuses_changed_recursive_helper_before_import(tmp_path: Path) -> None:
    """Every request-bound helper byte is rehashed before adapter import executes."""

    helper = tmp_path / "helper.py"
    helper.write_text("INCREMENT = 1\n", encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        _tiny_adapter()
        .replace(
            "import torch\n",
            "import torch\nfrom helper import INCREMENT\n",
        )
        .replace("return value + 1", "return value + INCREMENT"),
        encoding="utf-8",
    )
    members = [
        {"path": "adapter.py", "sha256": hash_bytes(adapter.read_bytes())},
        {"path": "helper.py", "sha256": hash_bytes(helper.read_bytes())},
    ]
    request = _worker_request(adapter, tmp_path, _adapter_revision(adapter))
    assert isinstance(request.recipe, dict)
    request.recipe.update(
        {
            "code_manifest": [
                {
                    "path": str(tmp_path / member["path"]),
                    "identity_path": member["path"],
                    "sha256": member["sha256"],
                }
                for member in members
            ],
            "code_manifest_sha256": stable_hash(members),
        }
    )
    helper.write_text("INCREMENT = 2\n", encoding="utf-8")

    receipt = run_worker(request)

    assert receipt["constructor_started"] is False
    assert receipt["observed_code_manifest_sha256"] != stable_hash(members)
    assert receipt["error"]["exception_type"] == "menagerie.crawler.recipe.RecipeError"
    assert "helper.py" in receipt["error"]["message"]


@pytest.mark.parametrize("suffix", [".bin", ".npz", ".pkl"])
def test_python_undeclared_weight_reads_poison_receipt(tmp_path: Path, suffix: str) -> None:
    """Every undeclared model-data read is denied independently of its suffix.

    Parameters
    ----------
    suffix:
        Representative hidden-weight container suffix.
    """

    hidden = tmp_path / f"weights{suffix}"
    hidden.write_bytes(b"not authorized model data")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        _tiny_adapter(f"Path({str(hidden)!r}).read_bytes()\nreturn Tiny()").replace(
            "import torch\n", "import torch\nfrom pathlib import Path\n"
        ),
        encoding="utf-8",
    )

    receipt = run_worker(_worker_request(adapter, tmp_path, _adapter_revision(adapter)))

    assert receipt["constructor_completed"] is False
    assert receipt["policy_observation"]["checkpoint_or_weight_read_attempted"] is True
    assert str(hidden) in receipt["policy_observation"]["checkpoint_paths"]
    assert receipt["error"]["reason_code"] == "checkpoint-read"


def test_pretrained_disable_fields_require_real_disabled_constructor_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real disabled parameter passes while absent or enabled claims are refused."""

    module = ModuleType("round3_recipe_fixture")

    def constructor(*, width: int, weights: object | None = None) -> dict[str, object]:
        """Return received constructor values for the declarative loader fixture."""

        return {"width": width, "weights": weights}

    setattr(module, "ExampleNet", constructor)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    base = {
        "distribution": "round3-fixture",
        "version": "1",
        "module": module.__name__,
        "symbol": "ExampleNet",
        "kwargs": {"width": 4, "weights": None},
        "pretrained_disable_fields": ["weights"],
    }

    loaded = load_declarative_recipe(base)

    assert loaded.build_model() == {"width": 4, "weights": None}
    with pytest.raises(RecipeError, match="absent from constructor kwargs"):
        load_declarative_recipe({**base, "pretrained_disable_fields": ["pretrained"]})
    with pytest.raises(RecipeError, match="does not carry a disabling value"):
        load_declarative_recipe({**base, "kwargs": {"width": 4, "weights": True}})


def test_proposal_refuses_bogus_pretrained_disable_field(tmp_path: Path) -> None:
    """R1 proposal validation binds every disable claim to an explicit disabled kwarg."""

    proposal, manifest = _ground_proposal(tmp_path)
    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    recipe["pretrained_disable_fields"] = ["pretrained"]

    with pytest.raises(ProposalValidationError, match="absent from constructor kwargs"):
        validate_author_proposal(
            proposal,
            allowed_model_dir=tmp_path,
            source_manifest=manifest,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux syscall-broker regression")
def test_native_undeclared_weight_probe_is_denied_and_reported_as_an_attempt(
    tmp_path: Path,
) -> None:
    """A namespace-denied libc probe returns no bytes and remains separate telemetry."""

    if shutil.which("strace") is None:
        pytest.skip("strace is unavailable")
    hidden = tmp_path / "native-weights.bin"
    hidden.write_bytes(b"native hidden weights")
    adapter = tmp_path / "adapter.py"
    constructor = (
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        f"descriptor = libc.open({str(hidden)!r}.encode(), os.O_RDONLY)\n"
        "if descriptor >= 0:\n"
        "    libc.close(descriptor)\n"
        "return Tiny()"
    )
    adapter.write_text(
        _tiny_adapter(constructor).replace(
            "import torch\n", "import ctypes\nimport os\nimport torch\n"
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    receipt_path = tmp_path / "result" / "receipt.json"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "stable_id": "m_native_hidden_read",
                "recipe": {
                    "kind": "typed-adapter",
                    "path": str(adapter),
                    "adapter_sha256": hash_bytes(adapter.read_bytes()),
                },
                "modality": "unknown",
                "input_spec": {"shape": [1, 2], "dtype": "float32"},
                "scratch_root": str(scratch),
                "meaningful_modes": ["eval"],
                "source_identity": "source-test",
                "recipe_revision": _adapter_revision(adapter),
            }
        ),
        encoding="utf-8",
    )

    result = supervise_worker(
        request_path,
        receipt_path,
        scratch / "supervisor",
        timeout_seconds=20,
        rss_limit_bytes=12 * 1024**3,
    )

    if result.receipt_error == "failed:sandbox-unavailable":
        pytest.skip("working Linux OS sandbox is unavailable")
    assert result.worker_receipt is not None
    policy = result.worker_receipt["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert str(hidden) in result.observation.failed_read_probe_paths
    assert result.worker_receipt["error"]["reason_code"] == "checkpoint-read"
    assert result.success_attestation_sha256 is None


@pytest.mark.parametrize("failure", ["missing", "truncated", "replaced"])
def test_parent_telemetry_integrity_failures_poison_closed(tmp_path: Path, failure: str) -> None:
    """Missing, truncated, or replaced syscall telemetry is always policy poison.

    Parameters
    ----------
    failure:
        Simulated parent telemetry integrity failure.
    """

    audit = tmp_path / "parent-audit.log"
    expected_identity: tuple[int, int] | None = None
    if failure != "missing":
        audit.write_text('1 openat(AT_FDCWD, "/tmp/x", O_RDONLY) = 3\n', encoding="utf-8")
        status = audit.stat()
        expected_identity = (status.st_dev, status.st_ino)
    if failure == "replaced":
        replacement = tmp_path / "replacement.log"
        replacement.write_text("1 +++ exited with 0 +++\n", encoding="utf-8")
        replacement.replace(audit)

    observation = _parse_linux_denial_audit(
        audit,
        tmp_path,
        (tmp_path / "scratch",),
        expected_identity=expected_identity,
    )

    assert observation.poisoned is True
    assert observation.checkpoint_or_weight_read_attempted is True
    assert observation.telemetry_failure == failure


def test_parent_telemetry_path_is_outside_child_writable_roots(tmp_path: Path) -> None:
    """The broker log is a parent-owned sibling, never a writable child bind."""

    scratch = tmp_path / "scratch" / "supervisor"
    result = tmp_path / "scratch" / "result"
    scratch.mkdir(parents=True)
    result.mkdir(parents=True)
    audit, _identity = _parent_owned_audit_path(scratch, (scratch, result))

    assert not audit.is_relative_to(scratch)
    assert not audit.is_relative_to(result)
    assert audit.name == "sandbox-syscalls.log"


def test_macos_caught_native_read_denial_is_policy_poison() -> None:
    """Seatbelt file-read-data denials poison a caught native model-data read."""

    observation = _macos_denial_audit(
        b"sandbox-exec: deny(1) file-read-data /tmp/hidden-native-weights.bin\n"
    )

    assert observation.poisoned is True
    assert observation.checkpoint_or_weight_read_attempted is True
    assert "hidden-native-weights.bin" in observation.checkpoint_paths[0]
