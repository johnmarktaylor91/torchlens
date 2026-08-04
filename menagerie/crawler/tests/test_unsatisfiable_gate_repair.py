"""Three enforced rules that no correct author could satisfy, and their repair.

A rule that cannot be satisfied is not a tripwire. The authoring stage runs
exactly ONCE per model, so an unsatisfiable check does not produce a retry: it
produces a permanently dead record, silently, for every model of that shape. The
2026-08-03 ten-model rung produced zero runs and made all three visible at once:

* Six models died on ``R1_LIBRARY must explicitly disable pretrained fields``.
  ``proposal`` refused an empty array and ``recipe`` required every named field to
  be a real constructor parameter carrying a disabling value, so a constructor with
  ZERO pretrained-capable parameters -- ``MiniMaxForCausalLM(config)``, every GNN
  layer, every SNN -- had no satisfying value at all.
* ``m8189`` vendored ``naver-ai/pit`` verbatim, as ``METHODOLOGY.md`` mandates,
  and was refused because upstream ``pit.py`` is not fully annotated. Essentially
  no real PyTorch repository is. R2_VENDOR, a locked rung of the source-fidelity
  ladder, was structurally unreachable.
* ``m9617`` emitted an honest ``BLOCKED`` citing thirteen evidence IDs. Ten
  grounded byte-for-byte, including the one carrying the ``blocked-prerequisite``
  predicate; three cited a response the broker never froze. All-or-nothing
  resolution discarded the ten, and the model terminalized
  ``terminal-disposition-unverifiable``. The author would have SURVIVED by citing
  fewer IDs.

Every test here comes in a pair. One shows the shape a correct author could not
express before and now can; its partner shows the case the rule exists to refuse
still being refused, so that opening the wall cannot be mistaken for removing the
protection behind it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    _validate_terminal_evidence_pack,
)
from menagerie.crawler.constants import SourceRung
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.proposal import (
    DEFAULT_GATED_CLAIMS,
    ProposalValidationError,
    _validate_code,
    validate_author_proposal,
)
from menagerie.crawler.recipe import (
    DeclarativeRecipe,
    RecipeError,
    assert_constructor_pretrained_assets_disabled,
)
from menagerie.crawler.terminal_evidence import (
    CHANNEL_DECLARED,
    GROUNDED,
    PARTIALLY_GROUNDED,
    UNRESOLVED,
    resolve_terminal_evidence,
)
from menagerie.crawler.tests.conftest import HASH, attach_paper_evidence, make_author_proposal

# --------------------------------------------------------------------------- #
# Shared R1 proposal fixture
# --------------------------------------------------------------------------- #

_GROUNDING_TEXT = (
    "Example Model introduced ExampleNet in TestConf 2020 by A. Author at Example Lab in "
    "the US. ExampleNet is an official PyTorch library CNN architecture for supervised computer "
    "vision classification in machine learning. This modern ExampleNet family uses vision modality "
    "and has the example and cnn keywords. It is a small source-grounded example network "
    "whose grounded contribution uses the Apache-2.0 license. It runs in PyTorch eval mode "
    "with no train eval divergence. The input contract is one small RGB image and the output "
    "is class scores."
)


def _r1_proposal(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a schema-valid, fully grounded R1_LIBRARY proposal.

    Parameters
    ----------
    tmp_path:
        Isolated model/CAS directory.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Proposal and its controlled-fetch source manifest.
    """

    proposal = make_author_proposal()
    source_path = tmp_path / "source.txt"
    source_path.write_text(_GROUNDING_TEXT)
    source_hash = hash_bytes(_GROUNDING_TEXT.encode())
    excerpt = proposal["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt.update(
        {
            "locator": f"bytes:0-{len(_GROUNDING_TEXT.encode())}",
            "text": _GROUNDING_TEXT,
            "text_sha256": source_hash,
            "supports": [*sorted(DEFAULT_GATED_CLAIMS), "implementation.architecture"],
            "family_level": True,
        }
    )
    proposal["proposed_facts"]["evidence"]["coverage"].update(
        {
            "all_agent_fields_have_support": True,
            "missing_support": [],
            "family_grounding_complete": True,
        }
    )
    manifest: dict[str, Any] = {
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://example.com/model",
                "revision": "v1",
                "content_sha256": source_hash,
                "cas_path": str(source_path),
                "retrieval_status": "fetched",
            }
        ]
    }
    manifest["manifest_sha256"] = stable_hash(manifest["sources"])
    proposal["verified_hashes"]["source_manifest"] = manifest["manifest_sha256"]
    attach_paper_evidence(proposal, manifest, tmp_path)
    return proposal, manifest


def _recipe(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable declarative recipe of an R1 proposal fixture."""

    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    assert isinstance(recipe, dict)
    return recipe


# --------------------------------------------------------------------------- #
# (A) the pretrained disposition
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_a_constructor_with_no_pretrained_keyword_can_now_state_that(tmp_path: Path) -> None:
    """``MiniMaxForCausalLM(config)`` finally has a spelling it can pass with.

    Its ``__init__`` is ``(self, config)``. There is no keyword to name, so the
    old non-empty-array rule admitted no satisfying value and the model became a
    permanent dead record. The positive assertion is that spelling, and it is
    distinguishable from silence, which is what makes it safe to accept.
    """

    proposal, manifest = _r1_proposal(tmp_path)
    recipe = _recipe(proposal)
    recipe["kwargs"] = {"num_labels": 2}
    recipe["pretrained_disable_fields"] = []
    recipe["pretrained_fields_absent"] = True

    report = validate_author_proposal(
        proposal, allowed_model_dir=tmp_path, source_manifest=manifest
    )
    assert report.rung is SourceRung.LIBRARY


@pytest.mark.smoke
def test_silence_about_pretrained_assets_is_still_refused(tmp_path: Path) -> None:
    """An empty array with no assertion is exactly "I did not think about it".

    The whole reason the blunt non-empty check was reached for is that ``[]``
    conflated that with "there is nothing to disable". Separating them must not
    quietly legalise the first one.
    """

    proposal, manifest = _r1_proposal(tmp_path)
    recipe = _recipe(proposal)
    recipe["kwargs"] = {"num_labels": 2}
    recipe["pretrained_disable_fields"] = []

    with pytest.raises(ProposalValidationError, match="must declare its pretrained disposition"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.smoke
def test_an_enabled_pretrained_flag_is_refused_even_beside_a_disabled_one(
    tmp_path: Path,
) -> None:
    """The protection the rule exists for, in the shape that used to defeat it.

    This proposal SATISFIES the old rule: ``pretrained_disable_fields`` is
    non-empty, its one name is a ``kwargs`` key, and that key carries a disabling
    value. Nothing looked at ``pretrained=True`` sitting beside it, so a capture
    would have downloaded weights the catalog does not describe -- the exact
    outcome the rule was written to prevent. It is refused now.
    """

    proposal, manifest = _r1_proposal(tmp_path)
    recipe = _recipe(proposal)
    recipe["kwargs"] = {"weights": None, "pretrained": True}
    recipe["pretrained_disable_fields"] = ["weights"]

    with pytest.raises(ProposalValidationError, match="leave known pretrained keywords enabled"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.smoke
def test_asserting_absence_while_naming_fields_is_a_refused_contradiction(
    tmp_path: Path,
) -> None:
    """The two spellings are alternatives, not a menu to pick both from."""

    proposal, manifest = _r1_proposal(tmp_path)
    recipe = _recipe(proposal)
    recipe["kwargs"] = {"weights": None}
    recipe["pretrained_disable_fields"] = ["weights"]
    recipe["pretrained_fields_absent"] = True

    with pytest.raises(ProposalValidationError, match="contradicts a non-empty"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=manifest)


@pytest.mark.smoke
def test_the_absence_assertion_is_re_derived_from_the_real_signature() -> None:
    """An author claim about a signature is checked against the signature.

    The proposal stage has no interpreter and must take the assertion on trust.
    The routed environment does not: it holds the real constructor, and it reads
    it immediately before construction. So the new spelling is not a loophole an
    author can talk its way through -- it is a claim with an oracle behind it.
    """

    def honest(config: object) -> object:
        """Constructor exposing no pretrained keyword at all."""

        return config

    def dishonest(config: object, pretrained: bool = True) -> object:
        """Constructor whose pretrained keyword defaults to enabled."""

        return (config, pretrained)

    assert_constructor_pretrained_assets_disabled(honest, {}, fields_absent=True)
    with pytest.raises(RecipeError, match="contradicted by the pinned constructor signature"):
        assert_constructor_pretrained_assets_disabled(dishonest, {}, fields_absent=True)


@pytest.mark.smoke
def test_an_enabling_default_nobody_overrode_is_refused_at_load() -> None:
    """The hole the declaration check could never see.

    ``MAnet(encoder_weights="imagenet")`` downloads ImageNet weights when the
    recipe simply says nothing about ``encoder_weights``. No declaration check
    can catch that, because there is nothing declared to inspect; only the real
    signature's DEFAULT reveals it.
    """

    def smp_shaped(
        encoder_weights: str = "imagenet", weights: object = None, aux_params: object = None
    ) -> tuple[object, object, object]:
        """Constructor with an enabling pretrained default."""

        return (encoder_weights, weights, aux_params)

    with pytest.raises(RecipeError, match="would resolve pretrained assets"):
        assert_constructor_pretrained_assets_disabled(smp_shaped, {"weights": None})

    assert_constructor_pretrained_assets_disabled(
        smp_shaped, {"weights": None, "encoder_weights": None}
    )


@pytest.mark.smoke
def test_a_lookalike_parameter_name_is_not_refused_for_its_spelling() -> None:
    """The known-name set is exact, so it cannot manufacture a new wall.

    ``pretrained_window_sizes`` is Swin geometry. A substring or prefix rule
    would refuse it -- turning the repair for one wall into a second wall.
    """

    def swin_shaped(
        pretrained: bool = False, pretrained_window_sizes: object = (0, 0, 0, 0)
    ) -> object:
        """Constructor with a pretrained-looking geometry parameter."""

        return (pretrained, pretrained_window_sizes)

    assert_constructor_pretrained_assets_disabled(swin_shaped, {})


@pytest.mark.smoke
def test_the_recipe_grammar_keeps_historical_recipe_revisions_stable() -> None:
    """The new leaf is emitted only when set, so pre-extension hashes do not move."""

    unchanged = DeclarativeRecipe.from_mapping(
        {
            "distribution": "example",
            "version": "1.0",
            "module": "example",
            "symbol": "ExampleNet",
            "kwargs": {"weights": None},
            "pretrained_disable_fields": ["weights"],
        }
    )
    assert "pretrained_fields_absent" not in unchanged.to_dict()

    asserted = DeclarativeRecipe.from_mapping(
        {
            "distribution": "example",
            "version": "1.0",
            "module": "example",
            "symbol": "ExampleNet",
            "kwargs": {"config": 1},
            "pretrained_disable_fields": [],
            "pretrained_fields_absent": True,
        }
    )
    assert asserted.to_dict()["pretrained_fields_absent"] is True


# --------------------------------------------------------------------------- #
# (B) verbatim vendored upstream and the annotation rule
# --------------------------------------------------------------------------- #

_TYPED_ADAPTER = '''"""Author-written adapter."""

from upstream import PitBlock


def build_model() -> object:
    """Construct the vendored model."""

    return PitBlock()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    """Return one call."""

    return ((seed, device),), {}
'''

_UNANNOTATED_UPSTREAM = '''# PiT
# Copyright 2021-present NAVER Corp.


class PitBlock:
    def __init__(self, base_dim=48, depth=2):
        self.base_dim = base_dim
        self.depth = depth

    def forward(self, x):
        return x
'''


def _stage_vendored(
    tmp_path: Path, *, upstream_text: str = _UNANNOTATED_UPSTREAM
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage a typed adapter beside verbatim upstream bytes the broker froze.

    Parameters
    ----------
    tmp_path:
        Model-local staging root.
    upstream_text:
        Bytes the broker froze AND the author staged.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Implementation block and its controlled-fetch source manifest.
    """

    (tmp_path / "adapter.py").write_text(_TYPED_ADAPTER, encoding="utf-8")
    (tmp_path / "upstream.py").write_text(upstream_text, encoding="utf-8")
    cas = tmp_path / "cas"
    cas.mkdir(exist_ok=True)
    frozen = upstream_text.encode("utf-8")
    digest = hash_bytes(frozen)
    (cas / "upstream.source").write_bytes(frozen)
    manifest = {
        "manifest_sha256": HASH,
        "sources": [
            {
                "source_id": "impl-pit",
                "url": "https://example.org/naver-ai/pit/pit.py",
                "content_sha256": digest,
                "cas_path": str(cas / "upstream.source"),
                "retrieval_status": "fetched",
            }
        ],
    }
    implementation = {
        "code_path": "adapter.py",
        "code_sha256": hash_bytes(_TYPED_ADAPTER.encode("utf-8")),
        "upstream_files": [
            {
                "source_id": "impl-pit",
                "path": "upstream.py",
                "sha256": digest,
                "use": "verbatim architecture",
            }
        ],
        "patches": [],
    }
    return implementation, manifest


@pytest.mark.smoke
def test_verbatim_vendored_upstream_is_exempt_from_the_annotation_rule(tmp_path: Path) -> None:
    """A perfectly annotated adapter can now carry the real source beside it.

    This is exactly the ``m8189`` shape: a typed ``build.py`` and an unannotated
    ``pit.py`` copied byte for byte from the upstream repository. Before, the
    closure check refused ``pit.py`` and no correct author could do anything
    about it without editing bytes ``METHODOLOGY.md`` forbids editing.
    """

    implementation, manifest = _stage_vendored(tmp_path)

    resolved = _validate_code(implementation, SourceRung.VENDOR, tmp_path, manifest)
    assert resolved == (tmp_path / "adapter.py").resolve()


@pytest.mark.smoke
def test_undeclared_bytes_get_no_exemption_however_they_were_obtained(tmp_path: Path) -> None:
    """The exemption is a declaration the record carries, not an inference.

    The same two files, with ``upstream_files`` empty: the proposal now claims
    every staged member as its own work, so every staged member is held to the
    rule for authored code.
    """

    implementation, manifest = _stage_vendored(tmp_path)
    implementation["upstream_files"] = []

    with pytest.raises(ProposalValidationError, match="must be fully typed"):
        _validate_code(implementation, SourceRung.VENDOR, tmp_path, manifest)


@pytest.mark.smoke
def test_one_edited_character_costs_the_exemption(tmp_path: Path) -> None:
    """"Vendored" means the frozen bytes, proven by digest, and nothing looser.

    The declaration is unchanged and still names a real frozen source; only the
    staged file differs. Author-written code cannot be laundered into the
    exemption by pointing at somebody else's hash.
    """

    implementation, manifest = _stage_vendored(tmp_path)
    (tmp_path / "upstream.py").write_text(
        _UNANNOTATED_UPSTREAM + "\n# one added comment\n", encoding="utf-8"
    )

    with pytest.raises(ProposalValidationError, match="must be fully typed"):
        _validate_code(implementation, SourceRung.VENDOR, tmp_path, manifest)


@pytest.mark.smoke
def test_the_author_entry_point_is_never_exempt(tmp_path: Path) -> None:
    """``build_model``/``make_dummy_call`` are the author's contract with the runner.

    Even bytes the broker really did freeze cannot buy the entry point out of the
    annotation rule, because the entry point is where the runner's contract is
    declared and the one file a reviewer must be able to read.
    """

    untyped_entry = "def build_model():\n    return object()\n"
    (tmp_path / "adapter.py").write_text(untyped_entry, encoding="utf-8")
    cas = tmp_path / "cas"
    cas.mkdir(exist_ok=True)
    digest = hash_bytes(untyped_entry.encode("utf-8"))
    (cas / "entry.source").write_bytes(untyped_entry.encode("utf-8"))
    manifest = {
        "manifest_sha256": HASH,
        "sources": [
            {
                "source_id": "impl-entry",
                "url": "https://example.org/entry.py",
                "content_sha256": digest,
                "cas_path": str(cas / "entry.source"),
                "retrieval_status": "fetched",
            }
        ],
    }
    implementation = {
        "code_path": "adapter.py",
        "code_sha256": digest,
        "upstream_files": [
            {
                "source_id": "impl-entry",
                "path": "adapter.py",
                "sha256": digest,
                "use": "verbatim entry point",
            }
        ],
        "patches": [],
    }

    with pytest.raises(ProposalValidationError, match="must be fully typed"):
        _validate_code(implementation, SourceRung.VENDOR, tmp_path, manifest)


@pytest.mark.smoke
def test_exempt_vendored_code_still_cannot_execute_dynamically(tmp_path: Path) -> None:
    """The safety half of the closure check keeps no exemption at all.

    Full annotation is a LEGIBILITY constraint on what the author wrote. Dynamic
    execution is a SAFETY constraint on what will run, and vendored bytes run
    exactly like authored ones, so ``_validate_calls_and_writes`` covers every
    closure member with no carve-out whatsoever.
    """

    hostile = _UNANNOTATED_UPSTREAM + '\n\ndef late(x):\n    return eval(x)\n'
    implementation, manifest = _stage_vendored(tmp_path, upstream_text=hostile)

    with pytest.raises(ProposalValidationError, match="forbidden dynamic execution call"):
        _validate_code(implementation, SourceRung.VENDOR, tmp_path, manifest)


@pytest.mark.smoke
def test_exempt_vendored_code_still_cannot_write_outside_the_sandbox(tmp_path: Path) -> None:
    """The same, for writes: the exemption buys legibility, never reach."""

    hostile = (
        _UNANNOTATED_UPSTREAM
        + '\n\ndef dump(x):\n    with open("/etc/menagerie.conf", "w") as handle:\n'
        '        handle.write(x)\n'
    )
    implementation, manifest = _stage_vendored(tmp_path, upstream_text=hostile)

    with pytest.raises(ProposalValidationError):
        _validate_code(implementation, SourceRung.VENDOR, tmp_path, manifest)


@pytest.mark.smoke
def test_entering_eval_mode_is_not_dynamic_execution(tmp_path: Path) -> None:
    """A fourth unsatisfiable rule, found by replaying ``m8189``'s real bytes.

    ``_FORBIDDEN_CALLS`` was matched against the LAST dotted segment, so
    ``model.eval()`` read as ``eval``. The author contract requires eval mode and
    ``torch`` spells it ``model.eval()``, so the prompt mandated the exact call
    the validator refused: no staged adapter that obeyed its instructions could
    ever be accepted. Reaching the real builtin through an attribute needs the
    builtin namespace, which is still refused, as is ``torch.compile``.
    """

    adapter = (
        '"""Adapter."""\n\nimport re\n\n\n'
        "def build_model() -> object:\n"
        '    """Build."""\n\n'
        "    model = object()\n"
        "    model.eval()\n"
        '    re.compile("x")\n'
        "    return model\n\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[object, ...], dict[str, object]]:\n"
        '    """Call."""\n\n'
        "    return ((seed, device),), {}\n"
    )
    (tmp_path / "adapter.py").write_text(adapter, encoding="utf-8")
    implementation = {
        "code_path": "adapter.py",
        "code_sha256": hash_bytes(adapter.encode("utf-8")),
        "upstream_files": [],
        "patches": [],
    }
    assert _validate_code(implementation, SourceRung.PORT, tmp_path, {"sources": []}) is not None

    for hostile in ("    return eval(seed)\n", "    return builtins.eval(seed)\n"):
        text = adapter.replace("    return ((seed, device),), {}\n", hostile)
        (tmp_path / "adapter.py").write_text(text, encoding="utf-8")
        implementation["code_sha256"] = hash_bytes(text.encode("utf-8"))
        with pytest.raises(ProposalValidationError, match="forbidden dynamic execution call"):
            _validate_code(implementation, SourceRung.PORT, tmp_path, {"sources": []})

    compiled = adapter.replace("    model.eval()\n", "    model = torch.compile(model)\n")
    (tmp_path / "adapter.py").write_text(compiled, encoding="utf-8")
    implementation["code_sha256"] = hash_bytes(compiled.encode("utf-8"))
    with pytest.raises(ProposalValidationError, match="torch.compile"):
        _validate_code(implementation, SourceRung.PORT, tmp_path, {"sources": []})


@pytest.mark.smoke
def test_known_string_evaluation_entry_points_stay_refused(tmp_path: Path) -> None:
    """Fixing the suffix rule must not legalise real dynamic evaluation.

    Refusing by last dotted segment is what made ``model.eval()`` unsatisfiable,
    but it also happened to catch ``pd.eval("...")`` -- genuine string
    evaluation. The exact-name denylist keeps those refused: the known
    module-level dynamic-evaluation entry points of pandas and numexpr, under
    their canonical names and their ubiquitous import aliases. It is a list of
    NAMES, deliberately: a pattern rule is what built the wall. What it cannot
    cover -- and does not claim to -- is a method spelling on an arbitrary
    receiver (``df.eval``) or a module nobody has heard of; the runtime sandbox
    is the containment boundary for those.
    """

    adapter = (
        '"""Adapter."""\n\n\n'
        "def build_model() -> object:\n"
        '    """Build."""\n\n'
        "    return object()\n\n\n"
        "def make_dummy_call(seed: int, device: str) -> "
        "tuple[tuple[object, ...], dict[str, object]]:\n"
        '    """Call."""\n\n'
        "    return ((seed, device),), {}\n"
    )
    hostile_calls = (
        'pd.eval("1+1")',
        'pandas.eval("1+1")',
        'numexpr.evaluate("1+1")',
        'numexpr.re_evaluate()',
        'ne.evaluate("1+1")',
    )
    for call in hostile_calls:
        text = adapter.replace(
            "    return ((seed, device),), {}\n",
            f"    return ({call},), {{}}\n",
        )
        (tmp_path / "adapter.py").write_text(text, encoding="utf-8")
        implementation = {
            "code_path": "adapter.py",
            "code_sha256": hash_bytes(text.encode("utf-8")),
            "upstream_files": [],
            "patches": [],
        }
        with pytest.raises(ProposalValidationError, match="forbidden dynamic execution call"):
            _validate_code(implementation, SourceRung.PORT, tmp_path, {"sources": []})


# --------------------------------------------------------------------------- #
# (C) per-record terminal evidence resolution
# --------------------------------------------------------------------------- #

_FROZEN = b"class MAnet(SegmentationModel):\n    encoder_weights = 'imagenet'\n"
_PREDICATE = "blocked-prerequisite"


def _terminal_fixture(tmp_path: Path) -> dict[str, Any]:
    """Freeze one source and return its manifest.

    Parameters
    ----------
    tmp_path:
        Author staging root.

    Returns
    -------
    dict[str, Any]
        Hash-bound source manifest.
    """

    cas = tmp_path / "source-cas"
    cas.mkdir(parents=True, exist_ok=True)
    digest = hash_bytes(_FROZEN)
    path = cas / f"{digest.removeprefix('sha256:')}.source"
    path.write_bytes(_FROZEN)
    return {
        "manifest_sha256": HASH,
        "sources": [
            {
                "source_id": "impl-manet",
                "url": "https://example.org/manet/model.py",
                "content_sha256": digest,
                "cas_path": str(path),
            }
        ],
    }


def _record(evidence_id: str, text: str, *, source_id: str = "impl-manet") -> dict[str, Any]:
    """Return one declared excerpt record."""

    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "locator": "model.py lines 1-2",
        "text": text,
        "supports": [_PREDICATE],
    }


@pytest.mark.smoke
def test_the_records_that_verified_survive_the_ones_that_did_not(tmp_path: Path) -> None:
    """The ``m9617`` shape: good citations are no longer destroyed by a bad one.

    All-or-nothing meant the author would have SURVIVED by citing LESS -- a
    system that wants grounding paying for its absence. Per-record settlement
    removes that inversion.
    """

    manifest = _terminal_fixture(tmp_path)
    resolved = resolve_terminal_evidence(
        source_manifest=manifest,
        evidence_ids=["ev-class", "ev-weights", "ev-paper"],
        predicate=_PREDICATE,
        author_root=tmp_path,
        declared_records=[
            _record("ev-class", "class MAnet(SegmentationModel):"),
            _record("ev-weights", "encoder_weights = 'imagenet'"),
            _record("ev-paper", "MAnet, IEEE Access 2020", source_id="paper-never-frozen"),
        ],
    )

    assert resolved.resolution == PARTIALLY_GROUNDED
    assert resolved.channel == CHANNEL_DECLARED
    assert [row["evidence_id"] for row in resolved.excerpts] == ["ev-class", "ev-weights"]
    assert resolved.unresolved_evidence_ids == ("ev-paper",)
    assert resolved.reason is not None and "paper-never-frozen" in resolved.reason
    assert not resolved.grounded


@pytest.mark.smoke
def test_an_invented_evidence_id_is_not_laundered_by_nine_good_neighbours(
    tmp_path: Path,
) -> None:
    """Partial acceptance must not become "grounded because most of it was".

    This is the counter-argument the all-or-nothing rule was defending, and it
    still holds: a fabricated citation never becomes an excerpt, it is named as a
    gap, and the presence of good rows beside it does not make the pack grounded.
    """

    manifest = _terminal_fixture(tmp_path)
    good_texts = [
        "class MAnet(SegmentationModel):",
        "class MAnet",
        "SegmentationModel",
        "encoder_weights = 'imagenet'",
        "encoder_weights",
        "'imagenet'",
        "MAnet(Segmentation",
        "weights = 'imagenet'",
        "class",
    ]
    good_ids = [f"ev-good-{index}" for index in range(len(good_texts))]
    resolved = resolve_terminal_evidence(
        source_manifest=manifest,
        evidence_ids=[*good_ids, "ev-invented"],
        predicate=_PREDICATE,
        author_root=tmp_path,
        declared_records=[
            *(
                _record(evidence_id, text)
                for evidence_id, text in zip(good_ids, good_texts)
            ),
            _record("ev-invented", "class MAnet uses a Fourier attention bottleneck"),
        ],
    )

    assert resolved.resolution == PARTIALLY_GROUNDED
    assert [row["evidence_id"] for row in resolved.excerpts] == good_ids
    assert resolved.unresolved_evidence_ids == ("ev-invented",)
    assert all("Fourier" not in str(row["text"]) for row in resolved.excerpts)


@pytest.mark.smoke
def test_a_pack_where_nothing_verified_still_shows_nothing(tmp_path: Path) -> None:
    """Zero grounded records is still a hard unresolved with an empty excerpt tuple."""

    manifest = _terminal_fixture(tmp_path)
    resolved = resolve_terminal_evidence(
        source_manifest=manifest,
        evidence_ids=["ev-one", "ev-two"],
        predicate=_PREDICATE,
        author_root=tmp_path,
        declared_records=[
            _record("ev-one", "text that is not in the frozen bytes"),
            _record("ev-two", "nor is this"),
        ],
    )

    assert resolved.resolution == UNRESOLVED
    assert resolved.excerpts == ()
    assert resolved.unresolved_evidence_ids == ("ev-one", "ev-two")


@pytest.mark.smoke
def test_full_grounding_is_unchanged(tmp_path: Path) -> None:
    """A pack where everything verifies still reads exactly as it did before."""

    manifest = _terminal_fixture(tmp_path)
    resolved = resolve_terminal_evidence(
        source_manifest=manifest,
        evidence_ids=["ev-class"],
        predicate=_PREDICATE,
        author_root=tmp_path,
        declared_records=[_record("ev-class", "class MAnet(SegmentationModel):")],
    )

    assert resolved.resolution == GROUNDED
    assert resolved.grounded
    assert resolved.unresolved_evidence_ids == ()
    assert resolved.reason is None


def _pack(**overrides: Any) -> dict[str, Any]:
    """Return a partially grounded envelope pack with optional overrides."""

    pack: dict[str, Any] = {
        "resolution": PARTIALLY_GROUNDED,
        "identity_preimage": [],
        "declared_evidence_ids": ["ev-a", "ev-b"],
        "excerpts": [
            {
                "evidence_id": "ev-a",
                "source_id": "impl-manet",
                "locator": "model.py line 1",
                "text": "class MAnet(SegmentationModel):",
            }
        ],
        "unresolved_evidence_ids": ["ev-b"],
        "unresolved_reason": "ev-b is outside the frozen source manifest",
    }
    pack.update(overrides)
    return pack


@pytest.mark.smoke
def test_the_envelope_accepts_a_partition_and_only_a_partition() -> None:
    """The structural invariant that makes laundering impossible by shape.

    Excerpt IDs and unresolved IDs are disjoint and together are exactly the
    declared set, so every declared ID lands in one bucket and never in both.
    There is no arrangement in which an unverified record reaches the checker
    labelled as verified.
    """

    _validate_terminal_evidence_pack(_pack())

    with pytest.raises(CheckerDispatchError, match="both ground and disclaim"):
        _validate_terminal_evidence_pack(_pack(unresolved_evidence_ids=["ev-a", "ev-b"]))

    with pytest.raises(CheckerDispatchError, match="exactly once"):
        _validate_terminal_evidence_pack(_pack(unresolved_evidence_ids=["ev-c"]))

    with pytest.raises(CheckerDispatchError, match="at least one excerpt"):
        _validate_terminal_evidence_pack(_pack(excerpts=[]))

    with pytest.raises(CheckerDispatchError, match="must name its unresolved"):
        _validate_terminal_evidence_pack(_pack(unresolved_evidence_ids=[]))


@pytest.mark.smoke
def test_a_pack_cannot_call_itself_grounded_while_naming_gaps() -> None:
    """"Grounded" keeps its exact old meaning: every declared ID, no exceptions."""

    with pytest.raises(CheckerDispatchError, match="cannot also name unresolved"):
        _validate_terminal_evidence_pack(_pack(resolution=GROUNDED, declared_evidence_ids=["ev-a"]))

    with pytest.raises(CheckerDispatchError, match="no literal excerpt for ev-b"):
        _validate_terminal_evidence_pack(
            _pack(resolution=GROUNDED, unresolved_evidence_ids=[])
        )


@pytest.mark.smoke
def test_an_unresolved_pack_still_ships_no_excerpts() -> None:
    """The oldest guarantee here is untouched."""

    with pytest.raises(CheckerDispatchError, match="cannot ship excerpts it did not verify"):
        _validate_terminal_evidence_pack(_pack(resolution=UNRESOLVED))
