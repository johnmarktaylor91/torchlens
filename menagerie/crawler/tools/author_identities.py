"""Derive an author proposal's five accepted identities from the author's own facts.

WHY THIS EXISTS
---------------
``driver_admission._validate_artifact_identities`` recomputes ``source_identity``,
``evidence_identity``, ``recipe_revision``, ``vet_identity``, and
``fidelity_identity`` from a proposal's ``proposed_facts`` and refuses a mismatch.
That recompute is a genuine tripwire: each of the five is a pure function of the
author's own declared facts, so a proposal whose identity does not follow from its
facts lifted that identity from somewhere, and the driver catches it. The check
must not be loosened and the identities must not be machine-stamped -- stamping
would make the driver compare itself.

What the author was actually missing was never the authority to make the claim. It
was the ARITHMETIC. The derivation is canonical JSON (``sort_keys``,
``separators=(",",":")``, ``ensure_ascii=False``) over a projected six-field excerpt
subset, then SHA-256, nested six deep. Expecting a language model to reproduce that
byte-exactly by hand is what produced ten straight refusals with three fields wrong
every time.

So the arithmetic is handed over as a calculator, and only the arithmetic. This
module runs the very same :func:`recompute_accepted_identities` the driver runs, on
facts the author supplies, and prints the result. It is the same relationship the
author already has with ``sha256sum``.

WHAT THIS IS NOT
----------------
It is a CALCULATOR, never a second author. Three boundaries make that structural
rather than aspirational:

1. **It computes only from facts the author supplies.** It never fetches a source,
   never consults the catalog, never infers a missing leaf, and never fills a
   default. Incomplete facts produce a typed refusal naming what is missing, not a
   guess that would launder an unmade claim into a real identity.
2. **It cannot be used to satisfy the check without the facts being real.** The
   driver's recompute is unchanged and runs over the PUBLISHED facts with the
   binding the machine holds. Feed this tool fabricated facts and it faithfully
   returns the identity of that fabrication -- which then fails against the real
   artifacts exactly as before. The tool removes arithmetic failure, not factual
   failure.
3. **It refuses to take an identity as input.** The five identities are stripped
   from the supplied facts before derivation, so a draft that already carries a
   guessed identity cannot have that guess echoed back as though it were computed.

The checker half of the vet/fidelity derivation (``checker_model``,
``checker_version``, and the frozen checker prompt digest) is a machine-held fact
about a dispatch the author has no view of, so it is read from the request
envelope's ``identity_inputs`` disclosure. The envelope's own ``envelope_sha256`` is
re-verified first, so a doctored binding cannot be smuggled in -- and even if one
were, the driver recomputes with the binding IT holds, so the only achievable
outcome is a refusal.

TWO MORE ARITHMETIC MODES, SAME BOUNDARY
----------------------------------------
The stage-2 Bash allowlist grants exactly ONE command: this calculator. Two more
pure-arithmetic needs therefore live here rather than widening the allowlist:

* ``--clock [--deadline <ISO>]`` prints the current UTC instant and, given the
  JOB FACTS deadline, the exact seconds remaining. Sessions cannot observe time
  any other way (``date`` is not granted), and in the 2026-08-05 rung eight of
  twenty sessions guessed -- publishing ``wall-exceeded`` with 63-88% of their
  grant unused. A wall claim must cite an observation, and this is the granted
  observer.
* ``--hash-file <path>`` / ``--hash-string <text>`` print the ``sha256:<hex>``
  digest of exact bytes. The proposal's ``excerpts[].text_sha256`` is authored,
  the stage brief used to instruct ``sha256sum`` -- a command the allowlist
  denies -- and a real session, unable to run it, published sequential
  placeholder digests over byte-perfect excerpts (m8189). The digest of a file
  ending in a newline is reported both with and without that final byte,
  because the one real observed mismatch class is a tool-appended trailing
  newline the quoted string never had.

Neither mode reads anything but its own operand, writes anything, or fetches
anything; the calculator remains a calculator.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Import bootstrap, and it has to run BEFORE the package imports below.
#
# The author session's working directory is its own attempt directory, and
# ``menagerie`` is not an installed distribution -- only ``torchlens`` is, so the
# package resolves through the current directory and nothing else. Every in-process
# caller (tests, the executor) happens to run from a repository root and therefore
# never notices. The author never does, and a live probe under the real permission
# recipe failed here with ``No module named 'menagerie'`` after the harness had
# already GRANTED the command: a capability that is reachable but not runnable.
#
# ``parents[3]`` is the repository root: tools -> crawler -> menagerie -> root.
# Guarded on an actual import failure so a normal ``-m`` invocation or an installed
# layout keeps whatever ``menagerie`` it already resolved.
try:  # pragma: no cover - exercised by the CLI probe, not by in-process callers
    import menagerie  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - see above
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from menagerie.crawler.author_dispatch import (
    AuthorEngineFaultError,
    checker_identity_binding,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.metadata import (
    MetadataValidationError,
    recompute_accepted_identities,
)
from menagerie.crawler.schema import MODEL_SCHEMA_VERSION_V3

#: The five proposal-level identities this tool derives, in report order. These are
#: exactly the keys ``_validate_artifact_identities`` compares.
DERIVED_IDENTITY_FIELDS = (
    "source_identity",
    "evidence_identity",
    "recipe_revision",
    "vet_identity",
    "fidelity_identity",
)


class IdentityToolError(RuntimeError):
    """Raised when the calculator cannot honestly derive an identity."""


def _read_json(path: Path, label: str) -> Any:
    """Read one UTF-8 JSON document or fail typed.

    Parameters
    ----------
    path:
        Path to read.
    label:
        Human name used in the refusal.

    Returns
    -------
    Any
        Parsed JSON document.

    Raises
    ------
    IdentityToolError
        If the file is unreadable or is not valid JSON.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IdentityToolError(f"{label} is unreadable at {path}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityToolError(f"{label} at {path} is not one UTF-8 JSON document: {exc}") from exc


def _verify_envelope(envelope: Any) -> Mapping[str, Any]:
    """Return the request envelope after re-verifying its own self-hash.

    The envelope is the ONLY input this tool takes that it did not get from the
    author, so it is the only one worth authenticating. ``build_author_envelope``
    hashes the complete body into ``envelope_sha256``; recomputing it here means a
    hand-edited ``identity_inputs`` block fails loudly at the calculator rather than
    silently producing an identity nobody can use.

    Parameters
    ----------
    envelope:
        Parsed request envelope document.

    Returns
    -------
    Mapping[str, Any]
        The verified envelope.

    Raises
    ------
    IdentityToolError
        If the document is not an object or its self-hash does not bind it.
    """

    if not isinstance(envelope, Mapping):
        raise IdentityToolError("request envelope must be one JSON object")
    expected = stable_hash({key: value for key, value in envelope.items() if key != "envelope_sha256"})
    if envelope.get("envelope_sha256") != expected:
        raise IdentityToolError(
            "envelope_sha256 does not bind the request envelope; refusing to derive an "
            "identity from an unverified machine binding"
        )
    return envelope


def _authored_facts(document: Any) -> Mapping[str, Any]:
    """Return the author's drafted ``proposed_facts`` from a facts or proposal file.

    Both spellings are accepted because both are the same authored object at a
    different nesting depth: a draft proposal, or the ``proposed_facts`` block on its
    own. Anything else is refused rather than searched -- guessing which sub-object
    the author meant is exactly the inference this tool must not perform.

    Parameters
    ----------
    document:
        Parsed facts document.

    Returns
    -------
    Mapping[str, Any]
        The drafted fact block, with any already-guessed identity removed.

    Raises
    ------
    IdentityToolError
        If no unambiguous fact block is present.
    """

    if not isinstance(document, Mapping):
        raise IdentityToolError("facts document must be one JSON object")
    facts = document.get("proposed_facts") if "proposed_facts" in document else document
    if not isinstance(facts, Mapping):
        raise IdentityToolError("facts document has no proposed_facts object")
    if "evidence" not in facts or "source_resolution" not in facts:
        raise IdentityToolError(
            "facts document is not a proposed_facts block: it declares neither "
            "source_resolution nor evidence. Pass your drafted proposal, or its "
            "proposed_facts object -- this tool derives identities from facts and "
            "never supplies a fact you did not write."
        )
    # An identity supplied on the way IN could only be a guess, and echoing a guess
    # back as a computed value is the one way a calculator could become a laundry.
    return {key: value for key, value in facts.items() if key not in DERIVED_IDENTITY_FIELDS}


def _recompute(
    facts: Mapping[str, Any], checker: Mapping[str, str], schema_version: str
) -> dict[str, Optional[str]]:
    """Run the REAL driver-side derivation once over one fact block.

    Reimplementing the derivation here to "verify" it would verify nothing: the two
    copies would only ever agree with each other.

    Parameters
    ----------
    facts, checker, schema_version:
        Fact block, checker preimage, and ownership policy version.

    Returns
    -------
    dict[str, str | None]
        The five identities.

    Raises
    ------
    IdentityToolError
        If the facts are too incomplete for any identity to follow from them.
    """

    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=checker["prompt_sha256"],
            checker_model=checker["model"],
            checker_version=checker["version"],
            schema_version=schema_version,
        )
    except MetadataValidationError as exc:
        raise IdentityToolError(
            f"facts are incomplete, so no identity follows from them: {exc}"
        ) from exc
    return {
        "source_identity": identities.source,
        "evidence_identity": identities.evidence,
        "recipe_revision": identities.recipe,
        "vet_identity": identities.vet,
        "fidelity_identity": identities.fidelity,
    }


def derive_identities(
    *, envelope: Mapping[str, Any], facts_document: Any
) -> dict[str, Optional[str]]:
    """Derive the five accepted identities from authored facts and the machine binding.

    **The derivation has to settle, and that is not a detail.** The driver also
    requires the two EMBEDDED copies inside the fact block --
    ``implementation.recipe_revision`` and ``evidence.evidence_identity`` -- to equal
    the identities it recomputes. Those two leaves are themselves authored leaves, so
    writing them moves ``vet_identity``, which projects every authored leaf. Deriving
    once from a draft that does not yet carry them therefore yields a ``vet_identity``
    that is wrong for the proposal the author will actually publish.

    It settles in exactly one write pass, and provably so rather than by luck:
    ``compute_recipe_revision`` excludes ``recipe_revision`` from its own input and
    ``compute_evidence_identity`` reads only ``excerpts``, so neither copy can move the
    value it holds. Only ``vet_identity`` moves, and it moves once. This function
    performs that one pass and then VERIFIES the invariant instead of assuming it --
    if either copy's value shifted, it refuses rather than iterating, because a
    derivation that did not settle is one nobody should publish an identity from.

    Parameters
    ----------
    envelope:
        Verified author request envelope.
    facts_document:
        The author's drafted proposal or ``proposed_facts`` object.

    Returns
    -------
    dict[str, str | None]
        The five identities as they will be recomputed from the SETTLED fact block;
        ``fidelity_identity`` is ``None`` when the rung and fidelity facts do not
        require one.

    Raises
    ------
    IdentityToolError
        If the envelope does not disclose a checker binding, the facts are too
        incomplete to derive from, or the derivation did not settle.
    """

    try:
        checker = checker_identity_binding(envelope)
    except AuthorEngineFaultError as exc:
        raise IdentityToolError(str(exc)) from exc
    inputs = envelope.get("identity_inputs")
    schema_version = MODEL_SCHEMA_VERSION_V3
    if isinstance(inputs, Mapping) and isinstance(inputs.get("model_schema_version"), str):
        schema_version = str(inputs["model_schema_version"])
    facts = _authored_facts(facts_document)
    first = _recompute(facts, checker, schema_version)

    settled = deepcopy(dict(facts))
    implementation = settled.get("implementation")
    evidence = settled.get("evidence")
    if not isinstance(implementation, dict) or not isinstance(evidence, dict):
        raise IdentityToolError(
            "facts must carry implementation and evidence objects to settle the "
            "embedded identity copies"
        )
    implementation["recipe_revision"] = first["recipe_revision"]
    evidence["evidence_identity"] = first["evidence_identity"]
    final = _recompute(settled, checker, schema_version)
    if (
        final["recipe_revision"] != first["recipe_revision"]
        or final["evidence_identity"] != first["evidence_identity"]
    ):
        raise IdentityToolError(
            "the embedded recipe/evidence copies did not settle in one pass; the "
            "derivation is not a fixed point for these facts"
        )
    return final


def _iso_utc(moment: datetime) -> str:
    """Return one instant in the campaign's canonical ``...Z`` ISO spelling."""

    return moment.isoformat().replace("+00:00", "Z")


def _parse_deadline(raw: str) -> datetime:
    """Parse one ISO-8601 UTC deadline, accepting the canonical ``Z`` suffix.

    Parameters
    ----------
    raw:
        Deadline string, exactly as the JOB FACTS line carries it.

    Returns
    -------
    datetime
        Timezone-aware deadline instant.

    Raises
    ------
    IdentityToolError
        If the string is not one ISO-8601 instant.
    """

    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IdentityToolError(
            f"deadline {raw!r} is not an ISO-8601 instant; pass the JOB FACTS "
            "wall-deadline value verbatim"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def observe_clock(deadline: Optional[str] = None) -> dict[str, Any]:
    """Report the current UTC instant, and the seconds left before a deadline.

    This is the session's only granted time observation: the stage allowlist
    carries no ``date``, so without it "the deadline is approaching" is a guess,
    and rung 8 measured what guessing costs -- eight sessions self-blocked with
    63-88% of their grant unused.

    Parameters
    ----------
    deadline:
        Optional ISO-8601 deadline (the JOB FACTS wall-deadline value).

    Returns
    -------
    dict[str, Any]
        ``now``, plus ``deadline`` and ``remaining_seconds`` when one was given.
        ``remaining_seconds`` goes negative once the deadline has passed.
    """

    now = datetime.now(timezone.utc)
    report: dict[str, Any] = {"now": _iso_utc(now)}
    if deadline is not None:
        parsed = _parse_deadline(deadline)
        report["deadline"] = _iso_utc(parsed)
        report["remaining_seconds"] = round((parsed - now).total_seconds(), 1)
    return report


def hash_report(data: bytes) -> dict[str, Any]:
    """Digest exact bytes, disarming the trailing-newline trap explicitly.

    Parameters
    ----------
    data:
        Exact bytes to digest.

    Returns
    -------
    dict[str, Any]
        ``sha256`` over the exact bytes and ``bytes_len``. When the bytes end in
        one newline the report ALSO carries ``sha256_without_trailing_newline``,
        because a file-writing tool that appends a final newline is the one
        observed way a byte-perfect excerpt still hashed wrong; the caller picks
        the digest of the string it actually quoted.
    """

    report: dict[str, Any] = {"bytes_len": len(data), "sha256": hash_bytes(data)}
    if data.endswith(b"\n"):
        report["trailing_newline"] = True
        report["sha256_without_trailing_newline"] = hash_bytes(data[:-1])
    return report


def _hash_file_report(path: Path) -> dict[str, Any]:
    """Digest one file's exact bytes."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IdentityToolError(f"hash target is unreadable at {path}: {exc}") from exc
    return {"path": str(path), **hash_report(data)}


def build_parser() -> argparse.ArgumentParser:
    """Build the calculator's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser taking the identity inputs, or exactly one arithmetic mode.
    """

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.author_identities",
        description=(
            "Derive a proposal's source/evidence/recipe/vet/fidelity identities from "
            "the facts you drafted. Computes only from what you supply; never fetches, "
            "infers, or fills a fact. Also carries the session's two other granted "
            "arithmetic modes: --clock (time observation) and --hash-file/--hash-string "
            "(exact-byte SHA-256 for excerpt digests)."
        ),
    )
    parser.add_argument(
        "--request",
        type=Path,
        help="The REQUEST envelope named in JOB FACTS (read for the checker binding).",
    )
    parser.add_argument(
        "--facts",
        type=Path,
        help="Your drafted proposal, or its proposed_facts object, as one JSON file.",
    )
    parser.add_argument(
        "--clock",
        action="store_true",
        help=(
            "Print the current UTC instant; with --deadline also print the exact "
            "seconds remaining. The only granted time observation."
        ),
    )
    parser.add_argument(
        "--deadline",
        type=str,
        default=None,
        help="ISO-8601 deadline (the JOB FACTS wall-deadline value), used with --clock.",
    )
    parser.add_argument(
        "--hash-file",
        type=Path,
        default=None,
        help=(
            "Print the sha256:<hex> digest of a file's exact bytes (plus the digest "
            "without one trailing newline when the file ends in one)."
        ),
    )
    parser.add_argument(
        "--hash-string",
        type=str,
        default=None,
        help="Print the sha256:<hex> digest of the argument's exact UTF-8 bytes.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the calculator.

    Parameters
    ----------
    argv:
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``2`` on a typed refusal.
    """

    args = build_parser().parse_args(argv)
    modes = [
        bool(args.clock),
        args.hash_file is not None,
        args.hash_string is not None,
        args.request is not None or args.facts is not None,
    ]
    try:
        if sum(modes) != 1:
            raise IdentityToolError(
                "pass exactly one mode: --request/--facts (identities), --clock, "
                "--hash-file, or --hash-string"
            )
        if args.clock:
            report: Mapping[str, Any] = observe_clock(args.deadline)
        elif args.hash_file is not None:
            report = _hash_file_report(args.hash_file)
        elif args.hash_string is not None:
            report = hash_report(args.hash_string.encode("utf-8"))
        else:
            if args.request is None or args.facts is None:
                raise IdentityToolError(
                    "identity derivation needs BOTH --request and --facts"
                )
            envelope = _verify_envelope(_read_json(args.request, "request envelope"))
            report = derive_identities(
                envelope=envelope,
                facts_document=_read_json(args.facts, "facts document"),
            )
    except IdentityToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
