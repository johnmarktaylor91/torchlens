"""Oracles wave-0 harness: registry, censuses, surface walk, static lints.

This package is the wrongness-defense machinery commissioned by the oracles
panel (plan of record: megaplan MEMO.md row P02; normative spec: the oracles
MEMO.md section 10 items 1-5). Doctrine in force here:

- Every defense is a predicate plus a MACHINE-DERIVED denominator plus a
  committed baseline; hand lists survive only as lockstep (a hand list plus a
  live walk assertion) (D1).
- The enumeration root is the REACHABLE public surface, not the declared one;
  ``__all__`` is a gated claim about a walked reality (D2, D3).
- New deltas BLOCK from this package's first merge: a live item absent from
  its committed baseline fails, and a stale baseline row whose item no longer
  exists also fails until the row is deleted (D25).
- Every defense ships a plant (a committed test that injects a known defect
  and asserts the defense goes red), and fixtures carry positive controls
  (D7).

Layout: ``_surface`` (reachable walk + classification), ``_registry``
(Table A obligations / Table B bindings / census-root descriptors),
``_censuses`` (the nine generator censuses), ``_known_gaps`` (the dated
monotone KNOWN-GAP manifest), ``_waivers`` (the D24 waiver schema),
``_invocation_templates`` (the purity/state harness skeleton),
``_lints`` (the five static lints), ``data/`` (committed baselines).
Every name here is a PLACEHOLDER pending the naming sprint.
"""
