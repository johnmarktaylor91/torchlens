"""Name lockstep tooling: keep public names, the glossary and the docs in step.

Four parts, one command (``python tools/name_lockstep <subcommand>``):

- ``inventory``: walk the importable public surface into rows keyed on stable ids
  (``root:tl.X``, ``export:tl.ns.X``, ``member:<module>:<Class>.<name>`` including the
  protocol dunders, ``param:<name>``), plus the spellings only the source shows
  (``code:<refusal code>``, ``receipt:<record>.<key>``; ``static_surface``), at the
  working tree or at any git ref. Every inventory records the commit it describes.
- ``diff``: compare two inventories and list new, removed, renamed and moved (same
  public spelling, other defining module or class: a mixin member folded onto the
  public class it was reached through) public names.
- ``check``: the LOCKED lockstep rule for every new or renamed name (glossary entry,
  agent-reference examples, ``notebooks/audit/`` and ``examples/``); exits 1 and lists
  the missing places, and exits 2 before diffing when the base is not a trustworthy
  diff base (``baseguard``: unknown ref, not an ancestor of head, or a branch-internal
  commit no merge target reaches; ``--target`` names the target, ``none`` skips). ``attribution`` keeps a contextual old spelling (``.member``,
  ``kw=``) out of the findings when the text is a live head name, an attribute of an
  imported third-party object, a URL or an assignment target.
- ``codemod``: apply proposed renames (rows carrying ``id`` and ``proposed``) across
  code, tests and docs. Dry run by default; ``--apply`` writes.

``gate-input`` turns a diff into the input of the new-name legibility check: each new
or renamed name with its signature, docstring line and call-site snippets.
"""
