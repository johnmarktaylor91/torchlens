"""Name lockstep tooling: keep public names, the glossary and the docs in step.

Four parts, one command (``python tools/name_lockstep <subcommand>``):

- ``inventory``: walk the importable public surface into rows keyed on stable ids
  (``root:tl.X``, ``export:tl.ns.X``, ``member:<module>:<Class>.<name>``,
  ``param:<name>``), at the working tree or at any git ref.
- ``diff``: compare two inventories and list new, removed, renamed and moved (same
  public spelling, other defining module) public names.
- ``check``: the LOCKED lockstep rule for every new or renamed name (glossary entry,
  agent-reference examples, ``notebooks/audit/`` and ``examples/``); exits 1 and lists
  the missing places. ``attribution`` keeps a contextual old spelling (``.member``,
  ``kw=``) out of the findings when the text is a live head name, an attribute of an
  imported third-party object, a URL or an assignment target.
- ``codemod``: apply proposed renames (rows carrying ``id`` and ``proposed``) across
  code, tests and docs. Dry run by default; ``--apply`` writes.

``gate-input`` turns a diff into the input of the new-name legibility check: each new
or renamed name with its signature, docstring line and call-site snippets.
"""
