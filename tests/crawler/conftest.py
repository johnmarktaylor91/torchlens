"""Expose the crawler's real-environment fixtures to the acceptance package.

The acceptance dry-run drives the production driver, which binds a real materialized
conda prefix rather than a stand-in. That prefix is built by the crawler suite's own
session fixture, so it is re-exported here instead of being re-derived: a second
builder would be a second definition of "real environment" and could drift into
accepting something weaker than the suite's.
"""

from __future__ import annotations

from menagerie.crawler.tests.conftest import (  # noqa: F401
    real_environment_fixture,
    real_environment_seal_counter,
)
