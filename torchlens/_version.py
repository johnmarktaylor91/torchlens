"""The package version string, readable below the public facade.

``torchlens.__version__`` re-exports this value. Modules that stamp the
version into artifacts import it from here instead of the package root, so
no layered module needs the facade for it. The release tooling bumps the
literal below (``pyproject.toml`` ``version_variables``).
"""

from __future__ import annotations

__tl_layer__ = "L0"

__version__ = "2.36.3"
