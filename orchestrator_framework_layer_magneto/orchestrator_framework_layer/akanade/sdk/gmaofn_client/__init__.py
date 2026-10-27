"""gmaofn_client — a small, external Python SDK for talking to a
deployed ALGO AKANADE chatbot service over its REST API. This is deliberately
separate from `akanade/backend/app/gmaofn/` (that package IS the
framework, used from inside the running server); this SDK is for a
THIRD PARTY / a script running somewhere else that wants to call the
already-deployed API — login, send chat messages (including "hybrid"
mode), manage RAG stores/documents, and register admin providers.

Published to PyPI as `gmaofn-client` (the package/import name is kept
stable across the framework's GMAOFN → ALGO AKANADE rename — it's a real,
already-installed external identifier, not internal branding); the
exported class names below now match the current ALGO AKANADE name.
See README.md in this directory for usage.
"""
from .client import AKANADEClient, AKANADEError

__all__ = ["AKANADEClient", "AKANADEError"]
