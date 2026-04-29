"""News subsystem for Karin.

V1: lightweight, lexical-only, JSON-persisted news ingestion and clustering.
Exposes a single `get_news` tool via bridge.tools; everything else is
internal plumbing.

Phase 1 (current): fetch, normalize, ledger.
Phase 2: clustering + independence counting.
Phase 3: confidence state engine + brief building.
Phase 4: bridge.tools integration.
"""
