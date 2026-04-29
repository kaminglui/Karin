"""Tiny pre-classifier for LLM tool routing.

Runs before each LLM turn to spot prompts that clearly map to one tool.
When we find such a prompt, we inject a ONE-LINE routing hint into the
system prompt so small models (3B-8B) pick the right tool without
needing the full routing table carried verbatim in karin.yaml.

The classifier is intentionally conservative: it returns None when a
prompt is ambiguous or chitchat. A missing hint is harmless; a wrong
hint derails routing. "Quiet unless confident" is the rule.
"""
from __future__ import annotations

from .classifier import classify, routing_hint
from .events import log_decision

__all__ = ["classify", "routing_hint", "log_decision"]
