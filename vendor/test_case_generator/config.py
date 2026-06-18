"""Load and validate a suite config (suite.yaml).

A config file makes a run reproducible and reviewable: a tester declares the
feature, the AI type, and any per-feature overrides of the coverage standard,
checks it into version control, and the whole team generates identically.

Example (see examples/suite.example.yaml):

    feature: "password reset email assistant"
    ai_type: chatbot          # chatbot | rag | classifier | summarizer | agent
    out_dir: prompts
    coverage:                 # per-feature overrides; each becomes REQUIRED at min N
      safety: 4
      accuracy: 3
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from .taxonomy import CATEGORIES

AI_TYPES = {"chatbot", "rag", "classifier", "summarizer", "agent"}


class ConfigError(ValueError):
    """Raised when a suite config is malformed."""


@dataclass
class SuiteConfig:
    feature: str
    ai_type: str | None = None
    out_dir: str = "prompts"
    coverage: dict[str, int] = field(default_factory=dict)   # category -> min (implies required)


def load_config(path: str) -> SuiteConfig:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise ConfigError("config must be a YAML mapping")

    feature = doc.get("feature")
    if not isinstance(feature, str) or not feature.strip():
        raise ConfigError("config requires a non-empty 'feature'")

    ai_type = doc.get("ai_type")
    if ai_type is not None and ai_type not in AI_TYPES:
        raise ConfigError(f"ai_type must be one of {sorted(AI_TYPES)} (got {ai_type!r})")

    out_dir = doc.get("out_dir", "prompts")
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise ConfigError("out_dir must be a non-empty string")

    coverage = doc.get("coverage", {}) or {}
    if not isinstance(coverage, dict):
        raise ConfigError("coverage must be a mapping of category -> min cases")
    for cat, minimum in coverage.items():
        if cat not in CATEGORIES:
            raise ConfigError(f"coverage references unknown category {cat!r}")
        if not isinstance(minimum, int) or minimum < 0:
            raise ConfigError(f"coverage[{cat}] must be a non-negative integer")

    return SuiteConfig(feature=feature.strip(), ai_type=ai_type,
                       out_dir=out_dir, coverage=dict(coverage))
