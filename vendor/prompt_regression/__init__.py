"""prompt-regression-suite: a small, honest harness for testing AI/LLM outputs.

Two things it does:
  1. Prompt-based testing  - run a suite of prompts against a model and judge
     each answer with a validator (accuracy, safety, consistency, edge cases).
  2. Regression testing     - save a baseline of results, re-run later, and
     diff to catch quality drops between model versions or prompt changes.

Runs fully offline against a built-in mock model by default. Set
ANTHROPIC_API_KEY to test the real Claude API instead.
"""

__version__ = "0.1.0"
