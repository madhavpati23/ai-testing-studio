"""ai-test-case-generator: turn a feature description into runnable test cases.

Give it a feature or requirement; it produces a set of test cases — across risk
categories, including edge cases and safety probes — as YAML that drops straight
into the `prompt-regression-suite` runner.

Two modes:
  * mock (default)  - deterministic template generator; no API key, CI-friendly.
  * claude          - asks Claude to design tailored cases when ANTHROPIC_API_KEY
                      is set. Every generated case is validated against a schema
                      before it is written; invalid cases are dropped and counted.
"""

__version__ = "0.1.0"
