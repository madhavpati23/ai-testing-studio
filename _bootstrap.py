"""Make the two framework packages importable.

Resolution order:
  1. already installed (pip)             -> use that
  2. bundled copy in ./vendor            -> used on deploy (self-contained, so the
                                            repo can be private with no external deps)
  3. sibling repos' src/ dirs            -> convenient for local dev

Import this before importing the framework packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = ("test_case_generator", "prompt_regression")
_VENDOR = os.path.join(_HERE, "vendor")
_SIBLINGS = {
    "test_case_generator": os.path.join(_HERE, "..", "ai-test-case-generator", "src"),
    "prompt_regression": os.path.join(_HERE, "..", "prompt-regression-suite", "src"),
}


def ensure_packages() -> None:
    for module in _MODULES:
        if importlib.util.find_spec(module) is not None:
            continue  # installed
        for base in (_VENDOR, os.path.abspath(_SIBLINGS[module])):
            if os.path.isdir(os.path.join(base, module)) and base not in sys.path:
                sys.path.insert(0, base)
                break


ensure_packages()
