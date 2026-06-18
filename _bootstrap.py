"""Make the two framework packages importable.

In deployment they're installed from git (see requirements.txt). For local dev
next to the sibling repos, this adds their `src/` dirs to sys.path so you don't
have to install anything. Import it before importing the framework packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLINGS = {
    "test_case_generator": os.path.join(_HERE, "..", "ai-test-case-generator", "src"),
    "prompt_regression": os.path.join(_HERE, "..", "prompt-regression-suite", "src"),
}


def ensure_packages() -> None:
    for module, src in _SIBLINGS.items():
        if importlib.util.find_spec(module) is None:   # not installed
            src = os.path.abspath(src)
            if os.path.isdir(src) and src not in sys.path:
                sys.path.insert(0, src)


ensure_packages()
