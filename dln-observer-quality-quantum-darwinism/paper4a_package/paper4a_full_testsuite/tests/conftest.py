import os
import sys
import pytest

# Ensure `src/` is importable when running tests from the repository root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Central place to control numeric tolerances for all tests.
TOL = 5e-6

@pytest.fixture(scope="session")
def tol():
    return TOL
