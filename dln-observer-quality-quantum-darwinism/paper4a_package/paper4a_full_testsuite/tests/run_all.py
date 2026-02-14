#!/usr/bin/env python
"""Convenience wrapper for running the full test suite.

Usage:
  python tests/run_all.py
"""

import sys

def main() -> int:
    try:
        import pytest
    except Exception as e:
        print("pytest is not available in this environment.")
        print("Install dependencies with: pip install -r requirements.txt")
        print(f"Import error: {e}")
        return 2
    return int(pytest.main(["-q"]))

if __name__ == "__main__":
    raise SystemExit(main())
