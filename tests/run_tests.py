"""Compatibility entry point; run the complete pytest suite."""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.exit(pytest.main([str(root / "tests"), *sys.argv[1:]]))
