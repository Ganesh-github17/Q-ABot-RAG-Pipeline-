import sys
from pathlib import Path

# Ensure the project root (parent of tests/) is on sys.path so that
# `from src.xxx import yyy` works regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
