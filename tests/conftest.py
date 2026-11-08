"""pytest configuration and sys.path setup for Go2 CHAMP testing."""

import os
import sys

# Add src, utils, and experiments to sys.path so tests can import directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in ("src", "utils", "experiments"):
    dir_path = os.path.join(ROOT, subdir)
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)
