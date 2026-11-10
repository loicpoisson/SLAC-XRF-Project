"""
Pytest configuration: put scripts/ on the import path so tests can do
`from utils.<module> import ...` exactly like the scripts do, and keep
matplotlib headless (the pipeline saves figures).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPTS = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
