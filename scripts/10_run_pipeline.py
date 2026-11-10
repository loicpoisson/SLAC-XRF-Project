"""Script 10 - production pipeline launcher.

The code lives in scripts/xrf_pipeline.py (importable: `import xrf_pipeline`);
this file keeps the historical command line working:

    python scripts/10_run_pipeline.py --coarse <250um.hdf5> --target Zn.Ka
    python scripts/10_run_pipeline.py --scan <next level> --resume outputs/<S>_cascade_state.pkl
    python scripts/10_run_pipeline.py --simulate --sample UB1_P1 --target Zn.Ka

See the module docstring of xrf_pipeline for the modes and outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from xrf_pipeline import main

if __name__ == "__main__":
    main()
