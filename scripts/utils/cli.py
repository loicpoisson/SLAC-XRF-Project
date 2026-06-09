"""
Shared argparse helpers.

The validation scripts (04, 07, 08, ...) all expose the same input/output
arguments (--coarse / --fine / --channels / --no-show) with identical
auto-detection defaults. `add_io_args` defines them once so the scripts only
declare their own script-specific options.
"""

from .paths import find_coarse, find_fine


def add_io_args(parser, with_fine=True):
    """
    Add the common I/O arguments to `parser`:
      --coarse   (auto-detected 250um file if omitted)
      --fine     (auto-detected 25um ground truth; omit with with_fine=False)
      --channels (comma-separated element list; default = all)
      --no-show  (save figures without opening a window)
    Returns the parser for chaining.
    """
    dc = find_coarse()
    parser.add_argument("--coarse", default=str(dc) if dc else None,
                        help="Coarse HDF5 (auto-detected from data/ if omitted)")
    if with_fine:
        df = find_fine()
        parser.add_argument("--fine", default=str(df) if df else None,
                            help="Fine HDF5 ground truth (auto-detected if omitted)")
    parser.add_argument("--channels", default=None,
                        help="Comma-separated element channels (default: all)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening a window")
    return parser
