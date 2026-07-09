"""
Shared argparse helpers.

The validation scripts (04, 07, 08, ...) all expose the same input/output
arguments (--coarse / --fine / --channels / --no-show) with identical
auto-detection defaults. `add_io_args` defines them once so the scripts only
declare their own script-specific options.
"""

from .paths import find_coarse, find_fine, sample_hint_from_path


def add_io_args(parser, with_fine=True):
    """
    Add the common I/O arguments to `parser`:
      --coarse   (auto-detected 250um file if omitted)
      --fine     (auto-detected 25um ground truth; omit with with_fine=False)
      --channels (comma-separated element list; default = all)
      --no-show  (save figures without opening a window)
    Returns the parser for chaining.

    The default fine file is constrained to the SAME sample as the default
    coarse file: a cross-sample pair would silently produce meaningless
    metrics (grid projection clamps to the nearest edge instead of erroring).
    If no same-sample fine scan exists, the default is None and the script's
    `require()` asks the user to pass --fine explicitly.
    """
    dc = find_coarse()
    parser.add_argument("--coarse", default=str(dc) if dc else None,
                        help="Coarse HDF5 (auto-detected from data/ if omitted)")
    if with_fine:
        hint = sample_hint_from_path(dc)
        df = find_fine(sample_hint=hint)
        if df is not None and hint and sample_hint_from_path(df) != hint:
            df = None    # only a different sample's fine scan exists
        parser.add_argument("--fine", default=str(df) if df else None,
                            help="Fine HDF5 ground truth (auto-detected if omitted)")
    parser.add_argument("--channels", default=None,
                        help="Comma-separated element channels (default: all)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening a window")
    return parser
