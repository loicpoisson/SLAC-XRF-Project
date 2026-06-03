"""
Tiny figure-finalization helper shared by all scripts.

Centralizes the savefig + show pattern so that:
  - figures are always saved to disk, and
  - plt.show() is skipped automatically on non-interactive backends (e.g. Agg),
    so batch / Colab / automated runs never block.

Pass show=False (wired to a --no-show CLI flag) to force save-only.
"""

import matplotlib
import matplotlib.pyplot as plt


def _interactive_backend():
    """True if the current matplotlib backend can display a window."""
    return matplotlib.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "cairo", "template"}


def save_and_show(fig, path, show=True, dpi=150):
    """
    Save `fig` to `path`, then display it only if `show` and the backend is
    interactive. Always closes non-shown figures to free memory.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Figure saved      -> {path}")
    if show and _interactive_backend():
        plt.show()
    else:
        plt.close(fig)
