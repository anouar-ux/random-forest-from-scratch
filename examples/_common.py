"""Shared helpers for the example scripts: import model.py from the repo root, save or show figures."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGURES = os.path.join(ROOT, "figures")

import matplotlib

SHOW = "--show" in sys.argv
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def finish(fig, name):
    """Save the figure in figures/ and open a window only when --show is passed."""
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("saved", os.path.relpath(path, ROOT))
    if SHOW:
        plt.show()
