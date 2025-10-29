"""
Shared plotting styles and constants.
Uses matplotlib native styling for publication-ready figures.
"""

import matplotlib as mpl

# Configure matplotlib for professional-looking plots
mpl.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "patch.linewidth": 0.5,
        "text.usetex": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
