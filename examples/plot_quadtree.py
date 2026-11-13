"""Generate the plots used in docs/example.md.

Run from the repository root: python examples/plot_quadtree.py
Requires matplotlib in addition to the packages in requirements.txt.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadtree_minimum_curvature import QuadtreeMinCurvatureGridder  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def truth(x, y):
    return np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.05) + 0.5 * np.sin(3 * np.pi * y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # Clustered sampling so the tree refines more where data are dense.
    n_dense, n_sparse = 400, 100
    x = np.concatenate([rng.normal(0.3, 0.08, n_dense), rng.uniform(0, 1, n_sparse)])
    y = np.concatenate([rng.normal(0.6, 0.08, n_dense), rng.uniform(0, 1, n_sparse)])
    keep = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    x, y = x[keep], y[keep]
    z = truth(x, y) + rng.normal(0, 0.02, x.size)

    gridder = QuadtreeMinCurvatureGridder(
        x, y, z, max_level=7, points_per_leaf=4, smooth_weight=0.1,
    ).fit()
    cells = gridder.quadtree()
    X, Y, Z = gridder.grid(nx=200, ny=200)

    fig, ax = plt.subplots(figsize=(6, 5))
    patches = [Rectangle((x0, y0), s, s) for x0, y0, s in
               zip(cells["x0"], cells["y0"], cells["size"])]
    pc = PatchCollection(patches, array=cells["level"], cmap="viridis",
                         edgecolor="k", linewidth=0.3, alpha=0.6)
    ax.add_collection(pc)
    ax.scatter(x, y, s=6, c="red", label="observations")
    fig.colorbar(pc, ax=ax, label="refinement level")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Adaptive quadtree ({len(cells['value'])} leaf cells)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "quadtree_cells.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    true_grid = truth(X, Y)
    vmin, vmax = true_grid.min(), true_grid.max()
    for ax, data, title in ((axes[0], true_grid, "Truth"),
                            (axes[1], Z, "Quadtree minimum-curvature fit")):
        im = ax.pcolormesh(X, Y, data, shading="auto", vmin=vmin, vmax=vmax,
                           cmap="magma")
        ax.scatter(x, y, s=4, c="cyan", edgecolor="none")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, label="z", shrink=0.9)
    fig.savefig(OUT / "surface_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
