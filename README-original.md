# Quadtree minimum-curvature gridding

Fit scattered observations on an adaptive quadtree and sample the result on a
regular grid. Install the dependency with `python -m pip install -r requirements.txt`.

```python
from quadtree_minimum_curvature import QuadtreeMinCurvatureGridder

gridder = QuadtreeMinCurvatureGridder(
    x=[0, 0.1, 0.5, 1], y=[0, 0.2, 0.8, 1], z=[1, 2, 3, 4],
    max_level=6, points_per_leaf=1,
).fit()
X, Y, Z = gridder.grid(nx=100, ny=100)
values = gridder.predict([0.25, 0.75], [0.25, 0.75])
```

The tree splits cells containing more than `points_per_leaf` observations,
up to `max_level` (0–30). Duplicate coordinates are allowed. The solver
minimizes the sum of `obs_weight * (leaf_value - z)**2` for every observation
and `smooth_weight * sum(cell_area * laplacian**2)`. The finite-volume
Laplacian uses shared-edge lengths and center distances, including at
coarse/fine interfaces, with zero normal flux at the root boundary. Coordinates
are normalized by the root size, so the weights do not depend on coordinate
units. This is an approximate cell-centered squared-Laplacian model, not an
exact thin-plate spline: boundary effects and coarse/fine interfaces can affect
accuracy. Predictions are piecewise constant, not interpolated between cells.

`obs_weight` must be positive; `smooth_weight` may be zero, in which case
populated leaves contain their observation mean and empty leaves use the global
mean. `iterations` limits conjugate-gradient iterations and `tolerance` is the
relative residual tolerance. Inspect `converged_` and `n_iterations_` after
fitting; a `RuntimeWarning` is emitted if the limit is reached. More refinement
can require many iterations due to conditioning.

Inputs must be nonempty, finite, equal-length one-dimensional arrays. Call
`fit()` before `predict()` or `grid()`. Query coordinates broadcast as NumPy
arrays; predictions outside the square root cell or at non-finite coordinates
are NaN. `grid(nx, ny)` returns arrays of shape `(ny, nx)` spanning the
observation bounds. `quadtree()` (after `fit()`) returns the leaf cells as a
dict of NumPy arrays (`x0`, `y0`, `size`, `level`, `count`, `value`). See
[docs/example.md](docs/example.md) for a worked example with plots.

Run tests with `python -m unittest discover -s tests`.
