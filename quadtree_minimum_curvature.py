"""Adaptive, cell-centred minimum-curvature gridding."""

from dataclasses import dataclass, field
import operator
import warnings

import numpy as np


@dataclass
class Leaf:
    x0: float
    y0: float
    size: float
    level: int
    value: float = 0.0
    points: list = field(default_factory=list)
    children: list = field(default_factory=list)
    _ix: int = field(default=0, repr=False)
    _iy: int = field(default=0, repr=False)

    @property
    def is_leaf(self):
        return not self.children

    def contains(self, x, y):
        return (self.x0 <= x < self.x0 + self.size and
                self.y0 <= y < self.y0 + self.size)


def _integer(name, value, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return value


class QuadtreeMinCurvatureGridder:
    """Fit a squared-Laplacian surface on an adaptive quadtree.

    Observation penalties are per point. The curvature penalty uses a
    finite-volume Laplacian with zero normal flux on the root boundary.
    Predictions are the constant value of the containing leaf.
    """

    def __init__(self, x, y, z, max_level=8, points_per_leaf=4,
                 obs_weight=100.0, smooth_weight=1.0, iterations=2000,
                 tolerance=1e-6):
        self.x, self.y, self.z = (np.asarray(v, dtype=float) for v in (x, y, z))
        if (any(v.ndim != 1 for v in (self.x, self.y, self.z)) or
                not self.x.size or
                self.x.size != self.y.size or self.x.size != self.z.size or
                not all(np.isfinite(v).all() for v in (self.x, self.y, self.z))):
            raise ValueError("x, y and z must be nonempty, finite, equal-length 1D arrays")
        self.max_level = _integer("max_level", max_level, 0, 30)
        self.points_per_leaf = _integer("points_per_leaf", points_per_leaf, 1)
        self.iterations = _integer("iterations", iterations, 1)
        self.obs_weight = float(obs_weight)
        self.smooth_weight = float(smooth_weight)
        self.tolerance = float(tolerance)
        if not np.isfinite(self.obs_weight) or self.obs_weight <= 0:
            raise ValueError("obs_weight must be finite and positive")
        if not np.isfinite(self.smooth_weight) or self.smooth_weight < 0:
            raise ValueError("smooth_weight must be finite and nonnegative")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self._fitted = False
        self._build_tree()

    def _build_tree(self):
        xmin, xmax = self.x.min(), self.x.max()
        ymin, ymax = self.y.min(), self.y.max()
        with np.errstate(over="ignore", invalid="ignore"):
            span = max(xmax - xmin, ymax - ymin)
            if span == 0:
                span = max(abs(xmin), abs(ymin), 1.0) * 1e-6
            size = span * 1.05
            x0, y0 = xmin - span * 0.025, ymin - span * 0.025
        if (not np.isfinite([x0, y0, size, x0 + size, y0 + size]).all()
                or not (x0 <= xmin <= xmax < x0 + size)
                or not (y0 <= ymin <= ymax < y0 + size)):
            raise ValueError("coordinate range cannot be represented by a root cell")
        self.root = Leaf(x0, y0, size, 0)
        self.root.points = list(zip(self.x, self.y, self.z))
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.level >= self.max_level or len(node.points) <= self.points_per_leaf:
                continue
            half = node.size / 2
            xm, ym = node.x0 + half, node.y0 + half
            if not (node.x0 < xm < node.x0 + node.size and
                    node.y0 < ym < node.y0 + node.size):
                continue
            node.children = [
                Leaf(node.x0, node.y0, half, node.level + 1,
                     _ix=node._ix * 2, _iy=node._iy * 2),
                Leaf(xm, node.y0, half, node.level + 1,
                     _ix=node._ix * 2 + 1, _iy=node._iy * 2),
                Leaf(node.x0, ym, half, node.level + 1,
                     _ix=node._ix * 2, _iy=node._iy * 2 + 1),
                Leaf(xm, ym, half, node.level + 1,
                     _ix=node._ix * 2 + 1, _iy=node._iy * 2 + 1),
            ]
            for point in node.points:
                index = int(point[0] >= xm) + 2 * int(point[1] >= ym)
                node.children[index].points.append(point)
            node.points = []
            stack.extend(node.children)

    def _find_leaf(self, node, x, y):
        if not node.contains(x, y):
            return None
        while node.children:
            half = node.size / 2
            index = int(x >= node.x0 + half) + 2 * int(y >= node.y0 + half)
            node = node.children[index]
        return node

    def _collect_leaves(self):
        leaves, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves

    def _edges(self, leaves):
        # Match opposing edges on the integer lattice at maximum depth.
        sides = [{}, {}, {}, {}]
        for i, leaf in enumerate(leaves):
            width = 1 << (self.max_level - leaf.level)
            x, y = leaf._ix * width, leaf._iy * width
            for side, key, start in ((0, x, y), (1, x + width, y),
                                     (2, y, x), (3, y + width, x)):
                sides[side].setdefault(key, []).append((start, start + width, i, width))
        edges = []
        for low, high in ((sides[0], sides[1]), (sides[2], sides[3])):
            for key in low.keys() & high.keys():
                a, b = sorted(low[key]), sorted(high[key])
                i = j = 0
                while i < len(a) and j < len(b):
                    start = max(a[i][0], b[j][0])
                    end = min(a[i][1], b[j][1])
                    if start < end:
                        edges.append((a[i][2], b[j][2],
                                      2.0 * (end - start) / (a[i][3] + b[j][3])))
                    if a[i][1] <= b[j][1]:
                        i += 1
                    else:
                        j += 1
        return edges

    def fit(self):
        leaves = self._collect_leaves()
        counts = np.array([len(leaf.points) for leaf in leaves], dtype=float)
        sums = np.array([sum(p[2] for p in leaf.points) for leaf in leaves])
        values = np.full(len(leaves), np.mean(self.z))
        populated = counts > 0
        values[populated] = sums[populated] / counts[populated]
        self.converged_ = True
        self.n_iterations_ = 0

        if self.smooth_weight and len(leaves) > 1:
            edges = self._edges(leaves)
            a = np.array([e[0] for e in edges], dtype=int)
            b = np.array([e[1] for e in edges], dtype=int)
            weights = np.array([e[2] for e in edges])
            areas = np.array([(leaf.size / self.root.size) ** 2 for leaf in leaves])
            degree = (np.bincount(a, weights, minlength=len(leaves)) +
                      np.bincount(b, weights, minlength=len(leaves)))

            def laplacian(v):
                flux = weights * (v[a] - v[b])
                return (np.bincount(a, flux, minlength=len(leaves)) -
                        np.bincount(b, flux, minlength=len(leaves)))

            def apply(v):
                return (self.obs_weight * counts * v +
                        self.smooth_weight * laplacian(laplacian(v) / areas))

            diagonal = self.obs_weight * counts + self.smooth_weight * (
                degree ** 2 / areas +
                np.bincount(a, weights ** 2 / areas[b], minlength=len(leaves)) +
                np.bincount(b, weights ** 2 / areas[a], minlength=len(leaves)))
            rhs = self.obs_weight * sums
            residual = rhs - apply(values)
            target = self.tolerance * max(np.linalg.norm(rhs), 1.0)
            self.converged_ = np.linalg.norm(residual) <= target
            if not self.converged_:
                preconditioned = residual / diagonal
                direction = preconditioned.copy()
                product = np.dot(residual, preconditioned)
                for iteration in range(1, self.iterations + 1):
                    applied = apply(direction)
                    step = product / np.dot(direction, applied)
                    values += step * direction
                    residual -= step * applied
                    self.n_iterations_ = iteration
                    if np.linalg.norm(residual) <= target:
                        self.converged_ = True
                        break
                    preconditioned = residual / diagonal
                    next_product = np.dot(residual, preconditioned)
                    direction = preconditioned + (next_product / product) * direction
                    product = next_product
            if not self.converged_:
                warnings.warn("curvature solver did not converge; increase iterations "
                              "or relax tolerance", RuntimeWarning, stacklevel=2)

        for leaf, value in zip(leaves, values):
            leaf.value = float(value)
        self._fitted = True
        return self

    def predict(self, xq, yq):
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        xq, yq = np.broadcast_arrays(np.atleast_1d(np.asarray(xq, dtype=float)),
                                     np.atleast_1d(np.asarray(yq, dtype=float)))
        values = np.full(xq.shape, np.nan)
        for index in np.ndindex(xq.shape):
            leaf = self._find_leaf(self.root, xq[index], yq[index])
            if leaf is not None:
                values[index] = leaf.value
        return values

    def grid(self, nx, ny):
        nx = _integer("nx", nx, 1)
        ny = _integer("ny", ny, 1)
        gx = np.linspace(self.x.min(), self.x.max(), nx)
        gy = np.linspace(self.y.min(), self.y.max(), ny)
        x, y = np.meshgrid(gx, gy)
        return x, y, self.predict(x, y)
