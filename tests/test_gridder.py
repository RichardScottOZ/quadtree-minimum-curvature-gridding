import unittest
import warnings

import numpy as np

from quadtree_minimum_curvature import QuadtreeMinCurvatureGridder


class GridderTests(unittest.TestCase):
    def setUp(self):
        self.x = [0, 0.02, 0.04, 0.1, 0.8, 1]
        self.y = [0, 0.02, 0.04, 0.1, 0.8, 1]
        self.z = [0, 0.1, 0.3, 0.5, 1.8, 2]

    def make_gridder(self, **kwargs):
        options = {"max_level": 4, "points_per_leaf": 1}
        options.update(kwargs)
        return QuadtreeMinCurvatureGridder(
            self.x, self.y, self.z, **options
        )

    def test_refines_and_assigns_each_point_once(self):
        gridder = self.make_gridder()
        leaves = gridder._collect_leaves()
        self.assertGreater(len(leaves), 4)
        self.assertGreater(len({leaf.level for leaf in leaves}), 1)
        self.assertEqual(sum(len(leaf.points) for leaf in leaves), len(self.x))
        for x, y in zip(self.x, self.y):
            self.assertTrue(gridder._find_leaf(gridder.root, x, y).contains(x, y))

    def test_edges_connect_all_levels_without_corner_neighbors(self):
        gridder = self.make_gridder()
        leaves = gridder._collect_leaves()
        edges = gridder._edges(leaves)
        self.assertTrue(any(leaves[a].level != leaves[b].level for a, b, _ in edges))
        seen = {0}
        while True:
            expanded = seen | {b for a, b, _ in edges if a in seen}
            expanded |= {a for a, b, _ in edges if b in seen}
            if expanded == seen:
                break
            seen = expanded
        self.assertEqual(len(seen), len(leaves))
        for a, b, weight in edges:
            self.assertGreater(weight, 0)
            self.assertNotEqual(a, b)

    def test_solver_minimizes_global_curvature_objective(self):
        gridder = self.make_gridder(tolerance=1e-10).fit()
        leaves = gridder._collect_leaves()
        n = len(leaves)
        laplacian = np.zeros((n, n))
        for a, b, weight in gridder._edges(leaves):
            laplacian[a, a] += weight
            laplacian[b, b] += weight
            laplacian[a, b] -= weight
            laplacian[b, a] -= weight
        areas = np.array([(leaf.size / gridder.root.size) ** 2 for leaf in leaves])
        counts = np.array([len(leaf.points) for leaf in leaves])
        sums = np.array([sum(p[2] for p in leaf.points) for leaf in leaves])
        matrix = (gridder.obs_weight * np.diag(counts) +
                  gridder.smooth_weight * laplacian @ np.diag(1 / areas) @ laplacian)
        expected = np.linalg.solve(matrix, gridder.obs_weight * sums)
        self.assertTrue(gridder.converged_)
        np.testing.assert_allclose([leaf.value for leaf in leaves], expected, atol=1e-8)

    def test_constant_data_and_scaled_values(self):
        for scale in (1, 1e-12):
            gridder = QuadtreeMinCurvatureGridder(
                self.x, self.y, np.array(self.z) * scale, max_level=4,
                points_per_leaf=1, tolerance=1e-9
            ).fit()
            reference = self.make_gridder(tolerance=1e-9).fit()
            np.testing.assert_allclose(gridder.predict(self.x, self.y),
                                       reference.predict(self.x, self.y) * scale,
                                       rtol=1e-7, atol=1e-20)
        constant = QuadtreeMinCurvatureGridder(
            self.x, self.y, [7] * len(self.x), points_per_leaf=1
        ).fit()
        np.testing.assert_allclose(constant.grid(7, 5)[2], 7)

    def test_grid_shapes_bounds_and_duplicates(self):
        gridder = QuadtreeMinCurvatureGridder([2, 2], [3, 3], [1, 3]).fit()
        np.testing.assert_allclose(gridder.predict(2, 3), [2])
        self.assertTrue(np.isnan(gridder.predict([0, np.nan], [3, 3])).all())
        x, y, z = gridder.grid(3, 2)
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(y.shape, (2, 3))
        np.testing.assert_allclose(z, 2)
        self.assertEqual(gridder.predict([[2], [2]], [3, 3]).shape, (2, 2))
        with self.assertRaises(ValueError):
            gridder.predict([1, 2], [1, 2, 3])

    def test_zero_smoothing_and_nonconvergence(self):
        gridder = self.make_gridder(smooth_weight=0).fit()
        for leaf in gridder._collect_leaves():
            if leaf.points:
                self.assertAlmostEqual(leaf.value, np.mean([p[2] for p in leaf.points]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gridder = self.make_gridder(iterations=1, tolerance=1e-12).fit()
        self.assertFalse(gridder.converged_)
        self.assertEqual(gridder.n_iterations_, 1)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_invalid_inputs(self):
        for args in (([], [], []), ([1], [1, 2], [1]),
                     ([np.nan], [1], [1]), ([[1]], [1], [1])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                QuadtreeMinCurvatureGridder(*args)
        for kwargs in ({"max_level": -1}, {"max_level": 31},
                       {"points_per_leaf": 0}, {"iterations": 0},
                       {"obs_weight": 0}, {"smooth_weight": -1},
                       {"tolerance": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_gridder(**kwargs)
        gridder = self.make_gridder()
        with self.assertRaises(RuntimeError):
            gridder.predict(0, 0)
        with self.assertRaises(ValueError):
            gridder.grid(0, 3)


if __name__ == "__main__":
    unittest.main()
