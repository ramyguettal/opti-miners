"""
tests/test_generator.py
========================
Unit tests for the instance generator.
"""
import unittest

from drone_delivery import config
from drone_delivery.data.generator import generate_instance
from drone_delivery.utils.distance import euclidean_distance


class TestGenerator(unittest.TestCase):
    """Tests for data/generator.py."""

    def setUp(self):
        """Generate a default instance for testing."""
        self.instance = generate_instance(
            n_customers=20, n_drones=4, seed=42,
        )

    def test_customer_count(self):
        """Generated instance has correct number of customers."""
        self.assertEqual(self.instance.n_customers, 20)

    def test_depot_position(self):
        """Depot is at the centre of the grid."""
        gw, gh = config.GRID_SIZE
        self.assertAlmostEqual(self.instance.depot[0], gw / 2)
        self.assertAlmostEqual(self.instance.depot[1], gh / 2)

    def test_demands_in_range(self):
        """All customer demands are positive and within payload limit."""
        for c in self.instance.customers:
            self.assertGreater(c.demand, 0.0)
            self.assertLessEqual(c.demand, config.MAX_PAYLOAD_KG)

    def test_no_customer_in_nfz(self):
        """No customer is located inside a no-fly zone."""
        for c in self.instance.customers:
            for nfz in self.instance.no_fly_zones:
                dist = euclidean_distance((c.x, c.y), nfz.center)
                self.assertGreater(
                    dist, nfz.radius,
                    f"Customer {c.id} at ({c.x:.1f},{c.y:.1f}) is inside "
                    f"{nfz.label} (centre={nfz.center}, r={nfz.radius:.1f})",
                )

    def test_distance_matrix_shape(self):
        """Distance matrix is (n+1) x (n+1) and symmetric."""
        n = self.instance.n_customers + 1
        self.assertEqual(self.instance.distance_matrix.shape, (n, n))
        for i in range(n):
            self.assertAlmostEqual(self.instance.distance_matrix[i, i], 0.0)
            for j in range(i + 1, n):
                self.assertAlmostEqual(
                    self.instance.distance_matrix[i, j],
                    self.instance.distance_matrix[j, i],
                )

    def test_feasible_arcs_no_self_loops(self):
        """Feasible arcs do not contain self-loops."""
        for i, j in self.instance.feasible_arcs:
            self.assertNotEqual(i, j)

    def test_reproducibility(self):
        """Same seed produces the same instance."""
        inst2 = generate_instance(n_customers=20, n_drones=4, seed=42)
        self.assertEqual(self.instance.n_customers, inst2.n_customers)
        for c1, c2 in zip(self.instance.customers, inst2.customers):
            self.assertAlmostEqual(c1.x, c2.x)
            self.assertAlmostEqual(c1.y, c2.y)
            self.assertAlmostEqual(c1.demand, c2.demand)


if __name__ == "__main__":
    unittest.main()
