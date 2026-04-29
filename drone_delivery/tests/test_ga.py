"""
tests/test_ga.py
================
Unit tests for the Genetic Algorithm.
"""
import unittest

from drone_delivery.data.generator import generate_instance
from drone_delivery.optimization.genetic_algorithm import run_ga


class TestGA(unittest.TestCase):
    """Tests for optimization/genetic_algorithm.py."""

    def setUp(self):
        """Generate a small instance for fast testing."""
        self.instance = generate_instance(n_customers=5, n_drones=2, seed=42)

    def test_small_instance(self):
        """GA solves a 5-customer, 2-drone instance without crashing."""
        solution, stats = run_ga(
            self.instance,
            pop_size=20,
            generations=30,
            seed=42,
            max_payload=5.0,
            battery=300.0,  # generous battery
            verbose=False,
        )
        self.assertIsNotNone(solution)
        self.assertGreater(stats.generations, 0)

    def test_all_customers_served(self):
        """With generous constraints, all customers should be served."""
        solution, _ = run_ga(
            self.instance,
            pop_size=20,
            generations=50,
            seed=42,
            max_payload=20.0,   # very generous
            battery=5000.0,     # very generous
            verbose=False,
        )
        self.assertEqual(len(solution.unserved), 0,
                         f"Unserved customers: {solution.unserved}")

    def test_convergence(self):
        """Fitness should decrease or stay same over generations."""
        _, stats = run_ga(
            self.instance,
            pop_size=20,
            generations=30,
            seed=42,
            max_payload=20.0,
            battery=5000.0,
            verbose=False,
        )
        curve = stats.convergence_curve
        self.assertGreater(len(curve), 1)
        # Best fitness should never increase (it's tracked as global best)
        for i in range(1, len(curve)):
            self.assertLessEqual(
                curve[i], curve[i - 1] + 1e-6,
                f"Fitness increased at gen {i}: {curve[i-1]:.4f} -> {curve[i]:.4f}"
            )


if __name__ == "__main__":
    unittest.main()
