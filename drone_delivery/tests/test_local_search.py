"""
tests/test_local_search.py
===========================
Unit tests for the Simulated Annealing local search.
"""
import unittest

from drone_delivery.data.generator import generate_instance
from drone_delivery.model.route import Route
from drone_delivery.model.solution import Solution
from drone_delivery.optimization.genetic_algorithm import decode_chromosome
from drone_delivery.optimization.local_search import local_search


class TestSimulatedAnnealing(unittest.TestCase):
    """Integration test for Simulated Annealing."""

    def test_sa_improves_or_maintains(self):
        """SA should improve or maintain overall solution quality."""
        instance = generate_instance(n_customers=8, n_drones=3, seed=42)
        chrom = list(range(1, 9))
        sol = decode_chromosome(chrom, instance, max_payload=20.0, battery=5000.0)
        energy_before = sol.total_energy

        sol = local_search(sol, instance, max_iter=50,
                           max_payload=20.0, battery=5000.0, seed=42)

        # SA with enough iterations should not worsen significantly
        # (best-so-far tracking ensures this)
        self.assertLessEqual(sol.total_energy, energy_before + 1e-6)

    def test_sa_preserves_customers(self):
        """SA should preserve all customer assignments."""
        instance = generate_instance(n_customers=10, n_drones=3, seed=42)
        chrom = list(range(1, 11))
        sol = decode_chromosome(chrom, instance, max_payload=20.0, battery=5000.0)

        before = set()
        for r in sol.routes:
            before.update(r.sequence)

        sol = local_search(sol, instance, max_iter=30,
                           max_payload=20.0, battery=5000.0, seed=42)

        after = set()
        for r in sol.routes:
            after.update(r.sequence)

        self.assertEqual(before, after)

    def test_sa_feasibility(self):
        """SA output routes must be feasible."""
        instance = generate_instance(n_customers=8, n_drones=3, seed=42)
        chrom = list(range(1, 9))
        sol = decode_chromosome(chrom, instance, max_payload=20.0, battery=5000.0)

        sol = local_search(sol, instance, max_iter=30,
                           max_payload=20.0, battery=5000.0, seed=42)

        for r in sol.routes:
            total_load = sum(instance.demand(c) for c in r.sequence)
            self.assertLessEqual(total_load, 20.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
