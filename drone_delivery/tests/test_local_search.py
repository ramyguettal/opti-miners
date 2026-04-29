"""
tests/test_local_search.py
===========================
Unit tests for local search operators.
"""
import unittest

from drone_delivery.data.generator import generate_instance
from drone_delivery.model.route import Route
from drone_delivery.model.solution import Solution
from drone_delivery.optimization.genetic_algorithm import decode_chromosome
from drone_delivery.optimization.local_search import (
    inter_route_swap,
    local_search,
    or_opt,
    two_opt_route,
)


class TestTwoOpt(unittest.TestCase):
    """Tests for the 2-opt intra-route operator."""

    def test_two_opt_improves(self):
        """2-opt should improve or maintain solution quality."""
        instance = generate_instance(n_customers=8, n_drones=2, seed=42)
        route = Route(drone_id=0, sequence=[1, 2, 3, 4])
        route.compute_metrics(instance)
        energy_before = route.total_energy

        two_opt_route(route, instance, max_payload=20.0, battery=5000.0)
        self.assertLessEqual(route.total_energy, energy_before + 1e-6)


class TestOrOpt(unittest.TestCase):
    """Tests for the or-opt relocate operator."""

    def test_or_opt_feasibility(self):
        """Or-opt output is always feasible (with generous constraints)."""
        instance = generate_instance(n_customers=8, n_drones=3, seed=42)
        chrom = list(range(1, 9))
        sol = decode_chromosome(chrom, instance, max_payload=20.0, battery=5000.0)

        or_opt(sol, instance, max_payload=20.0, battery=5000.0)

        # All customers should still be assigned somewhere
        all_served = set()
        for r in sol.routes:
            all_served.update(r.sequence)
        expected = set(range(1, 9))
        self.assertEqual(all_served, expected)


class TestInterRouteSwap(unittest.TestCase):
    """Tests for the inter-route swap operator."""

    def test_swap_correctness(self):
        """Swap preserves all customer assignments."""
        instance = generate_instance(n_customers=8, n_drones=3, seed=42)
        chrom = list(range(1, 9))
        sol = decode_chromosome(chrom, instance, max_payload=20.0, battery=5000.0)

        before_customers = set()
        for r in sol.routes:
            before_customers.update(r.sequence)

        inter_route_swap(sol, instance, max_payload=20.0, battery=5000.0)

        after_customers = set()
        for r in sol.routes:
            after_customers.update(r.sequence)

        self.assertEqual(before_customers, after_customers)


class TestLocalSearch(unittest.TestCase):
    """Integration test for the combined local search."""

    def test_local_search_improves(self):
        """Local search should improve or maintain overall solution."""
        instance = generate_instance(n_customers=8, n_drones=3, seed=42)
        chrom = list(range(1, 9))
        sol = decode_chromosome(chrom, instance, max_payload=20.0, battery=5000.0)
        energy_before = sol.total_energy

        sol = local_search(sol, instance, max_iter=20,
                           max_payload=20.0, battery=5000.0)

        self.assertLessEqual(sol.total_energy, energy_before + 1e-6)


if __name__ == "__main__":
    unittest.main()
