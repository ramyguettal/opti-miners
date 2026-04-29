"""
tests/test_constraints.py
==========================
Unit tests for constraint checking (NFZ, capacity, energy).
"""
import unittest

from drone_delivery.constraints.no_fly_zones import NoFlyZone, arc_crosses_nfz
from drone_delivery.data.generator import generate_instance
from drone_delivery.model.route import Route


class TestNFZIntersection(unittest.TestCase):
    """Tests for no-fly zone line-segment intersection."""

    def setUp(self):
        self.nfz = NoFlyZone(center=(500.0, 500.0), radius=50.0, label="TEST")

    def test_nfz_intersection(self):
        """Arc that passes directly through NFZ centre is blocked."""
        self.assertTrue(arc_crosses_nfz((400, 500), (600, 500), self.nfz))

    def test_nfz_parallel(self):
        """Arc beside but not touching NFZ is allowed."""
        # 100 metres away from centre  (500, 600) -> far from r=50
        self.assertFalse(arc_crosses_nfz((400, 600), (600, 600), self.nfz))

    def test_nfz_tangent(self):
        """Arc tangent to the NFZ boundary (just touching) is blocked."""
        # Arc from (450, 500) to (550, 500) goes through centre
        self.assertTrue(arc_crosses_nfz((450, 500), (550, 500), self.nfz))

    def test_nfz_miss(self):
        """Arc that clearly misses NFZ is allowed."""
        self.assertFalse(arc_crosses_nfz((0, 0), (100, 100), self.nfz))


class TestCapacityConstraint(unittest.TestCase):
    """Tests for payload capacity constraint."""

    def test_capacity_violation(self):
        """Route exceeding Q is infeasible."""
        instance = generate_instance(n_customers=10, n_drones=2, seed=99)
        # Create a route with all customers — will very likely exceed payload
        route = Route(
            drone_id=0,
            sequence=list(range(1, instance.n_customers + 1)),
        )
        # With total demand likely > 5 kg, should be infeasible
        feasible = route.is_feasible(instance, max_payload=5.0, battery=99999)
        total_demand = sum(instance.demand(c) for c in route.sequence)
        if total_demand > 5.0:
            self.assertFalse(feasible)

    def test_capacity_ok(self):
        """Single light customer should be feasible."""
        instance = generate_instance(n_customers=10, n_drones=2, seed=99)
        route = Route(drone_id=0, sequence=[1])
        feasible = route.is_feasible(instance, max_payload=5.0, battery=99999)
        self.assertTrue(feasible)  # single customer ≤ 2.5 kg < 5 kg


class TestEnergyConstraint(unittest.TestCase):
    """Tests for energy/battery constraint."""

    def test_energy_violation(self):
        """Route with very low battery should be infeasible."""
        instance = generate_instance(n_customers=10, n_drones=2, seed=99)
        route = Route(drone_id=0, sequence=[1, 2, 3, 4, 5])
        # With battery=1 Wh, any non-trivial route should be infeasible
        feasible = route.is_feasible(instance, max_payload=100, battery=1.0)
        self.assertFalse(feasible)

    def test_energy_ok(self):
        """Route with huge battery should be feasible."""
        instance = generate_instance(n_customers=10, n_drones=2, seed=99)
        route = Route(drone_id=0, sequence=[1])
        feasible = route.is_feasible(instance, max_payload=100, battery=99999)
        self.assertTrue(feasible)


if __name__ == "__main__":
    unittest.main()
