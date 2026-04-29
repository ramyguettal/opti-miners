"""
model/solution.py
=================
Solution dataclass — a complete set of routes (one per drone) with
cost evaluation and penalty-based fitness for the GA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from drone_delivery import config
from drone_delivery.model.route import Route

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance


@dataclass
class Solution:
    """A complete delivery solution consisting of one Route per drone.

    Attributes:
        routes:       List of Route objects (one per drone, some may be empty).
        total_energy: Sum of route energies [Wh].
        feasible:     True if all constraints are satisfied.
        unserved:     List of customer node-indices that were not assigned.
    """
    routes: list[Route] = field(default_factory=list)
    total_energy: float = 0.0
    feasible: bool = True
    unserved: list[int] = field(default_factory=list)

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self,
        instance: "DeliveryInstance",
        max_payload: float = config.MAX_PAYLOAD_KG,
        battery: float = config.BATTERY_WH,
    ) -> float:
        """Evaluate the solution and return fitness (lower = better).

        Steps:
            1. Recompute energy for each route (load-dependent).
            2. Identify unserved customers.
            3. Sum penalties for constraint violations.
            4. Return total_energy + penalty.

        Args:
            instance:    The problem instance.
            max_payload: Maximum payload capacity [kg].
            battery:     Battery capacity [Wh].

        Returns:
            Fitness value = total energy + penalties.
        """
        # Recompute route metrics
        for route in self.routes:
            route.compute_metrics(instance)

        self.total_energy = sum(r.total_energy for r in self.routes)

        # Check which customers are served
        served: set[int] = set()
        for route in self.routes:
            served.update(route.sequence)

        all_customers = set(range(1, instance.n_customers + 1))
        self.unserved = sorted(all_customers - served)

        # ── penalty computation ──────────────────────────────────────────
        penalty = 0.0

        # Unserved customers
        penalty += config.PENALTY_UNSERVED * len(self.unserved)

        # Per-route constraint violations
        self.feasible = True
        for route in self.routes:
            # Capacity overflow
            overflow_load = max(0.0, route.total_load - max_payload)
            if overflow_load > 1e-9:
                penalty += config.PENALTY_CAPACITY * overflow_load
                self.feasible = False

            # Energy overflow
            overflow_energy = max(0.0, route.total_energy - battery)
            if overflow_energy > 1e-9:
                penalty += config.PENALTY_ENERGY * overflow_energy
                self.feasible = False

            # NFZ violations
            if instance.feasible_arcs:
                prev = 0
                for cust_idx in route.sequence:
                    if (prev, cust_idx) not in instance.feasible_arcs:
                        penalty += config.PENALTY_UNSERVED
                        self.feasible = False
                    prev = cust_idx
                if route.sequence and (route.sequence[-1], 0) not in instance.feasible_arcs:
                    penalty += config.PENALTY_UNSERVED
                    self.feasible = False

        if self.unserved:
            self.feasible = False

        return self.total_energy + penalty

    @property
    def total_distance(self) -> float:
        """Sum of all route distances [m]."""
        return sum(r.total_distance for r in self.routes)
