"""
data/instance.py
================
DeliveryInstance dataclass — holds the complete problem instance including
depot, customers, distance/energy matrices, drones, and no-fly zones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drone_delivery.model.customer import Customer
    from drone_delivery.constraints.no_fly_zones import NoFlyZone


@dataclass
class DeliveryInstance:
    """Complete delivery-optimisation problem instance.

    Attributes:
        depot:          (x, y) coordinate of the central depot.
        customers:      List of Customer objects (index 0 → customer-list[0]).
        n_drones:       Number of available drones.
        distance_matrix: (n+1)×(n+1) Euclidean distance matrix where index 0 = depot.
        energy_matrix:  (n+1)×(n+1) base energy matrix (α×d, ignoring load).
        no_fly_zones:   List of NoFlyZone objects.
        feasible_arcs:  Set of (i, j) node-index pairs NOT blocked by any NFZ.
    """
    depot: tuple[float, float]
    customers: list["Customer"]
    n_drones: int
    distance_matrix: np.ndarray
    energy_matrix: np.ndarray
    no_fly_zones: list["NoFlyZone"] = field(default_factory=list)
    feasible_arcs: set[tuple[int, int]] = field(default_factory=set)

    # ── convenience helpers ──────────────────────────────────────────────
    @property
    def n_customers(self) -> int:
        """Number of customers (excludes depot)."""
        return len(self.customers)

    def node_coord(self, idx: int) -> tuple[float, float]:
        """Return (x, y) for a node index (0 = depot, 1..n = customers)."""
        if idx == 0:
            return self.depot
        return (self.customers[idx - 1].x, self.customers[idx - 1].y)

    def demand(self, idx: int) -> float:
        """Return demand for node *idx* (depot = 0)."""
        if idx == 0:
            return 0.0
        return self.customers[idx - 1].demand
