"""
model/route.py
==============
Route dataclass — represents one drone's delivery tour and provides
feasibility checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from drone_delivery import config
from drone_delivery.utils.distance import energy_for_arc

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance


@dataclass
class Route:
    """A single drone route (depot → customers → depot).

    Attributes:
        drone_id:       Index of the drone executing this route.
        sequence:       Customer *node indices* (1-based) in visit order.
                        The depot (0) is implicit at start and end.
        total_distance: Total Euclidean distance of the tour [m].
        total_energy:   Total energy consumed [Wh] (load-dependent).
        total_load:     Sum of demands served on this route [kg].
    """
    drone_id: int
    sequence: list[int] = field(default_factory=list)
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_load: float = 0.0

    # ── evaluation ───────────────────────────────────────────────────────
    def compute_metrics(self, instance: "DeliveryInstance") -> None:
        """Re-compute distance, energy (load-dependent), and load from scratch.

        Energy on each arc accounts for the *current* payload carried by the
        drone (decreasing as deliveries are made).

        Args:
            instance: The problem instance (for distance matrix & demands).
        """
        if not self.sequence:
            self.total_distance = 0.0
            self.total_energy = 0.0
            self.total_load = 0.0
            return

        total_demand = sum(instance.demand(c) for c in self.sequence)
        self.total_load = total_demand

        dist = 0.0
        energy = 0.0
        current_load = total_demand  # starts fully loaded
        prev = 0  # depot

        for cust_idx in self.sequence:
            d = instance.distance_matrix[prev, cust_idx]
            dist += d
            energy += energy_for_arc(d, current_load)
            current_load -= instance.demand(cust_idx)
            prev = cust_idx

        # Return to depot
        d = instance.distance_matrix[prev, 0]
        dist += d
        energy += energy_for_arc(d, current_load)  # load after all deliveries

        self.total_distance = dist
        self.total_energy = energy

    def segment_energies(self, instance: "DeliveryInstance") -> list[float]:
        """Return per-segment energy values (for JSON export).

        Args:
            instance: The problem instance.

        Returns:
            List of energy values, one per arc in [depot → c1 → c2 → … → depot].
        """
        if not self.sequence:
            return []

        total_demand = sum(instance.demand(c) for c in self.sequence)
        current_load = total_demand
        prev = 0
        energies: list[float] = []

        for cust_idx in self.sequence:
            d = instance.distance_matrix[prev, cust_idx]
            energies.append(energy_for_arc(d, current_load))
            current_load -= instance.demand(cust_idx)
            prev = cust_idx

        # Return to depot
        d = instance.distance_matrix[prev, 0]
        energies.append(energy_for_arc(d, current_load))
        return energies

    # ── feasibility ──────────────────────────────────────────────────────
    def is_feasible(
        self,
        instance: "DeliveryInstance",
        max_payload: float = config.MAX_PAYLOAD_KG,
        battery: float = config.BATTERY_WH,
    ) -> bool:
        """Check whether this route satisfies all hard constraints.

        Checks performed:
            1. Total load ≤ max_payload  (C4).
            2. Total energy ≤ battery    (C5).
            3. All arcs are in feasible_arcs (C6 — no NFZ violations).

        Args:
            instance:    The problem instance.
            max_payload: Maximum payload per trip [kg].
            battery:     Battery capacity [Wh].

        Returns:
            True if the route is feasible under all constraints.
        """
        self.compute_metrics(instance)

        # C4: payload
        if self.total_load > max_payload + 1e-9:
            return False

        # C5: energy
        if self.total_energy > battery + 1e-9:
            return False

        # C6: NFZ — check all arcs in feasible set
        if instance.feasible_arcs:
            prev = 0
            for cust_idx in self.sequence:
                if (prev, cust_idx) not in instance.feasible_arcs:
                    return False
                prev = cust_idx
            if (prev, 0) not in instance.feasible_arcs:
                return False

        return True
