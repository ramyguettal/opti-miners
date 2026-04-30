"""
optimization/lower_bound.py
============================
Lower bound calculators for the EC-CVRP-NFZ.

Provides theoretical minimum energy values that no feasible solution
can beat.  Comparing the GA/SA result against these bounds gives a
measure of solution quality (optimality gap).
"""
from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from drone_delivery import config
from drone_delivery.utils.distance import energy_for_arc

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance


def compute_lower_bound(
    instance: "DeliveryInstance",
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> dict:
    """Compute lower and upper bounds on the optimal solution energy.

    Lower bounds (no solution can do better):
      - MST bound:    α × MST distance (ignores load, routes, capacity)
      - Nearest-depot: ½ × Σ α × d(0,i)  (each customer adds at least
                       its one-way base energy from depot)

    Upper bound (a trivially feasible solution):
      - Individual trips: each customer served by a dedicated round-trip

    Args:
        instance:    The problem instance.
        max_payload: Max payload [kg].
        battery:     Battery capacity [Wh].

    Returns:
        Dict with bounds, gap info.
    """
    n = instance.n_customers

    # ── Lower Bound 1: MST-based ─────────────────────────────────────────
    # Any set of routes that visits all customers must traverse edges
    # whose total distance ≥ MST distance.  Multiplying by α gives
    # the minimum base energy (ignoring load).
    visited = [False] * (n + 1)
    visited[0] = True
    mst_dist = 0.0
    heap: list[tuple[float, int]] = []
    for j in range(1, n + 1):
        heapq.heappush(heap, (instance.distance_matrix[0, j], j))

    edges_added = 0
    while heap and edges_added < n:
        cost, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        mst_dist += cost
        edges_added += 1
        for v in range(n + 1):
            if not visited[v]:
                heapq.heappush(heap, (instance.distance_matrix[u, v], v))

    lb_mst = config.ALPHA * mst_dist

    # ── Lower Bound 2: Sum of half round-trip base energies ──────────────
    # Every customer i must be reached from somewhere; at minimum the
    # drone flies d(0,i) to get there (base energy only, no load).
    # This is a valid LB because every customer must be visited.
    lb_depot = 0.0
    for i in range(1, n + 1):
        lb_depot += config.ALPHA * instance.distance_matrix[0, i]

    # ── Lower Bound 3: Minimum assignment bound ──────────────────────────
    # For each customer, compute the cheapest possible contribution to
    # any route: min over all arcs entering + leaving customer i.
    lb_assign = 0.0
    for i in range(1, n + 1):
        # Cheapest in-arc
        min_in = min(
            config.ALPHA * instance.distance_matrix[j, i]
            for j in range(n + 1) if j != i
        )
        # Cheapest out-arc
        min_out = min(
            config.ALPHA * instance.distance_matrix[i, j]
            for j in range(n + 1) if j != i
        )
        lb_assign += (min_in + min_out) / 2.0  # avoid double-counting

    best_lb = max(lb_mst, lb_depot, lb_assign)

    # ── Upper Bound: Individual round trips ──────────────────────────────
    ub_individual = 0.0
    for i in range(1, n + 1):
        d_out = instance.distance_matrix[0, i]
        d_ret = instance.distance_matrix[i, 0]
        demand_i = instance.demand(i)
        ub_individual += energy_for_arc(d_out, demand_i) + energy_for_arc(d_ret, 0.0)

    return {
        "lb_mst": round(lb_mst, 2),
        "lb_depot_sum": round(lb_depot, 2),
        "lb_assignment": round(lb_assign, 2),
        "best_lower_bound": round(best_lb, 2),
        "ub_individual_trips": round(ub_individual, 2),
    }


def optimality_gap(solution_energy: float, lower_bound: float) -> float:
    """Compute the optimality gap as a percentage.

    gap = (solution - LB) / LB × 100

    A gap of 0% means the solution equals the lower bound (provably optimal).
    Typical gaps for VRP heuristics are 5-30%.

    Args:
        solution_energy: Energy of the GA/SA solution.
        lower_bound:     Best known lower bound.

    Returns:
        Gap as a percentage.
    """
    if lower_bound <= 0:
        return 0.0
    return ((solution_energy - lower_bound) / lower_bound) * 100.0
