"""
optimization/local_search.py
=============================
Local search operators for post-processing and hybridisation with the GA.

Three intra-/inter-route improvement operators:
  1. 2-opt (intra-route)
  2. Or-opt / Relocate (inter-route)
  3. Inter-route swap

Strategy: First Improvement — accept the first move that improves cost.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from drone_delivery import config
from drone_delivery.model.route import Route
from drone_delivery.model.solution import Solution
from drone_delivery.utils.distance import energy_for_arc

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _route_energy(
    sequence: list[int],
    instance: "DeliveryInstance",
) -> float:
    """Compute the load-dependent energy for a route sequence (depot implicit).

    Args:
        sequence: Customer node indices in visit order.
        instance: The problem instance.

    Returns:
        Total energy [Wh].
    """
    if not sequence:
        return 0.0
    total_demand = sum(instance.demand(c) for c in sequence)
    current_load = total_demand
    prev = 0
    energy = 0.0
    for c in sequence:
        d = instance.distance_matrix[prev, c]
        energy += energy_for_arc(d, current_load)
        current_load -= instance.demand(c)
        prev = c
    d = instance.distance_matrix[prev, 0]
    energy += energy_for_arc(d, current_load)
    return energy


def _route_load(sequence: list[int], instance: "DeliveryInstance") -> float:
    """Total demand of a route sequence."""
    return sum(instance.demand(c) for c in sequence)


def _is_route_feasible(
    sequence: list[int],
    instance: "DeliveryInstance",
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> bool:
    """Check if a route sequence is feasible (payload, energy, NFZ)."""
    if _route_load(sequence, instance) > max_payload + 1e-9:
        return False
    if _route_energy(sequence, instance) > battery + 1e-9:
        return False
    # NFZ arcs
    if instance.feasible_arcs:
        prev = 0
        for c in sequence:
            if (prev, c) not in instance.feasible_arcs:
                return False
            prev = c
        if sequence and (sequence[-1], 0) not in instance.feasible_arcs:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. 2-opt (intra-route)
# ─────────────────────────────────────────────────────────────────────────────

def two_opt_route(
    route: Route,
    instance: "DeliveryInstance",
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> bool:
    """Apply 2-opt improvement to a single route (first-improvement).

    For each pair (i, j) with i < j, reverse the segment between positions
    i and j.  Accept if the new route is feasible and has lower energy.

    Args:
        route:       The route to improve (modified in-place).
        instance:    The problem instance.
        max_payload: Max payload [kg].
        battery:     Battery capacity [Wh].

    Returns:
        True if an improvement was found.
    """
    seq = route.sequence
    n = len(seq)
    if n < 2:
        return False

    best_energy = _route_energy(seq, instance)

    for i in range(n - 1):
        for j in range(i + 1, n):
            new_seq = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
            if not _is_route_feasible(new_seq, instance, max_payload, battery):
                continue
            new_energy = _route_energy(new_seq, instance)
            if new_energy < best_energy - 1e-9:
                route.sequence = new_seq
                route.compute_metrics(instance)
                return True  # first improvement

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Or-opt (Relocate)
# ─────────────────────────────────────────────────────────────────────────────

def or_opt(
    solution: Solution,
    instance: "DeliveryInstance",
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> bool:
    """Or-opt relocate: move one customer from its route to the best position
    in any route (including its own).

    First improvement strategy.

    Args:
        solution:    The solution to improve (modified in-place).
        instance:    The problem instance.
        max_payload: Max payload [kg].
        battery:     Battery capacity [Wh].

    Returns:
        True if an improvement was found.
    """
    routes = solution.routes
    total_energy_before = sum(_route_energy(r.sequence, instance) for r in routes)

    for r1_idx, r1 in enumerate(routes):
        for pos, cust in enumerate(r1.sequence):
            # Remove customer from r1
            seq1_without = r1.sequence[:pos] + r1.sequence[pos + 1:]

            for r2_idx, r2 in enumerate(routes):
                seq2 = r2.sequence if r2_idx != r1_idx else seq1_without

                # Try inserting at every position
                for insert_pos in range(len(seq2) + 1):
                    new_seq2 = seq2[:insert_pos] + [cust] + seq2[insert_pos:]

                    # Skip if it's the same configuration
                    if r1_idx == r2_idx and new_seq2 == r1.sequence:
                        continue

                    # Check feasibility
                    if not _is_route_feasible(new_seq2, instance, max_payload, battery):
                        continue
                    if r1_idx != r2_idx and not _is_route_feasible(
                        seq1_without, instance, max_payload, battery
                    ):
                        continue

                    # Compute new total energy
                    new_total = 0.0
                    for k, r in enumerate(routes):
                        if k == r1_idx and k == r2_idx:
                            new_total += _route_energy(new_seq2, instance)
                        elif k == r1_idx:
                            new_total += _route_energy(seq1_without, instance)
                        elif k == r2_idx:
                            new_total += _route_energy(new_seq2, instance)
                        else:
                            new_total += _route_energy(r.sequence, instance)

                    if new_total < total_energy_before - 1e-9:
                        # Apply the move
                        r1.sequence = seq1_without if r1_idx != r2_idx else new_seq2
                        if r1_idx != r2_idx:
                            r2.sequence = new_seq2
                        for r in routes:
                            r.compute_metrics(instance)
                        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inter-route swap
# ─────────────────────────────────────────────────────────────────────────────

def inter_route_swap(
    solution: Solution,
    instance: "DeliveryInstance",
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> bool:
    """Swap one customer from route r1 with one from route r2.

    First improvement strategy.

    Args:
        solution:    The solution to improve (modified in-place).
        instance:    The problem instance.
        max_payload: Max payload [kg].
        battery:     Battery capacity [Wh].

    Returns:
        True if an improvement was found.
    """
    routes = solution.routes
    total_energy_before = sum(_route_energy(r.sequence, instance) for r in routes)

    for r1_idx in range(len(routes)):
        for r2_idx in range(r1_idx + 1, len(routes)):
            r1 = routes[r1_idx]
            r2 = routes[r2_idx]

            for i, c1 in enumerate(r1.sequence):
                for j, c2 in enumerate(r2.sequence):
                    # Swap c1 and c2
                    new_seq1 = list(r1.sequence)
                    new_seq2 = list(r2.sequence)
                    new_seq1[i] = c2
                    new_seq2[j] = c1

                    if not _is_route_feasible(new_seq1, instance, max_payload, battery):
                        continue
                    if not _is_route_feasible(new_seq2, instance, max_payload, battery):
                        continue

                    new_e1 = _route_energy(new_seq1, instance)
                    new_e2 = _route_energy(new_seq2, instance)
                    old_e1 = _route_energy(r1.sequence, instance)
                    old_e2 = _route_energy(r2.sequence, instance)

                    other_energy = total_energy_before - old_e1 - old_e2
                    new_total = other_energy + new_e1 + new_e2

                    if new_total < total_energy_before - 1e-9:
                        r1.sequence = new_seq1
                        r2.sequence = new_seq2
                        r1.compute_metrics(instance)
                        r2.compute_metrics(instance)
                        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Combined local search
# ─────────────────────────────────────────────────────────────────────────────

def local_search(
    solution: Solution,
    instance: "DeliveryInstance",
    max_iter: int = config.LS_MAX_ITER,
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> Solution:
    """Run all local search operators until convergence or max_iter.

    Order: 2-opt on each route → or-opt → inter-route swap.
    Repeat until no operator finds an improvement.

    Args:
        solution:    The solution to improve (modified in-place).
        instance:    The problem instance.
        max_iter:    Maximum total iterations.
        max_payload: Max payload [kg].
        battery:     Battery capacity [Wh].

    Returns:
        The (improved) solution.
    """
    for _ in range(max_iter):
        improved = False

        # 2-opt on each route
        for route in solution.routes:
            if two_opt_route(route, instance, max_payload, battery):
                improved = True

        # Or-opt relocate
        if or_opt(solution, instance, max_payload, battery):
            improved = True

        # Inter-route swap
        if inter_route_swap(solution, instance, max_payload, battery):
            improved = True

        if not improved:
            break

    # Recompute solution-level metrics
    solution.evaluate(instance, max_payload, battery)
    return solution
