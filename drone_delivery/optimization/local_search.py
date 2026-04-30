"""
optimization/local_search.py
=============================
Simulated Annealing (SA) for post-processing and hybridisation with the GA.

Uses three neighborhood operators:
  1. 2-opt (intra-route)
  2. Or-opt / Relocate (inter-route)
  3. Inter-route swap

SA accepts worse solutions with probability exp(-Δ/T), allowing escape
from local optima.  Temperature is cooled geometrically each iteration.
"""
from __future__ import annotations

import math
import random
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


def _solution_energy(solution: Solution, instance: "DeliveryInstance") -> float:
    """Total energy across all routes."""
    return sum(_route_energy(r.sequence, instance) for r in solution.routes)


# ─────────────────────────────────────────────────────────────────────────────
# Neighborhood moves (return new solution without modifying original)
# ─────────────────────────────────────────────────────────────────────────────

def _try_two_opt(
    solution: Solution,
    instance: "DeliveryInstance",
    rng: random.Random,
    max_payload: float,
    battery: float,
) -> Solution | None:
    """Try a random 2-opt move on a random route."""
    routes_with_len = [(idx, r) for idx, r in enumerate(solution.routes) if len(r.sequence) >= 2]
    if not routes_with_len:
        return None
    r_idx, route = rng.choice(routes_with_len)
    seq = route.sequence
    n = len(seq)
    i = rng.randint(0, n - 2)
    j = rng.randint(i + 1, n - 1)
    new_seq = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
    if not _is_route_feasible(new_seq, instance, max_payload, battery):
        return None
    # Build new solution
    new_routes = []
    for k, r in enumerate(solution.routes):
        nr = Route(drone_id=r.drone_id)
        nr.sequence = list(new_seq if k == r_idx else r.sequence)
        new_routes.append(nr)
    new_sol = Solution(routes=new_routes)
    new_sol.evaluate(instance, max_payload, battery)
    return new_sol


def _try_relocate(
    solution: Solution,
    instance: "DeliveryInstance",
    rng: random.Random,
    max_payload: float,
    battery: float,
) -> Solution | None:
    """Try moving a random customer from one route to another."""
    routes_with_cust = [(idx, r) for idx, r in enumerate(solution.routes) if r.sequence]
    if not routes_with_cust:
        return None
    r1_idx, r1 = rng.choice(routes_with_cust)
    pos = rng.randint(0, len(r1.sequence) - 1)
    cust = r1.sequence[pos]
    seq1_without = r1.sequence[:pos] + r1.sequence[pos + 1:]

    # Pick a target route (can be the same)
    r2_idx = rng.randint(0, len(solution.routes) - 1)
    r2 = solution.routes[r2_idx]
    seq2 = r2.sequence if r2_idx != r1_idx else seq1_without
    insert_pos = rng.randint(0, len(seq2))
    new_seq2 = seq2[:insert_pos] + [cust] + seq2[insert_pos:]

    if r1_idx == r2_idx and new_seq2 == r1.sequence:
        return None

    if not _is_route_feasible(new_seq2, instance, max_payload, battery):
        return None
    if r1_idx != r2_idx and not _is_route_feasible(seq1_without, instance, max_payload, battery):
        return None

    new_routes = []
    for k, r in enumerate(solution.routes):
        nr = Route(drone_id=r.drone_id)
        if k == r1_idx and k == r2_idx:
            nr.sequence = list(new_seq2)
        elif k == r1_idx:
            nr.sequence = list(seq1_without)
        elif k == r2_idx:
            nr.sequence = list(new_seq2)
        else:
            nr.sequence = list(r.sequence)
        new_routes.append(nr)
    new_sol = Solution(routes=new_routes)
    new_sol.evaluate(instance, max_payload, battery)
    return new_sol


def _try_swap(
    solution: Solution,
    instance: "DeliveryInstance",
    rng: random.Random,
    max_payload: float,
    battery: float,
) -> Solution | None:
    """Try swapping a customer between two routes."""
    routes_with_cust = [(idx, r) for idx, r in enumerate(solution.routes) if r.sequence]
    if len(routes_with_cust) < 2:
        return None
    (r1_idx, r1), (r2_idx, r2) = rng.sample(routes_with_cust, 2)
    i = rng.randint(0, len(r1.sequence) - 1)
    j = rng.randint(0, len(r2.sequence) - 1)

    new_seq1 = list(r1.sequence)
    new_seq2 = list(r2.sequence)
    new_seq1[i], new_seq2[j] = new_seq2[j], new_seq1[i]

    if not _is_route_feasible(new_seq1, instance, max_payload, battery):
        return None
    if not _is_route_feasible(new_seq2, instance, max_payload, battery):
        return None

    new_routes = []
    for k, r in enumerate(solution.routes):
        nr = Route(drone_id=r.drone_id)
        if k == r1_idx:
            nr.sequence = new_seq1
        elif k == r2_idx:
            nr.sequence = new_seq2
        else:
            nr.sequence = list(r.sequence)
        new_routes.append(nr)
    new_sol = Solution(routes=new_routes)
    new_sol.evaluate(instance, max_payload, battery)
    return new_sol


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def local_search(
    solution: Solution,
    instance: "DeliveryInstance",
    max_iter: int = config.LS_MAX_ITER,
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
    T_start: float = 500.0,
    T_end: float = 0.1,
    seed: int | None = None,
) -> Solution:
    """Simulated Annealing local search.

    Temperature starts at T_start and cools geometrically to T_end over
    max_iter iterations.  At each step a random neighbor is generated
    using one of {2-opt, relocate, swap}.  Better solutions are always
    accepted; worse solutions are accepted with probability exp(-Δ/T).

    Args:
        solution:    Starting solution (not modified — a copy is used).
        instance:    The problem instance.
        max_iter:    Maximum SA iterations.
        max_payload: Max payload [kg].
        battery:     Battery capacity [Wh].
        T_start:     Initial temperature.
        T_end:       Final temperature.
        seed:        Optional random seed.

    Returns:
        The best solution found during the SA run.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    # Deep-copy current solution
    current = _copy_solution(solution, instance, max_payload, battery)
    current_energy = _solution_energy(current, instance)

    best = _copy_solution(current, instance, max_payload, battery)
    best_energy = current_energy

    if max_iter <= 1:
        return best

    # Cooling rate
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    T = T_start

    move_funcs = [_try_two_opt, _try_relocate, _try_swap]

    for _ in range(max_iter):
        # Pick a random neighborhood
        move_fn = rng.choice(move_funcs)
        neighbor = move_fn(current, instance, rng, max_payload, battery)

        if neighbor is None:
            T *= cooling
            continue

        neighbor_energy = _solution_energy(neighbor, instance)
        delta = neighbor_energy - current_energy

        # SA acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            current = neighbor
            current_energy = neighbor_energy

            if current_energy < best_energy:
                best = _copy_solution(current, instance, max_payload, battery)
                best_energy = current_energy

        T *= cooling

    # Final metrics
    best.evaluate(instance, max_payload, battery)
    return best


def _copy_solution(
    solution: Solution,
    instance: "DeliveryInstance",
    max_payload: float,
    battery: float,
) -> Solution:
    """Create a deep copy of a solution."""
    new_routes = []
    for r in solution.routes:
        nr = Route(drone_id=r.drone_id)
        nr.sequence = list(r.sequence)
        new_routes.append(nr)
    new_sol = Solution(routes=new_routes)
    new_sol.evaluate(instance, max_payload, battery)
    return new_sol
