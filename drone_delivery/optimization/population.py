"""
optimization/population.py
===========================
Population initialization strategies for the Genetic Algorithm.

Two construction heuristics are provided:
  A) Nearest-Neighbor
  B) Clarke-Wright Savings Algorithm
Plus random permutations for diversity.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from drone_delivery import config
from drone_delivery.model.route import Route
from drone_delivery.model.solution import Solution
from drone_delivery.utils.distance import energy_for_arc

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — Nearest Neighbor Heuristic
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_init(
    instance: "DeliveryInstance",
    rng: random.Random | None = None,
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> list[int]:
    """Build a customer permutation using nearest-neighbour from a random start.

    The result is a flat permutation of customer indices (1..n) that can be
    decoded into drone routes by the GA decoder.

    Args:
        instance:    The problem instance.
        rng:         Random number generator (for shuffling start).
        max_payload: Max drone payload [kg].
        battery:     Battery capacity [Wh].

    Returns:
        Permutation of customer indices [1..n].
    """
    n = instance.n_customers
    if rng is None:
        rng = random.Random()

    unvisited = list(range(1, n + 1))
    rng.shuffle(unvisited)  # randomise starting customer

    permutation: list[int] = []
    current_pos = 0  # start at depot
    current_load = 0.0
    current_energy = 0.0

    while unvisited:
        best_idx = -1
        best_dist = float("inf")

        for i, cust in enumerate(unvisited):
            d = instance.distance_matrix[current_pos, cust]
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx == -1:
            break

        chosen = unvisited.pop(best_idx)
        permutation.append(chosen)
        current_pos = chosen

    return permutation


def savings_algorithm_init(
    instance: "DeliveryInstance",
    rng: random.Random | None = None,
) -> list[int]:
    """Build a customer permutation using the Clarke-Wright Savings algorithm.

    1. Compute savings s(i,j) = d(0,i) + d(0,j) - d(i,j)  for all i,j in C.
    2. Sort by savings descending.
    3. Greedily merge route-ends to form chains.
    4. Flatten chains into a single permutation.

    Args:
        instance: The problem instance.
        rng:      Random number generator (for tie-breaking).

    Returns:
        Permutation of customer indices [1..n].
    """
    n = instance.n_customers
    if rng is None:
        rng = random.Random()

    dm = instance.distance_matrix

    # Compute savings
    savings: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dm[0, i] + dm[0, j] - dm[i, j]
            savings.append((s, i, j))

    savings.sort(key=lambda x: -x[0])  # descending

    # Each customer starts in its own chain
    # chain_of[c] = id of the chain that customer c belongs to
    chain_of: dict[int, int] = {c: c for c in range(1, n + 1)}
    chains: dict[int, list[int]] = {c: [c] for c in range(1, n + 1)}

    for s_val, i, j in savings:
        ci, cj = chain_of[i], chain_of[j]
        if ci == cj:
            continue  # already in same chain

        chain_i = chains[ci]
        chain_j = chains[cj]

        # Can only merge if i is at an end of its chain and j at an end of its
        if chain_i[0] == i:
            if chain_j[-1] == j:
                # merge: chain_j + chain_i
                merged = chain_j + chain_i
            elif chain_j[0] == j:
                # merge: reversed(chain_j) + chain_i
                merged = chain_j[::-1] + chain_i
            else:
                continue
        elif chain_i[-1] == i:
            if chain_j[0] == j:
                # merge: chain_i + chain_j
                merged = chain_i + chain_j
            elif chain_j[-1] == j:
                # merge: chain_i + reversed(chain_j)
                merged = chain_i + chain_j[::-1]
            else:
                continue
        else:
            continue

        # Update bookkeeping
        new_id = ci
        chains[new_id] = merged
        if cj in chains:
            del chains[cj]
        for c in merged:
            chain_of[c] = new_id

    # Flatten all chains into a single permutation
    permutation: list[int] = []
    for chain in chains.values():
        permutation.extend(chain)

    return permutation


# ─────────────────────────────────────────────────────────────────────────────
# Population factory
# ─────────────────────────────────────────────────────────────────────────────

def create_initial_population(
    instance: "DeliveryInstance",
    pop_size: int = config.GA_POP_SIZE,
    seed: int = config.RANDOM_SEED,
) -> list[list[int]]:
    """Create an initial population of customer permutations.

    Composition (as specified in Prompt.txt §3.2):
        - 30 % nearest-neighbour heuristic with random starts
        - 30 % savings algorithm variants
        - 40 % random valid permutations

    Args:
        instance: The problem instance.
        pop_size: Total population size.
        seed:     Base random seed.

    Returns:
        List of permutations (each is a list of customer indices 1..n).
    """
    rng = random.Random(seed)
    n = instance.n_customers
    population: list[list[int]] = []

    nn_count = int(pop_size * 0.30)
    sw_count = int(pop_size * 0.30)
    rand_count = pop_size - nn_count - sw_count

    # 30 % nearest-neighbour
    for i in range(nn_count):
        sub_rng = random.Random(seed + i * 7)
        perm = nearest_neighbor_init(instance, sub_rng)
        population.append(perm)

    # 30 % savings algorithm
    for i in range(sw_count):
        sub_rng = random.Random(seed + 1000 + i * 13)
        perm = savings_algorithm_init(instance, sub_rng)
        # Slight perturbation for diversity
        if i > 0:
            idx1 = rng.randint(0, n - 1)
            idx2 = rng.randint(0, n - 1)
            perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
        population.append(perm)

    # 40 % random permutations
    for i in range(rand_count):
        perm = list(range(1, n + 1))
        sub_rng = random.Random(seed + 2000 + i * 3)
        sub_rng.shuffle(perm)
        population.append(perm)

    return population
