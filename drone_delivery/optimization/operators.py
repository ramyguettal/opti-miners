"""
optimization/operators.py
=========================
Genetic operators: Order Crossover (OX) and three mutation strategies.
"""
from __future__ import annotations

import random


# ─────────────────────────────────────────────────────────────────────────────
# Crossover — Order Crossover (OX)
# ─────────────────────────────────────────────────────────────────────────────

def order_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Order Crossover (OX) producing two children.

    1. Select a random segment from parent 1 → copy to child 1 at same positions.
    2. Fill remaining positions with genes from parent 2 in order, skipping
       those already present.
    3. Symmetric for child 2.

    Args:
        parent1: First parent permutation.
        parent2: Second parent permutation.
        rng:     Random number generator.

    Returns:
        Tuple of two child permutations.
    """
    n = len(parent1)
    if n < 2:
        return list(parent1), list(parent2)

    # Pick two crossover points
    pt1 = rng.randint(0, n - 2)
    pt2 = rng.randint(pt1 + 1, n - 1)

    child1 = _ox_child(parent1, parent2, pt1, pt2)
    child2 = _ox_child(parent2, parent1, pt1, pt2)
    return child1, child2


def _ox_child(
    donor: list[int],
    filler: list[int],
    pt1: int,
    pt2: int,
) -> list[int]:
    """Create one OX child.

    Args:
        donor:  Parent whose segment is copied.
        filler: Parent providing the remaining order.
        pt1:    Start of copied segment (inclusive).
        pt2:    End of copied segment (inclusive).

    Returns:
        Child permutation.
    """
    n = len(donor)
    child: list[int | None] = [None] * n
    child[pt1:pt2 + 1] = donor[pt1:pt2 + 1]
    segment_set = set(donor[pt1:pt2 + 1])

    fill_order = [g for g in filler if g not in segment_set]
    pos = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill_order[pos]
            pos += 1

    return child  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Mutation operators
# ─────────────────────────────────────────────────────────────────────────────

def swap_mutation(perm: list[int], rng: random.Random) -> list[int]:
    """Swap mutation: pick 2 random positions, swap their genes.

    Args:
        perm: Input permutation (not modified in-place).
        rng:  Random number generator.

    Returns:
        Mutated copy of the permutation.
    """
    child = list(perm)
    n = len(child)
    if n < 2:
        return child
    i, j = rng.sample(range(n), 2)
    child[i], child[j] = child[j], child[i]
    return child


def inversion_mutation(perm: list[int], rng: random.Random) -> list[int]:
    """Inversion mutation: reverse a random sub-sequence.

    Args:
        perm: Input permutation.
        rng:  Random number generator.

    Returns:
        Mutated copy of the permutation.
    """
    child = list(perm)
    n = len(child)
    if n < 2:
        return child
    i = rng.randint(0, n - 2)
    j = rng.randint(i + 1, n - 1)
    child[i:j + 1] = reversed(child[i:j + 1])
    return child


def insertion_mutation(perm: list[int], rng: random.Random) -> list[int]:
    """Insertion mutation: remove a gene and re-insert at a random position.

    Args:
        perm: Input permutation.
        rng:  Random number generator.

    Returns:
        Mutated copy of the permutation.
    """
    child = list(perm)
    n = len(child)
    if n < 2:
        return child
    i = rng.randint(0, n - 1)
    gene = child.pop(i)
    j = rng.randint(0, len(child))
    child.insert(j, gene)
    return child


def mutate(perm: list[int], rng: random.Random) -> list[int]:
    """Apply one of the three mutation operators at random.

    Args:
        perm: Input permutation.
        rng:  Random number generator.

    Returns:
        Mutated permutation.
    """
    choice = rng.randint(0, 2)
    if choice == 0:
        return swap_mutation(perm, rng)
    elif choice == 1:
        return inversion_mutation(perm, rng)
    else:
        return insertion_mutation(perm, rng)
