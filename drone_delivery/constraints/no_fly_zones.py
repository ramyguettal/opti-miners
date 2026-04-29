"""
constraints/no_fly_zones.py
===========================
No-Fly Zone geometry, line-segment intersection tests, and feasible-arc
construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class NoFlyZone:
    """A circular no-fly zone.

    Attributes:
        center: (x, y) centre of the forbidden area [metres].
        radius: Radius of the forbidden area [metres].
        label:  Human-readable label (e.g. 'NFZ-A').
    """
    center: tuple[float, float]
    radius: float
    label: str


def arc_crosses_nfz(
    p1: tuple[float, float],
    p2: tuple[float, float],
    nfz: NoFlyZone,
) -> bool:
    """Return True if the straight-line segment p1→p2 enters *nfz*.

    Uses the standard line-segment to circle intersection test:
        1. Parameterise the segment as  P(t) = p1 + t·(p2 − p1),  t ∈ [0, 1].
        2. Compute the minimum distance from nfz.center to the segment.
        3. If that distance < nfz.radius the arc is blocked.

    Args:
        p1:  Start point (x, y).
        p2:  End point (x, y).
        nfz: The no-fly zone to test against.

    Returns:
        True if the segment intersects (enters) the NFZ circle.
    """
    cx, cy = nfz.center
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - cx, p1[1] - cy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - nfz.radius * nfz.radius

    if a == 0.0:
        # p1 == p2 — degenerate segment; just check if point is inside
        return c < 0.0

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False  # no intersection with the full line

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # The segment is parameterised for t ∈ [0, 1].
    # Intersection exists if the interval [t1, t2] overlaps [0, 1].
    return t1 < 1.0 and t2 > 0.0


def build_feasible_arcs(
    nodes: list[tuple[float, float]],
    nfz_list: list[NoFlyZone],
) -> set[tuple[int, int]]:
    """Pre-compute all feasible (i, j) arcs not blocked by any NFZ.

    Args:
        nodes:    Ordered list of node coordinates; index 0 = depot.
        nfz_list: List of NoFlyZone objects.

    Returns:
        Set of (i, j) tuples where the straight-line arc is free of NFZ
        intersections.
    """
    n = len(nodes)
    feasible: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            blocked = any(
                arc_crosses_nfz(nodes[i], nodes[j], nfz)
                for nfz in nfz_list
            )
            if not blocked:
                feasible.add((i, j))
    return feasible
