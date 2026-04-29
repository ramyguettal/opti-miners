"""
utils/distance.py
=================
Distance and energy computation utilities used throughout the project.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from drone_delivery import config


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points.

    Args:
        a: (x, y) of the first point.
        b: (x, y) of the second point.

    Returns:
        Distance in the same unit as the coordinates (metres).
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def energy_for_arc(
    distance: float,
    current_load: float,
    alpha: float = config.ALPHA,
    beta: float = config.BETA,
) -> float:
    """Energy consumed to fly a single arc.

    E(arc) = α × distance + β × current_load × distance

    Args:
        distance:      Length of the arc [m].
        current_load:  Payload the drone is carrying on this arc [kg].
        alpha:         Base energy coefficient [Wh/m].
        beta:          Weight-dependent coefficient [Wh/(kg·m)].

    Returns:
        Energy consumed [Wh].
    """
    return (alpha + beta * current_load) * distance


def build_distance_matrix(
    nodes: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Build an (n×n) Euclidean distance matrix for *nodes*.

    Args:
        nodes: Ordered sequence of (x, y); index 0 should be the depot.

    Returns:
        2-D numpy array of pairwise distances.
    """
    n = len(nodes)
    coords = np.array(nodes, dtype=np.float64)         # (n, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n,n,2)
    return np.sqrt((diff ** 2).sum(axis=2))             # (n, n)


def build_energy_matrix(
    distance_matrix: np.ndarray,
    alpha: float = config.ALPHA,
) -> np.ndarray:
    """Build a base energy matrix (no load component, just α×d).

    The full energy on an arc also depends on the drone's current load,
    which changes during the route.  This matrix stores only the
    distance-proportional part so callers can add β×load×d at runtime.

    Args:
        distance_matrix: Pre-computed distance matrix.
        alpha:           Base energy coefficient [Wh/m].

    Returns:
        2-D numpy array of base energies [Wh].
    """
    return alpha * distance_matrix
