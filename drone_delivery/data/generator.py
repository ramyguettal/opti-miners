"""
data/generator.py
=================
Random instance generator for the Drone-Based Delivery Optimization problem.
"""
from __future__ import annotations

import random

import numpy as np

from drone_delivery import config
from drone_delivery.constraints.no_fly_zones import NoFlyZone, build_feasible_arcs
from drone_delivery.data.instance import DeliveryInstance
from drone_delivery.model.customer import Customer
from drone_delivery.utils.distance import (
    build_distance_matrix,
    build_energy_matrix,
    euclidean_distance,
)


def generate_instance(
    n_customers: int = config.NUM_CUSTOMERS,
    n_drones: int = config.DRONE_COUNT,
    seed: int = config.RANDOM_SEED,
    battery_wh: float = config.BATTERY_WH,
    max_payload_kg: float = config.MAX_PAYLOAD_KG,
) -> DeliveryInstance:
    """Generate a random delivery-optimisation instance.

    - Depot placed at the centre of the grid.
    - Customers placed uniformly at random, guaranteed outside NFZs.
    - Demands drawn uniformly from [0.5, 2.5] kg.
    - No-fly zones are circular forbidden areas placed randomly,
      verified not to cover the depot or any customer.

    Args:
        n_customers:    Number of customer nodes to generate.
        n_drones:       Number of drones in the fleet.
        seed:           Random seed for reproducibility.
        battery_wh:     Battery capacity [Wh] (unused here, stored in config).
        max_payload_kg: Max payload [kg] (unused here, stored in config).

    Returns:
        A fully constructed DeliveryInstance ready for optimisation.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    gw, gh = config.GRID_SIZE
    depot = (gw / 2.0, gh / 2.0)  # centre of grid

    # ── generate no-fly zones ────────────────────────────────────────────
    nfz_list: list[NoFlyZone] = []
    labels = [f"NFZ-{chr(65 + i)}" for i in range(config.NFZ_COUNT)]
    for i in range(config.NFZ_COUNT):
        for _ in range(500):
            cx = rng.uniform(100, gw - 100)
            cy = rng.uniform(100, gh - 100)
            radius = rng.uniform(50, 90)
            # NFZ must not cover depot
            if euclidean_distance((cx, cy), depot) < radius + 20:
                continue
            # NFZ must not overlap another NFZ excessively
            overlap = False
            for prev in nfz_list:
                if euclidean_distance((cx, cy), prev.center) < radius + prev.radius:
                    overlap = True
                    break
            if overlap:
                continue
            nfz_list.append(NoFlyZone(center=(cx, cy), radius=radius, label=labels[i]))
            break

    # ── generate customers ───────────────────────────────────────────────
    customers: list[Customer] = []
    cid = 1
    attempts = 0
    while len(customers) < n_customers and attempts < 5000:
        attempts += 1
        x = rng.uniform(30, gw - 30)
        y = rng.uniform(30, gh - 30)
        # Must be outside every NFZ (with small buffer)
        inside = any(
            euclidean_distance((x, y), nfz.center) < nfz.radius + 10
            for nfz in nfz_list
        )
        if inside:
            continue
        demand = round(rng.uniform(0.5, 2.5), 2)
        customers.append(Customer(id=cid, x=x, y=y, demand=demand))
        cid += 1

    # ── build node list: index 0 = depot, 1..n = customers ──────────────
    nodes = [depot] + [(c.x, c.y) for c in customers]

    # ── precompute matrices ──────────────────────────────────────────────
    dist_matrix = build_distance_matrix(nodes)
    energy_matrix = build_energy_matrix(dist_matrix, alpha=config.ALPHA)

    # ── precompute feasible arcs ─────────────────────────────────────────
    feasible_arcs = build_feasible_arcs(nodes, nfz_list)

    return DeliveryInstance(
        depot=depot,
        customers=customers,
        n_drones=n_drones,
        distance_matrix=dist_matrix,
        energy_matrix=energy_matrix,
        no_fly_zones=nfz_list,
        feasible_arcs=feasible_arcs,
    )
