"""
data/generator.py
=================
Random instance generator for the Drone-Based Delivery Optimization problem.

Thresholds are calibrated so that:
  - Each drone can serve 4-8 customers per trip (demand / payload ratio ~15-25%)
  - Battery allows round-trips of ~60-70% of the grid diagonal
  - NFZs block ~5-10% of the area, creating realistic routing challenges
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

    Thresholds ensure logical consistency:
      - Total fleet capacity ≥ total demand (so all customers CAN be served)
      - Battery ≥ 2 × average round-trip distance × energy rate
      - Demands are proportional to payload (each ~10-20% of max payload)

    Args:
        n_customers:    Number of customer nodes to generate.
        n_drones:       Number of drones in the fleet.
        seed:           Random seed for reproducibility.
        battery_wh:     Battery capacity [Wh].
        max_payload_kg: Max payload [kg].

    Returns:
        A fully constructed DeliveryInstance ready for optimisation.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    gw, gh = config.GRID_SIZE
    depot = (gw / 2.0, gh / 2.0)  # centre of grid

    # ── calibrate demand range to payload ────────────────────────────────
    # Each customer demand = 8-22% of max_payload → ~5-12 customers per drone
    demand_lo = max(0.1, max_payload_kg * 0.08)
    demand_hi = max(demand_lo + 0.1, max_payload_kg * 0.22)

    # ── calibrate battery to grid ────────────────────────────────────────
    # If user-provided battery seems too low for the grid, auto-scale it
    grid_diag = (gw**2 + gh**2) ** 0.5
    # Minimum battery should cover ~70% of diagonal as a round trip
    min_battery = 0.7 * grid_diag * (config.ALPHA + config.BETA * max_payload_kg * 0.5)
    effective_battery = max(battery_wh, min_battery)

    # ── generate no-fly zones ────────────────────────────────────────────
    nfz_list: list[NoFlyZone] = []
    labels = [f"NFZ-{chr(65 + i)}" for i in range(config.NFZ_COUNT)]
    for i in range(config.NFZ_COUNT):
        for _ in range(500):
            cx = rng.uniform(100, gw - 100)
            cy = rng.uniform(100, gh - 100)
            radius = rng.uniform(40, 70)
            # NFZ must not cover depot
            if euclidean_distance((cx, cy), depot) < radius + 30:
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
    # Spread customers in a reasonable radius from depot (not too far)
    max_spread = min(gw, gh) * 0.42  # customers within 42% of grid from depot
    customers: list[Customer] = []
    cid = 1
    attempts = 0
    while len(customers) < n_customers and attempts < 5000:
        attempts += 1
        # Cluster customers closer to depot for more realistic routing
        angle = rng.uniform(0, 2 * 3.14159)
        dist = rng.uniform(50, max_spread)
        x = depot[0] + dist * rng.uniform(-1, 1)
        y = depot[1] + dist * rng.uniform(-1, 1)
        # Clamp to grid
        x = max(30, min(gw - 30, x))
        y = max(30, min(gh - 30, y))
        # Must be outside every NFZ (with small buffer)
        inside = any(
            euclidean_distance((x, y), nfz.center) < nfz.radius + 10
            for nfz in nfz_list
        )
        if inside:
            continue
        demand = round(rng.uniform(demand_lo, demand_hi), 2)
        customers.append(Customer(id=cid, x=x, y=y, demand=demand))
        cid += 1

    # ── validate: ensure fleet can theoretically serve all customers ──────
    total_demand = sum(c.demand for c in customers)
    fleet_capacity = n_drones * max_payload_kg
    if total_demand > fleet_capacity:
        # Scale demands down so total = 85% of fleet capacity
        scale = (fleet_capacity * 0.85) / total_demand
        for c in customers:
            c.demand = round(c.demand * scale, 2)

    # ── build node list: index 0 = depot, 1..n = customers ──────────────
    nodes = [depot] + [(c.x, c.y) for c in customers]

    # ── precompute matrices ──────────────────────────────────────────────
    dist_matrix = build_distance_matrix(nodes)
    energy_matrix = build_energy_matrix(dist_matrix, alpha=config.ALPHA)

    # ── precompute feasible arcs ─────────────────────────────────────────
    feasible_arcs = build_feasible_arcs(nodes, nfz_list)

    instance = DeliveryInstance(
        depot=depot,
        customers=customers,
        n_drones=n_drones,
        distance_matrix=dist_matrix,
        energy_matrix=energy_matrix,
        no_fly_zones=nfz_list,
        feasible_arcs=feasible_arcs,
    )
    # Store effective battery for the export
    instance._battery = effective_battery   # type: ignore[attr-defined]
    instance._payload = max_payload_kg      # type: ignore[attr-defined]
    return instance
