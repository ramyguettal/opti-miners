"""
data/loader.py
==============
Loads the pre-processed real dataset from CSV/JSON files produced by
``data pre processing/adapt_dataset.py`` and builds a DeliveryInstance
ready for the GA optimiser.

The UI sliders for ``drones``, ``battery``, and ``payload`` act as
**query / override parameters** that let the user experiment with
different fleet configurations on top of the fixed customer locations
and no-fly zones from the real data.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from drone_delivery.model.customer import Customer
from drone_delivery.constraints.no_fly_zones import NoFlyZone
from drone_delivery.data.instance import DeliveryInstance


# ── helpers ──────────────────────────────────────────────────────────────────

def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _energy(dist: float, demand: float, max_payload: float,
            alpha: float, beta: float, mode: str) -> float:
    """Energy for one loaded arc.

    Modes
    -----
    distance_x_weight  : alpha * dist * (demand / max_payload)
    distance           : alpha * dist
    """
    if mode == "distance_x_weight" and max_payload > 0:
        weight_factor = min(demand / max_payload, 1.0)
    else:
        weight_factor = 1.0
    return alpha * dist * weight_factor + beta * dist


# ── main loader ───────────────────────────────────────────────────────────────

def load_instance(
    data_dir: str | Path,
    *,
    n_drones: int | None = None,
    battery_override: float | None = None,
    payload_override: float | None = None,
    max_customers: int | None = None,
    alpha: float = 0.05,
    beta: float = 0.002,
) -> DeliveryInstance:
    """Build a :class:`DeliveryInstance` from pre-processed CSV/JSON files.

    Parameters
    ----------
    data_dir:
        Directory containing ``customers.csv``, ``no_fly_zones.csv`` and
        ``parameters.json``.
    n_drones:
        Override the number of drones (from UI slider). If *None*, the
        value from ``parameters.json`` is used.
    battery_override:
        Override the battery capacity (from UI slider).  If *None*, the
        value from ``parameters.json`` is used (scaled to Wh units).
    payload_override:
        Override the maximum payload (from UI slider).  If *None*, the
        value from ``parameters.json`` is used (scaled to kg).
    max_customers:
        Truncate to the first *max_customers* rows (useful for quick runs).
    alpha, beta:
        Energy-model coefficients (Wh per metre).
    """
    data_dir = Path(data_dir)

    # ── 1. load parameters ────────────────────────────────────────────────
    params_path = data_dir / "parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(f"parameters.json not found in {data_dir}")
    with open(params_path) as f:
        params = json.load(f)

    grid_rows = params["grid"]["rows"]
    grid_cols = params["grid"]["cols"]
    depot_xy  = (float(params["depot"]["x"]), float(params["depot"]["y"]))
    grid_size = (grid_cols, grid_rows)          # (width, height)

    # Drone specs — UI sliders may override
    drone_params  = params.get("drones", {})
    n_drones      = n_drones      if n_drones      is not None else int(drone_params.get("count", 4))
    max_payload   = payload_override if payload_override is not None else float(drone_params.get("max_payload", 200))
    battery_cap   = battery_override if battery_override is not None else float(drone_params.get("battery_capacity", 1178.78))

    energy_mode   = params.get("energy_model", {}).get("mode", "distance_x_weight")

    # ── 2. load customers ─────────────────────────────────────────────────
    cust_path = data_dir / "customers.csv"
    if not cust_path.exists():
        raise FileNotFoundError(f"customers.csv not found in {data_dir}")

    customers: list[Customer] = []
    with open(cust_path, newline="") as f:
        for row in csv.DictReader(f):
            cid    = int(row["id"])
            cx     = float(row["x"])
            cy     = float(row["y"])
            demand = float(row["demand_weight"])
            # Skip customers whose single-trip demand already exceeds payload
            if demand > max_payload:
                continue
            customers.append(Customer(id=cid, x=cx, y=cy, demand=demand))
            if max_customers and len(customers) >= max_customers:
                break

    if not customers:
        raise ValueError("No feasible customers found (all demands exceed payload).")

    # ── 3. load no-fly zones ──────────────────────────────────────────────
    nfz_path = data_dir / "no_fly_zones.csv"
    nfzs: list[NoFlyZone] = []
    if nfz_path.exists():
        with open(nfz_path, newline="") as f:
            for row in csv.DictReader(f):
                r_min = float(row["r_min"])
                r_max = float(row["r_max"])
                c_min = float(row["c_min"])
                c_max = float(row["c_max"])
                # Convert bounding box → circumscribed circle
                cx = (c_min + c_max) / 2.0
                cy = (r_min + r_max) / 2.0
                radius = math.sqrt(((c_max - c_min) / 2) ** 2 + ((r_max - r_min) / 2) ** 2)
                nfzs.append(NoFlyZone(
                    center=(cx, cy),
                    radius=radius,
                    label=f"NFZ-{row['zone_id']}",
                ))

    # ── 4. build distance + energy matrices ───────────────────────────────
    n = len(customers)
    # Index 0 = depot, indices 1..n = customers
    coords = [depot_xy] + [(c.x, c.y) for c in customers]
    demands = [0.0] + [c.demand for c in customers]

    dist_mat   = np.zeros((n + 1, n + 1))
    energy_mat = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            d = _euclidean(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dist_mat[i, j] = dist_mat[j, i] = d
            # Use the heavier load direction for energy (conservative)
            load = max(demands[i], demands[j])
            e = _energy(d, load, max_payload, alpha, beta, energy_mode)
            energy_mat[i, j] = energy_mat[j, i] = e

    # ── 5. compute feasible arcs (those not blocked by any NFZ) ──────────
    from drone_delivery.constraints.no_fly_zones import build_feasible_arcs
    coords_list = [depot_xy] + [(c.x, c.y) for c in customers]
    feasible_arcs = build_feasible_arcs(coords_list, nfzs)

    instance = DeliveryInstance(
        depot=depot_xy,
        customers=customers,
        n_drones=n_drones,
        distance_matrix=dist_mat,
        energy_matrix=energy_mat,
        no_fly_zones=nfzs,
        feasible_arcs=feasible_arcs,
    )
    # Store grid for export
    instance._grid = grid_size          # type: ignore[attr-defined]
    instance._battery = battery_cap     # type: ignore[attr-defined]
    instance._payload = max_payload     # type: ignore[attr-defined]
    return instance
