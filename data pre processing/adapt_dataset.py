"""
adapt_dataset.py
================
Converts a HashCode 2016 drone-delivery .in file into a clean dataset
for the Drone-Based Delivery Optimization project.

Key adaptations
---------------
1. Single depot  → warehouse 0 is used as the unique depot.
2. Customers     → each HashCode order becomes one customer node,
                   characterised by (x, y, demand_weight).
3. Energy model  → energy consumed on a leg = Euclidean distance × segment_weight_factor
                   (weight_factor = demand / max_load, normalised 0-1).
                   Pure-distance mode is also available (see ENERGY_MODE).
4. Battery cap   → derived so that a drone can cover roughly BATTERY_FACTOR × (average
                   round-trip distance from depot) before needing to return.
5. Payload cap   → kept from the original file (max_load).
6. No-fly zones  → N_NO_FLY rectangular zones are generated randomly, avoiding the
                   depot and all customer locations.
7. Output        → four files
                     <input_name>.csv – id, x, y, n_items, demand_weight, product_types (ALL orders)
                     customers.csv   – id, x, y, demand_weight (feasible orders only)
                     parameters.json – depot, drone specs, energy model, no-fly zones
                     no_fly_zones.csv – zone_id, r_min, r_max, c_min, c_max

Usage
-----
    python adapt_dataset.py busy_day.in [--output-dir ./output]

Dependencies: standard library only (no numpy/pandas needed).
"""

import argparse
import csv
import json
import math
import os
import random
import sys

# ─── tuneable constants ────────────────────────────────────────────────────────
ENERGY_MODE     = "distance_x_weight"   # "distance" | "distance_x_weight"
BATTERY_FACTOR  = 3.5     # battery = BATTERY_FACTOR × avg round-trip distance
N_NO_FLY        = 5       # number of random no-fly rectangular zones
NO_FLY_SIZE     = 0.04    # each zone side ≈ NO_FLY_SIZE × grid dimension
RANDOM_SEED     = 42
# ───────────────────────────────────────────────────────────────────────────────


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def parse_hashcode(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    idx = 0

    # ── simulation header ──────────────────────────────────────────────────────
    h = lines[idx].split(); idx += 1
    rows, cols   = int(h[0]), int(h[1])
    n_drones     = int(h[2])
    deadline     = int(h[3])
    max_load     = int(h[4])

    # ── products ───────────────────────────────────────────────────────────────
    P = int(lines[idx]); idx += 1
    weights = list(map(int, lines[idx].split())); idx += 1
    assert len(weights) == P

    # ── warehouses ─────────────────────────────────────────────────────────────
    W = int(lines[idx]); idx += 1
    warehouses = []
    for _ in range(W):
        coord = tuple(map(int, lines[idx].split())); idx += 1
        stock = list(map(int, lines[idx].split())); idx += 1
        warehouses.append({"coord": coord, "stock": stock})

    # ── orders ─────────────────────────────────────────────────────────────────
    C = int(lines[idx]); idx += 1
    orders = []
    for oid in range(C):
        loc   = tuple(map(int, lines[idx].split())); idx += 1
        n     = int(lines[idx]); idx += 1
        types = list(map(int, lines[idx].split())); idx += 1
        demand_weight = sum(weights[t] for t in types)
        orders.append({"id": oid, "loc": loc, "n_items": n,
                       "product_types": types, "demand_weight": demand_weight})

    return {
        "rows": rows, "cols": cols,
        "n_drones": n_drones, "deadline": deadline, "max_load": max_load,
        "product_weights": weights,
        "warehouses": warehouses,
        "orders": orders,
    }


def compute_battery_capacity(depot, orders, factor):
    """Average round-trip distance from depot, scaled by factor."""
    if not orders:
        return 1.0
    avg_rt = sum(2 * euclidean(depot, o["loc"]) for o in orders) / len(orders)
    return round(avg_rt * factor, 2)


def generate_no_fly_zones(n, rows, cols, depot, customers, seed):
    """Random rectangles that do NOT cover depot or any customer."""
    rng   = random.Random(seed)
    h_size = max(1, int(rows * NO_FLY_SIZE))
    w_size = max(1, int(cols * NO_FLY_SIZE))
    blocked = {depot} | {(c["x"], c["y"]) for c in customers}
    zones = []
    attempts = 0
    while len(zones) < n and attempts < 10_000:
        attempts += 1
        r0 = rng.randint(0, rows - h_size - 1)
        c0 = rng.randint(0, cols - w_size - 1)
        r1, c1 = r0 + h_size, c0 + w_size
        # reject if any blocked point falls inside the zone
        if any(r0 <= p[0] <= r1 and c0 <= p[1] <= c1 for p in blocked):
            continue
        zones.append({"zone_id": len(zones),
                      "r_min": r0, "r_max": r1,
                      "c_min": c0, "c_max": c1})
    return zones


def adapt(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(RANDOM_SEED)

    print(f"[1/6] Parsing {input_path} ...")
    data = parse_hashcode(input_path)

    depot   = data["warehouses"][0]["coord"]   # warehouse 0 → single depot
    orders  = data["orders"]
    n_drones = data["n_drones"]
    max_load = data["max_load"]

    # ── write raw orders CSV (ALL orders, before filtering) ───────────────────
    print("[2/6] Writing raw input data as CSV ...")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    raw_csv_path = os.path.join(output_dir, f"{base_name}.csv")
    with open(raw_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "n_items", "demand_weight", "product_types"])
        for o in orders:
            w.writerow([
                o["id"],
                o["loc"][0],
                o["loc"][1],
                o["n_items"],
                o["demand_weight"],
                ";".join(str(t) for t in o["product_types"])
            ])

    # ── filter out orders whose demand_weight > max_load ──────────────────────
    feasible = [o for o in orders if o["demand_weight"] <= max_load]
    skipped  = len(orders) - len(feasible)
    if skipped:
        print(f"  [!] Skipped {skipped} orders whose demand exceeds max payload "
              f"({max_load}). Remaining: {len(feasible)}")
    customers = [{"id": o["id"], "x": o["loc"][0], "y": o["loc"][1],
                  "demand_weight": o["demand_weight"]} for o in feasible]

    # ── energy + battery ──────────────────────────────────────────────────────
    print("[3/6] Computing energy model and battery capacity ...")
    battery_cap = compute_battery_capacity(depot, feasible, BATTERY_FACTOR)

    if ENERGY_MODE == "distance_x_weight":
        energy_formula = "energy = distance × (demand_weight / max_load)"
    else:
        energy_formula = "energy = distance"

    # ── no-fly zones ──────────────────────────────────────────────────────────
    print("[4/6] Generating no-fly zones ...")
    nfz = generate_no_fly_zones(N_NO_FLY, data["rows"], data["cols"],
                                depot, customers, RANDOM_SEED)

    # ── write customers.csv ──────────────────────────────────────────────────
    print("[5/6] Writing output files ...")
    cust_path = os.path.join(output_dir, "customers.csv")
    with open(cust_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "x", "y", "demand_weight"])
        w.writeheader()
        w.writerows(customers)

    # ── write no_fly_zones.csv ────────────────────────────────────────────────
    nfz_path = os.path.join(output_dir, "no_fly_zones.csv")
    with open(nfz_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zone_id", "r_min", "r_max", "c_min", "c_max"])
        w.writeheader()
        w.writerows(nfz)

    # ── write parameters.json ─────────────────────────────────────────────────
    params = {
        "source_file": os.path.basename(input_path),
        "grid": {"rows": data["rows"], "cols": data["cols"]},
        "depot": {"x": depot[0], "y": depot[1]},
        "drones": {
            "count": n_drones,
            "max_payload": max_load,
            "battery_capacity": battery_cap,
            "battery_unit": "energy units (see energy_model)"
        },
        "energy_model": {
            "mode": ENERGY_MODE,
            "formula": energy_formula,
            "note": "distance = Euclidean; weight_factor = demand_weight / max_load"
        },
        "simulation": {
            "deadline_turns": data["deadline"]
        },
        "dataset_stats": {
            "total_orders_in_source": len(orders),
            "feasible_customers": len(customers),
            "skipped_overweight_orders": skipped,
            "no_fly_zones": len(nfz)
        },
        "assumptions": [
            "Each drone starts and ends its route at the depot (warehouse 0).",
            "Each customer must be served exactly once.",
            "A drone cannot exceed its max_payload on any single trip.",
            "A drone cannot exceed its battery_capacity on a single route.",
            "No-fly zones are rectangular areas; paths crossing them are forbidden.",
            "Energy on a leg = Euclidean_distance × (demand_weight / max_load) "
            "when mode is distance_x_weight, else pure Euclidean distance.",
            f"Battery capacity derived as {BATTERY_FACTOR}× the average depot "
            "round-trip distance across all customers.",
            f"No-fly zones are randomly generated (seed={RANDOM_SEED}), sized "
            f"~{int(NO_FLY_SIZE*100)}% of each grid dimension, and guaranteed "
            "not to cover the depot or any customer location."
        ]
    }
    params_path = os.path.join(output_dir, "parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    print("[6/6] Done!\n")
    print("=" * 58)
    print("  Output files")
    print(f"    {raw_csv_path}")
    print(f"    {cust_path}")
    print(f"    {nfz_path}")
    print(f"    {params_path}")
    print("=" * 58)
    print(f"  Grid              : {data['rows']} × {data['cols']}")
    print(f"  Depot (warehouse 0): {depot}")
    print(f"  Drones            : {n_drones}")
    print(f"  Max payload       : {max_load}")
    print(f"  Battery capacity  : {battery_cap:.2f} energy units")
    print(f"  Customers         : {len(customers)}")
    print(f"  No-fly zones      : {len(nfz)}")
    print(f"  Energy mode       : {ENERGY_MODE}")
    print("=" * 58)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adapt a HashCode 2016 .in file for drone delivery optimisation."
    )
    parser.add_argument("input", help="Path to the .in file (e.g. busy_day.in)")
    parser.add_argument("--output-dir", default="./adapted_dataset",
                        help="Directory to write the output files (default: ./adapted_dataset)")
    args = parser.parse_args()
    adapt(args.input, args.output_dir)
