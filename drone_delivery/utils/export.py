"""
utils/export.py
===============
Export a Solution + Instance to JSON for the React UI dashboard.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from drone_delivery import config

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance
    from drone_delivery.model.solution import Solution
    from drone_delivery.optimization.genetic_algorithm import GAStats


def export_solution_json(
    instance: "DeliveryInstance",
    solution: "Solution",
    ga_stats: "GAStats",
    output_path: str = "results/solution.json",
    report=None,
) -> dict:
    """Export the optimisation results to a JSON file.

    Schema matches Prompt.txt §6.

    Args:
        instance:    The problem instance.
        solution:    The best solution found.
        ga_stats:    GA runtime statistics.
        output_path: Path to write the JSON file.
        report:      Optional ConstraintReport from checker.

    Returns:
        The dict that was serialised to JSON.
    """
    # ── instance data ────────────────────────────────────────────────────
    customers_json = [
        {
            "id": c.id,
            "x": round(c.x, 2),
            "y": round(c.y, 2),
            "demand": round(c.demand, 2),
        }
        for c in instance.customers
    ]

    nfz_json = [
        {
            "cx": round(nfz.center[0], 2),
            "cy": round(nfz.center[1], 2),
            "r": round(nfz.radius, 2),
            "label": nfz.label,
        }
        for nfz in instance.no_fly_zones
    ]

    # ── routes ───────────────────────────────────────────────────────────
    routes_json = []
    for route in solution.routes:
        # Build full coordinate path: depot → customers → depot
        coords = [list(instance.depot)]
        for c_idx in route.sequence:
            coords.append(list(instance.node_coord(c_idx)))
        coords.append(list(instance.depot))

        seg_energies = route.segment_energies(instance)

        routes_json.append({
            "drone_id": route.drone_id,
            "sequence": [instance.customers[c_idx - 1].id for c_idx in route.sequence],
            "coordinates": [[round(x, 2), round(y, 2)] for x, y in coords],
            "energy_wh": round(route.total_energy, 4),
            "distance_m": round(route.total_distance, 4),
            "load_kg": round(route.total_load, 4),
            "segment_energies": [round(e, 4) for e in seg_energies],
        })

    # ── collision placeholder ────────────────────────────────────────────
    collisions: list[dict] = []  # §5 — optional advanced feature

    # ── served count ─────────────────────────────────────────────────────
    served_ids = set()
    for r in solution.routes:
        served_ids.update(r.sequence)
    served_count = len(served_ids)

    # ── constraints ──────────────────────────────────────────────────────
    constraints_json = None
    if report is not None:
        constraints_json = {
            "feasible": report.feasible,
            "checks": [
                {"name": "All customers served", "passed": report.all_served},
                {"name": "Payload feasible",     "passed": report.payload_feasible},
                {"name": "Energy feasible",      "passed": report.energy_feasible},
                {"name": "No NFZ violations",    "passed": report.nfz_feasible},
                {"name": "No collisions",        "passed": report.no_collisions},
            ],
            "violations": report.violations,
        }

    # ── assemble output ──────────────────────────────────────────────────
    output = {
        "instance": {
            "depot": [round(instance.depot[0], 2), round(instance.depot[1], 2)],
            "customers": customers_json,
            "no_fly_zones": nfz_json,
            "grid": list(getattr(instance, '_grid', config.GRID_SIZE)),
            "n_drones": instance.n_drones,
        },
        "solution": {
            "total_energy_wh": round(solution.total_energy, 4),
            "total_distance_m": round(solution.total_distance, 4),
            "feasible": solution.feasible,
            "served_count": served_count,
            "routes": routes_json,
            "collisions": collisions,
        },
        "optimization": {
            "algorithm": "SA + Local Search",
            "crossover": "Order Crossover (OX)",
            "mutation": "Swap / Inversion / Insert",
            "local_search": "2-opt, Or-opt, Swap",
            "generations": ga_stats.generations,
            "population_size": ga_stats.population_size,
            "convergence_curve": [round(v, 4) for v in ga_stats.convergence_curve],
            "runtime_seconds": ga_stats.runtime_seconds,
            "parameters": ga_stats.parameters,
        },
    }

    if constraints_json is not None:
        output["constraints"] = constraints_json

    # Write to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output

