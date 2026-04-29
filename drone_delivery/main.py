"""
main.py
=======
Entry point for the Drone-Based Delivery Optimization System.

Generates (or loads) an instance, runs the Genetic Algorithm, validates
the solution, exports JSON, and prints a summary.

Usage:
    python -m drone_delivery.main [--customers N] [--drones N] [--seed N]
                                   [--battery F] [--payload F]
                                   [--generations N] [--pop-size N]
                                   [--output PATH]
"""
from __future__ import annotations

import argparse
import sys

from drone_delivery import config
from drone_delivery.constraints.checker import check_solution
from drone_delivery.data.generator import generate_instance
from drone_delivery.optimization.genetic_algorithm import run_ga
from drone_delivery.utils.export import export_solution_json


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, run optimisation, and export results."""
    parser = argparse.ArgumentParser(
        description="Drone-Based Delivery Optimization (EC-CVRP-NFZ)"
    )
    parser.add_argument("--customers", type=int, default=config.NUM_CUSTOMERS,
                        help="Number of customers (default: %(default)s)")
    parser.add_argument("--drones", type=int, default=config.DRONE_COUNT,
                        help="Number of drones (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--battery", type=float, default=config.BATTERY_WH,
                        help="Battery capacity in Wh (default: %(default)s)")
    parser.add_argument("--payload", type=float, default=config.MAX_PAYLOAD_KG,
                        help="Max payload in kg (default: %(default)s)")
    parser.add_argument("--generations", type=int, default=config.GA_GENERATIONS,
                        help="GA generations (default: %(default)s)")
    parser.add_argument("--pop-size", type=int, default=config.GA_POP_SIZE,
                        help="GA population size (default: %(default)s)")
    parser.add_argument("--output", type=str, default="results/solution.json",
                        help="Output JSON path (default: %(default)s)")
    args = parser.parse_args(argv)

    # ── Step 1: generate instance ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DRONE-BASED DELIVERY OPTIMIZATION")
    print(f"{'='*60}")
    print(f"  Customers : {args.customers}")
    print(f"  Drones    : {args.drones}")
    print(f"  Battery   : {args.battery} Wh")
    print(f"  Payload   : {args.payload} kg")
    print(f"  Seed      : {args.seed}")
    print(f"{'='*60}\n")

    instance = generate_instance(
        n_customers=args.customers,
        n_drones=args.drones,
        seed=args.seed,
        battery_wh=args.battery,
        max_payload_kg=args.payload,
    )
    print(f"Instance generated: {instance.n_customers} customers, "
          f"{len(instance.no_fly_zones)} NFZs, "
          f"{len(instance.feasible_arcs)} feasible arcs\n")

    # ── Step 2: run GA ───────────────────────────────────────────────────
    best_solution, ga_stats = run_ga(
        instance,
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed,
        max_payload=args.payload,
        battery=args.battery,
        verbose=True,
    )

    # ── Step 3: validate ─────────────────────────────────────────────────
    report = check_solution(best_solution, instance,
                            max_payload=args.payload, battery=args.battery)
    print(f"\n  Constraint Report:")
    print(f"    All served      : {'PASS' if report.all_served else 'FAIL'}")
    print(f"    Payload feasible: {'PASS' if report.payload_feasible else 'FAIL'}")
    print(f"    Energy feasible : {'PASS' if report.energy_feasible else 'FAIL'}")
    print(f"    NFZ feasible    : {'PASS' if report.nfz_feasible else 'FAIL'}")
    print(f"    Overall         : {'FEASIBLE' if report.feasible else 'INFEASIBLE'}")
    if report.violations:
        for v in report.violations:
            print(f"    !! {v}")

    # ── Step 4: export JSON ──────────────────────────────────────────────
    export_solution_json(instance, best_solution, ga_stats, output_path=args.output)
    print(f"\n  Solution exported to: {args.output}")

    # ── Route summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ROUTE SUMMARY")
    print(f"{'='*60}")
    for route in best_solution.routes:
        if not route.sequence:
            continue
        seq_str = " -> ".join(["0"] + [str(c) for c in route.sequence] + ["0"])
        print(f"  Drone {route.drone_id}: {seq_str}")
        print(f"    Energy: {route.total_energy:.2f} Wh | "
              f"Load: {route.total_load:.2f} kg | "
              f"Dist: {route.total_distance:.1f} m | "
              f"Stops: {len(route.sequence)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
