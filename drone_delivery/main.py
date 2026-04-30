"""
main.py
=======
Entry point for the Drone-Based Delivery Optimization System.

Two modes
---------
1. **Real data mode** (default):
        python -m drone_delivery.main --data-dir "data pre processing"
   Loads customers, NFZs, and baseline drone specs from the pre-processed
   CSV/JSON files.  UI sliders for drones / battery / payload override the
   dataset defaults so users can experiment with different fleet configs.

2. **Synthetic mode** (for unit-testing / demo):
        python -m drone_delivery.main --random --customers 20 --drones 4
   Generates a random instance.

Common options
--------------
    --drones N        Number of drones (overrides dataset default)
    --battery F       Battery capacity in Wh (overrides dataset default)
    --payload F       Max payload in kg (overrides dataset default)
    --generations N   GA generations
    --pop-size N      GA population size
    --output PATH     Where to write solution.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drone_delivery import config
from drone_delivery.constraints.checker import check_solution
from drone_delivery.optimization.genetic_algorithm import run_ga
from drone_delivery.utils.export import export_solution_json

# Default real-data directory (relative to project root)
DEFAULT_DATA_DIR = "data pre processing"


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, run optimisation, and export results."""
    parser = argparse.ArgumentParser(
        description="Opti Miners — Drone-Based Delivery Optimization (EC-CVRP-NFZ)"
    )

    # ── data source ──────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Directory with customers.csv / no_fly_zones.csv / parameters.json "
             "(default: '%(default)s')"
    )
    src.add_argument(
        "--random", action="store_true",
        help="Generate a random synthetic instance instead of loading real data"
    )

    # ── fleet overrides (work in both modes) ─────────────────────────────
    parser.add_argument("--drones",    type=int,   default=None,
                        help="Override number of drones")
    parser.add_argument("--battery",   type=float, default=None,
                        help="Override battery capacity (energy units)")
    parser.add_argument("--payload",   type=float, default=None,
                        help="Override max payload per drone")
    parser.add_argument("--max-customers", type=int, default=None,
                        help="Limit to this many customers from the dataset")

    # ── synthetic-mode extras ─────────────────────────────────────────────
    parser.add_argument("--customers", type=int, default=config.NUM_CUSTOMERS,
                        help="[random mode] Number of synthetic customers")
    parser.add_argument("--seed",      type=int, default=config.RANDOM_SEED,
                        help="[random mode] Random seed")

    # ── GA hyper-params ───────────────────────────────────────────────────
    parser.add_argument("--generations", type=int, default=config.GA_GENERATIONS,
                        help="GA generations (default: %(default)s)")
    parser.add_argument("--pop-size",    type=int, default=config.GA_POP_SIZE,
                        help="GA population size (default: %(default)s)")
    parser.add_argument("--output",      type=str, default="results/solution.json",
                        help="Output JSON path (default: %(default)s)")

    args = parser.parse_args(argv)

    # Resolve effective fleet params
    eff_drones  = args.drones   or config.DRONE_COUNT
    eff_battery = args.battery  or config.BATTERY_WH
    eff_payload = args.payload  or config.MAX_PAYLOAD_KG

    # ── header ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  OPTI MINERS — DRONE DELIVERY OPTIMIZATION")
    print(f"{'='*60}")

    # ── build instance ────────────────────────────────────────────────────
    if args.random:
        # Synthetic mode — kept for testing / CI
        from drone_delivery.data.generator import generate_instance
        print(f"  Mode     : Synthetic (random)")
        print(f"  Customers: {args.customers}  Drones: {eff_drones}")
        print(f"  Battery  : {eff_battery} Wh  Payload: {eff_payload} kg")
        print(f"  Seed     : {args.seed}")
        print(f"{'='*60}\n")
        instance = generate_instance(
            n_customers=args.customers,
            n_drones=eff_drones,
            seed=args.seed,
            battery_wh=eff_battery,
            max_payload_kg=eff_payload,
        )
    else:
        # Real data mode
        from drone_delivery.data.loader import load_instance
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"[ERROR] Data directory not found: {data_dir.resolve()}")
            sys.exit(1)

        print(f"  Mode     : Real dataset ({data_dir})")
        print(f"  Drones   : {args.drones or 'from dataset'}")
        print(f"  Battery  : {args.battery or 'from dataset'}")
        print(f"  Payload  : {args.payload or 'from dataset'}")
        if args.max_customers:
            print(f"  Customers: up to {args.max_customers}")
        print(f"{'='*60}\n")

        instance = load_instance(
            data_dir,
            n_drones=args.drones,
            battery_override=args.battery,
            payload_override=args.payload,
            max_customers=args.max_customers,
        )
        # Resolve effective values from instance (dataset may have set them)
        eff_battery = getattr(instance, '_battery', eff_battery)
        eff_payload = getattr(instance, '_payload', eff_payload)

    print(f"Instance loaded: {instance.n_customers} customers, "
          f"{len(instance.no_fly_zones)} NFZs, "
          f"{len(instance.feasible_arcs)} feasible arcs\n")

    # ── run GA ────────────────────────────────────────────────────────────
    best_solution, ga_stats = run_ga(
        instance,
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed if args.random else config.RANDOM_SEED,
        max_payload=eff_payload,
        battery=eff_battery,
        verbose=False,
    )

    # ── validate ──────────────────────────────────────────────────────────
    report = check_solution(best_solution, instance,
                            max_payload=eff_payload, battery=eff_battery)

    print(f"\n{'='*60}")
    print(f"  GA finished in {ga_stats.runtime_seconds:.2f}s over "
          f"{ga_stats.generations} generations")
    print(f"  Best energy : {best_solution.total_energy:.2f} energy units")
    print(f"  Feasible    : {report.feasible}")
    served = sum(len(r.sequence) for r in best_solution.routes)
    print(f"  Served      : {served}/{instance.n_customers}")
    print(f"{'='*60}")
    print(f"\n  Constraint Report:")
    print(f"    All served      : {'PASS' if report.all_served else 'FAIL'}")
    print(f"    Payload feasible: {'PASS' if report.payload_feasible else 'FAIL'}")
    print(f"    Energy feasible : {'PASS' if report.energy_feasible else 'FAIL'}")
    print(f"    NFZ feasible    : {'PASS' if report.nfz_feasible else 'FAIL'}")
    print(f"    Overall         : {'FEASIBLE' if report.feasible else 'INFEASIBLE'}")
    if report.violations:
        for v in report.violations:
            print(f"    !! {v}")

    # ── lower bound & optimality gap ──────────────────────────────────────
    from drone_delivery.optimization.lower_bound import compute_lower_bound, optimality_gap
    bounds = compute_lower_bound(instance, max_payload=eff_payload, battery=eff_battery)
    gap = optimality_gap(best_solution.total_energy, bounds["best_lower_bound"])
    print(f"\n  Optimality Analysis:")
    print(f"    Lower bound (MST)              : {bounds['lb_mst']:.2f} Wh")
    print(f"    Lower bound (assignment)       : {bounds['lb_assignment']:.2f} Wh")
    print(f"    Best lower bound               : {bounds['best_lower_bound']:.2f} Wh")
    print(f"    GA+SA solution                 : {best_solution.total_energy:.2f} Wh")
    print(f"    Upper bound (individual trips) : {bounds['ub_individual_trips']:.2f} Wh")
    print(f"    Optimality gap                 : {gap:.1f}%")
    if gap < 30:
        print(f"    Quality                        : EXCELLENT (within 30% of theoretical optimum)")
    elif gap < 60:
        print(f"    Quality                        : GOOD (within 60% of theoretical optimum)")
    else:
        print(f"    Quality                        : ACCEPTABLE")

    # ── export ────────────────────────────────────────────────────────────
    export_solution_json(instance, best_solution, ga_stats,
                         output_path=args.output, report=report)
    print(f"\n  Solution exported to: {args.output}")

    # ── route summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ROUTE SUMMARY")
    print(f"{'='*60}")
    for route in best_solution.routes:
        if not route.sequence:
            continue
        seq_str = " -> ".join(["0"] + [str(c) for c in route.sequence] + ["0"])
        print(f"  Drone {route.drone_id}: {seq_str}")
        print(f"    Energy: {route.total_energy:.2f} | "
              f"Load: {route.total_load:.2f} | "
              f"Dist: {route.total_distance:.1f} m | "
              f"Stops: {len(route.sequence)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
