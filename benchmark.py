"""
benchmark.py — Run 10 instances and collect results for the LaTeX report.
"""
import time
import sys
sys.path.insert(0, ".")

from drone_delivery.data.generator import generate_instance
from drone_delivery.data.loader import load_instance
from drone_delivery.optimization.genetic_algorithm import run_ga
from drone_delivery.optimization.lower_bound import compute_lower_bound, optimality_gap
from drone_delivery.constraints.checker import check_solution

# ── 10 test instances ────────────────────────────────────────────────────
INSTANCES = [
    # (label, mode, n_cust, n_drones, payload, battery, seed)
    ("I1",  "random", 10, 3, 5,   150,  42),
    ("I2",  "random", 15, 3, 5,   150,  77),
    ("I3",  "random", 20, 4, 5,   150,  123),
    ("I4",  "random", 25, 5, 5,   200,  256),
    ("I5",  "random", 30, 5, 5,   200,  314),
    ("I6",  "random", 35, 6, 5,   250,  500),
    ("I7",  "random", 40, 6, 5,   250,  667),
    ("I8",  "real",   20, 4, 200, 1200, 42),
    ("I9",  "real",   30, 8, 200, 1200, 42),
    ("I10", "real",   40, 10,200, 1200, 42),
]

print(f"\n{'='*90}")
print("  OPTI MINERS — BENCHMARK: 10 INSTANCES")
print(f"{'='*90}")
print(f"{'Inst':<5} {'n':>3} {'K':>3} {'Q':>5} {'B':>6}  {'Served':>8} {'Energy':>8} "
      f"{'LB':>8} {'UB':>8} {'Gap%':>7} {'Time':>7} {'Feas':>5}")
print("-" * 90)

results = []

for label, mode, n_cust, n_drones, payload, battery, seed in INSTANCES:
    try:
        if mode == "random":
            instance = generate_instance(
                n_customers=n_cust, n_drones=n_drones, seed=seed,
                battery_wh=battery, max_payload_kg=payload,
            )
        else:
            from pathlib import Path
            instance = load_instance(
                Path("data pre processing"),
                max_customers=n_cust,
            )
            instance.n_drones = n_drones

        t0 = time.time()
        best_sol, stats = run_ga(
            instance,
            pop_size=60,
            generations=80,
            seed=seed,
            max_payload=payload,
            battery=battery,
            verbose=False,
        )
        runtime = time.time() - t0

        report = check_solution(best_sol, instance, max_payload=payload, battery=battery)
        bounds = compute_lower_bound(instance, max_payload=payload, battery=battery)
        gap = optimality_gap(best_sol.total_energy, bounds["best_lower_bound"])

        served = sum(len(r.sequence) for r in best_sol.routes)
        feas = "YES" if report.feasible else "NO"

        print(f"{label:<5} {n_cust:>3} {n_drones:>3} {payload:>5} {battery:>6}  "
              f"{served:>3}/{n_cust:<4} {best_sol.total_energy:>8.1f} "
              f"{bounds['best_lower_bound']:>8.1f} {bounds['ub_individual_trips']:>8.1f} "
              f"{gap:>6.1f}% {runtime:>6.1f}s {feas:>5}")

        results.append({
            "label": label, "n": n_cust, "K": n_drones, "Q": payload,
            "B": battery, "served": served, "total": n_cust,
            "energy": round(best_sol.total_energy, 1),
            "lb": bounds["best_lower_bound"],
            "ub": round(bounds["ub_individual_trips"], 1),
            "gap": round(gap, 1),
            "time": round(runtime, 1),
            "feasible": feas,
        })

    except Exception as e:
        print(f"{label:<5} {n_cust:>3} {n_drones:>3} {payload:>5} {battery:>6}  ERROR: {e}")

print(f"{'='*90}")

# ── LaTeX table output ───────────────────────────────────────────────────
print("\n\n% -- Copy this into your LaTeX report --")
print("\\begin{table}[H]\\centering")
print("\\begin{tabular}{lrrrrrrrrl}\\toprule")
print("\\textbf{Inst} & $n$ & $K$ & $Q$ & $B$ & \\textbf{Served} & "
      "\\textbf{Energy} & \\textbf{LB} & \\textbf{Gap} & \\textbf{Time} \\\\\\midrule")
for r in results:
    print(f"{r['label']} & {r['n']} & {r['K']} & {r['Q']} & {r['B']} & "
          f"{r['served']}/{r['total']} & {r['energy']} & {r['lb']} & "
          f"{r['gap']}\\% & {r['time']}s \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Experimental results on 10 benchmark instances. "
      "LB = best lower bound, Gap = (Energy$-$LB)/LB.}")
print("\\end{table}")
