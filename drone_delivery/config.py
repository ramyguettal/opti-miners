"""
config.py
=========
All constants and hyperparameters for the Drone-Based Delivery Optimization
System.  Every tunable value lives here — no magic numbers elsewhere.
"""

# ── Fleet & drone specs ──────────────────────────────────────────────────────
DRONE_COUNT: int = 4
MAX_PAYLOAD_KG: float = 5.0        # kg  — maximum weight per trip
BATTERY_WH: float = 150.0          # Wh  — maximum energy per route
DRONE_SPEED_MS: float = 15.0       # m/s — cruise speed (constant)
SAFE_DISTANCE_M: float = 50.0      # m   — minimum separation between drones

# ── Energy model ─────────────────────────────────────────────────────────────
ALPHA: float = 0.05                # Wh/m       — base energy per metre flown
BETA: float = 0.002                # Wh/(kg·m)  — weight-dependent energy

# ── Grid / instance ──────────────────────────────────────────────────────────
GRID_SIZE: tuple[int, int] = (1000, 1000)   # metres (width, height)
NUM_CUSTOMERS: int = 20
NFZ_COUNT: int = 3                           # number of no-fly zones

# ── Genetic Algorithm ────────────────────────────────────────────────────────
GA_POP_SIZE: int = 100
GA_GENERATIONS: int = 300
GA_CROSSOVER_RATE: float = 0.85
GA_MUTATION_RATE: float = 0.15
GA_ELITE_SIZE: int = 10
GA_TOURNAMENT_SIZE: int = 5
GA_STAGNATION_LIMIT: int = 50      # stop after N gens w/o improvement

# ── Local Search ─────────────────────────────────────────────────────────────
LS_MAX_ITER: int = 500

# ── Penalty weights (for infeasible solutions in the GA) ─────────────────────
PENALTY_UNSERVED: float = 1e6      # per unserved customer
PENALTY_CAPACITY: float = 1e4      # per kg of capacity overflow
PENALTY_ENERGY: float = 1e4        # per Wh of energy overflow

# ── Repair ───────────────────────────────────────────────────────────────────
MAX_REPAIR_ITER: int = 20

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
