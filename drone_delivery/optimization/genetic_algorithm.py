"""
optimization/genetic_algorithm.py
==================================
Full Genetic Algorithm (Memetic Algorithm) for the EC-CVRP-NFZ.

Chromosome = permutation of customer indices.
Decoding   = greedy left-to-right split into drone routes when adding the
             next customer would violate capacity or energy.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from drone_delivery import config
from drone_delivery.model.route import Route
from drone_delivery.model.solution import Solution
from drone_delivery.optimization.local_search import local_search
from drone_delivery.optimization.operators import mutate, order_crossover
from drone_delivery.optimization.population import create_initial_population
from drone_delivery.utils.distance import energy_for_arc

if TYPE_CHECKING:
    from drone_delivery.data.instance import DeliveryInstance


@dataclass
class GAStats:
    """Statistics collected during the GA run.

    Attributes:
        generations:       Number of generations completed.
        convergence_curve: Best fitness per generation.
        runtime_seconds:   Wall-clock time [s].
        population_size:   Population size used.
        parameters:        Dict of all GA/LS hyperparameters.
    """
    generations: int = 0
    convergence_curve: list[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    population_size: int = config.GA_POP_SIZE
    parameters: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome → Solution decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_chromosome(
    chromosome: list[int],
    instance: "DeliveryInstance",
    n_drones: int | None = None,
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
) -> Solution:
    """Decode a customer permutation into a Solution with drone routes.

    Scan the permutation left to right.  For each customer, first try the
    current drone.  If it can't fit (payload or energy), try all other
    drones using a best-fit strategy.  If no drone can take it, advance
    the current drone pointer and try a fresh route.

    Args:
        chromosome: Permutation of customer indices [1..n].
        instance:   The problem instance.
        n_drones:   Number of drones (defaults to instance.n_drones).
        max_payload: Max payload [kg].
        battery:    Battery capacity [Wh].

    Returns:
        A Solution object with routes assigned to drones.
    """
    if n_drones is None:
        n_drones = instance.n_drones

    routes: list[Route] = [Route(drone_id=k) for k in range(n_drones)]
    current_drone = 0

    for cust_idx in chromosome:
        placed = False

        # Try the current drone first
        if current_drone < n_drones:
            if _try_add_customer(routes[current_drone], cust_idx, instance,
                                  max_payload, battery):
                placed = True

        if not placed:
            # Try all drones (best-fit: pick the one with the highest load
            # that can still accommodate the customer)
            best_drone = -1
            best_load = -1.0
            for k in range(n_drones):
                trial_seq = routes[k].sequence + [cust_idx]
                trial_load = sum(instance.demand(c) for c in trial_seq)
                if trial_load > max_payload + 1e-9:
                    continue
                trial_energy = _estimate_route_energy(trial_seq, instance)
                if trial_energy > battery + 1e-9:
                    continue
                # Prefer the drone already carrying the most (best-fit packing)
                if routes[k].total_load > best_load:
                    best_load = routes[k].total_load
                    best_drone = k

            if best_drone >= 0:
                routes[best_drone].sequence = routes[best_drone].sequence + [cust_idx]
                routes[best_drone].total_load = sum(
                    instance.demand(c) for c in routes[best_drone].sequence
                )
                placed = True
            else:
                # Advance current_drone to the next empty one
                current_drone += 1
                while current_drone < n_drones and routes[current_drone].sequence:
                    current_drone += 1
                if current_drone < n_drones:
                    if _try_add_customer(routes[current_drone], cust_idx, instance,
                                          max_payload, battery):
                        placed = True

        # If still not placed, the penalty system handles it via unserved

    # Compute metrics for all routes
    sol = Solution(routes=routes)
    sol.evaluate(instance, max_payload, battery)
    return sol


def _try_add_customer(
    route: Route,
    cust_idx: int,
    instance: "DeliveryInstance",
    max_payload: float,
    battery: float,
) -> bool:
    """Try to append a customer to a route.  Returns True if successful."""
    trial_seq = route.sequence + [cust_idx]
    trial_load = sum(instance.demand(c) for c in trial_seq)
    if trial_load > max_payload + 1e-9:
        return False
    trial_energy = _estimate_route_energy(trial_seq, instance)
    if trial_energy > battery + 1e-9:
        return False
    route.sequence = trial_seq
    route.total_load = trial_load
    return True


def _estimate_route_energy(
    sequence: list[int],
    instance: "DeliveryInstance",
) -> float:
    """Quick energy estimate for a route (load-dependent)."""
    if not sequence:
        return 0.0
    total_demand = sum(instance.demand(c) for c in sequence)
    current_load = total_demand
    prev = 0
    energy = 0.0
    for c in sequence:
        d = instance.distance_matrix[prev, c]
        energy += energy_for_arc(d, current_load)
        current_load -= instance.demand(c)
        prev = c
    d = instance.distance_matrix[prev, 0]
    energy += energy_for_arc(d, current_load)
    return energy


# ─────────────────────────────────────────────────────────────────────────────
# Selection — Tournament
# ─────────────────────────────────────────────────────────────────────────────

def tournament_selection(
    population: list[list[int]],
    fitness: list[float],
    rng: random.Random,
    tournament_size: int = config.GA_TOURNAMENT_SIZE,
) -> list[int]:
    """Tournament selection: pick *tournament_size* random individuals,
    return the one with the lowest fitness.

    Args:
        population:      List of chromosomes.
        fitness:         Parallel list of fitness values (lower = better).
        rng:             Random number generator.
        tournament_size: Number of competitors.

    Returns:
        The winning chromosome (copy).
    """
    indices = rng.sample(range(len(population)), min(tournament_size, len(population)))
    best = min(indices, key=lambda i: fitness[i])
    return list(population[best])


# ─────────────────────────────────────────────────────────────────────────────
# Repair operator
# ─────────────────────────────────────────────────────────────────────────────

def repair_chromosome(chromosome: list[int], n_customers: int) -> list[int]:
    """Ensure the chromosome is a valid permutation of [1..n].

    Removes duplicates and adds any missing customers.

    Args:
        chromosome:  Possibly invalid permutation.
        n_customers: Expected number of customers.

    Returns:
        Valid permutation of [1..n_customers].
    """
    seen: set[int] = set()
    clean: list[int] = []
    for g in chromosome:
        if 1 <= g <= n_customers and g not in seen:
            clean.append(g)
            seen.add(g)

    missing = [c for c in range(1, n_customers + 1) if c not in seen]
    clean.extend(missing)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Main GA loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(
    instance: "DeliveryInstance",
    pop_size: int = config.GA_POP_SIZE,
    generations: int = config.GA_GENERATIONS,
    crossover_rate: float = config.GA_CROSSOVER_RATE,
    mutation_rate: float = config.GA_MUTATION_RATE,
    elite_size: int = config.GA_ELITE_SIZE,
    seed: int = config.RANDOM_SEED,
    max_payload: float = config.MAX_PAYLOAD_KG,
    battery: float = config.BATTERY_WH,
    verbose: bool = True,
) -> tuple[Solution, GAStats]:
    """Run the full Genetic Algorithm (Memetic) for the EC-CVRP-NFZ.

    Args:
        instance:       The problem instance.
        pop_size:       Population size.
        generations:    Maximum number of generations.
        crossover_rate: Probability of applying crossover.
        mutation_rate:  Probability of applying mutation.
        elite_size:     Number of elite individuals preserved each generation.
        seed:           Random seed.
        max_payload:    Max payload [kg].
        battery:        Battery capacity [Wh].
        verbose:        If True, print progress to console.

    Returns:
        Tuple of (best Solution, GAStats).
    """
    rng = random.Random(seed)
    start_time = time.time()

    # ── initialise population ────────────────────────────────────────────
    population = create_initial_population(instance, pop_size, seed)
    n_cust = instance.n_customers

    # Evaluate fitness for the whole population
    fitness: list[float] = []
    solutions: list[Solution] = []
    for chrom in population:
        sol = decode_chromosome(chrom, instance, max_payload=max_payload, battery=battery)
        fit = sol.evaluate(instance, max_payload, battery)
        fitness.append(fit)
        solutions.append(sol)

    convergence: list[float] = []
    best_fitness = min(fitness)
    best_idx = fitness.index(best_fitness)
    best_chromosome = list(population[best_idx])
    best_solution = solutions[best_idx]
    stagnation = 0

    if verbose:
        elapsed = time.time() - start_time
        print(f"Gen   0 | Best fitness: {best_fitness:12.2f} | Time: {elapsed:6.2f}s")

    convergence.append(best_fitness)

    # ── main loop ────────────────────────────────────────────────────────
    for gen in range(1, generations + 1):
        # Sort by fitness for elitism
        ranked = sorted(range(len(population)), key=lambda i: fitness[i])

        new_population: list[list[int]] = []
        new_fitness: list[float] = []
        new_solutions: list[Solution] = []

        # Elitism — keep top individuals
        for idx in ranked[:elite_size]:
            new_population.append(list(population[idx]))
            new_fitness.append(fitness[idx])
            new_solutions.append(solutions[idx])

        # Apply SA to top 3 elite individuals (Memetic part)
        for i in range(min(3, len(new_solutions))):
            sol = new_solutions[i]
            sol = local_search(sol, instance, max_iter=10,
                               max_payload=max_payload, battery=battery,
                               seed=seed + gen * 100 + i)
            fit = sol.evaluate(instance, max_payload, battery)
            new_fitness[i] = fit
            new_solutions[i] = sol
            # Rebuild chromosome from solution routes
            new_chrom: list[int] = []
            for r in sol.routes:
                new_chrom.extend(r.sequence)
            # Ensure all customers present
            new_chrom = repair_chromosome(new_chrom, n_cust)
            new_population[i] = new_chrom

        # Fill rest of population
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitness, rng)
            parent2 = tournament_selection(population, fitness, rng)

            # Crossover
            if rng.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2, rng)
            else:
                child1, child2 = list(parent1), list(parent2)

            # Mutation
            if rng.random() < mutation_rate:
                child1 = mutate(child1, rng)
            if rng.random() < mutation_rate:
                child2 = mutate(child2, rng)

            # Repair
            child1 = repair_chromosome(child1, n_cust)
            child2 = repair_chromosome(child2, n_cust)

            for child in [child1, child2]:
                if len(new_population) >= pop_size:
                    break
                sol = decode_chromosome(child, instance, max_payload=max_payload,
                                        battery=battery)
                fit = sol.evaluate(instance, max_payload, battery)
                new_population.append(child)
                new_fitness.append(fit)
                new_solutions.append(sol)

        population = new_population
        fitness = new_fitness
        solutions = new_solutions

        # Track best
        gen_best = min(fitness)
        gen_best_idx = fitness.index(gen_best)

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_chromosome = list(population[gen_best_idx])
            best_solution = solutions[gen_best_idx]
            stagnation = 0
        else:
            stagnation += 1

        convergence.append(best_fitness)

        if verbose and (gen % 20 == 0 or gen == generations):
            elapsed = time.time() - start_time
            print(f"Gen {gen:3d} | Best fitness: {best_fitness:12.2f} | "
                  f"Stag: {stagnation:3d} | Time: {elapsed:6.2f}s")

        # Stagnation check
        if stagnation >= config.GA_STAGNATION_LIMIT:
            if verbose:
                print(f"  >> Stopped early at gen {gen} (stagnation = {stagnation})")
            break

    # ── final local search on best ───────────────────────────────────────
    best_solution = local_search(best_solution, instance, max_iter=100,
                                  max_payload=max_payload, battery=battery,
                                  seed=seed + 999999)
    best_fitness = best_solution.evaluate(instance, max_payload, battery)
    convergence.append(best_fitness)

    elapsed = time.time() - start_time

    stats = GAStats(
        generations=len(convergence) - 1,
        convergence_curve=convergence,
        runtime_seconds=round(elapsed, 3),
        population_size=pop_size,
        parameters={
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elite_size": elite_size,
            "tournament_size": config.GA_TOURNAMENT_SIZE,
            "stagnation_limit": config.GA_STAGNATION_LIMIT,
            "ls_max_iter": config.LS_MAX_ITER,
            "max_payload_kg": max_payload,
            "battery_wh": battery,
            "alpha": config.ALPHA,
            "beta": config.BETA,
        },
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GA finished in {elapsed:.2f}s over {stats.generations} generations")
        print(f"  Best energy : {best_solution.total_energy:.2f} Wh")
        print(f"  Feasible    : {best_solution.feasible}")
        print(f"  Unserved    : {len(best_solution.unserved)}")
        print(f"{'='*60}")

    return best_solution, stats
