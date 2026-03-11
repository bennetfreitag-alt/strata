"""
STRATA Evolution Engine — Population Manager (v4)
Island Model: 3 sub-populations with distinct strategies.
  - Exploiter: low mutation, refines current best
  - Explorer: high mutation, broad search
  - AI-Directed: Claude-created genomes targeting diagnosed problems
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from genome import Genome, GROUPS
from simulator import (
    simulate_genome_full,
    compute_fitness,
    FITNESS_WEIGHTS,
    SCENARIO_PROFILES,
)
from database import save_genome_eval, save_elite, save_stagnation_event

logger = logging.getLogger("strata.population")

_executor = ThreadPoolExecutor(max_workers=36)


# ---------------------------------------------------------------------------
# Island configuration
# ---------------------------------------------------------------------------

class IslandRole(Enum):
    EXPLOITER = "exploiter"
    EXPLORER = "explorer"
    AI_DIRECTED = "ai_directed"


@dataclass
class IslandConfig:
    role: IslandRole
    size: int
    mutation_strength: float
    crossover_fraction: float
    elite_fraction: float
    random_fraction: float


ISLAND_CONFIGS: dict[IslandRole, IslandConfig] = {
    IslandRole.EXPLOITER: IslandConfig(
        role=IslandRole.EXPLOITER, size=12,
        mutation_strength=0.08, crossover_fraction=0.35,
        elite_fraction=0.25, random_fraction=0.08,
    ),
    IslandRole.EXPLORER: IslandConfig(
        role=IslandRole.EXPLORER, size=12,
        mutation_strength=0.35, crossover_fraction=0.25,
        elite_fraction=0.08, random_fraction=0.25,
    ),
    IslandRole.AI_DIRECTED: IslandConfig(
        role=IslandRole.AI_DIRECTED, size=12,
        mutation_strength=0.15, crossover_fraction=0.20,
        elite_fraction=0.15, random_fraction=0.15,
    ),
}


# ---------------------------------------------------------------------------
# Island: self-contained sub-population
# ---------------------------------------------------------------------------

class Island:
    def __init__(self, config: IslandConfig):
        self.config = config
        self.population: list[Genome] = []
        self.best_fitness: float = float("-inf")
        self.best_genome: Genome | None = None
        self.hall_of_fame: list[Genome] = []  # top 3 per island
        self.stagnation_counter: int = 0
        self.fitness_history: list[float] = []
        self.per_category_scores: dict[int, dict[str, float]] = {}

    def initialize(self, seed_genomes: list[Genome] | None = None) -> None:
        """Fill island population from seeds or random."""
        self.population = []
        if seed_genomes:
            for sg in seed_genomes[:self.config.size]:
                g = Genome.from_dict(sg.to_dict())
                if not g.genome_id:
                    g.genome_id = str(uuid.uuid4())[:8]
                self.population.append(g)
        # Fill remaining with mutations of first or seed
        base = self.population[0] if self.population else Genome.seed()
        while len(self.population) < self.config.size:
            if self.config.role == IslandRole.EXPLORER:
                g = Genome.random()  # explorers start diverse
            else:
                g = base.mutate(self.config.mutation_strength * 2)
            g.genome_id = str(uuid.uuid4())[:8]
            self.population.append(g)

    def breed_next_generation(self) -> None:
        """Breed according to island role config."""
        ranked = sorted(self.population, key=lambda g: g.fitness or 0, reverse=True)
        size = self.config.size

        elite_count = max(1, int(size * self.config.elite_fraction))
        crossover_count = max(1, int(size * self.config.crossover_fraction))
        random_count = max(1, int(size * self.config.random_fraction))
        mutation_count = size - elite_count - crossover_count - random_count
        if mutation_count < 0:
            mutation_count = 0
            crossover_count = size - elite_count - random_count

        new_pop: list[Genome] = []

        # Elites
        for i in range(min(elite_count, len(ranked))):
            elite = Genome.from_dict(ranked[i].to_dict())
            elite.genome_id = str(uuid.uuid4())[:8]
            new_pop.append(elite)

        # Crossover
        top_half = ranked[:max(2, size // 2)]
        for _ in range(crossover_count):
            if len(top_half) >= 2:
                p1 = random.choice(top_half[:3])
                p2 = random.choice(top_half)
                child = p1.crossover(p2)
            else:
                child = ranked[0].mutate(self.config.mutation_strength)
            child.genome_id = str(uuid.uuid4())[:8]
            new_pop.append(child)

        # Mutations
        for _ in range(mutation_count):
            base = ranked[0] if ranked else Genome.seed()
            # Use hall of fame occasionally
            if self.hall_of_fame and random.random() < 0.3:
                base = random.choice(self.hall_of_fame)
            child = base.mutate(self.config.mutation_strength)
            child.genome_id = str(uuid.uuid4())[:8]
            new_pop.append(child)

        # Random newcomers
        for _ in range(random_count):
            g = Genome.random()
            g.genome_id = str(uuid.uuid4())[:8]
            new_pop.append(g)

        self.population = new_pop[:size]
        # Pad if needed
        while len(self.population) < size:
            g = Genome.random()
            g.genome_id = str(uuid.uuid4())[:8]
            self.population.append(g)

    def inject_genome(self, genome: Genome) -> None:
        """Inject a genome (from migration or AI strategy), replacing worst."""
        if not self.population:
            self.population.append(genome)
            return
        worst_idx = min(
            range(len(self.population)),
            key=lambda i: self.population[i].fitness or float("-inf"),
        )
        self.population[worst_idx] = genome

    def get_best(self) -> Genome | None:
        if not self.population:
            return None
        return max(self.population, key=lambda g: g.fitness or float("-inf"))

    def update_hall_of_fame(self, genome: Genome) -> None:
        self.hall_of_fame.append(genome)
        self.hall_of_fame.sort(key=lambda g: g.fitness or 0, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:3]


# ---------------------------------------------------------------------------
# Sync evaluation wrapper for thread pool
# ---------------------------------------------------------------------------

def _eval_full_sync(
    genome: Genome,
    generation: int,
    variant: int,
    num_workloads: int,
    difficulty_boost: float,
    stress_category: int | None,
):
    """Synchronous wrapper for full simulation (runs in thread pool)."""
    metrics, fitness, per_cat = simulate_genome_full(
        genome, generation, variant, num_workloads,
        difficulty_boost, stress_category, use_concept_drift=True,
    )
    return genome, metrics, fitness, per_cat


# ---------------------------------------------------------------------------
# Population Manager: orchestrates 3 islands
# ---------------------------------------------------------------------------

class PopulationManager:
    def __init__(
        self,
        island_configs: dict[IslandRole, IslandConfig] | None = None,
        workloads_per_genome: int = 60,
        stagnation_patience: int = 15,
        migration_interval: int = 20,
        **kwargs,  # backward compat
    ):
        self.configs = island_configs or ISLAND_CONFIGS
        self.workloads = workloads_per_genome
        self.stagnation_patience = stagnation_patience
        self.migration_interval = migration_interval

        self.islands: dict[IslandRole, Island] = {
            role: Island(config) for role, config in self.configs.items()
        }

        self.generation: int = 0
        self.best_fitness: float = float("-inf")
        self.best_genome: Genome | None = None
        self.global_hall_of_fame: list[Genome] = []  # top 5 across all islands
        self.stagnation_counter: int = 0
        self.escape_index: int = 0

        # Diagnostic data
        self.weakest_category: int | None = None
        self.per_category_global: dict[int, dict[str, float]] = {}

    @property
    def population(self) -> list[Genome]:
        """Flat list of all genomes across islands (backward compat)."""
        all_genomes: list[Genome] = []
        for island in self.islands.values():
            all_genomes.extend(island.population)
        return all_genomes

    @property
    def pop_size(self) -> int:
        return sum(island.config.size for island in self.islands.values())

    def initialize(self, from_elites: list[dict] | None = None) -> None:
        """Initialize all islands."""
        if from_elites and len(from_elites) > 0:
            logger.info(f"Restoring from {len(from_elites)} elites across 3 islands")
            # Distribute elites across islands
            per_island = max(1, len(from_elites) // 3)
            island_list = list(self.islands.values())
            for i, island in enumerate(island_list):
                start = i * per_island
                end = start + per_island if i < 2 else len(from_elites)
                island_elites = from_elites[start:end]
                seed_genomes = [Genome.from_dict(e) for e in island_elites]
                island.initialize(seed_genomes=seed_genomes)

            # Restore global best
            best_elite = max(from_elites, key=lambda e: e.get("fitness", 0))
            self.best_fitness = best_elite.get("fitness", float("-inf"))
            self.best_genome = Genome.from_dict(best_elite)
            self.generation = best_elite.get("generation", 0)
        else:
            seed = Genome.seed()
            for island in self.islands.values():
                island.initialize(seed_genomes=[seed])

        # Set generation on all genomes
        for g in self.population:
            g.generation = self.generation

    async def evaluate_all(self) -> None:
        """Evaluate all islands in parallel. Uses simulate_genome_full()."""
        loop = asyncio.get_event_loop()
        tasks = []
        task_meta: list[tuple[IslandRole, int]] = []

        difficulty = self._compute_difficulty_boost()
        stress_cat = self.weakest_category

        for role, island in self.islands.items():
            for i, genome in enumerate(island.population):
                tasks.append(
                    loop.run_in_executor(
                        _executor,
                        _eval_full_sync,
                        genome, self.generation, len(task_meta),
                        self.workloads, difficulty, stress_cat,
                    )
                )
                task_meta.append((role, i))

        results = await asyncio.gather(*tasks)

        # Accumulate per-category data
        category_accum: dict[int, dict[str, list[float]]] = {
            i: {k: [] for k in FITNESS_WEIGHTS} for i in range(8)
        }

        for idx, (genome, metrics, fitness, per_cat) in enumerate(results):
            role, gi = task_meta[idx]
            island = self.islands[role]
            island.population[gi].metrics = metrics
            island.population[gi].fitness = fitness

            # Accumulate per-category for global diagnostics
            for cat_id, cat_metrics in per_cat.items():
                for k, v in cat_metrics.items():
                    if k in category_accum.get(cat_id, {}):
                        category_accum[cat_id][k].append(v)

            await save_genome_eval(
                genome_id=genome.genome_id,
                generation=self.generation,
                fitness=fitness,
                metrics=metrics,
                values=genome.values,
                parent_ids=genome.parent_ids,
            )

        # Compute global per-category averages
        self.per_category_global = {}
        for cat_id, cat_data in category_accum.items():
            cat_avg: dict[str, float] = {}
            for k, vals in cat_data.items():
                cat_avg[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
            self.per_category_global[cat_id] = cat_avg

        # Find weakest category
        cat_fitnesses: dict[int, float] = {}
        for cat_id, cat_avg in self.per_category_global.items():
            cat_fitnesses[cat_id] = compute_fitness(cat_avg)
        if cat_fitnesses:
            self.weakest_category = min(cat_fitnesses, key=lambda k: cat_fitnesses[k])

        # Update per-island bests and global best
        for island in self.islands.values():
            island_best = max(
                island.population,
                key=lambda g: g.fitness or float("-inf"),
            )
            island.fitness_history.append(island_best.fitness or 0)

            if island_best.fitness and island_best.fitness > island.best_fitness:
                island.best_fitness = island_best.fitness
                island.best_genome = island_best
                island.stagnation_counter = 0
                island.update_hall_of_fame(island_best)
            else:
                island.stagnation_counter += 1

        # Global best
        gen_best = max(self.population, key=lambda g: g.fitness or float("-inf"))
        if gen_best.fitness is not None and gen_best.fitness > self.best_fitness:
            self.best_fitness = gen_best.fitness
            self.best_genome = gen_best
            self.stagnation_counter = 0
            await save_elite(
                genome_id=gen_best.genome_id,
                generation=self.generation,
                fitness=gen_best.fitness,
                metrics=gen_best.metrics,
                values=gen_best.values,
            )
            self._update_hof(gen_best)
        else:
            self.stagnation_counter += 1

    def _update_hof(self, genome: Genome) -> None:
        self.global_hall_of_fame.append(genome)
        self.global_hall_of_fame.sort(key=lambda g: g.fitness or 0, reverse=True)
        self.global_hall_of_fame = self.global_hall_of_fame[:5]

    def migrate(self) -> list[str]:
        """Best from each island migrates to others."""
        log: list[str] = []
        island_list = list(self.islands.values())
        migrants = [
            (island.config.role.value, island.get_best())
            for island in island_list
        ]

        for i, (source_role, migrant) in enumerate(migrants):
            if migrant is None:
                continue
            for j, target_island in enumerate(island_list):
                if i == j:
                    continue
                clone = Genome.from_dict(migrant.to_dict())
                clone.genome_id = str(uuid.uuid4())[:8]
                clone.generation = self.generation
                target_island.inject_genome(clone)
                log.append(f"{source_role} -> {target_island.config.role.value}")

        return log

    def breed_all_islands(self) -> None:
        """Breed each island according to its own strategy."""
        self.generation += 1
        for island in self.islands.values():
            island.breed_next_generation()
            for g in island.population:
                g.generation = self.generation

    def is_stagnating(self) -> bool:
        return self.stagnation_counter >= self.stagnation_patience

    async def escape_stagnation(self) -> str:
        """Apply stagnation escape to the most stagnant island."""
        strategies = [
            "inject_random",
            "group_reset_conflict",
            "group_reset_uncertainty",
            "group_reset_meta",
            "group_reset_anchors",
            "group_reset_memory",
            "crossover_elite_random",
            "big_mutation",
        ]
        strategy = strategies[self.escape_index % len(strategies)]
        self.escape_index += 1
        self.stagnation_counter = 0

        # Find most stagnant island
        target_island = max(
            self.islands.values(),
            key=lambda isl: isl.stagnation_counter,
        )
        island_name = target_island.config.role.value

        logger.info(f"Stagnation escape: {strategy} on island {island_name}")

        pop = target_island.population

        if strategy == "inject_random":
            half = len(pop) // 2
            pop.sort(key=lambda g: g.fitness or 0, reverse=True)
            for i in range(half, len(pop)):
                pop[i] = Genome.random()
                pop[i].genome_id = str(uuid.uuid4())[:8]

        elif strategy.startswith("group_reset_"):
            group = strategy.replace("group_reset_", "")
            for i in range(len(pop)):
                pop[i] = pop[i].random_reset_params(group)
                pop[i].genome_id = str(uuid.uuid4())[:8]

        elif strategy == "crossover_elite_random":
            best = target_island.best_genome or self.best_genome
            if best:
                for i in range(1, len(pop)):
                    partner = Genome.random()
                    pop[i] = best.crossover(partner)
                    pop[i].genome_id = str(uuid.uuid4())[:8]

        elif strategy == "big_mutation":
            best = target_island.best_genome or self.best_genome
            if best:
                for i in range(len(pop)):
                    pop[i] = best.mutate(0.5)
                    pop[i].genome_id = str(uuid.uuid4())[:8]

        target_island.stagnation_counter = 0

        await save_stagnation_event(
            self.generation,
            f"{strategy}@{island_name}",
            self.best_fitness,
        )

        return f"{strategy}@{island_name}"

    def _compute_difficulty_boost(self) -> float:
        """Adaptive difficulty based on global best fitness."""
        if self.best_fitness <= 0:
            return 0.0
        return min(0.5, max(0.0, (self.best_fitness - 3.0) / 10.0))

    def get_sorted_population(self) -> list[Genome]:
        """All genomes sorted by fitness (backward compat)."""
        return sorted(self.population, key=lambda g: g.fitness or 0, reverse=True)

    def get_island_states(self) -> dict[str, Any]:
        """Return per-island state for dashboard."""
        states: dict[str, Any] = {}
        for role, island in self.islands.items():
            sorted_pop = sorted(
                island.population,
                key=lambda g: g.fitness or 0,
                reverse=True,
            )
            avg_fit = (
                sum(g.fitness or 0 for g in island.population) / len(island.population)
                if island.population else 0
            )
            states[role.value] = {
                "size": island.config.size,
                "mutation_strength": round(island.config.mutation_strength, 4),
                "best_fitness": round(island.best_fitness, 4) if island.best_fitness > float("-inf") else 0,
                "avg_fitness": round(avg_fit, 4),
                "best_genome_id": island.best_genome.genome_id if island.best_genome else "",
                "stagnation_counter": island.stagnation_counter,
                "population": [
                    {"genome_id": g.genome_id, "fitness": round(g.fitness, 4) if g.fitness else 0}
                    for g in sorted_pop
                ],
            }
        return states
