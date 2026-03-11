"""
STRATA Evolution Engine — Evolution Loop (v5)
Island Model orchestration with 15-step generation loop:
  1. Evaluate all islands
  2. Check pending strategy results
  3. Adaptive Claude frequency
  4. Regular Claude analysis (every 100 gens — cost-efficient)
  5. Multi-strategy diagnosis (every 200 gens)
  6. Migration
  7. Stagnation check
  8. Breed all islands
  9. Adaptive mutation rates
  10. Island snapshots
  11. Synthesis (every 1000 gens)
  12. Convergence check (window: 5000 gens)
  13. Email notifications
  14. Ollama validation (every 50 gens — FREE)
  15. System prompt export
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from genome import Genome, PARAM_SPECS
from population import PopulationManager, IslandRole
from analyst import (
    analyze_generation,
    apply_adjustments,
    synthesize_final,
    diagnose_and_strategize,
    build_strategy_genomes,
)
from database import (
    get_top_elites,
    get_fitness_history,
    get_recent_insights,
    save_strategy,
    update_strategy_result,
    save_island_snapshot,
    save_migration_event,
)
from simulator import SCENARIO_PROFILES, compute_fitness
from notifier import send_evolution_summary
from prompt_compiler import compile_system_prompt, get_prompt_summary

logger = logging.getLogger("strata.evolution")


class EvolutionEngine:
    def __init__(self):
        self.workloads = int(os.environ.get("WORKLOADS_PER_GENOME", "60"))
        self.claude_every_n = int(os.environ.get("CLAUDE_EVERY_N", "100"))
        self.stagnation_patience = int(os.environ.get("STAGNATION_PATIENCE", "15"))
        self.sleep_between = float(os.environ.get("SLEEP_BETWEEN_GENS", "0.05"))
        self.migration_interval = int(os.environ.get("MIGRATION_INTERVAL", "20"))
        self.diagnosis_every_n = int(os.environ.get("DIAGNOSIS_EVERY_N", "200"))
        self.notify_every_n = int(os.environ.get("NOTIFY_EVERY_N", "500"))
        self.synthesis_every_n = int(os.environ.get("SYNTHESIS_EVERY_N", "1000"))
        self.max_generations = int(os.environ.get("MAX_GENERATIONS", "10000"))
        self.ollama_every_n = int(os.environ.get("OLLAMA_EVERY_N", "100"))
        self.ollama_top_n = int(os.environ.get("OLLAMA_TOP_N", "1"))
        self.ollama_cases = int(os.environ.get("OLLAMA_CASES", "4"))

        self.convergence_window = int(os.environ.get("CONVERGENCE_WINDOW", "5000"))
        self.convergence_threshold = float(os.environ.get("CONVERGENCE_THRESHOLD", "0.003"))

        # Claude API — disabled when no credits
        self.claude_enabled = os.environ.get("CLAUDE_ENABLED", "false").lower() == "true"

        # Adaptive Claude frequency — less aggressive for cost control
        self.claude_base_every_n = self.claude_every_n
        self.claude_min_every_n = 50   # never more often than every 50 gens
        self.claude_max_every_n = 200  # never less often than every 200 gens

        self.pm = PopulationManager(
            workloads_per_genome=self.workloads,
            stagnation_patience=self.stagnation_patience,
            migration_interval=self.migration_interval,
        )

        self.running = False
        self.converged = False
        self.converged_at_gen: int | None = None
        self.start_time: float = 0
        self.fitness_history: list[float] = []
        self.last_insight: dict | None = None
        self.last_spec: str | None = None
        self.gen_log: list[dict] = []
        self.emergent_principles: list[str] = []

        # V4: Strategy tracking
        self.strategy_ledger: list[dict] = []
        self.pending_strategies: dict[str, dict] = {}  # genome_id -> info

        # V5: Real LLM evaluation
        self.real_evaluations: list[dict] = []
        self.best_system_prompt: str = ""
        self.prompt_summary: dict = {}
        self._ollama_task: asyncio.Task | None = None  # non-blocking validation

    @property
    def state(self) -> dict[str, Any]:
        """State dict read by FastAPI."""
        best = self.pm.best_genome
        sorted_pop = self.pm.get_sorted_population()
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Per-category breakdown
        per_cat_state: dict[str, dict] = {}
        for cat_id, metrics in self.pm.per_category_global.items():
            name = SCENARIO_PROFILES.get(cat_id, {}).get("name", str(cat_id))
            per_cat_state[name] = {
                "fitness": round(compute_fitness(metrics), 4),
                "halluc_rate": round(metrics.get("halluc_rate", 0), 4),
                "robustness": round(metrics.get("robustness", 0), 4),
            }

        weakest_name = None
        if self.pm.weakest_category is not None:
            weakest_name = SCENARIO_PROFILES.get(
                self.pm.weakest_category, {}
            ).get("name", "unknown")

        return {
            "running": self.running,
            "generation": self.pm.generation,
            "best_fitness": self.pm.best_fitness,
            "best_metrics": best.metrics if best else {},
            "best_values": best.values if best else {},
            "best_genome_id": best.genome_id if best else "",
            # V4: Islands
            "islands": self.pm.get_island_states(),
            "per_category": per_cat_state,
            "weakest_category": weakest_name,
            # V4: Strategies
            "strategy_ledger": self.strategy_ledger[-20:],
            "pending_strategies": len(self.pending_strategies),
            # Existing
            "population": [
                {
                    "genome_id": g.genome_id,
                    "fitness": g.fitness,
                    "metrics": g.metrics,
                }
                for g in sorted_pop[:20]
            ],
            "stagnation_counter": self.pm.stagnation_counter,
            "total_evals": self.pm.generation * self.pm.pop_size,
            "fitness_history": self.fitness_history[-200:],
            "last_insight": self.last_insight,
            "last_spec": self.last_spec,
            "gen_log": self.gen_log[-50:],
            "emergent_principles": self.emergent_principles[-20:],
            "elapsed_seconds": elapsed,
            "converged": self.converged,
            "converged_at_gen": self.converged_at_gen,
            "claude_every_n": self.claude_every_n,
            # V5: Real evaluation
            "real_evaluations": self.real_evaluations[-10:],
            "best_system_prompt": self.best_system_prompt,
            "prompt_summary": self.prompt_summary,
            "max_generations": self.max_generations,
            "config": {
                "pop_size": self.pm.pop_size,
                "workloads": self.workloads,
                "claude_every_n": self.claude_every_n,
                "stagnation_patience": self.stagnation_patience,
                "convergence_window": self.convergence_window,
                "migration_interval": self.migration_interval,
                "diagnosis_every_n": self.diagnosis_every_n,
                "max_generations": self.max_generations,
                "ollama_every_n": self.ollama_every_n,
                "synthesis_every_n": self.synthesis_every_n,
            },
        }

    async def start(self) -> None:
        """Initialize and run the eternal loop."""
        self.running = True
        self.start_time = time.time()

        # Try to restore from DB elites
        elites = await get_top_elites(limit=self.pm.pop_size)
        if elites:
            self.pm.initialize(from_elites=elites)
            history = await get_fitness_history(limit=500)
            self.fitness_history = [h["best"] for h in history]
            insights = await get_recent_insights(limit=20)
            if insights:
                self.last_insight = insights[0]
                for ins in insights:
                    if ins.get("emergent_principle"):
                        self.emergent_principles.append(ins["emergent_principle"])
            logger.info(
                f"Restored from generation {self.pm.generation}, "
                f"best fitness {self.pm.best_fitness:.4f}"
            )
        else:
            self.pm.initialize()
            logger.info("Starting fresh from seed genome (3 islands x 12)")

        # The eternal loop
        while self.running:
            try:
                await self._run_generation()
                await asyncio.sleep(self.sleep_between)
            except asyncio.CancelledError:
                logger.info("Evolution loop cancelled")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                await asyncio.sleep(2)

    async def _run_generation(self) -> None:
        gen = self.pm.generation
        t0 = time.time()

        # ── Step 1: Evaluate all islands ──
        await self.pm.evaluate_all()

        best = self.pm.get_sorted_population()[0]
        self.fitness_history.append(best.fitness or 0)

        log_entry: dict[str, Any] = {
            "gen": gen,
            "best_fitness": best.fitness,
            "best_halluc": best.metrics.get("halluc_rate", 0),
            "avg_fitness": sum(g.fitness or 0 for g in self.pm.population) / len(self.pm.population),
            "stagnation": self.pm.stagnation_counter,
            "time": round(time.time() - t0, 2),
            "island_bests": {
                role.value: round(island.best_fitness, 4) if island.best_fitness > float("-inf") else 0
                for role, island in self.pm.islands.items()
            },
        }

        # ── Step 2: Check pending strategy results ──
        await self._evaluate_pending_strategies()

        # ── Step 3: Adaptive Claude frequency ──
        self._adapt_claude_frequency()

        # ── Step 4: Regular Claude analysis (adaptive frequency) ──
        if self.claude_enabled and gen > 0 and gen % self.claude_every_n == 0:
            diversity = self._compute_diversity()
            insight = await analyze_generation(
                genome=best,
                generation=gen,
                fitness_history=self.fitness_history,
                stagnation_counter=self.pm.stagnation_counter,
                population_diversity=diversity,
            )
            if insight:
                self.last_insight = insight
                log_entry["claude"] = True

                # Apply adjustments to exploiter island
                adjustments = insight.get("adjustments", {})
                if adjustments and self.pm.best_genome:
                    adjusted = apply_adjustments(self.pm.best_genome, adjustments)
                    adjusted.genome_id = str(uuid.uuid4())[:8]
                    adjusted.generation = gen
                    self.pm.islands[IslandRole.EXPLOITER].inject_genome(adjusted)

                # Track emergent principles
                ep = insight.get("emergent_principle")
                if ep and ep != "null" and ep not in self.emergent_principles:
                    self.emergent_principles.append(ep)

        # ── Step 5: Multi-strategy diagnosis ──
        should_diagnose = self.claude_enabled and (
            (gen > 0 and gen % self.diagnosis_every_n == 0)
            or self.pm.stagnation_counter >= self.pm.stagnation_patience // 2
        )
        if should_diagnose and gen > 5:
            diagnosis = await diagnose_and_strategize(
                genome=best,
                generation=gen,
                per_category_scores=self.pm.per_category_global,
                strategy_ledger=self.strategy_ledger[-20:],
                fitness_history=self.fitness_history,
                stagnation_counter=self.pm.stagnation_counter,
                island_states=self.pm.get_island_states(),
            )
            if diagnosis:
                strategies = diagnosis.get("strategies", [])
                strategy_genomes = build_strategy_genomes(best, strategies, gen)
                ai_island = self.pm.islands[IslandRole.AI_DIRECTED]
                for i, sg in enumerate(strategy_genomes):
                    ai_island.inject_genome(sg)
                    strategy_id = f"s{gen}_{i}"
                    self.pending_strategies[sg.genome_id] = {
                        "strategy_id": strategy_id,
                        "description": strategies[i].get("description", ""),
                        "fitness_before": best.fitness or 0,
                        "generation_proposed": gen,
                        "source_category": strategies[i].get("target_failure_mode", -1),
                    }
                    await save_strategy(
                        strategy_id=strategy_id,
                        generation=gen,
                        description=strategies[i].get("description", ""),
                        target_params=json.dumps(strategies[i].get("parameter_values", {})),
                        source_category=strategies[i].get("target_failure_mode", -1),
                        fitness_before=best.fitness or 0,
                    )
                log_entry["diagnosis"] = True
                log_entry["strategies_injected"] = len(strategy_genomes)

        # ── Step 6: Migration ──
        if gen > 0 and gen % self.migration_interval == 0:
            migration_log = self.pm.migrate()
            log_entry["migration"] = migration_log
            # Save migration events
            for msg in migration_log:
                parts = msg.split(" -> ")
                if len(parts) == 2:
                    src, tgt = parts
                    src_island = self.pm.islands.get(
                        IslandRole(src), list(self.pm.islands.values())[0]
                    )
                    migrant = src_island.get_best()
                    await save_migration_event(
                        gen, src, tgt,
                        migrant.genome_id if migrant else "",
                        migrant.fitness or 0 if migrant else 0,
                    )

        # ── Step 7: Stagnation check ──
        if self.pm.is_stagnating():
            strategy = await self.pm.escape_stagnation()
            log_entry["escape"] = strategy

        # ── Step 8: Breed all islands ──
        self.pm.breed_all_islands()

        # ── Step 9: Adaptive mutation rates ──
        self._adapt_island_mutation_rates()

        # ── Step 10: Island snapshots ──
        for role, island in self.pm.islands.items():
            avg_fit = (
                sum(g.fitness or 0 for g in island.population) / len(island.population)
                if island.population else 0
            )
            await save_island_snapshot(
                gen, role.value, island.best_fitness, avg_fit, island.stagnation_counter,
            )

        # ── Step 11: Synthesis every N gens ──
        if self.claude_enabled and gen > 0 and gen % self.synthesis_every_n == 0:
            insights = await get_recent_insights(limit=50)
            spec = await synthesize_final(
                genome=best, generation=gen, insights=insights,
            )
            if spec:
                self.last_spec = spec
                log_entry["synthesis"] = True

        # ── Step 12: Convergence check ──
        if not self.converged and self._check_convergence():
            self.converged = True
            self.converged_at_gen = gen
            log_entry["converged"] = True
            logger.info(
                f"CONVERGED at generation {gen} with fitness {self.pm.best_fitness:.4f}"
            )
            await self._export_final_spec(best, gen)

        # ── Step 13: Email notifications ──
        should_notify = (
            (gen > 0 and gen % self.notify_every_n == 0)
            or log_entry.get("converged")
        )
        if should_notify:
            elapsed = time.time() - self.start_time if self.start_time else 0
            try:
                sent = send_evolution_summary(
                    generation=gen,
                    fitness=best.fitness or 0,
                    halluc_rate=best.metrics.get("halluc_rate", 0),
                    metrics=best.metrics,
                    insight=self.last_insight,
                    emergent_principles=self.emergent_principles,
                    stagnation_counter=self.pm.stagnation_counter,
                    elapsed_seconds=elapsed,
                    converged=self.converged,
                    spec_text=self.last_spec if self.converged else None,
                )
                if sent:
                    log_entry["email_sent"] = True
            except Exception as e:
                logger.error(f"Email notification failed: {e}")

        # ── Step 14: Ollama validation (FREE, NON-BLOCKING) ──
        if gen > 0 and gen % self.ollama_every_n == 0:
            # Only start if previous validation is done
            if self._ollama_task is None or self._ollama_task.done():
                # Snapshot the population for background eval
                pop_snapshot = [g for g in self.pm.population]
                self._ollama_task = asyncio.create_task(
                    self._run_ollama_validation(pop_snapshot, gen)
                )
                log_entry["ollama_started"] = True
            else:
                log_entry["ollama_skipped"] = "previous still running"

        # Collect any completed ollama results
        if self._ollama_task and self._ollama_task.done():
            try:
                bg_results = self._ollama_task.result()
                if bg_results:
                    for r in bg_results:
                        self.real_evaluations.append(r)
                    log_entry["ollama_validation"] = len(bg_results)
                    log_entry["best_real_fitness"] = max(r["real_fitness"] for r in bg_results)
                    logger.info(
                        f"Ollama validation complete: {len(bg_results)} results, "
                        f"best real fitness: {max(r['real_fitness'] for r in bg_results):.4f}"
                    )
            except Exception as e:
                logger.info(f"Ollama validation failed: {e}")
            self._ollama_task = None

        # ── Step 15: System prompt export ──
        if gen > 0 and gen % 100 == 0 and best:
            try:
                self.best_system_prompt = compile_system_prompt(best)
                self.prompt_summary = get_prompt_summary(best)
            except Exception as e:
                logger.warning(f"Prompt compilation failed: {e}")

        # ── Max generations check ──
        if gen >= self.max_generations:
            logger.info(f"Maximum generations ({self.max_generations}) reached. Stopping.")
            self.running = False
            await self._export_final_spec(best, gen)

        self.gen_log.append(log_entry)

        if gen % 10 == 0:
            island_info = ", ".join(
                f"{r.value}:{i.best_fitness:.3f}" if i.best_fitness > float("-inf") else f"{r.value}:--"
                for r, i in self.pm.islands.items()
            )
            logger.info(
                f"Gen {gen}: fitness={best.fitness:.4f} "
                f"halluc={best.metrics.get('halluc_rate', 0):.4f} "
                f"stag={self.pm.stagnation_counter} "
                f"islands=[{island_info}]"
            )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def _run_ollama_validation(
        self, population: list[Genome], gen: int,
    ) -> list[dict]:
        """Run Ollama validation in background (non-blocking)."""
        try:
            from llm_evaluator import evaluate_top_genomes
            results = await evaluate_top_genomes(
                population,
                top_n=self.ollama_top_n,
                n_cases=self.ollama_cases,
                seed=gen,
            )
            return [
                {
                    "generation": gen,
                    "genome_id": r.genome_id,
                    "model": r.model_used,
                    "sim_fitness": round(r.simulated_fitness, 4),
                    "real_fitness": round(r.real_fitness, 4),
                    "delta": round(r.real_fitness - r.simulated_fitness, 4),
                    "elapsed": round(r.elapsed_seconds, 1),
                }
                for r in results
            ]
        except Exception as e:
            logger.info(f"Ollama background validation error: {e}")
            return []

    async def _evaluate_pending_strategies(self) -> None:
        """Check if any pending strategy genomes have been evaluated."""
        evaluated: list[str] = []
        for genome_id, info in self.pending_strategies.items():
            for island in self.pm.islands.values():
                for g in island.population:
                    if g.genome_id == genome_id and g.fitness is not None:
                        improvement = g.fitness - info["fitness_before"]
                        success = improvement > 0
                        record = {
                            **info,
                            "fitness_after": round(g.fitness, 4),
                            "improvement": round(improvement, 4),
                            "success": success,
                        }
                        self.strategy_ledger.append(record)
                        await update_strategy_result(
                            info["strategy_id"],
                            self.pm.generation,
                            g.fitness,
                            improvement,
                            1 if success else 0,
                        )
                        evaluated.append(genome_id)
                        break
        for gid in evaluated:
            del self.pending_strategies[gid]

    def _adapt_claude_frequency(self) -> None:
        """More frequent during stagnation, less during progress."""
        if self.pm.stagnation_counter >= self.pm.stagnation_patience // 2:
            self.claude_every_n = self.claude_min_every_n
        elif self.pm.stagnation_counter == 0 and len(self.fitness_history) > 10:
            recent = self.fitness_history[-10:]
            improving = recent[-1] > recent[0]
            if improving:
                self.claude_every_n = self.claude_max_every_n
            else:
                self.claude_every_n = self.claude_base_every_n
        else:
            self.claude_every_n = self.claude_base_every_n

    def _adapt_island_mutation_rates(self) -> None:
        """Adjust per-island mutation rates based on fitness improvement."""
        for role, island in self.pm.islands.items():
            if len(island.fitness_history) >= 10:
                recent = island.fitness_history[-5:]
                old = island.fitness_history[-10:-5]
                recent_avg = sum(recent) / len(recent)
                old_avg = sum(old) / len(old)
                improvement_rate = (recent_avg - old_avg) / max(abs(old_avg), 0.01)

                if improvement_rate < 0.001:
                    # Not improving: increase mutation
                    island.config.mutation_strength = min(
                        0.5, island.config.mutation_strength * 1.1
                    )
                elif improvement_rate > 0.01:
                    # Improving well: decrease mutation (refine)
                    island.config.mutation_strength = max(
                        0.03, island.config.mutation_strength * 0.95
                    )

    def _check_convergence(self) -> bool:
        """Check if fitness improvement over last N generations is < threshold."""
        window = self.convergence_window
        if len(self.fitness_history) < window:
            return False
        recent = self.fitness_history[-window:]
        old_fitness = sum(recent[:30]) / 30
        new_fitness = sum(recent[-30:]) / 30
        if old_fitness <= 0:
            return False
        improvement = (new_fitness - old_fitness) / abs(old_fitness)
        return improvement < self.convergence_threshold

    async def _export_final_spec(self, best: Genome, gen: int) -> None:
        """Generate and save the final STRATA specification."""
        data_dir = os.environ.get("STRATA_DATA_DIR", "/app/data")
        spec_path = Path(data_dir) / "STRATA_SPEC_FINAL.md"

        insights = await get_recent_insights(limit=100)
        spec = await synthesize_final(genome=best, generation=gen, insights=insights)

        if spec:
            self.last_spec = spec

        # Also export the compiled system prompt
        try:
            compiled_prompt = compile_system_prompt(best)
            prompt_path = Path(data_dir) / "STRATA_SYSTEM_PROMPT.md"
            prompt_path.write_text(compiled_prompt, encoding="utf-8")
            logger.info(f"System prompt exported to {prompt_path}")
            self.best_system_prompt = compiled_prompt
            self.prompt_summary = get_prompt_summary(best)
        except Exception as e:
            logger.warning(f"System prompt export failed: {e}")

        lines = [
            f"# STRATA Evolution Engine — Finale Spezifikation (v5)",
            f"",
            f"**Konvergiert in Generation {gen}**",
            f"**Fitness: {self.pm.best_fitness:.4f}**",
            f"**Halluzinationsrate: {best.metrics.get('halluc_rate', 0):.4f}**",
            f"**Island Model: 3 Inseln x 12 Genome**",
            f"",
            f"---",
            f"",
            f"## Konvergierte Parameter (40 total)",
            f"",
        ]
        for group_name in sorted(set(s.group for s in PARAM_SPECS)):
            lines.append(f"### {group_name.upper()}")
            lines.append("")
            lines.append("| Parameter | Wert |")
            lines.append("|-----------|------|")
            for s in PARAM_SPECS:
                if s.group == group_name:
                    val = best.values.get(s.name, 0)
                    if s.dtype == "int":
                        lines.append(f"| {s.name} | {round(val)} |")
                    else:
                        lines.append(f"| {s.name} | {val:.4f} |")
            lines.append("")

        lines.append("## Metriken")
        lines.append("")
        lines.append("| Metrik | Wert | Gewicht |")
        lines.append("|--------|------|---------|")
        from simulator import FITNESS_WEIGHTS
        for k, w in FITNESS_WEIGHTS.items():
            val = best.metrics.get(k, 0)
            lines.append(f"| {k} | {val:.4f} | {w:+.1f} |")

        lines.append("")
        lines.append("## Emergente Prinzipien")
        lines.append("")
        for i, p in enumerate(self.emergent_principles, 1):
            lines.append(f"{i}. {p}")

        lines.append("")
        lines.append("## Strategy Ledger (erfolgreich)")
        lines.append("")
        successful = [s for s in self.strategy_ledger if s.get("success")]
        for s in successful[-10:]:
            lines.append(f"- Gen {s.get('generation_proposed', '?')}: {s.get('description', '?')} (improvement: {s.get('improvement', 0):+.4f})")

        if spec:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Claude-generierte Spezifikation")
            lines.append("")
            lines.append(spec)

        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Final spec exported to {spec_path}")

        genome_path = Path(data_dir) / "best_genome_final.json"
        genome_path.write_text(
            json.dumps(best.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Best genome exported to {genome_path}")

    def _compute_diversity(self) -> dict:
        """Compute population diversity metrics."""
        if not self.pm.population:
            return {}

        fitnesses = [g.fitness or 0 for g in self.pm.population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        fitness_spread = max(fitnesses) - min(fitnesses)

        param_stds: dict[str, float] = {}
        for spec in PARAM_SPECS:
            vals = [g.values.get(spec.name, 0) for g in self.pm.population]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            param_stds[spec.name] = round(variance ** 0.5, 4)

        avg_param_std = sum(param_stds.values()) / len(param_stds) if param_stds else 0

        return {
            "avg_fitness": round(avg_fitness, 4),
            "fitness_spread": round(fitness_spread, 4),
            "avg_param_std": round(avg_param_std, 4),
            "top_diverse_params": sorted(
                param_stds.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def stop(self) -> None:
        self.running = False
