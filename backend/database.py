"""
STRATA Evolution Engine — Database Module
Async SQLite persistence with aiosqlite.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import aiosqlite

import os

_data_dir = os.environ.get("STRATA_DATA_DIR", "/app/data")
DB_PATH = Path(_data_dir) / "strata.db"


async def init_db() -> None:
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS genome_evals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                genome_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                fitness REAL NOT NULL,
                metrics TEXT NOT NULL,
                values_json TEXT NOT NULL,
                parent_ids TEXT DEFAULT '[]',
                timestamp REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS claude_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                analysis TEXT NOT NULL,
                bottleneck_metric TEXT,
                strategy TEXT,
                adjustments TEXT,
                emergent_principle TEXT,
                convergence_assessment TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS elite_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                genome_id TEXT NOT NULL UNIQUE,
                generation INTEGER NOT NULL,
                fitness REAL NOT NULL,
                metrics TEXT NOT NULL,
                values_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stagnation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                escape_strategy TEXT NOT NULL,
                fitness_before REAL,
                timestamp REAL NOT NULL
            )
        """)
        # V4: Strategy tracking
        await db.execute("""
            CREATE TABLE IF NOT EXISTS strategy_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                generation_proposed INTEGER NOT NULL,
                generation_evaluated INTEGER,
                description TEXT,
                target_params TEXT,
                source_category INTEGER,
                fitness_before REAL,
                fitness_after REAL,
                improvement REAL,
                success INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            )
        """)
        # V4: Island snapshots
        await db.execute("""
            CREATE TABLE IF NOT EXISTS island_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                island_role TEXT NOT NULL,
                best_fitness REAL,
                avg_fitness REAL,
                stagnation_counter INTEGER,
                timestamp REAL NOT NULL
            )
        """)
        # V4: Migration events
        await db.execute("""
            CREATE TABLE IF NOT EXISTS migration_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                source_island TEXT NOT NULL,
                target_island TEXT NOT NULL,
                genome_id TEXT NOT NULL,
                fitness REAL,
                timestamp REAL NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_evals_gen ON genome_evals(generation)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_evals_fitness ON genome_evals(fitness DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_elite_fitness ON elite_archive(fitness DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategy_gen ON strategy_ledger(generation_proposed)"
        )
        await db.commit()


async def save_genome_eval(
    genome_id: str,
    generation: int,
    fitness: float,
    metrics: dict,
    values: dict,
    parent_ids: list[str],
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO genome_evals
               (genome_id, generation, fitness, metrics, values_json, parent_ids, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                genome_id,
                generation,
                fitness,
                json.dumps(metrics),
                json.dumps(values),
                json.dumps(parent_ids),
                time.time(),
            ),
        )
        await db.commit()


async def save_insight(
    generation: int,
    analysis: str,
    bottleneck_metric: str | None,
    strategy: str | None,
    adjustments: str | None,
    emergent_principle: str | None,
    convergence_assessment: str | None,
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO claude_insights
               (generation, analysis, bottleneck_metric, strategy, adjustments,
                emergent_principle, convergence_assessment, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                generation,
                analysis,
                bottleneck_metric,
                strategy,
                adjustments,
                emergent_principle,
                convergence_assessment,
                time.time(),
            ),
        )
        await db.commit()


async def save_elite(
    genome_id: str,
    generation: int,
    fitness: float,
    metrics: dict,
    values: dict,
) -> None:
    """Save to elite archive, maintaining top 50."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT OR REPLACE INTO elite_archive
               (genome_id, generation, fitness, metrics, values_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (genome_id, generation, fitness, json.dumps(metrics), json.dumps(values), time.time()),
        )
        # Keep only top 50
        await db.execute("""
            DELETE FROM elite_archive WHERE id NOT IN (
                SELECT id FROM elite_archive ORDER BY fitness DESC LIMIT 50
            )
        """)
        await db.commit()


async def save_stagnation_event(
    generation: int, escape_strategy: str, fitness_before: float | None
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO stagnation_events
               (generation, escape_strategy, fitness_before, timestamp)
               VALUES (?, ?, ?, ?)""",
            (generation, escape_strategy, fitness_before, time.time()),
        )
        await db.commit()


async def get_top_elites(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM elite_archive ORDER BY fitness DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "genome_id": r["genome_id"],
                "generation": r["generation"],
                "fitness": r["fitness"],
                "metrics": json.loads(r["metrics"]),
                "values": json.loads(r["values_json"]),
            }
            for r in rows
        ]


async def get_fitness_history(limit: int = 500) -> list[dict]:
    """Get best fitness per generation."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT generation, MAX(fitness) as best_fitness,
                      AVG(fitness) as avg_fitness, MIN(fitness) as worst_fitness
               FROM genome_evals
               GROUP BY generation
               ORDER BY generation DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "generation": r["generation"],
                "best": r["best_fitness"],
                "avg": r["avg_fitness"],
                "worst": r["worst_fitness"],
            }
            for r in reversed(rows)
        ]


async def get_recent_insights(limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM claude_insights ORDER BY generation DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "generation": r["generation"],
                "analysis": r["analysis"],
                "bottleneck_metric": r["bottleneck_metric"],
                "strategy": r["strategy"],
                "adjustments": r["adjustments"],
                "emergent_principle": r["emergent_principle"],
                "convergence_assessment": r["convergence_assessment"],
            }
            for r in rows
        ]


async def get_stats() -> dict:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM genome_evals")
        total_evals = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT MAX(generation) FROM genome_evals")
        row = await cursor.fetchone()
        max_gen = row[0] if row[0] is not None else 0

        cursor = await db.execute("SELECT MAX(fitness) FROM genome_evals")
        row = await cursor.fetchone()
        best_fitness = row[0] if row[0] is not None else 0.0

        cursor = await db.execute("SELECT COUNT(*) FROM claude_insights")
        insight_count = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(*) FROM stagnation_events")
        stagnation_count = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(*) FROM elite_archive")
        elite_count = (await cursor.fetchone())[0]

        return {
            "total_evals": total_evals,
            "max_generation": max_gen,
            "best_fitness_ever": best_fitness,
            "insight_count": insight_count,
            "stagnation_count": stagnation_count,
            "elite_count": elite_count,
        }


# ---------------------------------------------------------------------------
# V4: Strategy ledger CRUD
# ---------------------------------------------------------------------------

async def save_strategy(
    strategy_id: str,
    generation: int,
    description: str,
    target_params: str,
    source_category: int,
    fitness_before: float,
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO strategy_ledger
               (strategy_id, generation_proposed, description, target_params,
                source_category, fitness_before, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (strategy_id, generation, description, target_params,
             source_category, fitness_before, time.time()),
        )
        await db.commit()


async def update_strategy_result(
    strategy_id: str,
    generation_evaluated: int,
    fitness_after: float,
    improvement: float,
    success: int,
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """UPDATE strategy_ledger
               SET generation_evaluated = ?, fitness_after = ?,
                   improvement = ?, success = ?
               WHERE strategy_id = ?""",
            (generation_evaluated, fitness_after, improvement, success, strategy_id),
        )
        await db.commit()


async def get_strategy_ledger(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM strategy_ledger ORDER BY generation_proposed DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "strategy_id": r["strategy_id"],
                "generation_proposed": r["generation_proposed"],
                "generation_evaluated": r["generation_evaluated"],
                "description": r["description"],
                "source_category": r["source_category"],
                "fitness_before": r["fitness_before"],
                "fitness_after": r["fitness_after"],
                "improvement": r["improvement"],
                "success": r["success"],
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# V4: Island snapshots
# ---------------------------------------------------------------------------

async def save_island_snapshot(
    generation: int,
    island_role: str,
    best_fitness: float,
    avg_fitness: float,
    stagnation_counter: int,
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO island_snapshots
               (generation, island_role, best_fitness, avg_fitness,
                stagnation_counter, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (generation, island_role, best_fitness, avg_fitness,
             stagnation_counter, time.time()),
        )
        await db.commit()


async def get_island_history(island_role: str, limit: int = 200) -> list[dict]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM island_snapshots
               WHERE island_role = ?
               ORDER BY generation DESC LIMIT ?""",
            (island_role, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "generation": r["generation"],
                "best_fitness": r["best_fitness"],
                "avg_fitness": r["avg_fitness"],
                "stagnation_counter": r["stagnation_counter"],
            }
            for r in reversed(rows)
        ]


# ---------------------------------------------------------------------------
# V4: Migration events
# ---------------------------------------------------------------------------

async def save_migration_event(
    generation: int,
    source_island: str,
    target_island: str,
    genome_id: str,
    fitness: float,
) -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO migration_events
               (generation, source_island, target_island, genome_id, fitness, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (generation, source_island, target_island, genome_id, fitness, time.time()),
        )
        await db.commit()
