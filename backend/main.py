"""
STRATA Evolution Engine — FastAPI Server
Serves the dashboard and exposes API endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from database import (
    init_db, get_fitness_history, get_recent_insights, get_top_elites,
    get_stats, get_strategy_ledger, get_island_history,
)
from evolution import EvolutionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("strata.main")

engine = EvolutionEngine()
_evo_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _evo_task
    logger.info("STRATA Evolution Engine starting...")
    await init_db()
    _evo_task = asyncio.create_task(engine.start())
    logger.info("Evolution loop launched")
    yield
    logger.info("Shutting down evolution loop...")
    engine.stop()
    if _evo_task:
        _evo_task.cancel()
        try:
            await _evo_task
        except asyncio.CancelledError:
            pass
    logger.info("STRATA Evolution Engine stopped")


app = FastAPI(title="STRATA Evolution Engine", lifespan=lifespan)

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/health")
async def health():
    return {"status": "ok", "running": engine.running}


@app.get("/state")
async def get_state():
    return JSONResponse(engine.state)


@app.get("/stats")
async def stats():
    return await get_stats()


@app.get("/history")
async def history(limit: int = 500):
    return await get_fitness_history(limit=limit)


@app.get("/insights")
async def insights(limit: int = 20):
    return await get_recent_insights(limit=limit)


@app.get("/elites")
async def elites(limit: int = 50):
    return await get_top_elites(limit=limit)


@app.get("/spec")
async def spec():
    return {"spec": engine.last_spec}


@app.get("/system-prompt")
async def system_prompt():
    return {
        "prompt": engine.best_system_prompt,
        "summary": engine.prompt_summary,
        "generation": engine.pm.generation,
        "fitness": engine.pm.best_fitness,
    }


@app.get("/real-evals")
async def real_evals():
    return engine.real_evaluations


@app.get("/strategies")
async def strategies(limit: int = 50):
    return await get_strategy_ledger(limit=limit)


@app.get("/islands/{role}")
async def island_history(role: str, limit: int = 200):
    return await get_island_history(role, limit=limit)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
