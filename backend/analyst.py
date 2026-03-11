"""
STRATA Evolution Engine — Analyst Module (v4)
Claude API integration for genome analysis, multi-strategy diagnosis,
and STRATA spec synthesis.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import uuid

import anthropic

from genome import Genome, PARAM_SPECS, GROUPS
from database import save_insight
from simulator import SCENARIO_PROFILES, compute_fitness, FITNESS_WEIGHTS

logger = logging.getLogger("strata.analyst")

ANALYSIS_SYSTEM_PROMPT = """Du bist der STRATA-Architekt — ein Experte für KI-Memory-Architekturen und evolutionäre Optimierung.

Du analysierst Simulationsdaten eines evolutionären Optimierungsloops, der ein 40-Parameter-Genom optimiert. Das Genom beschreibt, wie ein Sprachmodell Kontext speichern, Unsicherheit propagieren und Halluzinationen vermeiden soll.

Deine Aufgabe:
1. Analysiere die aktuellen Metriken und identifiziere den Flaschenhals
2. Schlage gezielte Parameter-Anpassungen vor (als Deltas)
3. Erkenne emergente Prinzipien

Antworte IMMER als valides JSON mit dieser Struktur:
{
  "analysis": "Kurze Analyse der aktuellen Situation (2-3 Sätze)",
  "bottleneck_metric": "Name der schwächsten Metrik",
  "bottleneck_params": ["param1", "param2"],
  "adjustments": {"param_name": delta_value, ...},
  "strategy": "Beschreibung der Optimierungsstrategie",
  "targeted_param_group": "Gruppe die fokussiert werden sollte",
  "emergent_principle": "Ein emergentes Prinzip das sich zeigt (oder null)",
  "convergence_assessment": "Einschätzung wie nah an Konvergenz (0-100%)"
}

Wichtig:
- adjustments sind DELTAS (z.B. +0.05, -0.1), keine absoluten Werte
- Maximal 5 Adjustments pro Analyse
- Halluzinationsrate ist die wichtigste Metrik
- Sei konservativ mit Änderungen — kleine Schritte"""


SYNTHESIS_SYSTEM_PROMPT = """Du bist der STRATA-Spezifikations-Architekt. Du schreibst die definitive STRATA-Spezifikation basierend auf den konvergierten Parametern des evolutionären Optimierungsloops.

Erstelle eine vollständige, praxistaugliche Spezifikation die beschreibt:
1. Die optimale Memory-Architektur (Layer, Kompression, Retrieval)
2. Das Confidence-Management (Schwellenwerte, Aggregation, Decay)
3. Die Unsicherheitspropagation (Raten, Compounding, Epistemic vs Aleatoric)
4. Source Anchoring und Conflict Resolution
5. Meta-Awareness und Self-Correction
6. Adaptive Strategien

Schließe ab mit einem fertigen System-Prompt der alle Prinzipien kodifiziert.

Format: Markdown mit klaren Abschnitten."""


def _extract_json(text: str) -> dict:
    """Robustly extract JSON from Claude's response, handling markdown wrapping,
    comments, trailing commas, +signs, multi-line comments, and other issues."""
    # Strip markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # Find JSON object boundaries (handle nested braces)
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    # Find matching closing brace
    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        end = text.rfind("}")
    if end == -1:
        raise json.JSONDecodeError("No closing brace found", text, 0)

    text = text[start : end + 1]

    # Remove multi-line comments (/* ... */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove single-line comments (// ...) but NOT inside strings
    # Split by lines to avoid breaking string values containing //
    cleaned_lines = []
    for line in text.split('\n'):
        # Only remove // comments that are outside of string values
        # Simple heuristic: if // appears after the value part
        in_str = False
        comment_pos = -1
        for i, c in enumerate(line):
            if c == '"' and (i == 0 or line[i - 1] != '\\'):
                in_str = not in_str
            elif c == '/' and i + 1 < len(line) and line[i + 1] == '/' and not in_str:
                comment_pos = i
                break
        if comment_pos >= 0:
            line = line[:comment_pos]
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Fix +sign in numbers: "+0.05" → "0.05" (JSON doesn't allow leading +)
    text = re.sub(r':\s*\+(\d)', r': \1', text)
    # Fix unquoted null/true/false (shouldn't be needed but just in case)
    text = re.sub(r':\s*None\b', ': null', text)
    text = re.sub(r':\s*True\b', ': true', text)
    text = re.sub(r':\s*False\b', ': false', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to fix common remaining issues
        # Remove any non-JSON text after the object
        text = text.strip()
        # Try again
        return json.loads(text)


async def analyze_generation(
    genome: Genome,
    generation: int,
    fitness_history: list[float],
    stagnation_counter: int,
    population_diversity: dict,
) -> dict | None:
    """Ask Claude to analyze the current best genome and suggest adjustments."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY set, skipping Claude analysis")
        return None

    metrics_str = json.dumps(genome.metrics, indent=2)
    values_str = json.dumps(genome.values, indent=2)
    recent_fitness = fitness_history[-20:] if fitness_history else []

    user_msg = f"""Generation {generation} | Fitness: {genome.fitness:.4f} | Stagnation: {stagnation_counter}

## Aktuelle Metriken (Best Genome)
{metrics_str}

## Aktuelle Parameter
{values_str}

## Fitness-Verlauf (letzte {len(recent_fitness)} Generationen)
{json.dumps(recent_fitness)}

## Population Diversity
{json.dumps(population_diversity, indent=2)}

Analysiere und schlage Optimierungen vor."""

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text.strip()
        result = _extract_json(text)

        await save_insight(
            generation=generation,
            analysis=result.get("analysis", ""),
            bottleneck_metric=result.get("bottleneck_metric"),
            strategy=result.get("strategy"),
            adjustments=json.dumps(result.get("adjustments", {})),
            emergent_principle=result.get("emergent_principle"),
            convergence_assessment=result.get("convergence_assessment"),
        )

        logger.info(
            f"Claude analysis: bottleneck={result.get('bottleneck_metric')}, "
            f"adjustments={len(result.get('adjustments', {}))}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None


def apply_adjustments(genome: Genome, adjustments: dict[str, float]) -> Genome:
    """Apply Claude's suggested delta adjustments to a genome."""
    adjusted = Genome(
        values=copy.deepcopy(genome.values),
        parent_ids=[genome.genome_id],
        genome_id="",
        generation=genome.generation,
    )
    for param, delta in adjustments.items():
        if param in adjusted.values:
            adjusted.values[param] += delta
    adjusted.clamp()
    return adjusted


async def synthesize_final(
    genome: Genome,
    generation: int,
    insights: list[dict],
) -> str | None:
    """Every 100 generations: write a complete STRATA specification."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    values_str = json.dumps(genome.values, indent=2)
    metrics_str = json.dumps(genome.metrics, indent=2)

    # Collect emergent principles from insights
    principles = [
        i.get("emergent_principle", "")
        for i in insights
        if i.get("emergent_principle")
    ]

    user_msg = f"""## Konvergiertes Genom (Generation {generation})

### Parameter
{values_str}

### Metriken
{metrics_str}

### Fitness: {genome.fitness:.4f}

### Emergente Prinzipien aus {len(insights)} Claude-Analysen
{chr(10).join(f'- {p}' for p in principles[-15:])}

Erstelle die vollständige STRATA v{generation // 100 + 3} Spezifikation mit fertigem System-Prompt."""

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        spec = response.content[0].text.strip()
        logger.info(f"Synthesized STRATA spec at generation {generation}")
        return spec

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return None


# ---------------------------------------------------------------------------
# V4: Multi-Strategy Diagnosis
# ---------------------------------------------------------------------------

DIAGNOSIS_SYSTEM_PROMPT = """Du bist der STRATA-Diagnostiker — ein Experte fuer KI-Memory-Architekturen.

Du erhaeltst:
1. Per-Szenario-Kategorie Fitness-Breakdown (8 Kategorien)
2. Aktuelle Parameter des besten Genoms
3. History welche Strategien funktioniert/versagt haben
4. Island-Status (3 Sub-Populationen mit unterschiedlichen Rollen)

Deine Aufgabe:
1. Identifiziere die Top 3 Failure Modes (z.B. "adversarial Szenarien verursachen hohe Halluzinationsraten")
2. Schlage 3 VERSCHIEDENE Strategien vor, jede mit spezifischen Parameter-Aenderungen
3. Jede Strategie soll ein anderes Problem addressieren

Antworte IMMER als valides JSON:
{
  "failure_modes": [
    {"category": 4, "category_name": "adversarial", "issue": "Beschreibung", "severity": 0.8},
    {"category": 1, "category_name": "reasoning_chain", "issue": "Beschreibung", "severity": 0.6}
  ],
  "strategies": [
    {
      "name": "Strategie-Name",
      "description": "Was diese Strategie macht und warum",
      "target_failure_mode": 0,
      "parameter_values": {"param_name": absoluter_wert},
      "expected_improvement": "Welche Metriken sich verbessern sollten"
    }
  ],
  "meta_observation": "Uebergeordnete Beobachtung zum Evolutionszustand"
}

Wichtig:
- parameter_values sind ABSOLUTE Werte (nicht Deltas)
- Maximal 6 Parameter pro Strategie
- Strategien sollen sich DEUTLICH unterscheiden
- Beruecksichtige die Strategy-History: wiederhole keine gescheiterten Ansaetze
- Halluzinationsrate hat das hoechste Gewicht (-3.2)"""


async def diagnose_and_strategize(
    genome: Genome,
    generation: int,
    per_category_scores: dict[int, dict[str, float]],
    strategy_ledger: list[dict],
    fitness_history: list[float],
    stagnation_counter: int,
    island_states: dict,
) -> dict | None:
    """Ask Claude to diagnose problems and propose 3 distinct strategies.
    Returns dict with 'failure_modes' and 'strategies' keys.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    # Format per-category data
    category_summary: dict[str, dict] = {}
    for cat_id, metrics in per_category_scores.items():
        name = SCENARIO_PROFILES.get(cat_id, {}).get("name", str(cat_id))
        cat_fitness = compute_fitness(metrics)
        # Find worst metric for this category
        worst_metric = "none"
        worst_val = float("inf")
        for k, v in metrics.items():
            w = FITNESS_WEIGHTS.get(k, 0)
            effective = v * w
            if w > 0 and v < worst_val:
                worst_val = v
                worst_metric = k
        category_summary[name] = {
            "fitness": round(cat_fitness, 4),
            "halluc_rate": round(metrics.get("halluc_rate", 0), 4),
            "robustness": round(metrics.get("robustness", 0), 4),
            "worst_metric": worst_metric,
        }

    # Format strategy history (last 10)
    recent_strategies = strategy_ledger[-10:]
    strategy_history_str = (
        json.dumps(recent_strategies, indent=2, default=str)
        if recent_strategies else "Noch keine Strategien getestet."
    )

    user_msg = f"""Generation {generation} | Best Fitness: {genome.fitness:.4f} | Stagnation: {stagnation_counter}

## Per-Category Performance
{json.dumps(category_summary, indent=2)}

## Aktuelle Parameter (Best Genome)
{json.dumps(genome.values, indent=2)}

## Strategy History (was funktioniert/versagt hat)
{strategy_history_str}

## Island States
{json.dumps(island_states, indent=2, default=str)}

## Fitness-Trend (letzte 20 Generationen)
{json.dumps(fitness_history[-20:] if fitness_history else [])}

Diagnose die Top Failure Modes und schlage 3 verschiedene Verbesserungsstrategien vor."""

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            system=DIAGNOSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        result = _extract_json(text)

        # Save as insight too
        meta_obs = result.get("meta_observation", "")
        failure_modes = result.get("failure_modes", [])
        await save_insight(
            generation=generation,
            analysis=meta_obs,
            bottleneck_metric=failure_modes[0].get("issue", "") if failure_modes else "",
            strategy=json.dumps(result.get("strategies", []), default=str),
            adjustments="{}",
            emergent_principle=meta_obs if meta_obs else None,
            convergence_assessment=None,
        )

        logger.info(
            f"Diagnosis: {len(failure_modes)} failure modes, "
            f"{len(result.get('strategies', []))} strategies proposed"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse diagnosis JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Diagnosis API error: {e}")
        return None


def build_strategy_genomes(
    base_genome: Genome,
    strategies: list[dict],
    generation: int,
) -> list[Genome]:
    """Convert Claude's strategy proposals into actual Genome objects."""
    genomes: list[Genome] = []
    for strategy in strategies[:3]:  # max 3
        values = copy.deepcopy(base_genome.values)
        param_values = strategy.get("parameter_values", {})
        for param, val in param_values.items():
            if param in values:
                try:
                    values[param] = float(val)
                except (TypeError, ValueError):
                    pass

        g = Genome(
            values=values,
            genome_id=str(uuid.uuid4())[:8],
            generation=generation,
            parent_ids=[base_genome.genome_id],
        )
        g.clamp()
        genomes.append(g)
    return genomes
