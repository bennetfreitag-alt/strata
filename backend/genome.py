"""
STRATA Evolution Engine — Genome Module (v4)
40 parameters in 8 groups defining how an LLM should handle context,
uncertainty propagation, and hallucination avoidance.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    name: str
    group: str
    lo: float
    hi: float
    dtype: str  # "float" or "int"
    description: str


PARAM_SPECS: list[ParamSpec] = [
    # --- layers (5) ---
    ParamSpec("context_window_pct", "layers", 0.1, 1.0, "float",
             "Fraction of context window actively used for retrieval"),
    ParamSpec("layer_depth", "layers", 1, 8, "int",
             "Number of memory hierarchy layers"),
    ParamSpec("compression_ratio", "layers", 0.05, 0.95, "float",
             "Compression factor per layer transition"),
    ParamSpec("recency_weight", "layers", 0.0, 1.0, "float",
             "Weight given to recency vs relevance"),
    ParamSpec("cross_layer_bleed", "layers", 0.0, 0.5, "float",
             "Information leakage between layers"),

    # --- confidence (4) ---
    ParamSpec("base_confidence", "confidence", 0.1, 0.99, "float",
             "Starting confidence for new assertions"),
    ParamSpec("confidence_decay", "confidence", 0.001, 0.1, "float",
             "Per-hop confidence decay rate"),
    ParamSpec("min_confidence_threshold", "confidence", 0.01, 0.5, "float",
             "Below this confidence, abstain from claiming"),
    ParamSpec("confidence_aggregation_mode", "confidence", 0, 2, "int",
             "0=min, 1=geometric_mean, 2=weighted_harmonic"),

    # --- uncertainty (5) ---
    ParamSpec("uncertainty_propagation_rate", "uncertainty", 0.0, 1.0, "float",
             "How fast uncertainty spreads through derivation chains"),
    ParamSpec("uncertainty_floor", "uncertainty", 0.0, 0.3, "float",
             "Minimum uncertainty never reduced below this"),
    ParamSpec("epistemic_weight", "uncertainty", 0.0, 1.0, "float",
             "Weight of epistemic vs aleatoric uncertainty"),
    ParamSpec("uncertainty_compounding", "uncertainty", 1.0, 3.0, "float",
             "Exponent for multi-hop uncertainty growth"),
    ParamSpec("hedging_verbosity", "uncertainty", 0.0, 1.0, "float",
             "How explicitly uncertainty is communicated"),

    # --- anchors (4) ---
    ParamSpec("source_anchor_strength", "anchors", 0.0, 1.0, "float",
             "How strongly original sources anchor beliefs"),
    ParamSpec("anchor_decay_rate", "anchors", 0.0, 0.2, "float",
             "How fast anchor strength decays with context distance"),
    ParamSpec("max_anchor_count", "anchors", 1, 20, "int",
             "Maximum number of active source anchors"),
    ParamSpec("anchor_conflict_resolution", "anchors", 0, 2, "int",
             "0=newest_wins, 1=strongest_wins, 2=abstain"),
    ParamSpec("source_freshness_weight", "anchors", 0.0, 1.0, "float",
             "Preference for recent vs old source material"),

    # --- conflict (5) ---
    ParamSpec("conflict_detection_sensitivity", "conflict", 0.0, 1.0, "float",
             "Sensitivity to detecting contradictions"),
    ParamSpec("conflict_escalation_threshold", "conflict", 0.1, 0.9, "float",
             "Contradiction level that triggers explicit flagging"),
    ParamSpec("conflict_resolution_strategy", "conflict", 0, 3, "int",
             "0=flag, 1=prefer_recent, 2=prefer_authoritative, 3=abstain"),
    ParamSpec("multi_doc_conflict_weight", "conflict", 0.0, 1.0, "float",
             "Extra weight for cross-document contradictions"),
    ParamSpec("contradiction_memory_depth", "conflict", 1, 10, "int",
             "How deep contradiction tracking goes in derivation chains"),

    # --- meta (5) ---
    ParamSpec("meta_awareness_depth", "meta", 0, 3, "int",
             "Levels of self-reflective reasoning"),
    ParamSpec("calibration_sensitivity", "meta", 0.0, 1.0, "float",
             "How aggressively calibration adjustments are applied"),
    ParamSpec("self_correction_rate", "meta", 0.0, 1.0, "float",
             "Probability of catching own errors before output"),
    ParamSpec("hallucination_vigilance", "meta", 0.0, 1.0, "float",
             "Aggressiveness of hallucination self-detection"),
    ParamSpec("abstraction_level", "meta", 0.0, 1.0, "float",
             "How abstract vs concrete the reasoning is"),

    # --- memory (6) ---
    ParamSpec("working_memory_slots", "memory", 3, 20, "int",
             "Active items in working memory"),
    ParamSpec("retrieval_precision_bias", "memory", 0.0, 1.0, "float",
             "Precision vs recall tradeoff in memory retrieval"),
    ParamSpec("consolidation_interval", "memory", 1, 50, "int",
             "Steps between memory consolidation sweeps"),
    ParamSpec("forgetting_curve_steepness", "memory", 0.01, 0.5, "float",
             "How aggressively old information is forgotten"),
    ParamSpec("semantic_similarity_threshold", "memory", 0.1, 0.95, "float",
             "Minimum similarity for memory retrieval match"),
    ParamSpec("parallel_retrieval_paths", "memory", 1, 8, "int",
             "Concurrent memory retrieval channels"),

    # --- adaptive (5) ---
    ParamSpec("learning_rate", "adaptive", 0.001, 0.3, "float",
             "Speed of within-session adaptation"),
    ParamSpec("exploration_temperature", "adaptive", 0.0, 2.0, "float",
             "Temperature for exploring vs exploiting known strategies"),
    ParamSpec("feedback_integration_speed", "adaptive", 0.0, 1.0, "float",
             "How quickly feedback modifies behavior"),
    ParamSpec("domain_transfer_coefficient", "adaptive", 0.0, 1.0, "float",
             "Ability to transfer strategies across domains"),
    ParamSpec("attention_head_allocation", "adaptive", 0.0, 1.0, "float",
             "How attention distributes across context positions"),
]

PARAM_INDEX: dict[str, int] = {s.name: i for i, s in enumerate(PARAM_SPECS)}
GROUPS: list[str] = sorted(set(s.group for s in PARAM_SPECS))


# V3-converged seed values (empirically strong starting point)
_SEED_VALUES: dict[str, float] = {
    "context_window_pct": 0.82,
    "layer_depth": 4,
    "compression_ratio": 0.35,
    "recency_weight": 0.45,
    "cross_layer_bleed": 0.12,
    "base_confidence": 0.72,
    "confidence_decay": 0.035,
    "min_confidence_threshold": 0.18,
    "confidence_aggregation_mode": 1,
    "uncertainty_propagation_rate": 0.65,
    "uncertainty_floor": 0.08,
    "epistemic_weight": 0.70,
    "uncertainty_compounding": 1.8,
    "hedging_verbosity": 0.55,
    "source_anchor_strength": 0.78,
    "anchor_decay_rate": 0.06,
    "max_anchor_count": 8,
    "anchor_conflict_resolution": 2,
    "conflict_detection_sensitivity": 0.72,
    "conflict_escalation_threshold": 0.40,
    "conflict_resolution_strategy": 3,
    "multi_doc_conflict_weight": 0.65,
    "meta_awareness_depth": 2,
    "calibration_sensitivity": 0.60,
    "self_correction_rate": 0.55,
    "hallucination_vigilance": 0.80,
    "working_memory_slots": 10,
    "retrieval_precision_bias": 0.65,
    "consolidation_interval": 12,
    "forgetting_curve_steepness": 0.08,
    "semantic_similarity_threshold": 0.55,
    "learning_rate": 0.05,
    "exploration_temperature": 0.7,
    "feedback_integration_speed": 0.45,
    "domain_transfer_coefficient": 0.50,
    "attention_head_allocation": 0.60,
    "source_freshness_weight": 0.50,
    "contradiction_memory_depth": 4,
    "abstraction_level": 0.55,
    "parallel_retrieval_paths": 3,
}


@dataclass
class Genome:
    values: dict[str, float] = field(default_factory=dict)
    genome_id: str = ""
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    fitness: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def seed(cls) -> Genome:
        """Create the V3-converged seed genome."""
        return cls(values=dict(_SEED_VALUES))

    @classmethod
    def random(cls) -> Genome:
        """Create a fully random genome within bounds."""
        values: dict[str, float] = {}
        for spec in PARAM_SPECS:
            if spec.dtype == "int":
                values[spec.name] = random.randint(int(spec.lo), int(spec.hi))
            else:
                values[spec.name] = random.uniform(spec.lo, spec.hi)
        return cls(values=values)

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def mutate(self, strength: float = 0.15, targeted_params: list[str] | None = None) -> Genome:
        """Return a mutated copy. strength ∈ [0, 1] controls mutation range."""
        child = Genome(values=copy.deepcopy(self.values), parent_ids=[self.genome_id])
        targets = targeted_params or [s.name for s in PARAM_SPECS]
        for name in targets:
            if random.random() > 0.4:  # 60% chance to mutate each targeted param
                continue
            spec = PARAM_SPECS[PARAM_INDEX[name]]
            span = spec.hi - spec.lo
            delta = random.gauss(0, strength * span * 0.3)
            new_val = child.values[name] + delta
            new_val = max(spec.lo, min(spec.hi, new_val))
            if spec.dtype == "int":
                new_val = round(new_val)
            child.values[name] = new_val
        return child

    def crossover(self, other: Genome) -> Genome:
        """Uniform crossover: each param randomly from self or other."""
        child_values: dict[str, float] = {}
        for spec in PARAM_SPECS:
            if random.random() < 0.5:
                child_values[spec.name] = self.values[spec.name]
            else:
                child_values[spec.name] = other.values[spec.name]
        return Genome(
            values=child_values,
            parent_ids=[self.genome_id, other.genome_id],
        )

    def random_reset_params(self, group: str) -> Genome:
        """Return copy with all params in group randomized."""
        child = Genome(values=copy.deepcopy(self.values), parent_ids=[self.genome_id])
        for spec in PARAM_SPECS:
            if spec.group != group:
                continue
            if spec.dtype == "int":
                child.values[spec.name] = random.randint(int(spec.lo), int(spec.hi))
            else:
                child.values[spec.name] = random.uniform(spec.lo, spec.hi)
        return child

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def clamp(self) -> None:
        """Clamp all values to their bounds (in-place)."""
        for spec in PARAM_SPECS:
            v = self.values.get(spec.name, (spec.lo + spec.hi) / 2)
            v = max(spec.lo, min(spec.hi, v))
            if spec.dtype == "int":
                v = round(v)
            self.values[spec.name] = v

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genome:
        g = cls(
            values=d.get("values", {}),
            genome_id=d.get("genome_id", ""),
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
            fitness=d.get("fitness"),
            metrics=d.get("metrics", {}),
        )
        g.clamp()
        return g

    def params_by_group(self) -> dict[str, dict[str, float]]:
        groups: dict[str, dict[str, float]] = {}
        for spec in PARAM_SPECS:
            groups.setdefault(spec.group, {})[spec.name] = self.values[spec.name]
        return groups
