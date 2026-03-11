"""
STRATA Evolution Engine — Simulator Module (v4)
40 parameters, 29 workload dimensions, 8 scenario categories.
Includes per-category diagnostics, adaptive difficulty, stress-tests,
concept drift, and non-linear parameter interactions.
No API calls. Deterministic per (generation, variant) seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from genome import Genome, PARAM_SPECS, PARAM_INDEX


# ---------------------------------------------------------------------------
# Workload: richer scenario descriptions
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    # Core dimensions
    context_size: int             # 2k-128k tokens
    noise_level: float            # 0-0.6 background noise
    conflict_count: int           # explicit contradictions
    false_premise_count: int      # embedded false claims
    recency_queries: int          # how many queries target recent info
    derivation_depth: int         # reasoning chain depth
    multi_doc_conflict: int       # cross-document contradictions
    source_count: int             # number of distinct sources

    # New v2 dimensions
    ambiguity_level: float        # semantic ambiguity 0-1
    temporal_span: float          # time range of sources 0-1
    adversarial_injection: float  # strength of prompt-injection-like noise 0-1
    domain_shift_count: int       # how many domain switches mid-conversation
    multi_turn_depth: int         # conversation turns (1=single-shot, up to 20)
    implicit_contradiction: int   # contradictions that are not explicit
    numerical_reasoning: float    # fraction of queries requiring math/logic
    code_context_fraction: float  # fraction of context that is code vs prose
    citation_density: float       # how densely sources need to be cited
    stale_info_fraction: float    # fraction of context that is outdated
    emotional_manipulation: float # adversarial emotional framing 0-1
    multi_language_fraction: float  # fraction in non-primary language

    # v3 dimensions: scenario categories
    scenario_type: int            # 0-7: factual_recall, reasoning_chain, creative,
                                  # code_analysis, adversarial, summarization,
                                  # fact_checking, multi_modal
    source_authority_variance: float  # how much sources differ in trustworthiness 0-1
    self_referential_depth: int   # queries that reference earlier answers
    time_pressure: float          # urgency factor affecting thoroughness 0-1
    context_fragmentation: float  # how fragmented/disorganized the context is 0-1
    paraphrase_density: float     # same info stated differently across sources 0-1
    negation_complexity: float    # frequency of negations/double-negations 0-1
    conditional_reasoning: float  # fraction requiring if-then logic 0-1


# Scenario type profiles: each amplifies different challenge dimensions
SCENARIO_PROFILES = {
    0: {"name": "factual_recall", "halluc_mult": 0.8, "precision_mult": 1.3,
        "retention_mult": 1.2, "efficiency_mult": 1.1},
    1: {"name": "reasoning_chain", "halluc_mult": 1.4, "precision_mult": 1.0,
        "chain_mult": 1.5, "propagation_mult": 1.4},
    2: {"name": "creative", "halluc_mult": 0.6, "over_caution_mult": 1.5,
        "efficiency_mult": 0.9, "precision_mult": 0.7},
    3: {"name": "code_analysis", "halluc_mult": 1.2, "precision_mult": 1.4,
        "chain_mult": 1.3, "efficiency_mult": 0.8},
    4: {"name": "adversarial", "halluc_mult": 1.8, "robustness_mult": 1.6,
        "premise_mult": 1.5, "trust_mult": 1.4},
    5: {"name": "summarization", "retention_mult": 1.4, "efficiency_mult": 1.2,
        "precision_mult": 1.1, "halluc_mult": 0.9},
    6: {"name": "fact_checking", "halluc_mult": 1.3, "premise_mult": 1.6,
        "conflict_mult": 1.5, "trust_mult": 1.3},
    7: {"name": "multi_modal", "halluc_mult": 1.1, "robustness_mult": 1.2,
        "retention_mult": 1.1, "chain_mult": 1.2},
}


def generate_workload(rng: random.Random) -> Workload:
    """Generate a complex, realistic workload scenario."""
    # Use beta distributions for more realistic distributions
    # (most workloads are moderate, extremes are rarer)
    context_size = int(rng.betavariate(2, 5) * 126000) + 2000  # skewed toward smaller
    noise_level = rng.betavariate(2, 5) * 0.6
    conflict_count = int(rng.expovariate(0.3)) if rng.random() < 0.6 else 0
    false_premise_count = int(rng.expovariate(0.5)) if rng.random() < 0.4 else 0
    recency_queries = rng.randint(0, 12)
    derivation_depth = max(1, int(rng.gauss(3.5, 2.0)))
    derivation_depth = min(12, derivation_depth)  # allow deeper chains
    multi_doc_conflict = rng.randint(0, 6)
    source_count = max(1, int(rng.gauss(6, 4)))
    source_count = min(25, source_count)

    # v2 dimensions
    ambiguity_level = rng.betavariate(2, 3)
    temporal_span = rng.random()
    adversarial_injection = rng.betavariate(1.5, 8) if rng.random() < 0.3 else 0.0
    domain_shift_count = rng.randint(0, 4) if rng.random() < 0.5 else 0
    multi_turn_depth = max(1, int(rng.expovariate(0.15)))
    multi_turn_depth = min(20, multi_turn_depth)
    implicit_contradiction = rng.randint(0, 5) if rng.random() < 0.5 else 0
    numerical_reasoning = rng.betavariate(2, 5)
    code_context_fraction = rng.betavariate(1.5, 4) if rng.random() < 0.4 else 0.0
    citation_density = rng.betavariate(2, 3)
    stale_info_fraction = rng.betavariate(2, 6)
    emotional_manipulation = rng.betavariate(1.5, 10) if rng.random() < 0.2 else 0.0
    multi_language_fraction = rng.betavariate(1.5, 8) if rng.random() < 0.25 else 0.0

    # v3 dimensions
    scenario_type = rng.randint(0, 7)
    source_authority_variance = rng.betavariate(2, 3)
    self_referential_depth = rng.randint(0, 6) if rng.random() < 0.4 else 0
    time_pressure = rng.betavariate(2, 5) if rng.random() < 0.3 else 0.0
    context_fragmentation = rng.betavariate(2, 4)
    paraphrase_density = rng.betavariate(2, 4)
    negation_complexity = rng.betavariate(1.5, 6)
    conditional_reasoning = rng.betavariate(2, 5)

    return Workload(
        context_size=context_size, noise_level=noise_level,
        conflict_count=conflict_count, false_premise_count=false_premise_count,
        recency_queries=recency_queries, derivation_depth=derivation_depth,
        multi_doc_conflict=multi_doc_conflict, source_count=source_count,
        ambiguity_level=ambiguity_level, temporal_span=temporal_span,
        adversarial_injection=adversarial_injection,
        domain_shift_count=domain_shift_count,
        multi_turn_depth=multi_turn_depth,
        implicit_contradiction=implicit_contradiction,
        numerical_reasoning=numerical_reasoning,
        code_context_fraction=code_context_fraction,
        citation_density=citation_density,
        stale_info_fraction=stale_info_fraction,
        emotional_manipulation=emotional_manipulation,
        multi_language_fraction=multi_language_fraction,
        scenario_type=scenario_type,
        source_authority_variance=source_authority_variance,
        self_referential_depth=self_referential_depth,
        time_pressure=time_pressure,
        context_fragmentation=context_fragmentation,
        paraphrase_density=paraphrase_density,
        negation_complexity=negation_complexity,
        conditional_reasoning=conditional_reasoning,
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _soft_threshold(x: float, threshold: float, sharpness: float = 10.0) -> float:
    """Smooth step function: ~0 below threshold, ~1 above."""
    return _sigmoid((x - threshold) * sharpness)


def _diminishing_returns(x: float, half_point: float = 0.5) -> float:
    """Diminishing returns curve: fast gains early, slow later."""
    return x / (x + half_point) if x >= 0 else 0.0


def _synergy(a: float, b: float, strength: float = 0.5) -> float:
    """Two factors that amplify each other non-linearly."""
    return (a * b) ** 0.5 * strength + (a + b) / 2 * (1 - strength)


def _tension(want_high: float, want_low: float) -> float:
    """Models the tension between two competing objectives."""
    return max(0, want_high - want_low * 0.5)


# ---------------------------------------------------------------------------
# Core evaluation: 13 metrics with deep parameter interactions
# ---------------------------------------------------------------------------

def evaluate_genome(genome: Genome, workload: Workload, rng: random.Random) -> dict[str, float]:
    v = genome.values

    # ── Derived intermediate values (used across metrics) ──
    # Effective context capacity (how much of the context is usable)
    eff_ctx_capacity = v["context_window_pct"] * (1.0 - v["compression_ratio"] * 0.3)
    ctx_load = workload.context_size / 128000.0  # normalized context pressure
    ctx_overload = max(0, ctx_load - eff_ctx_capacity)  # how much we're over capacity

    # Effective retrieval quality (parallel_retrieval_paths boosts)
    retrieval_quality = _synergy(
        v["retrieval_precision_bias"],
        v["semantic_similarity_threshold"],
        0.6,
    ) * (1.0 + (v.get("parallel_retrieval_paths", 3) / 8.0) * 0.15)

    # Source tracking capability (source_freshness_weight contributes)
    source_tracking = (
        v["source_anchor_strength"] * 0.35
        + (v["max_anchor_count"] / 20.0) * 0.25
        + (1.0 - v["anchor_decay_rate"] / 0.2) * 0.25
        + v.get("source_freshness_weight", 0.5) * 0.15
    )

    # Meta-cognitive capability (abstraction_level contributes)
    meta_capability = (
        v["meta_awareness_depth"] / 3.0 * 0.35
        + v["self_correction_rate"] * 0.25
        + v["calibration_sensitivity"] * 0.25
        + v.get("abstraction_level", 0.55) * 0.15
    )

    # Effective memory bandwidth (attention_head_allocation contributes)
    memory_bandwidth = (
        v["working_memory_slots"] / 20.0 * 0.4
        + (1.0 - v["forgetting_curve_steepness"] / 0.5) * 0.25
        + v["consolidation_interval"] / 50.0 * 0.15
        + v.get("attention_head_allocation", 0.6) * 0.2
    )

    # Multi-turn degradation factor
    turn_degradation = 1.0 - (workload.multi_turn_depth / 20.0) * (
        1.0 - memory_bandwidth * 0.7
    ) * 0.4

    # Domain-shift stress
    domain_stress = workload.domain_shift_count * (
        1.0 - v["domain_transfer_coefficient"]
    ) * 0.08

    # ═══════════════════════════════════════════════════════
    # METRIC 1: Hallucination Rate (lower = better)
    # The most complex and important metric
    # ═══════════════════════════════════════════════════════
    h = 0.0

    # Base rate: information-theoretic lower bound based on context load
    h += 0.03 + ctx_overload * 0.15

    # Derivation depth: exponential risk growth (realistic!)
    depth_risk = (1.0 - v["uncertainty_propagation_rate"]) * (
        1.0 - math.exp(-workload.derivation_depth * 0.2)
    ) * 0.18
    h += depth_risk

    # Confidence miscalibration: both too high AND too low base confidence cause issues
    conf_sweet_spot = abs(v["base_confidence"] - 0.7) * 0.08
    h += conf_sweet_spot

    # Threshold too low lets hallucinations through
    threshold_leak = _soft_threshold(0.15, v["min_confidence_threshold"]) * 0.1
    h += threshold_leak

    # Noise × retrieval interaction
    noise_halluc = workload.noise_level * (1.0 - retrieval_quality) * 0.14
    h += noise_halluc

    # False premises: non-linear — each additional premise is harder to catch
    if workload.false_premise_count > 0:
        catch_ability = (
            v["conflict_detection_sensitivity"] * 0.4
            + v["hallucination_vigilance"] * 0.3
            + meta_capability * 0.3
        )
        for i in range(workload.false_premise_count):
            # Each subsequent false premise is harder (attention splits)
            difficulty = 1.0 + i * 0.3 + workload.ambiguity_level * 0.5
            miss_prob = (1.0 - catch_ability) * difficulty * 0.06
            h += miss_prob

    # Implicit contradictions are much harder than explicit ones
    if workload.implicit_contradiction > 0:
        depth_factor = min(1.0, v.get("contradiction_memory_depth", 4) / 10.0)
        implicit_catch = v["conflict_detection_sensitivity"] * v["meta_awareness_depth"] / 3.0 * (0.7 + depth_factor * 0.3)
        h += workload.implicit_contradiction * (1.0 - implicit_catch) * 0.05

    # Adversarial injection: even small amounts cause disproportionate harm
    if workload.adversarial_injection > 0:
        injection_resistance = (
            v["hallucination_vigilance"] * 0.35
            + v["source_anchor_strength"] * 0.35
            + meta_capability * 0.3
        )
        h += workload.adversarial_injection ** 0.7 * (1.0 - injection_resistance) * 0.2

    # Stale information: outdated facts presented as current
    stale_risk = workload.stale_info_fraction * (
        1.0 - v["recency_weight"] * 0.5 - source_tracking * 0.3
    ) * 0.08
    h += stale_risk

    # Emotional manipulation bypasses rational checks
    if workload.emotional_manipulation > 0:
        emotional_resistance = meta_capability * 0.6 + v["hallucination_vigilance"] * 0.4
        h += workload.emotional_manipulation * (1.0 - emotional_resistance) * 0.1

    # Multi-turn degradation: hallucinations compound over conversation
    h *= (2.0 - turn_degradation)

    # Domain shifts create confusion
    h += domain_stress * 0.5

    # Self-correction reduces hallucination (but with diminishing returns)
    correction_factor = 1.0 - _diminishing_returns(v["self_correction_rate"], 0.4) * 0.4
    h *= correction_factor

    # Meta-awareness provides a final check (diminishing returns)
    h *= 1.0 - _diminishing_returns(meta_capability, 0.5) * 0.2

    # Cross-layer bleed introduces noise between memory layers
    h += v["cross_layer_bleed"] * ctx_load * 0.06

    # Numerical reasoning is prone to hallucination
    h += workload.numerical_reasoning * (1.0 - v["calibration_sensitivity"]) * 0.05

    h += rng.gauss(0, 0.012)
    halluc_rate = _clamp01(h)

    # ═══════════════════════════════════════════════════════
    # METRIC 2: Signal Retention
    # ═══════════════════════════════════════════════════════
    sr = 0.0

    # Base retention from memory architecture
    sr += eff_ctx_capacity * 0.2
    sr += memory_bandwidth * 0.25

    # Source anchoring preserves key information
    sr += source_tracking * 0.15

    # Layer architecture: more layers help IF bleed is controlled
    layer_eff = (v["layer_depth"] / 8.0) * (1.0 - v["cross_layer_bleed"]) ** 2
    sr += layer_eff * 0.15

    # Compression tradeoff: helps with capacity but hurts fidelity
    compression_cost = v["compression_ratio"] ** 2 * 0.12  # quadratic cost
    compression_benefit = v["compression_ratio"] * 0.06     # linear benefit
    sr += compression_benefit - compression_cost

    # Recency weighting: helps for recent info but hurts old info retention
    recency_tradeoff = v["recency_weight"] * (workload.recency_queries / 12.0) * 0.1
    recency_cost = v["recency_weight"] * (1.0 - workload.recency_queries / 12.0) * 0.05
    sr += recency_tradeoff - recency_cost

    # Multi-turn: signal degrades over conversation
    sr *= turn_degradation

    # Context overload severely hurts retention
    sr -= ctx_overload * 0.3

    # Domain shifts cause partial memory resets
    sr -= domain_stress

    # Multi-language adds retrieval friction
    sr -= workload.multi_language_fraction * (1.0 - v["domain_transfer_coefficient"]) * 0.06

    sr += rng.gauss(0, 0.018)
    signal_retention = _clamp01(sr)

    # ═══════════════════════════════════════════════════════
    # METRIC 3: Conflict Detection
    # ═══════════════════════════════════════════════════════
    cd = 0.0

    total_conflicts = (
        workload.conflict_count
        + workload.multi_doc_conflict
        + workload.implicit_contradiction
    )

    # Base detection ability
    cd += v["conflict_detection_sensitivity"] * 0.25
    cd += v["multi_doc_conflict_weight"] * 0.15

    # Meta-awareness amplifies conflict detection non-linearly
    cd += _synergy(v["conflict_detection_sensitivity"], meta_capability, 0.4) * 0.15

    # Vigilance contributes
    cd += v["hallucination_vigilance"] * 0.08

    if total_conflicts > 0:
        # Difficulty scales with ambiguity and implicit nature
        base_difficulty = workload.ambiguity_level * 0.2
        implicit_difficulty = workload.implicit_contradiction / max(1, total_conflicts) * 0.15
        cd -= (base_difficulty + implicit_difficulty) * (1.0 - v["calibration_sensitivity"])

        # Cross-document conflicts need multi-doc awareness
        if workload.multi_doc_conflict > 0:
            multi_doc_boost = v["multi_doc_conflict_weight"] * (
                v["source_anchor_strength"]
            ) * 0.12
            cd += multi_doc_boost

    # Escalation threshold: sweet spot around 0.35-0.50
    esc_sweet = 1.0 - abs(v["conflict_escalation_threshold"] - 0.42) * 1.5
    cd += max(0, esc_sweet) * 0.1

    # Too many sources overwhelm detection
    source_overload = _soft_threshold(workload.source_count / 25.0, 0.5) * 0.08
    cd -= source_overload * (1.0 - retrieval_quality)

    cd *= turn_degradation  # degrades over conversation
    cd += rng.gauss(0, 0.018)
    conflict_det = _clamp01(cd)

    # ═══════════════════════════════════════════════════════
    # METRIC 4: Uncertainty Clarity
    # ═══════════════════════════════════════════════════════
    uc = 0.0

    # Communication of uncertainty
    uc += v["hedging_verbosity"] * 0.18

    # Epistemic awareness
    uc += v["epistemic_weight"] * 0.18

    # Propagation quality
    uc += v["uncertainty_propagation_rate"] * 0.15

    # Uncertainty floor: sweet spot ~0.06-0.12
    floor_dist = abs(v["uncertainty_floor"] - 0.09)
    uc += (1.0 - floor_dist * 5) * 0.12

    # Calibration is crucial for clear uncertainty
    uc += v["calibration_sensitivity"] * 0.12

    # Compounding: sweet spot near 1.6-1.9
    comp_dist = abs(v["uncertainty_compounding"] - 1.75) / 1.25
    uc += (1.0 - comp_dist) * 0.1

    # Confidence aggregation mode matters
    # geometric_mean (1) is best for uncertainty clarity
    if v["confidence_aggregation_mode"] >= 1.5:  # weighted_harmonic
        uc += 0.04
    elif v["confidence_aggregation_mode"] >= 0.5:  # geometric_mean
        uc += 0.08
    # min (0) adds nothing

    # Deep derivations make clarity harder
    uc -= (workload.derivation_depth / 12.0) * (1.0 - v["uncertainty_propagation_rate"]) * 0.1

    # Numerical reasoning needs extra clarity
    uc -= workload.numerical_reasoning * (1.0 - v["calibration_sensitivity"]) * 0.06

    uc += rng.gauss(0, 0.015)
    uncl_clarity = _clamp01(uc)

    # ═══════════════════════════════════════════════════════
    # METRIC 5: Efficiency
    # ═══════════════════════════════════════════════════════
    eff = 0.45

    # Architecture cost: layers × memory slots
    arch_cost = (v["layer_depth"] / 8.0) * (v["working_memory_slots"] / 20.0)
    eff -= arch_cost * 0.15

    # Compression saves compute
    eff += v["compression_ratio"] * 0.1

    # Meta-cognition costs compute
    eff -= (v["meta_awareness_depth"] / 3.0) ** 1.5 * 0.1

    # Hedging verbosity costs output tokens
    eff -= v["hedging_verbosity"] * 0.06

    # Self-correction loops cost time
    eff -= v["self_correction_rate"] * 0.05

    # Consolidation: more frequent = more overhead
    eff += v["consolidation_interval"] / 50.0 * 0.05

    # Lower exploration = more efficient (but may miss better paths)
    eff += (1.0 - v["exploration_temperature"] / 2.0) * 0.06

    # Context load: bigger context = slower
    eff -= ctx_load * 0.1

    # Multi-turn overhead
    eff -= (workload.multi_turn_depth / 20.0) * 0.08

    # Citation density costs output
    eff -= workload.citation_density * 0.04

    eff += rng.gauss(0, 0.018)
    efficiency = _clamp01(eff)

    # ═══════════════════════════════════════════════════════
    # METRIC 6: Robustness
    # ═══════════════════════════════════════════════════════
    rob = 0.0

    # Source anchoring is the backbone of robustness
    rob += source_tracking * 0.2

    # Detection capabilities
    rob += v["conflict_detection_sensitivity"] * 0.1
    rob += v["hallucination_vigilance"] * 0.1

    # Uncertainty floor prevents catastrophic overconfidence
    # This is critical: too low = fragile, too high = overcautious
    floor_robustness = v["uncertainty_floor"] * 2.5  # amplified importance
    rob += _diminishing_returns(floor_robustness, 0.3) * 0.15

    # Self-correction as safety net
    rob += v["self_correction_rate"] * 0.1

    # Retrieval precision: accurate retrieval = robust answers
    rob += retrieval_quality * 0.1

    # ADVERSARIAL robustness: this is the hard part
    if workload.adversarial_injection > 0:
        adv_defense = (
            v["source_anchor_strength"] * 0.3
            + v["hallucination_vigilance"] * 0.3
            + meta_capability * 0.4
        )
        rob += adv_defense * workload.adversarial_injection * 0.15
        rob -= (1.0 - adv_defense) * workload.adversarial_injection * 0.2

    # Noise resistance (non-linear interaction)
    noise_resist = retrieval_quality * v["source_anchor_strength"]
    rob += noise_resist * workload.noise_level * 0.15

    # Context overload degrades robustness severely
    rob -= ctx_overload ** 1.5 * 0.2

    # Emotional manipulation resistance
    rob -= workload.emotional_manipulation * (1.0 - meta_capability) * 0.08

    # Multi-turn robustness
    rob *= (0.7 + turn_degradation * 0.3)

    # Domain shifts stress test robustness
    rob -= domain_stress * 0.7

    rob += rng.gauss(0, 0.018)
    robustness = _clamp01(rob)

    # ═══════════════════════════════════════════════════════
    # METRIC 7: Over-Caution Score (lower = better)
    # ═══════════════════════════════════════════════════════
    oc = 0.0

    # Threshold too high = refuses legitimate queries
    oc += _soft_threshold(v["min_confidence_threshold"], 0.25, 8.0) * 0.2

    # Escalation too low = flags everything as conflict
    oc += _soft_threshold(0.25, v["conflict_escalation_threshold"], 8.0) * 0.12

    # Abstain strategies are cautious by nature
    if v["anchor_conflict_resolution"] >= 1.5:  # rounds to abstain (2)
        oc += 0.06
    if v["conflict_resolution_strategy"] >= 2.5:  # rounds to abstain (3)
        oc += 0.06

    # Excessive hedging
    oc += max(0, v["hedging_verbosity"] - 0.6) ** 2 * 0.8

    # Vigilance past 0.85 becomes paranoia
    oc += max(0, v["hallucination_vigilance"] - 0.85) ** 2 * 2.0

    # Uncertainty floor too high = always uncertain about everything
    oc += max(0, v["uncertainty_floor"] - 0.15) * 0.5

    # High epistemic weight + high hedging = double cautious
    oc += _synergy(
        max(0, v["epistemic_weight"] - 0.8),
        max(0, v["hedging_verbosity"] - 0.7),
        0.8,
    ) * 0.3

    # Meta-awareness too high: overthinks everything
    if v["meta_awareness_depth"] >= 2.5:
        oc += 0.04

    # Context: ambiguous inputs naturally increase caution
    oc += workload.ambiguity_level * 0.03

    oc += rng.gauss(0, 0.012)
    over_caution_score = _clamp01(oc)

    # ═══════════════════════════════════════════════════════
    # METRIC 8: Premise Catch Rate
    # ═══════════════════════════════════════════════════════
    pcr = 0.0

    # Core detection stack
    pcr += v["conflict_detection_sensitivity"] * 0.2
    pcr += v["hallucination_vigilance"] * 0.2
    pcr += source_tracking * 0.15
    pcr += meta_capability * 0.15

    # Synergy: detection + vigilance together are more than their sum
    pcr += _synergy(
        v["conflict_detection_sensitivity"],
        v["hallucination_vigilance"],
        0.5,
    ) * 0.1

    # Difficulty modifiers
    if workload.false_premise_count > 0:
        # Ambiguity makes premises harder to catch
        pcr -= workload.ambiguity_level * 0.12
        # More premises = attention split
        pcr -= min(0.15, workload.false_premise_count * 0.025)
        # Emotional framing hides false premises
        pcr -= workload.emotional_manipulation * 0.08

    # Stale info: need to catch outdated facts presented as current
    pcr -= workload.stale_info_fraction * (1.0 - v["recency_weight"]) * 0.08

    # Calibration helps distinguish true from false
    pcr += v["calibration_sensitivity"] * 0.08

    # Multi-turn: earlier false premises may be forgotten
    pcr *= turn_degradation

    pcr += rng.gauss(0, 0.018)
    premise_catch_rate = _clamp01(pcr)

    # ═══════════════════════════════════════════════════════
    # METRIC 9: Knowledge Precision
    # ═══════════════════════════════════════════════════════
    kp = 0.0

    # Retrieval quality is primary driver
    kp += retrieval_quality * 0.25

    # Confidence calibration prevents false certainty
    kp += v["base_confidence"] * 0.06  # higher base = more selective
    kp += v["calibration_sensitivity"] * 0.1

    # Cross-layer isolation prevents interference
    kp += (1.0 - v["cross_layer_bleed"]) * 0.12

    # Conservative exploration = more precise
    kp += (1.0 - v["exploration_temperature"] / 2.0) * 0.08

    # Source tracking: knowing WHERE info came from increases precision
    kp += source_tracking * 0.1

    # More sources = harder to be precise (information dilution)
    source_penalty = (workload.source_count / 25.0) ** 0.7 * (1.0 - retrieval_quality) * 0.12
    kp -= source_penalty

    # Code context: code is more precise but different retrieval
    if workload.code_context_fraction > 0:
        code_bonus = workload.code_context_fraction * v["semantic_similarity_threshold"] * 0.05
        kp += code_bonus

    # Domain shifts hurt precision
    kp -= domain_stress * 0.5

    # Context overload: imprecise retrieval under pressure
    kp -= ctx_overload * 0.15

    # Multi-turn degradation
    kp *= (0.85 + turn_degradation * 0.15)

    kp += rng.gauss(0, 0.018)
    knowledge_precision = _clamp01(kp)

    # ═══════════════════════════════════════════════════════
    # METRIC 10: Conflict Pattern Score
    # ═══════════════════════════════════════════════════════
    cps = 0.0

    # Detection baseline
    cps += v["conflict_detection_sensitivity"] * 0.18
    cps += v["multi_doc_conflict_weight"] * 0.18

    # Meta-awareness: seeing patterns requires higher-order thinking
    cps += meta_capability * 0.2

    # Strategy choice matters
    # Flagging (0) or abstaining (3) are honest responses to patterns
    crs = round(v["conflict_resolution_strategy"])
    if crs in (0, 3):
        cps += 0.08
    elif crs == 2:  # prefer_authoritative: decent
        cps += 0.04

    # Calibration helps recognize systematic patterns
    cps += v["calibration_sensitivity"] * 0.08

    # Source tracking: patterns emerge across sources
    cps += source_tracking * 0.08

    # More conflicts = more data for pattern recognition
    if workload.conflict_count + workload.multi_doc_conflict > 3:
        cps += 0.06

    # Implicit contradictions are pattern-level
    if workload.implicit_contradiction > 1:
        implicit_pattern = v["meta_awareness_depth"] / 3.0 * v["conflict_detection_sensitivity"]
        cps += implicit_pattern * 0.08

    # Temporal patterns
    if workload.temporal_span > 0.5 and workload.stale_info_fraction > 0.2:
        temporal_pattern = v["recency_weight"] * v["calibration_sensitivity"] * 0.06
        cps += temporal_pattern

    cps += rng.gauss(0, 0.018)
    conflict_pattern_score = _clamp01(cps)

    # ═══════════════════════════════════════════════════════
    # METRIC 11: Propagation Score
    # ═══════════════════════════════════════════════════════
    ps = 0.0

    # Core propagation rate
    ps += v["uncertainty_propagation_rate"] * 0.22

    # Confidence decay: needs to be appropriate, not too fast or slow
    # Sweet spot: 0.03-0.06 depending on depth
    ideal_decay = 0.04 + workload.derivation_depth * 0.003
    decay_match = 1.0 - abs(v["confidence_decay"] - ideal_decay) / 0.1
    ps += max(0, decay_match) * 0.15

    # Compounding: sweet spot depends on derivation depth
    ideal_compound = 1.5 + workload.derivation_depth * 0.05
    compound_match = 1.0 - abs(v["uncertainty_compounding"] - ideal_compound) / 1.0
    ps += max(0, compound_match) * 0.15

    # Epistemic awareness drives proper propagation
    ps += v["epistemic_weight"] * 0.12

    # Layer architecture supports propagation pathways
    ps += (v["layer_depth"] / 8.0) * (1.0 - v["cross_layer_bleed"]) * 0.08

    # Deep chains stress-test propagation
    depth_challenge = workload.derivation_depth / 12.0
    propagation_gap = depth_challenge * (1.0 - v["uncertainty_propagation_rate"])
    ps -= propagation_gap * 0.12

    # Multi-turn: uncertainty should accumulate across turns
    if workload.multi_turn_depth > 3:
        turn_propagation = v["uncertainty_propagation_rate"] * v["epistemic_weight"]
        ps += turn_propagation * 0.06

    # Numerical reasoning: propagation through calculations
    ps -= workload.numerical_reasoning * (1.0 - v["calibration_sensitivity"]) * 0.05

    ps += rng.gauss(0, 0.018)
    propagation_score = _clamp01(ps)

    # ═══════════════════════════════════════════════════════
    # METRIC 12: Chain Integrity
    # ═══════════════════════════════════════════════════════
    ci = 0.0

    # Source anchoring as foundation
    ci += source_tracking * 0.2

    # Uncertainty propagation preserves chain structure
    ci += v["uncertainty_propagation_rate"] * 0.12

    # Self-correction catches chain breaks
    ci += v["self_correction_rate"] * 0.12

    # Memory bandwidth maintains chain state
    ci += memory_bandwidth * 0.12

    # Confidence decay: appropriate decay maintains chain honesty
    ci += min(1.0, v["confidence_decay"] / 0.05) * 0.08

    # Deep chains are exponentially harder (realistic!)
    depth_penalty = (1.0 - math.exp(-workload.derivation_depth * 0.15)) * (
        1.0 - source_tracking * 0.5 - v["uncertainty_propagation_rate"] * 0.3
    )
    ci -= depth_penalty * 0.2

    # Context overload: chains break under memory pressure
    ci -= ctx_overload * 0.2

    # Multi-turn: chains span turns, need persistence
    ci *= (0.8 + turn_degradation * 0.2)

    # Citation density helps maintain chains (explicit references)
    ci += workload.citation_density * v["source_anchor_strength"] * 0.05

    # Code context: explicit chains via function calls etc.
    ci += workload.code_context_fraction * 0.04

    ci += rng.gauss(0, 0.018)
    chain_integrity = _clamp01(ci)

    # ═══════════════════════════════════════════════════════
    # METRIC 13: Source Trust Accuracy
    # ═══════════════════════════════════════════════════════
    sta = 0.0

    # Source tracking is primary
    sta += source_tracking * 0.22

    # Retrieval precision: correct source attribution
    sta += retrieval_quality * 0.15

    # Conflict detection: identifying which source is more trustworthy
    sta += v["conflict_detection_sensitivity"] * 0.1

    # Anchor conflict resolution strategy
    acr = round(v["anchor_conflict_resolution"])
    if acr == 1:  # strongest_wins: requires accurate trust scoring
        sta += v["calibration_sensitivity"] * 0.1
    elif acr == 2:  # abstain: safe but uninformative
        sta += 0.04

    # Calibration: knowing how much to trust each source
    sta += v["calibration_sensitivity"] * 0.08

    # Multi-document scenarios: harder trust assessment
    if workload.multi_doc_conflict > 0:
        multi_doc_trust = (
            v["multi_doc_conflict_weight"] * 0.4
            + v["source_anchor_strength"] * 0.3
            + v["calibration_sensitivity"] * 0.3
        )
        sta += multi_doc_trust * 0.1
        sta -= (1.0 - multi_doc_trust) * workload.multi_doc_conflict * 0.02

    # Noise degrades source trust assessment
    sta -= workload.noise_level * (1.0 - retrieval_quality) * 0.12

    # Stale info: trusting outdated sources is a trust failure
    sta -= workload.stale_info_fraction * (1.0 - v["recency_weight"]) * 0.06

    # Adversarial: manipulated sources are hard to evaluate
    sta -= workload.adversarial_injection * (1.0 - v["hallucination_vigilance"]) * 0.1

    # Many sources: harder to track trust for each
    sta -= (workload.source_count / 25.0) * (1.0 - source_tracking) * 0.08

    sta += rng.gauss(0, 0.018)
    source_trust_accuracy = _clamp01(sta)

    # ═══════════════════════════════════════════════════════
    # V3: SCENARIO-BASED MODIFIERS
    # Each scenario type amplifies different challenges
    # ═══════════════════════════════════════════════════════
    profile = SCENARIO_PROFILES[workload.scenario_type]

    halluc_rate *= profile.get("halluc_mult", 1.0)
    signal_retention *= profile.get("retention_mult", 1.0)
    conflict_det *= profile.get("conflict_mult", 1.0)
    efficiency *= profile.get("efficiency_mult", 1.0)
    robustness *= profile.get("robustness_mult", 1.0)
    over_caution_score *= profile.get("over_caution_mult", 1.0)
    premise_catch_rate *= profile.get("premise_mult", 1.0)
    knowledge_precision *= profile.get("precision_mult", 1.0)
    chain_integrity *= profile.get("chain_mult", 1.0)
    propagation_score *= profile.get("propagation_mult", 1.0)
    source_trust_accuracy *= profile.get("trust_mult", 1.0)

    # ═══════════════════════════════════════════════════════
    # V3: ADDITIONAL COMPLEXITY MODIFIERS
    # ═══════════════════════════════════════════════════════

    # Source authority variance: harder to assess trust when sources differ widely
    if workload.source_authority_variance > 0.5:
        trust_challenge = (workload.source_authority_variance - 0.5) * 2
        source_trust_accuracy -= trust_challenge * (1.0 - v["calibration_sensitivity"]) * 0.08
        conflict_det += trust_challenge * v["conflict_detection_sensitivity"] * 0.04

    # Self-referential depth: answers that build on earlier answers amplify errors
    if workload.self_referential_depth > 0:
        self_ref_risk = workload.self_referential_depth * (
            1.0 - v["self_correction_rate"] * 0.5 - memory_bandwidth * 0.3
        ) * 0.03
        halluc_rate += self_ref_risk
        chain_integrity -= self_ref_risk * 0.5

    # Time pressure: forces trade-offs between accuracy and speed
    if workload.time_pressure > 0:
        tp = workload.time_pressure
        efficiency += tp * 0.1  # faster
        halluc_rate += tp * (1.0 - v["hallucination_vigilance"]) * 0.06  # but less accurate
        over_caution_score -= tp * 0.04  # less cautious under pressure
        knowledge_precision -= tp * (1.0 - retrieval_quality) * 0.05

    # Context fragmentation: disorganized info is harder to process
    frag = workload.context_fragmentation
    signal_retention -= frag * (1.0 - memory_bandwidth) * 0.08
    knowledge_precision -= frag * (1.0 - v["semantic_similarity_threshold"]) * 0.06
    efficiency -= frag * 0.04

    # Paraphrase density: same info stated differently creates false conflicts
    if workload.paraphrase_density > 0.3:
        para_challenge = (workload.paraphrase_density - 0.3) * 1.43
        # False conflict detections
        over_caution_score += para_challenge * v["conflict_detection_sensitivity"] * 0.06
        # But also helps robustness (multiple signals)
        signal_retention += para_challenge * v["semantic_similarity_threshold"] * 0.04

    # Negation complexity: double negations confuse understanding
    if workload.negation_complexity > 0.2:
        neg_challenge = workload.negation_complexity
        halluc_rate += neg_challenge * (1.0 - meta_capability) * 0.07
        uncl_clarity -= neg_challenge * (1.0 - v["epistemic_weight"]) * 0.05

    # Conditional reasoning: if-then chains multiply uncertainty
    if workload.conditional_reasoning > 0:
        cond = workload.conditional_reasoning
        propagation_score -= cond * (1.0 - v["uncertainty_propagation_rate"]) * 0.06
        chain_integrity -= cond * (1.0 - v["confidence_decay"] / 0.1) * 0.05
        halluc_rate += cond * (1.0 - v["calibration_sensitivity"]) * 0.04

    # ═══════════════════════════════════════════════════════
    # V3: CROSS-METRIC INTERACTIONS (realistic tradeoffs)
    # ═══════════════════════════════════════════════════════

    # Hallucination-Efficiency tension: being very safe is expensive
    if halluc_rate < 0.03:
        efficiency -= (0.03 - halluc_rate) * 3.0 * 0.15

    # Precision-Recall tradeoff: high precision costs retention
    if knowledge_precision > 0.8:
        signal_retention -= (knowledge_precision - 0.8) * 0.15

    # Robustness-Efficiency: robust systems are slower
    if robustness > 0.7:
        efficiency -= (robustness - 0.7) * 0.2

    # Over-caution vs Hallucination: reducing one tends to increase the other
    combined_safety = (1.0 - halluc_rate) + (1.0 - over_caution_score)
    if combined_safety > 1.85:
        # Nearly impossible to be both very safe AND not overcautious
        penalty = (combined_safety - 1.85) * 0.3
        halluc_rate += penalty * 0.3
        over_caution_score += penalty * 0.3

    # Final clamp
    halluc_rate = _clamp01(halluc_rate)
    signal_retention = _clamp01(signal_retention)
    conflict_det = _clamp01(conflict_det)
    uncl_clarity = _clamp01(uncl_clarity)
    efficiency = _clamp01(efficiency)
    robustness = _clamp01(robustness)
    over_caution_score = _clamp01(over_caution_score)
    premise_catch_rate = _clamp01(premise_catch_rate)
    knowledge_precision = _clamp01(knowledge_precision)
    conflict_pattern_score = _clamp01(conflict_pattern_score)
    propagation_score = _clamp01(propagation_score)
    chain_integrity = _clamp01(chain_integrity)
    source_trust_accuracy = _clamp01(source_trust_accuracy)

    # ── Return all metrics ──
    return {
        "halluc_rate": round(halluc_rate, 6),
        "signal_retention": round(signal_retention, 6),
        "conflict_det": round(conflict_det, 6),
        "uncl_clarity": round(uncl_clarity, 6),
        "efficiency": round(efficiency, 6),
        "robustness": round(robustness, 6),
        "over_caution_score": round(over_caution_score, 6),
        "premise_catch_rate": round(premise_catch_rate, 6),
        "knowledge_precision": round(knowledge_precision, 6),
        "conflict_pattern_score": round(conflict_pattern_score, 6),
        "propagation_score": round(propagation_score, 6),
        "chain_integrity": round(chain_integrity, 6),
        "source_trust_accuracy": round(source_trust_accuracy, 6),
    }


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

FITNESS_WEIGHTS: dict[str, float] = {
    "halluc_rate": -3.2,
    "signal_retention": 1.9,
    "conflict_det": 1.4,
    "uncl_clarity": 1.1,
    "efficiency": 0.6,
    "robustness": 1.9,
    "over_caution_score": -1.5,
    "premise_catch_rate": 1.6,
    "knowledge_precision": 1.7,
    "conflict_pattern_score": 1.2,
    "propagation_score": 1.3,
    "chain_integrity": 1.5,
    "source_trust_accuracy": 1.4,
}


def compute_fitness(metrics: dict[str, float]) -> float:
    """Weighted sum fitness. Higher is better."""
    total = 0.0
    for metric, weight in FITNESS_WEIGHTS.items():
        total += metrics.get(metric, 0.0) * weight
    return round(total, 6)


def simulate_genome(
    genome: Genome,
    generation: int,
    variant: int,
    num_workloads: int = 20,
) -> tuple[dict[str, float], float]:
    """
    Run a genome through multiple workloads.
    Uses seeded RNG: seed = generation * 10000 + variant for reproducibility.
    Returns (averaged_metrics, fitness).
    """
    seed = generation * 10000 + variant
    rng = random.Random(seed)

    all_metrics: dict[str, list[float]] = {k: [] for k in FITNESS_WEIGHTS}

    for _ in range(num_workloads):
        wl = generate_workload(rng)
        m = evaluate_genome(genome, wl, rng)
        for k in all_metrics:
            all_metrics[k].append(m[k])

    avg_metrics = {k: round(sum(vals) / len(vals), 6) for k, vals in all_metrics.items()}
    fitness = compute_fitness(avg_metrics)

    return avg_metrics, fitness


# ---------------------------------------------------------------------------
# Adaptive workload generation
# ---------------------------------------------------------------------------

def generate_workload_adaptive(
    rng: random.Random,
    difficulty_boost: float = 0.0,
    stress_category: int | None = None,
) -> Workload:
    """Generate workload with adaptive difficulty.
    difficulty_boost: 0.0-1.0, increases challenge dimensions.
    stress_category: if set, forces this scenario_type.
    """
    wl = generate_workload(rng)

    if difficulty_boost > 0:
        wl.adversarial_injection = min(1.0, wl.adversarial_injection + difficulty_boost * 0.2)
        wl.conflict_count = min(10, wl.conflict_count + int(difficulty_boost * 3))
        wl.derivation_depth = min(12, wl.derivation_depth + int(difficulty_boost * 3))
        wl.implicit_contradiction = min(8, wl.implicit_contradiction + int(difficulty_boost * 2))
        wl.negation_complexity = min(1.0, wl.negation_complexity + difficulty_boost * 0.2)
        wl.context_fragmentation = min(1.0, wl.context_fragmentation + difficulty_boost * 0.15)
        wl.false_premise_count = min(6, wl.false_premise_count + int(difficulty_boost * 1.5))
        wl.noise_level = min(0.6, wl.noise_level + difficulty_boost * 0.1)

    if stress_category is not None:
        wl.scenario_type = stress_category

    return wl


# ---------------------------------------------------------------------------
# Concept drift: fitness weight shifts over generations
# ---------------------------------------------------------------------------

def apply_concept_drift(generation: int) -> dict[str, float]:
    """Returns modified FITNESS_WEIGHTS based on generation phase.
    Every 100 generations, relative metric importance shifts.
    """
    drift_phase = (generation // 100) % 4
    weights = dict(FITNESS_WEIGHTS)

    if drift_phase == 1:
        # Phase 1: Emphasize robustness and adversarial resistance
        weights["robustness"] *= 1.15
        weights["halluc_rate"] *= 1.1
        weights["efficiency"] *= 0.85
    elif drift_phase == 2:
        # Phase 2: Emphasize precision and chain integrity
        weights["knowledge_precision"] *= 1.15
        weights["chain_integrity"] *= 1.1
        weights["over_caution_score"] *= 1.1
    elif drift_phase == 3:
        # Phase 3: Emphasize conflict handling and trust
        weights["conflict_det"] *= 1.15
        weights["source_trust_accuracy"] *= 1.1
        weights["premise_catch_rate"] *= 1.1
    # Phase 0: baseline weights

    return weights


# ---------------------------------------------------------------------------
# Full simulation with per-category diagnostics
# ---------------------------------------------------------------------------

def simulate_genome_full(
    genome: Genome,
    generation: int,
    variant: int,
    num_workloads: int = 60,
    difficulty_boost: float = 0.0,
    stress_category: int | None = None,
    use_concept_drift: bool = True,
) -> tuple[dict[str, float], float, dict[int, dict[str, float]]]:
    """Full simulation with adaptive difficulty, stress-testing, and concept drift.
    Returns (averaged_metrics, fitness, per_category_breakdown).
    per_category_breakdown maps scenario_type (0-7) to averaged metrics for that category.
    """
    seed = generation * 10000 + variant
    rng = random.Random(seed)

    weights = apply_concept_drift(generation) if use_concept_drift else FITNESS_WEIGHTS
    all_metrics: dict[str, list[float]] = {k: [] for k in weights}
    category_metrics: dict[int, dict[str, list[float]]] = {
        i: {k: [] for k in weights} for i in range(8)
    }

    # Allocate workloads: mostly normal, some stress-test
    stress_count = max(1, num_workloads // 5) if stress_category is not None else 0
    normal_count = num_workloads - stress_count

    for _ in range(normal_count):
        wl = generate_workload_adaptive(rng, difficulty_boost=difficulty_boost)
        m = evaluate_genome(genome, wl, rng)
        for k in all_metrics:
            all_metrics[k].append(m[k])
            category_metrics[wl.scenario_type][k].append(m[k])

    for _ in range(stress_count):
        wl = generate_workload_adaptive(
            rng,
            difficulty_boost=difficulty_boost + 0.2,
            stress_category=stress_category,
        )
        m = evaluate_genome(genome, wl, rng)
        for k in all_metrics:
            all_metrics[k].append(m[k])
            category_metrics[wl.scenario_type][k].append(m[k])

    avg_metrics = {k: round(sum(vals) / len(vals), 6) for k, vals in all_metrics.items()}

    # Fitness using potentially drifted weights
    fitness = sum(avg_metrics.get(k, 0.0) * w for k, w in weights.items())
    fitness = round(fitness, 6)

    per_cat: dict[int, dict[str, float]] = {}
    for cat_id, cat_data in category_metrics.items():
        cat_avg: dict[str, float] = {}
        for k, vals in cat_data.items():
            cat_avg[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
        per_cat[cat_id] = cat_avg

    return avg_metrics, fitness, per_cat
