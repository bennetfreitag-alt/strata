# STRATA — System for Transparent Retrieval and Accuracy Through Architecture

Evolutionary optimization engine that evolves system prompts to reduce AI hallucinations. Runs 10,000+ generations of mathematical simulation validated by real local LLM testing — completely free using Ollama.

**Result:** A battle-tested system prompt with 40 optimized parameters covering confidence management, uncertainty propagation, conflict detection, source anchoring, and hallucination prevention.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USER/strata.git
cd strata

# Start (Docker — includes Ollama automatically)
cp .env.example .env
docker compose up --build -d

# Dashboard
open http://localhost:7860
```

The system starts evolving immediately. No API key needed — runs 100% free with local Ollama.

## How It Works

```
                    STRATA Evolution Engine (v5)

    ┌─────────────────────────────────────────────────────┐
    │              3 Islands x 12 Genomes                 │
    │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
    │  │Exploiter │ │ Explorer │ │   AI-Directed      │  │
    │  │(refine)  │ │(explore) │ │(strategy injection)│  │
    │  └────┬─────┘ └────┬─────┘ └────────┬───────────┘  │
    │       └──────┬──────┘───────────────┘               │
    │              ↓ Migration every 20 gens              │
    │  ┌───────────────────────────────────────────────┐  │
    │  │        Mathematical Simulator (FREE)          │  │
    │  │  13 metrics × 60 workloads per genome         │  │
    │  └───────────────────┬───────────────────────────┘  │
    │                      ↓                              │
    │  ┌───────────────────────────────────────────────┐  │
    │  │    Ollama Validation every 100 gens (FREE)    │  │
    │  │  Real LLM testing with llama3.2:1b            │  │
    │  └───────────────────┬───────────────────────────┘  │
    │                      ↓                              │
    │  ┌───────────────────────────────────────────────┐  │
    │  │         Prompt Compiler                       │  │
    │  │  40 parameters → deployable system prompt     │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
```

**Evolutionary Cycle per Generation:**
1. Evaluate all 36 genomes across 60 simulation workloads
2. Breed within islands (crossover + mutation)
3. Migrate best genomes between islands
4. Escape stagnation with adaptive strategies
5. Validate top genome with real Ollama LLM (every 100 gens)
6. Compile optimized system prompt (every 100 gens)
7. Export final spec at convergence or max generations

## 40 Evolved Parameters (8 Groups)

| Group | Parameters | What It Controls |
|-------|-----------|-----------------|
| **Layers** | 5 | Memory hierarchy depth, compression, recency |
| **Confidence** | 4 | Starting confidence, decay rate, minimum threshold |
| **Uncertainty** | 5 | Propagation, compounding, epistemic weighting |
| **Anchors** | 5 | Source anchoring strength, freshness, conflict resolution |
| **Conflict** | 5 | Detection sensitivity, escalation, contradiction memory |
| **Meta** | 5 | Hallucination vigilance, self-correction, calibration |
| **Memory** | 6 | Working memory slots, retrieval precision, forgetting |
| **Adaptive** | 5 | Learning rate, exploration, feedback, domain transfer |

## 13 Fitness Metrics

| Metric | Weight | Target |
|--------|--------|--------|
| halluc_rate | **-3.2** | Minimize hallucinations |
| signal_retention | +1.9 | Preserve information through layers |
| robustness | +1.9 | Resist noise and adversarial inputs |
| knowledge_precision | +1.7 | Precise knowledge retrieval |
| premise_catch_rate | +1.6 | Detect false premises |
| chain_integrity | +1.5 | Sound reasoning chains |
| over_caution_score | **-1.5** | Avoid being too cautious |
| conflict_det | +1.4 | Detect contradictions |
| source_trust_accuracy | +1.4 | Evaluate source reliability |
| propagation_score | +1.3 | Propagate uncertainty correctly |
| conflict_pattern_score | +1.2 | Recognize conflict patterns |
| uncl_clarity | +1.1 | Clear uncertainty communication |
| efficiency | +0.6 | Computational efficiency |

## Results (10,000 Generations)

After 4.7 hours and 360,000 evaluations:

- **Sim Fitness:** 12.00 (near-maximum)
- **Hallucination Rate:** 8.7%
- **Signal Retention:** 79.6%
- **Conflict Detection:** 84.8%
- **Convergence:** Generation 5,225
- **Cost:** $0

The generated system prompt is saved to `data/STRATA_SYSTEM_PROMPT.md`.

## Using the System Prompt

Copy the contents of `data/STRATA_SYSTEM_PROMPT.md` and use it as a system prompt with any LLM:

- **ChatGPT / GPT-4:** Paste as "Custom Instructions" or system message
- **Claude:** Use as system prompt in the API
- **Local models:** Prepend as system message in Ollama, llama.cpp, etc.

The prompt instructs the model to:
- Start with max 77% confidence, decaying 5% per reasoning step
- Never state anything below 1% confidence
- Anchor every claim to provided sources
- Flag contradictions with [WIDERSPRUCH ERKANNT]
- Run an anti-hallucination checklist before every response
- Use calibrated uncertainty language (5 levels)

## Configuration

All settings via `.env` file or Docker environment variables. See `.env.example` for full reference.

Key settings:
| Variable | Default | Description |
|----------|---------|------------|
| `CLAUDE_ENABLED` | false | Enable Claude API for AI-guided evolution |
| `MAX_GENERATIONS` | 10000 | Total generations to run |
| `OLLAMA_EVERY_N` | 100 | Real LLM validation frequency |
| `OLLAMA_MODEL` | llama3.2:1b | Local model for validation |

## Local Development

```bash
# Without Docker
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Install Ollama
brew install ollama    # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2:1b

# Run
cd backend
export STRATA_DATA_DIR=../data
export CLAUDE_ENABLED=false
python main.py
```

## Dashboard

The web dashboard at `http://localhost:7860` includes 10 tabs:

- **Overview:** Real-time fitness, hallucination rate, generation counter
- **Population:** All 36 genomes with fitness scores
- **Genome:** Detailed parameter view of the best genome
- **Fitness:** Historical fitness chart
- **Categories:** Per-category performance breakdown
- **Islands:** Island model status (Exploiter/Explorer/AI-Directed)
- **Strategies:** AI-proposed strategy tracking
- **Log:** Generation-by-generation event log
- **System-Prompt:** Live compiled system prompt with copy button
- **Real vs Sim:** Ollama validation results comparing simulated vs real fitness

## License

MIT
