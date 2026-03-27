# Iteration Log: Research Ideation

---

## Cycle 1: Initial Generation

### Self-Critique (Brutal)

**Idea 1: WorldSearch (Inference-Time Scaling) — NOT oral material**
- A reviewer would say: *"More search helps. We knew this from AlphaGo (2016). TD-MPC2 already uses CEM planning. RoboMonkey/CoVer (Stanford, 2025-2026) characterized power-law scaling for VLA test-time compute. This paper applies the same concept to world model search — an obvious next step."*
- Novelty: 6/10 — recombination of MPC + scaling law measurement
- The "insight" that more search = better results is not surprising to any reviewer who has read MuZero
- Verdict: **Poster at best.** Reject for oral.

**Idea 2: PhysLang (Language-Grounded Physics) — Spotlight at best**
- A reviewer would say: *"Why language? If you're in simulation, you HAVE the physics parameters. The Oracle-Numerical baseline will crush PhysLang. Language adds noise. The practical use case ('tell the robot it's heavy') is contrived."*
- Novelty: 7/10 — creative intersection but the "why not just use numbers?" critique is devastating
- The Oracle baseline undermines the thesis: language is a lossy encoding of information you already have
- Verdict: **Poster.** The Oracle baseline problem kills the narrative.

**Idea 3: CounterFact (Counterfactual Trajectories) — Poster**
- A reviewer would say: *"This is MBPO-style branching + gradient saliency for picking branch points. Ctrl-World already imagines successes. DiffStitch already stitches trajectories. The 'counterfactual' framing is rebranding, not innovation."*
- Novelty: 6/10 — every component exists; combination is not surprising
- Verdict: **Weak poster.** An Area Chair would down-rank this.

### Key Failure Mode in Cycle 1
All ideas were "combine technique A with domain B." None had a fundamental insight that would surprise a reviewer. Oral papers need: (1) a surprising finding, (2) a new paradigm, or (3) a shockingly good result.

---

## Cycle 2: Deep Novelty Search + New Ideas

### Additional Literature Found (Cycle 2 Searches)

**Dynamics-aligned representations (closest to DynaCLIP):**
- MCR (ICLR 2025): aligns vision with co-occurring proprioception — temporal, NOT dynamics similarity
- CLASS (CoRL 2025): contrastive pairs by action-sequence DTW — behavioral, NOT physical dynamics
- PSE (ICLR 2021 Spotlight): policy similarity embeddings — policy-based, NOT physics-based
- DynaMo: dynamics prediction as pretext — predictive, NOT contrastive by similarity
- R3M, VIP, MVP, Voltron: all use temporal/semantic/value alignment, NOT dynamics
- **Gap confirmed: no paper uses physics dynamics similarity as contrastive metric**

**Failures-only learning:**
- Grollman & Billard (2011): only paper to try failure-only learning, low-dimensional only
- Hertel & Ahmadzadeh (2021): failure-only gives avoidance info but no directional guidance
- CQL: trajectory stitching from zero-success data (navigation, not manipulation)
- **Gap confirmed: no paper shows failure-only + world model imagination for manipulation**

**Latest March 2026 papers (27 additional):**
- PlayWorld (Princeton): world models from self-play
- Latent Particle World Models (CMU): object-centric stochastic dynamics
- SAE for VLAs (Stanford): interpretable/steerable features in VLAs
- Simulation Distillation: distilling simulator into latent world model
- GigaWorld-Policy: decoupled video/action architectures

### New Ideas

#### Idea A: DynaCLIP — Contrastive Dynamics-Vision Alignment (PRIMARY)

**Title:** *DynaCLIP: Learning Physics-Grounded Visual Representations via Contrastive Dynamics Alignment*

**One-line:** CLIP aligned vision with language. DynaCLIP aligns vision with physics — two objects that behave the same physically get similar representations, regardless of how they look.

**Why it's different from everything that exists:**
- MCR aligns vision with co-occurring proprioception (temporal coincidence)
- CLASS aligns vision with action sequences (behavioral)
- PSE aligns with policy similarity (optimal policy)
- R3M/VIP align with time/value
- **DynaCLIP aligns with PHYSICAL DYNAMICS** — how the environment responds to interactions

**The killer experiment ("Invisible Physics Test"):**
Create visually IDENTICAL objects with DIFFERENT physical properties (same texture, different mass/friction). DINOv2/CLIP/R3M representations are IDENTICAL for these objects. DynaCLIP representations are DIFFERENT — because their dynamics differ. Show that policies using DynaCLIP can handle these objects appropriately, while policies using DINOv2 cannot.

**Self-evaluation:**
- Novelty: 9/10 — genuinely new paradigm, confirmed by exhaustive search
- Significance: 9/10 — could replace DINOv2 as standard robotics backbone
- Feasibility: 8/10 — contrastive learning well-understood, simulation data is free
- Technical depth: 9/10 — formulation + probing + downstream evaluation + the invisible physics test
- Clarity: 9/10 — "CLIP for physics" is immediately understandable
- Surprise: 8/10 — the invisible physics test would be memorable
- **Oral potential: 8/10** — strong narrative, clean experiments, broad impact

#### Idea B: Zero-Success Learning via World Model Imagination (HIGH-RISK)

**Title:** *Learning from Zero Successes: Robot Manipulation via World Model Imagination from Failure Data Alone*

**One-line:** A robot learns to succeed at manipulation using ONLY failed demonstrations — the world model learns dynamics from failures, identifies near-success states, and imagines corrective actions to complete the task.

**Why it's surprising:** Everyone assumes you need demonstrations of success. Failures contain ALL the dynamics information — the only thing missing is the right action at the critical moment.

**Self-evaluation:**
- Novelty: 9/10 — confirmed nobody has done this with world models
- Significance: 9/10 — if it works, failures become free training data
- Feasibility: 6/10 — world model from failure-only data may have poor coverage
- Technical depth: 7/10 — mechanism is relatively simple
- Clarity: 10/10 — "zero successes → 80% success" is instantly compelling
- Surprise: 10/10 — would be SHOCKING
- **Oral potential: IF it works, 9/10. If results are mediocre, 4/10.**

### Honest Assessment: Do I have a 9/10 novelty + significance idea?

**DynaCLIP: Novelty 9 + Significance 9 = 18/20.** YES, this meets the bar.

The remaining question is execution: can the results be dramatic enough? The invisible physics test provides the killer demo. The physics property probing shows a clear capability gap vs. DINOv2. The downstream policy improvements on physics-varying tasks would seal it.

**My genuine confidence level for NeurIPS oral:** 60-70% for DynaCLIP if executed well with strong results. That's about as high as you can get before running experiments.

### Why DynaCLIP > all Cycle 1 ideas

| Dimension | WorldSearch | PhysLang | CounterFact | **DynaCLIP** |
|-----------|:---:|:---:|:---:|:---:|
| Novelty | 6 | 7 | 6 | **9** |
| Significance | 8 | 7 | 7 | **9** |
| Feasibility | 9 | 8 | 7 | **8** |
| Surprise factor | 5 | 6 | 6 | **8** |
| Oral potential | 5 | 5 | 4 | **8** |
| TOTAL | 33 | 33 | 30 | **42** |

DynaCLIP is strictly better on every dimension except feasibility (tied with PhysLang). The gap is largest on novelty and significance — the dimensions that matter most for an oral.
