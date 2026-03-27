# Top 5 Oral-Level Ideas for NeurIPS 2026

*Cycle 3 output, updated with DreamZero/Cosmos/MolmoBot landscape analysis. Each idea verified novel against 20-75+ papers. Ranked by oral potential.*

## Landscape Context (Critical)

The field as of March 2026 is dominated by:
- **DreamZero** (NVIDIA, 2602.15922): 14B World Action Model, >2× SOTA VLAs, open-sourced
- **DreamDojo** (NVIDIA, 2602.06949): 44K hrs human video pretraining, open-sourced
- **Cosmos Policy** (NVIDIA, 2601.16163): world foundation model as manipulation policy
- **Dreamer 4** (DeepMind, 2509.24527): 2B world model, Nature publication
- **V-JEPA 2** (Meta, 2506.09985): non-generative world model, 62 hrs robot data for zero-shot MPC
- **MolmoBot** (AI2, 2603.16861): pure sim training matches π0, fully open-source
- **π*0.6** (Physical Intelligence, 2511.14759): VLA self-improvement via RL

**Our 5 ideas operate at a different abstraction level** — representations, pre-training paradigms, data efficiency, interpretability, and curricula — that is complementary to these systems, not competing with them. DynaCLIP could improve DreamZero's backbone. PhysSAE could analyze DreamZero's internals. Zero-Success could train DreamZero without demonstrations.

---

## Idea 1: DynaCLIP — Contrastive Dynamics-Vision Alignment ⭐⭐⭐⭐⭐

**Title:** *DynaCLIP: Physics-Grounded Visual Representations via Contrastive Dynamics Alignment*

**Pitch:** CLIP aligned vision with language. DynaCLIP aligns vision with physics. Two objects that behave the same physically get similar representations, regardless of appearance.

**The gap:** Every robotics backbone (DINOv2, CLIP, SigLIP, R3M, VIP, MCR) aligns with the WRONG similarity for manipulation. Verified against 25+ representation learning papers — no prior work uses physics dynamics similarity as the contrastive metric.

**Killer experiment:** The Invisible Physics Test — visually identical objects with different mass/friction. All existing encoders produce identical embeddings. DynaCLIP distinguishes them. Policies using DynaCLIP handle them correctly; DINOv2-based policies fail.

**Why oral:** New paradigm (not recombination). Foundation model contribution (pre-train once, use everywhere). Clean CLIP analogy. Memorable demo.

| Dimension | Score |
|-----------|:-----:|
| Novelty | 9 |
| Significance | 9 |
| Feasibility | 8 |
| Technical depth | 9 |
| Clarity | 9 |
| Oral potential | **8/10** |

---

## Idea 2: PhysFoundation — Physics Pre-training from Non-Robot Simulation ⭐⭐⭐⭐½

**Title:** *PhysFoundation: Pre-training Dynamics Models on Diverse Physics Simulation for Zero-Shot Manipulation Transfer*

**Pitch:** Pre-train a dynamics model on millions of pure physics simulations (objects falling, bouncing, sliding, colliding — no robot involved). Then fine-tune for robot manipulation. Physics pre-training provides 10× data efficiency over training from scratch.

**The gap:** MPP (2023) pre-trains on multiple physics systems but only for PDE surrogate modeling, not robotics. GNS (2020) trains on physics particles but doesn't transfer to manipulation. SimDist (2026) pre-trains in sim but with robot-specific data. Nobody has done "massive non-robot physics → manipulation transfer." Verified against 25+ papers.

**Key contribution:** Not just a method — a FINDING. We characterize: (1) What types of physics pre-training data matter most (contact > free-space > fluid). (2) Scaling law for physics diversity (more diverse interactions > more volume of same interaction). (3) Minimum pre-training required for manipulation transfer.

**Why oral:** "ImageNet for physics" narrative. Pre-training is free (simulation is unlimited). Connects to cognitive science (infants learn physics by OBSERVING objects before manipulating them). Scaling laws are the hottest topic in ML.

| Dimension | Score |
|-----------|:-----:|
| Novelty | 9 |
| Significance | 9 |
| Feasibility | 7 |
| Technical depth | 8 |
| Clarity | 8 |
| Oral potential | **8/10** |

**Method sketch:**
- Pre-training data: Generate 10M+ physics episodes in Isaac Lab / MuJoCo MJX. Scenarios: (a) rigid body contact (push, stack, topple), (b) projectile motion (throw, bounce), (c) friction (slide on varied surfaces), (d) mass effects (heavy vs light collisions), (e) multi-body interactions (chain reactions, dominos). No robot arm — just objects interacting.
- Model: GNN-based dynamics model (following GNS architecture but with modern improvements — Transformer-based message passing, learned edge types, multi-scale). Input: object states (position, velocity, shape embedding, physical properties). Output: next-state prediction.
- Transfer: Add robot embodiment as an additional node in the graph. Fine-tune the robot-related edges while keeping the object-object interaction network mostly frozen.
- Baselines: (a) from-scratch GNN, (b) DreamDojo-style video pre-training + IDM, (c) SimDist-style robot-sim pre-training, (d) Dreamer-v3 from scratch.
- Benchmarks: LIBERO, CALVIN, ManiSkill3 (with physical property variation).

---

## Idea 3: Zero-Success Learning — Manipulation from Failures Alone ⭐⭐⭐⭐

**Title:** *Zero-Success Learning: Robot Manipulation from Failure Data Alone via World Model Imagination*

**Pitch:** A robot learns to succeed using ZERO successful demonstrations — only failures. The world model learns dynamics from failures, identifies near-success states, and imagines corrective actions to synthesize success.

**The gap:** Only Grollman & Billard (ICRA 2011) attempted failure-only learning, and only in low-dimensional settings. No paper combines failure-only data + world model imagination for modern manipulation. Verified against 20+ papers.

**Why oral (IF it works):** "Zero successes → 80% success" is the most shocking result imaginable. Failures are FREE to collect. Would democratize robot learning. Connects to fundamental question: what information do failures contain?

**Risk assessment:** High. World model trained only on failures may lack coverage of success-state dynamics. Feasibility: 6/10. But if the proof-of-concept shows even 50% success from zero-success data, it's publishable and surprising.

| Dimension | Score |
|-----------|:-----:|
| Novelty | 9 |
| Significance | 10 |
| Feasibility | 6 |
| Technical depth | 7 |
| Clarity | 10 |
| Oral potential | **7/10** (9/10 if works) |

**Method sketch:**
- Failure data: Random + scripted exploration policies in simulation. 10K trajectories per task, ALL failures (filtered by task success checker).
- World model: Latent dynamics model (RSSM or Transformer) trained on failure trajectories. Learns: object physics, robot kinematics, contact dynamics — all present in failures.
- Near-success identification: Train a goal-proximity function g(s) = visual similarity to goal image + learned progress predictor. Identify timesteps where g(s) > threshold in failed trajectories.
- Imagination-based completion: From near-success states, CEM planning through world model to find action sequences that reach the goal. Short horizon search (3-10 steps) since we're already near success.
- Trajectory synthesis: Stitch (failure approach) + (imagined completion) = full synthetic success.
- Policy training: Standard BC or diffusion policy on synthetic successes.
- Key experiment: Compare {10, 50, 100 demos, all failures} + world model imagination vs. {10, 50, 100 demos, all successes} standard BC. Plot success rate vs. number of demos for each.

---

## Idea 4: PhysSAE — Discovering Steerable Physics Features in World Models ⭐⭐⭐⭐

**Title:** *PhysSAE: Discovering and Steering Emergent Physics Features in Manipulation World Models*

**Pitch:** Apply Sparse Autoencoders to world model hidden activations. Discover that individual latent features spontaneously encode physical quantities (mass, friction, contact force). Show these features are STEERABLE — modifying the "mass feature" changes predicted dynamics consistently.

**The gap:** SAE for VLAs (Stanford, March 2026, arXiv 2603.19183) discovered steerable motion primitives in VLAs. Nobody has applied SAEs to WORLD MODELS. Nobody has shown that world model representations spontaneously develop interpretable physics features. Nobody has demonstrated physics steerability.

**Why oral:** Discovery + method paper. "World models develop physics neurons" is a headline result (analogous to "LLMs develop reasoning circuits"). Steerability is immediately useful for controllable simulation. Connects to the hot trend of mechanistic interpretability.

| Dimension | Score |
|-----------|:-----:|
| Novelty | 8 |
| Significance | 8 |
| Feasibility | 8 |
| Technical depth | 9 |
| Clarity | 8 |
| Oral potential | **7.5/10** |

**Method sketch:**
- Train world model: Dreamer-v3 or latent diffusion world model on ManiSkill3 data WITH physical property variation (objects of varying mass, friction, restitution).
- Extract activations: Record hidden states at each layer of the dynamics model during rollouts. Collect 1M+ activation vectors.
- Train SAE: Sparse autoencoder (following Anthropic/OpenAI mechanistic interpretability methodology) on these activations. Vary dictionary size: 1K, 4K, 16K, 64K features.
- Analyze features: For each SAE feature, compute correlation with ground-truth physical quantities (mass, friction, velocity, force, contact binary). Identify "physics features" — features with high correlation to specific physical quantities.
- Test steerability: Clamp a physics feature to a high/low value during world model rollout. Does the predicted dynamics change consistently? (e.g., clamping "mass feature" high → predicted object moves slower when pushed, requires more force to lift).
- Downstream application: Use steerable physics features for (a) zero-shot physical property inference from a single interaction, (b) sim-to-real adaptation by adjusting physics features to match real-world observations, (c) controllable world simulation for policy training.
- Compare across world model architectures: Do physics features emerge in RSSM (Dreamer), Transformer, or diffusion-based dynamics models? Which architecture develops the most interpretable physics?

---

## Idea 5: DevPhys — Developmental Physics Curriculum for Robot Foundation Models ⭐⭐⭐⭐

**Title:** *DevPhys: Does Training Order Matter? A Developmental Physics Curriculum for Robot World Models*

**Pitch:** Infants learn physics (object permanence, gravity, solidity) BEFORE vision (object recognition) and language. Current robot foundation models train in reverse order: language first (LLM), then vision (VLM), then actions (VLA). We show that training a world model with a physics-first developmental curriculum dramatically improves physical reasoning and manipulation performance.

**The gap:** V-JEPA (Meta, 2025) showed physics emerges from video pre-training, but this is accidental, not curriculum-designed. QuasiSim (ECCV 2024) does physics fidelity curriculum for sim-to-real but not for world model pre-training order. PLATO (DeepMind, 2022) implements developmental psychology but not for robot manipulation. No paper has systematically compared physics-first vs. language-first vs. simultaneous training orders for robot foundation models. Verified against 30+ papers.

**Why oral:** Clean, testable hypothesis derived from cognitive science. Practical implications (redesigning training curricula is cheaper than scaling data). Surprising if true: training ORDER matters more than training DATA. Connects AI to developmental psychology in a concrete way.

| Dimension | Score |
|-----------|:-----:|
| Novelty | 8 |
| Significance | 8 |
| Feasibility | 7 |
| Technical depth | 8 |
| Clarity | 9 |
| Oral potential | **7/10** |

**Method sketch:**
- Same model architecture throughout: Transformer backbone (e.g., 300M params)
- Same total training data: physics sim data + internet video + robot demos + language descriptions
- Curriculum A (physics-first / developmental): Phase 1: physics sim only (objects interacting, forces, contacts) → Phase 2: visual prediction (internet video) → Phase 3: language-conditioned action prediction (robot demos with instructions)
- Curriculum B (language-first / standard VLA): Phase 1: language modeling → Phase 2: vision-language alignment → Phase 3: action prediction with physics
- Curriculum C (simultaneous): All data mixed uniformly throughout training
- Curriculum D (video-first / V-JEPA style): Phase 1: video prediction → Phase 2: physics sim → Phase 3: language + actions
- Evaluation: (a) Intuitive physics benchmarks (IntPhys, PhyBench), (b) world model prediction accuracy, (c) manipulation success on LIBERO/CALVIN, (d) compositional generalization on physics-varying tasks, (e) data efficiency (how many robot demos needed to reach X% success?)
- Key finding: Curriculum A (physics-first) achieves X% higher physical reasoning accuracy and Y% better manipulation success than Curriculum B (language-first) with the SAME compute and data.

---

## Final Rankings

| Rank | Idea | Oral Potential | Risk | Key Strength |
|------|------|:---:|:---:|---|
| **1** | **DynaCLIP** | 8/10 | Low | New paradigm, killer demo, clean narrative |
| **2** | **PhysFoundation** | 8/10 | Medium | "ImageNet for physics", scaling laws |
| **3** | **Zero-Success** | 7-9/10 | High | Most shocking result if it works |
| **4** | **PhysSAE** | 7.5/10 | Low-Med | Connects to hot interpretability trend |
| **5** | **DevPhys** | 7/10 | Medium | Cognitive science × ML, clean experiment |

### Recommendation for Parallel Execution

With 48 H200 GPUs and 6 independent nodes, pursue top 3 in parallel:

| Nodes | Idea | PoC Timeline |
|-------|------|-------------|
| 1-2 | DynaCLIP (data gen + contrastive pre-training) | 2 weeks |
| 3-4 | PhysFoundation (physics data gen + GNN pre-training) | 2 weeks |
| 5-6 | Zero-Success (failure data + world model + imagination) | 2 weeks |

After 2-week PoC: double down on whichever shows strongest initial results. PhysSAE and DevPhys can be pursued later with the same infrastructure.

### Honest Assessment

Am I genuinely confident ANY of these would be accepted as NeurIPS oral?

**DynaCLIP:** 60-70% confidence. The invisible physics test is compelling. The "CLIP for physics" narrative is clean. The risk is that gains on standard benchmarks (where physics variation is minimal) might be modest.

**PhysFoundation:** 50-60% confidence. The "ImageNet for physics" narrative is strong. The risk is that physics pre-training from non-robot data might not transfer as well as hoped — there's a domain gap between "objects bouncing" and "robot grasping."

**Zero-Success:** 40% confidence (but 80% if it works). The result would be phenomenal, but it might simply not work.

**PhysSAE:** 50% confidence. Depends entirely on whether world models actually develop interpretable physics features. If they do, it's a landmark finding. If they don't, the paper is dead.

**DevPhys:** 45% confidence. The result might be that training order doesn't matter much at sufficient scale — which is a negative result that's publishable but not oral-worthy.

None of these is a guaranteed oral. But DynaCLIP and PhysFoundation have the strongest combination of novelty, feasibility, and narrative clarity. If I had to bet on one, it's DynaCLIP.
