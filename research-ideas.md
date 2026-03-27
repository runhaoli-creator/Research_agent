# Research Ideas: World Models for Robotics

*Generated from gap analysis, March 2026. Targeting NeurIPS 2026.*

---

## Idea 1: Inference-Time World Model Scaling for Robotic Manipulation

**Title:** *Think Before You Act: Inference-Time Scaling Laws for Robotic Manipulation via World Model Search*

**One-line pitch:** Instead of training larger policies, give a small policy more "thinking time" by searching over actions in a world model — revealing the first inference-time scaling law for robotic manipulation.

**Motivation:** LLMs show clear inference-time scaling (chain-of-thought, reasoning tokens, tree search). VLA-RL and SimpleVLA-RL hint at similar effects in robotics but haven't characterized a scaling law. No paper has systematically studied how more test-time compute via world model simulation translates to manipulation performance. This is Gap 5 in our analysis.

**Key insight:** A trained world model can simulate the outcomes of multiple candidate action sequences. By increasing the number of simulated rollouts at test time, a small policy should improve monotonically — and the performance gain should follow a predictable scaling law. This is analogous to how MCTS improved AlphaGo, but applied to continuous visual manipulation with modern world models.

**Method sketch:**
Train a latent world model (diffusion-based or RSSM-based) on manipulation demonstrations. At test time, instead of executing the policy's single best action, generate N candidate action chunks, simulate each forward through the world model, and select the one with the highest predicted value (from a learned value function) or lowest predicted distance to the goal. Systematically vary N from 1 to 10,000 and measure success rate.

The key technical challenge is making the value estimation reliable enough that more search actually helps (rather than finding adversarial inputs to the value function). We address this with: (1) ensemble value functions to filter unreliable estimates, (2) world model confidence weighting, (3) progressive narrowing (sample broadly first, then refine around promising candidates).

We also test tree-structured search: instead of evaluating independent rollouts, build a search tree where promising branches are expanded (analogous to beam search). Compare flat sampling vs. tree search to understand what search structure works for manipulation.

**Expected experiments:**
- **Simulator:** LIBERO (130 tasks), CALVIN (34 tasks, long-horizon chains)
- **Baselines:** OpenVLA, OpenVLA-OFT, Octo, π0-style flow matching VLA (without search)
- **Key experiment:** Fix policy size (e.g., 300M params), vary test-time compute budget (1, 10, 100, 1000, 10000 rollouts). Plot success rate vs. compute on log scale. Fit power law.
- **Ablations:** (a) search strategy (random, CEM, tree search), (b) value function accuracy, (c) world model fidelity, (d) action chunk length
- **GPU hours (proof-of-concept):** ~800 H200-hours. World model training: 200h. Policy training: 100h. Scaling experiments: 500h (many evaluations at different compute budgets).
- **Node allocation:**
  - Nodes 1-2: Train world model + value function (8×H200 DDP)
  - Node 3: Train baseline policies
  - Node 4: Scaling curve experiments on LIBERO
  - Node 5: Scaling curve experiments on CALVIN
  - Node 6: Ablation studies (search strategies, value function variants)

**Why it's NeurIPS-worthy:**
- First demonstration of inference-time scaling laws in robotic manipulation — a foundational result
- Connects two major threads: scaling laws (from LLMs) and world models (from embodied AI)
- Practical implication: you can deploy smaller, cheaper policies and compensate with compute at test time
- Clean, reproducible experimental setup in simulation

**Risks:**
- Value function may not be accurate enough — more search could find adversarial examples rather than better actions
- World model errors may accumulate over rollouts, making longer horizons worse with more compute
- The scaling law might plateau quickly (diminishing returns)

---

## Idea 2: Language-Grounded World Models — Teaching Dynamics through Physical Descriptions

**Title:** *PhysLang: Language-Grounded World Models for Zero-Shot Physical Reasoning in Manipulation*

**One-line pitch:** Condition a world model on language descriptions of physical properties ("heavy", "fragile", "slippery") so it can predict dynamics for novel objects described only in words.

**Motivation:** Current world models have NO mechanism to incorporate knowledge about physical properties — they must learn friction, mass, and material behavior purely from visual experience. Meanwhile, LLMs encode substantial physical commonsense. Language could serve as a bridge: describe an object's properties in words, and the world model adjusts its dynamics predictions accordingly. This is Gap 7.

**Key insight:** Physical properties (mass, friction, elasticity, fragility) are often describable in language and dramatically affect manipulation dynamics, but are NOT always visually distinguishable. A white ceramic mug and a white plastic mug look identical but behave differently when dropped. Language conditioning gives the world model access to physical knowledge that vision alone cannot provide.

**Method sketch:**
Build a world model that takes as input: (1) current visual observation, (2) robot action, (3) language description of physical properties of objects in the scene. The language input conditions the dynamics prediction through cross-attention.

Training data comes from physics simulation (Isaac Lab or ManiSkill3) where we programmatically vary physical properties (mass: 0.1–10kg, friction: 0.1–1.0, elasticity: 0.1–0.9, fragility: binary) and generate corresponding language descriptions via templates and paraphrasing. The world model is trained to predict the next state conditioned on all three inputs.

At test time, you describe a new object ("this is a heavy, slippery metal cube") and the world model predicts appropriate dynamics — slower movement when pushing heavy objects, sliding on low-friction surfaces, etc. — without ever having seen that specific object-property combination during training.

We use a factored architecture: a visual encoder (DINOv2) processes the scene, a language encoder (frozen CLIP or SigLIP text encoder) processes property descriptions, and a dynamics core (transformer or diffusion) predicts next-state, conditioned on both via cross-attention and FiLM conditioning.

**Expected experiments:**
- **Simulator:** ManiSkill3 (GPU-parallel, controllable physics) + Isaac Lab (photorealistic rendering)
- **Tasks:** Push, pick-and-place, pour, stack — all with varying object physical properties
- **Key experiment:** Train on N physical property combinations, test on held-out combinations. Measure prediction accuracy and downstream policy success. Compare: (a) language-conditioned world model, (b) vision-only world model, (c) oracle (ground-truth physics parameters as input)
- **Ablations:** (a) which properties matter most for language grounding, (b) template vs. free-form language descriptions, (c) scaling number of training property combinations
- **GPU hours (proof-of-concept):** ~600 H200-hours. Sim data generation: 100h. World model training: 300h. Evaluation + ablations: 200h.
- **Node allocation:**
  - Nodes 1-2: Isaac Lab / ManiSkill3 data generation (massive parallel envs)
  - Nodes 3-4: World model training (main method + baselines)
  - Node 5: Evaluation on held-out property combinations
  - Node 6: Ablation studies (architecture, language encoding, property types)

**Why it's NeurIPS-worthy:**
- Entirely new paradigm: using language as a physical property interface for world models
- Bridges two major fields: physical reasoning in NLP and dynamics prediction in robotics
- Practical value: a robot could ask a human "is this object heavy?" and incorporate the answer into its predictions
- Clean compositional generalization story (novel property combinations)

**Risks:**
- Language descriptions of physics may be too coarse ("heavy" doesn't specify exactly how heavy)
- The model might learn to ignore language and rely on visual shortcuts
- Gap between template-generated language and natural language descriptions

---

## Idea 3: Cross-Embodiment World Models via Environment-Embodiment Factorization

**Title:** *FactorWorld: Factorized World Models for Cross-Embodiment Transfer in Robotic Manipulation*

**One-line pitch:** Decompose world model dynamics into transferable environment physics and embodiment-specific kinematics, enabling few-shot world model adaptation to new robots.

**Motivation:** Every world model is trained for a single robot embodiment. Meanwhile, VLAs have shown cross-embodiment transfer (Octo, X-VLA). But the world models they could use for planning are NOT cross-embodiment. This means every new robot needs its own world model trained from scratch. Factoring dynamics into embodiment-independent (object physics) and embodiment-dependent (robot kinematics/contacts) would allow sharing the expensive-to-learn physics component. This is Gap 2.

**Key insight:** When a Franka pushes a block, the block's sliding dynamics depend on friction and mass (environment physics), not on whether a Franka or a WidowX is pushing it. The robot-specific part is how the end-effector contacts the block and how the arm moves. By factoring these, the environment physics module transfers for free.

**Method sketch:**
Architecture has two modules: (1) Environment Dynamics Module (EDM) — predicts how objects in the scene evolve given abstract "interaction forces" (contact point, force direction/magnitude). Trained across multiple embodiments on shared object interactions. (2) Embodiment Dynamics Module (EbDM) — maps robot joint commands to abstract interaction forces and predicts robot state evolution. Specific to each robot.

Pre-train EDM on 3+ robot embodiments interacting with shared objects in simulation. Then, to adapt to a new embodiment: freeze EDM, train only EbDM on small amount of new-robot data. Evaluate: does the factored world model with few-shot EbDM training match or exceed a monolithic world model trained from scratch?

The factorization is enforced architecturally: EDM receives only object states + abstract interaction descriptors (contact location, force vector), NOT raw robot state. This forces it to learn embodiment-agnostic physics.

**Expected experiments:**
- **Simulator:** ManiSkill3 (supports multiple robot models: Franka, WidowX, xArm, Sawyer)
- **Tasks:** Push, pick-and-place, stack across all embodiments
- **Key experiment:** Pre-train EDM on 3 robots, adapt to 4th robot with {10, 50, 100, 500} demos. Compare: (a) FactorWorld, (b) monolithic world model trained from scratch, (c) monolithic world model fine-tuned from single-robot pre-training
- **GPU hours (proof-of-concept):** ~700 H200-hours. Multi-robot data generation: 100h. EDM pre-training: 200h. EbDM adaptation experiments: 200h. Baselines + ablations: 200h.
- **Node allocation:**
  - Nodes 1-2: Multi-robot data generation in ManiSkill3
  - Node 3: EDM pre-training (8×H200 DDP)
  - Node 4: Monolithic baselines
  - Node 5: Few-shot adaptation experiments (vary data amount)
  - Node 6: Ablations (factorization architecture, interaction representation)

**Why it's NeurIPS-worthy:**
- First cross-embodiment world model — fills a clear gap between VLA progress and world model progress
- Clean technical contribution (factorization + transfer)
- Practical impact: deploying world models to new robots becomes cheap
- Connects to broader "foundation model" narrative for robotics

**Risks:**
- The factorization might be too lossy — some dynamics are truly entangled between robot and environment (e.g., how the gripper shape affects grasp stability)
- ManiSkill3 robots may be too similar (all single arms) to show meaningful cross-embodiment transfer
- Abstract interaction representation may not capture enough information for accurate prediction

---

## Idea 4: Counterfactual World Models for Sample-Efficient Policy Learning

**Title:** *CounterFact: Learning from What Could Have Been — Counterfactual Trajectory Synthesis for Robot Policy Improvement*

**One-line pitch:** Use a world model to generate "what-if" alternative trajectories at decision points in failed demonstrations, turning every failure into dozens of synthetic successes.

**Motivation:** Failures in robot learning are discarded or used only for negative reward signal. But a failure contains rich information about what went wrong — and a world model can imagine what would have happened with different actions. This counterfactual data could provide orders of magnitude more training signal from the same number of demonstrations. This is Gap 6.

**Key insight:** At each timestep of a failed trajectory, the world model can branch: "what if the robot had moved 2cm to the left?" By generating multiple counterfactual continuations per decision point, we create a tree of trajectories. The ones that succeed (as judged by a reward model) become additional positive training data. The branching points where small action changes flip failure to success are the most informative training examples.

**Method sketch:**
Phase 1: Train a world model and a binary success classifier on demonstration data. Phase 2: For each failed trajectory, identify "critical decision points" where the world model predicts that small action perturbations could change the outcome (high reward sensitivity). Phase 3: At each critical point, sample K alternative action chunks, roll out through the world model, and classify outcomes. Phase 4: Successful counterfactual trajectories become additional training data for the policy. Phase 5: Retrain policy on original + counterfactual data.

Critical decision point identification uses the gradient of the success classifier with respect to actions through the world model: points where ∂success/∂action is large are decision-critical. This focuses counterfactual generation where it matters most.

**Expected experiments:**
- **Simulator:** LIBERO-Long (hard long-horizon tasks), CALVIN ABC-D
- **Key experiment:** Fix demonstration budget (10, 25, 50 demos per task). Compare: (a) BC on original demos, (b) BC on original + random augmented demos, (c) BC on original + counterfactual demos, (d) DAgger. Show counterfactual data provides 5–10× effective sample efficiency.
- **Ablations:** (a) number of counterfactuals per decision point, (b) decision point identification methods, (c) world model fidelity requirements
- **GPU hours (proof-of-concept):** ~500 H200-hours. World model + classifier training: 150h. Counterfactual generation: 150h. Policy training + evaluation: 200h.
- **Node allocation:**
  - Node 1: World model training
  - Node 2: Success classifier training
  - Node 3: Counterfactual generation (LIBERO)
  - Node 4: Counterfactual generation (CALVIN)
  - Node 5: Policy training with counterfactual data
  - Node 6: Baselines (BC, DAgger, random augmentation)

**Why it's NeurIPS-worthy:**
- Novel use of world models — counterfactual reasoning hasn't been applied to manipulation
- Clear practical value: dramatically more training signal from expensive demonstrations
- Connects to causal reasoning literature in ML (counterfactual inference, SCMs)
- Clean, controlled experiments with clear metrics (sample efficiency curves)

**Risks:**
- World model errors in counterfactual branches may compound, producing physically implausible trajectories
- Success classifier may be unreliable, accepting failed counterfactuals or rejecting good ones
- Critical decision points may be hard to identify accurately — manipulation success often depends on accumulated precision, not single decision points

---

## Idea 5: Physics-Constrained Latent World Models via Differentiable Simulation

**Title:** *PhysDreamer: Constraining Latent World Model Dynamics with Differentiable Physics*

**One-line pitch:** Train a latent world model whose dynamics are directly constrained by a differentiable physics simulator, guaranteeing physically plausible predictions without post-hoc filtering.

**Motivation:** Physics hallucination is the #1 barrier to deploying video world models (Gap 3). Current approaches patch this post-hoc: ABot-PhysWorld uses DPO, MIND-V uses V-JEPA2 as validator. These are band-aids. We propose constraining the dynamics representation ITSELF to follow physical laws, using gradients from a differentiable physics engine during training.

**Key insight:** Differentiable physics simulators (DiffTaichi, Warp, Brax, MuJoCo MJX) can provide gradients of physical predictions with respect to state. By sharing a state representation between the latent world model and the differentiable simulator, we can add a "physics consistency" loss that penalizes the world model when its latent dynamics diverge from what physics predicts. The latent space learns to encode physically meaningful quantities (positions, velocities, forces) because that's what the physics engine needs.

**Method sketch:**
Architecture: (1) Visual encoder maps observation → latent state z, (2) Physics decoder maps z → physical state (positions, velocities, masses of scene objects), (3) Differentiable physics engine predicts next physical state given current physical state + action, (4) Latent dynamics model predicts next z given current z + action.

Training objective: standard world model loss (reconstruction + latent prediction) + physics consistency loss (||physics_decoder(z_{t+1,predicted}) - physics_engine(physics_decoder(z_t), action)||²).

The physics consistency loss forces the latent dynamics to be "explainable" by Newtonian physics. During inference, the physics decoder can be discarded — the latent dynamics have already internalized physical constraints.

**Expected experiments:**
- **Simulator:** Isaac Lab (has differentiable physics via Warp) or MuJoCo MJX
- **Tasks:** Object pushing (varying mass/friction), stacking (stability), pouring (fluid-like dynamics)
- **Key experiment:** Compare long-horizon prediction quality: (a) PhysDreamer vs. (b) unconstrained Dreamer-v3, (c) ABot-PhysWorld-style DPO post-training, (d) RoboScape-style depth+keypoint regularization. Measure physics violation rate and downstream policy success.
- **GPU hours (proof-of-concept):** ~800 H200-hours. Differentiable sim setup + data: 100h. PhysDreamer training: 300h. Baselines: 200h. Evaluation: 200h.
- **Node allocation:**
  - Nodes 1-2: PhysDreamer training (need GPU for both world model + differentiable physics)
  - Node 3: Unconstrained Dreamer-v3 baseline
  - Node 4: DPO-based physics alignment baseline
  - Node 5: Long-horizon evaluation + physics violation metrics
  - Node 6: Ablations (physics loss weight, which physical quantities to constrain)

**Why it's NeurIPS-worthy:**
- Principled approach to physics grounding (constraints, not patches)
- Novel combination: latent world models + differentiable physics
- Addresses the most important open problem in the field (physics hallucination)
- Strong technical depth: the physics consistency loss formulation, latent-to-physical mapping, and gradient flow analysis are all substantive contributions

**Risks:**
- Differentiable physics simulators may not cover all relevant physics (deformable objects, fluids, articulated mechanisms)
- The physics decoder might become a bottleneck — mapping latent states to physical states accurately is itself hard
- Training may be slow or unstable due to coupled optimization of world model + physics decoder

---

## Idea 6: World Models as Targeted Data Engines for VLA Training

**Title:** *DataDreamer: World Model-Guided Data Generation for Closing the Long Tail of Robot Manipulation*

**One-line pitch:** Use a trained world model to identify where a VLA policy will fail (distribution gaps), then generate targeted synthetic demonstrations to fill those gaps — closing the long tail 10× faster than random data augmentation.

**Motivation:** VLA policies fail on the long tail of manipulation scenarios — unusual object poses, edge-case grasps, rare configurations. Current data augmentation (random crops, color jitter) doesn't target these gaps. A world model can identify failure modes by simulating the policy forward and detecting low-confidence or failing trajectories. It can then generate synthetic demonstrations specifically for those scenarios. This is Gap 9.

**Key insight:** Random augmentation is like searching for a needle in a haystack. A world model tells you WHERE the needles are: it simulates the policy's behavior, finds states where it fails or is uncertain, and generates corrective demonstrations anchored at those states. The world model acts as a "data curriculum" that evolves as the policy improves.

**Method sketch:**
Iterative loop: (1) Train VLA policy on current dataset. (2) Deploy policy in world model (imagination rollouts) across diverse initial conditions. (3) Identify failure states using success classifier + world model uncertainty. (4) Generate synthetic demonstrations starting from or passing through failure states (using MPC through the world model or by querying a privileged expert in sim). (5) Add synthetic demonstrations to training set. (6) Repeat.

This is distinct from DAgger (which requires real environment interaction) and from random augmentation (which doesn't target gaps). The world model provides both the failure identification AND the synthetic demonstration generation.

**Expected experiments:**
- **Simulator:** LIBERO-PRO (specifically designed to expose VLA brittleness), LIBERO-90/10
- **Baselines:** Random augmentation, DAgger (with sim oracle), curriculum learning, RoboTransfer
- **Key experiment:** Start with 10 demos per task. Compare data efficiency: how many additional synthetic demos does each method need to reach 90% success? DataDreamer should need 5–10× fewer.
- **GPU hours (proof-of-concept):** ~600 H200-hours. World model training: 150h. Iterative loop (5 iterations): 300h. Baselines: 150h.
- **Node allocation:**
  - Node 1: World model training
  - Node 2: VLA policy training (iterative)
  - Node 3: Failure identification + synthetic demo generation (LIBERO)
  - Node 4: Failure identification + synthetic demo generation (LIBERO-PRO)
  - Node 5: Baselines (DAgger, random augmentation)
  - Node 6: Ablations (failure identification methods, synthetic demo quality)

**Why it's NeurIPS-worthy:**
- Practical, immediate value for robotics teams struggling with data efficiency
- Clean framework: world model as both diagnoser and treatment
- Novel combination of active learning ideas with world model capabilities
- Connects to broader "data engine" narrative (Tesla, scale.ai) but makes it autonomous

**Risks:**
- World model errors may cause misidentification of failure modes (false positives = wasted data, false negatives = missed gaps)
- Synthetic demonstrations may have systematic biases from the world model
- The iterative loop might be unstable (model improves, changes failure modes, world model becomes stale)

---

## Idea 7: Object-Centric Diffusion World Models for Compositional Manipulation

**Title:** *SlotDyn: Object-Centric Diffusion Dynamics for Compositional Generalization in Robotic Manipulation*

**One-line pitch:** Decompose scenes into object slots (via DINOv2 + slot attention), then predict per-object dynamics with a graph diffusion model — enabling zero-shot generalization to novel object combinations.

**Motivation:** Current world models predict entire scenes monolithically. This means they memorize scene-level correlations rather than learning object-level physics. When objects are rearranged or new objects are introduced, monolithic models fail. Object-centric representations with interaction-based dynamics should enable compositional generalization: if you know how a cup behaves and how a plate behaves, you can predict what happens when they interact. This is Gap 4.

**Key insight:** Object-centric world models (SlotFormer, C-SWM) proved the concept but were limited to toy scenes with simple shapes. Modern vision backbones (DINOv2) can extract high-quality object features from realistic scenes. By combining DINOv2-based slot attention with graph neural network diffusion dynamics, we can build the first object-centric world model that works on realistic manipulation scenes.

**Method sketch:**
(1) Object discovery: DINOv2 features → slot attention → K object slots (position, appearance, shape embeddings). (2) Interaction graph: construct a graph where objects are nodes and edges represent potential interactions (based on proximity). (3) Graph diffusion dynamics: a GNN-based diffusion model predicts the evolution of all slots simultaneously, with message passing capturing inter-object interactions. (4) Action conditioning: robot action is injected as an additional node connected to objects near the end-effector.

Key architectural choice: slots are decoded independently (per-object rendering via spatial broadcast decoder), ensuring the world model truly represents individual objects and not scene-level textures.

Compositional generalization test: train on scenes with objects {A, B, C} in pairs ({A,B}, {A,C}, {B,C}), test on the unseen triple {A,B,C} and novel objects {D,E}.

**Expected experiments:**
- **Simulator:** LIBERO (controlled object sets), ManiSkill3 (diverse objects with segmentation masks)
- **Key experiment:** Systematic compositional generalization test. Train on subset of object combinations, evaluate on held-out combinations. Compare: (a) SlotDyn, (b) monolithic video prediction (SVD-based), (c) monolithic latent prediction (Dreamer-style), (d) SlotFormer (original object-centric baseline)
- **GPU hours (proof-of-concept):** ~700 H200-hours. Object discovery module: 100h. Graph dynamics training: 300h. Baselines: 200h. Evaluation: 100h.
- **Node allocation:**
  - Node 1: DINOv2 slot attention training on manipulation scenes
  - Node 2: Graph diffusion dynamics training
  - Nodes 3-4: Monolithic baselines (video diffusion, Dreamer)
  - Node 5: Compositional generalization evaluation
  - Node 6: Ablations (number of slots, graph structure, diffusion steps)

**Why it's NeurIPS-worthy:**
- Bridges object-centric representation learning (well-studied in ML) with practical world models for manipulation
- Clean compositional generalization result would be a first for manipulation world models
- Technically deep: slot attention + graph diffusion + compositional evaluation
- Connects to cognitive science (humans understand the world through objects, not pixels)

**Risks:**
- Slot attention may fail on cluttered, realistic manipulation scenes (overlapping objects, partial occlusion)
- Graph diffusion may be overkill — simpler GNN dynamics might suffice (reducing the novelty)
- Compositional generalization might require more objects than simulation can efficiently provide

---

## Idea 8: Multi-Granularity World Models — Pixels, Points, and Semantics

**Title:** *GrainWorld: Multi-Granularity World Models with Cross-Level Consistency for Robust Manipulation*

**One-line pitch:** A single world model that simultaneously predicts at pixel, point-cloud, and semantic levels — with cross-level consistency constraints making each granularity reinforce the others.

**Motivation:** Existing world models operate at a single granularity: pixels (video diffusion), latent vectors (Dreamer), 3D points (PointWorld), or semantic states. Each has trade-offs: pixels are rich but hallucinate physics; points capture geometry but miss appearance; semantics are compact but lose spatial detail. No world model leverages multiple granularities simultaneously with mutual consistency constraints. This would be more robust than any single granularity. See Gaps 3 and 4.

**Key insight:** Physics violations in pixel space often correspond to geometric inconsistencies in point-cloud space and logical inconsistencies in semantic space. By requiring predictions to be consistent ACROSS granularities, errors at one level are caught and corrected by the others. The semantic level catches "the block passed through the table" (semantic: block should be ON table); the point-cloud level catches "the object teleported" (geometric discontinuity); the pixel level catches "the texture changed" (appearance inconsistency).

**Method sketch:**
Three parallel prediction heads share a common latent backbone:
- **Pixel head:** Predicts next RGB frame (lightweight decoder from shared latent)
- **Point head:** Predicts next point cloud / depth map (geometric decoder)
- **Semantic head:** Predicts next object states and relations (graph-structured output)

Cross-level consistency losses:
- Pixel ↔ Point: rendered depth from point cloud must match pixel-predicted depth
- Pixel ↔ Semantic: object presence/position from semantic head must match segmentation from pixel head
- Point ↔ Semantic: point-cloud centroids must match semantic position predictions

Training: Standard prediction loss per head + weighted sum of consistency losses. The consistency losses are differentiable and don't require additional supervision.

**Expected experiments:**
- **Simulator:** ManiSkill3 (provides RGB + depth + segmentation + object states natively)
- **Key experiment:** Compare long-horizon prediction and downstream policy learning: (a) GrainWorld, (b) pixel-only (video diffusion), (c) point-only (PointWorld-style), (d) latent-only (Dreamer-v3). Measure physics violation rate, geometric accuracy, and policy success.
- **GPU hours (proof-of-concept):** ~800 H200-hours. Multi-modal data generation: 100h. GrainWorld training: 300h. Single-granularity baselines: 200h. Evaluation: 200h.
- **Node allocation:**
  - Nodes 1-2: GrainWorld training (multi-head model, 8×H200 DDP)
  - Node 3: Pixel-only baseline (video diffusion)
  - Node 4: Point-only baseline + latent-only baseline
  - Node 5: Downstream policy learning experiments
  - Node 6: Ablations (which consistency losses matter, single vs. multi-granularity)

**Why it's NeurIPS-worthy:**
- Novel architecture: no prior world model operates at multiple granularities simultaneously
- Cross-level consistency is a principled and testable hypothesis
- Addresses both physics hallucination (via geometric constraints) and robustness (via multi-view redundancy)
- Comprehensive evaluation across existing single-granularity approaches

**Risks:**
- Training three heads + consistency losses may be complex to optimize (loss balancing)
- The consistency losses may over-constrain the model, reducing expressiveness
- Point cloud prediction quality may bottleneck the system if depth estimation is poor
- May be seen as "just multi-task learning" without a deeper insight

---

## Self-Evaluation Scores

| Dimension | Idea 1: Inference Scaling | Idea 2: PhysLang | Idea 3: FactorWorld | Idea 4: CounterFact | Idea 5: PhysDreamer | Idea 6: DataDreamer | Idea 7: SlotDyn | Idea 8: GrainWorld |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Novelty** | 9 | 9 | 8 | 8 | 7 | 7 | 7 | 7 |
| **Significance** | 10 | 8 | 8 | 8 | 9 | 8 | 7 | 7 |
| **Feasibility** | 9 | 8 | 7 | 7 | 6 | 8 | 7 | 6 |
| **Technical depth** | 8 | 8 | 8 | 8 | 9 | 7 | 8 | 8 |
| **Clarity** | 10 | 9 | 8 | 9 | 7 | 9 | 8 | 7 |
| **Sim-only validatable** | 10 | 10 | 10 | 10 | 10 | 9 | 10 | 10 |
| **TOTAL** | **56** | **52** | **49** | **50** | **48** | **48** | **47** | **45** |

### Rankings

| Rank | Idea | Score | Verdict |
|------|------|-------|---------|
| 1 | Inference-Time World Model Scaling | 56/60 | **Strong submit** — timely, clean, high impact |
| 2 | PhysLang (Language-Grounded World Models) | 52/60 | **Strong submit** — novel paradigm, creative |
| 3 | CounterFact (Counterfactual Trajectories) | 50/60 | **Submit** — clean story, but risks with WM accuracy |
| 4 | FactorWorld (Cross-Embodiment) | 49/60 | **Submit** — fills clear gap, execution matters |
| 5 | PhysDreamer (Physics-Constrained) | 48/60 | **Borderline** — principled but technically complex |
| 6 | DataDreamer (Targeted Data Engine) | 48/60 | **Borderline** — practical value but may seem incremental |
| 7 | SlotDyn (Object-Centric Diffusion) | 47/60 | **Borderline** — known ingredients, novelty in combination |
| 8 | GrainWorld (Multi-Granularity) | 45/60 | **Weak** — engineering-heavy, may lack crisp insight |

### Honest Assessment

- **Ideas 1-2** have the strongest narratives. Idea 1 ("inference-time scaling for robotics") is the clearest NeurIPS contribution because it's a fundamental result about scaling laws. Idea 2 is the most creative — nobody has thought to use language as a physics interface for world models.
- **Ideas 3-4** are solid contributions but depend heavily on execution quality.
- **Ideas 5-8** risk being seen as incremental: combining known techniques (differentiable physics + world models, active learning + world models, slot attention + diffusion, multi-task learning with consistency losses). They need very strong results to overcome the "just combining A and B" critique.
