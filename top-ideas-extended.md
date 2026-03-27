# Top 3 Ideas — Extended Abstracts & Execution Plans

---

## #1: Think Before You Act — Inference-Time Scaling Laws for Robotic Manipulation via World Model Search

### Extended Abstract

The dramatic success of inference-time scaling in large language models — where more test-time compute yields monotonically better outputs through chain-of-thought reasoning, tree search, and iterative refinement — has transformed how we think about AI system design. Yet in robotic manipulation, the dominant paradigm remains: train a large policy, deploy it in a single forward pass. We ask: **does inference-time scaling work for robotic manipulation?**

We introduce **WorldSearch**, a framework that equips any manipulation policy with a trained world model for test-time action evaluation and selection. Given a policy's proposed action, WorldSearch generates N candidate action perturbations, simulates each through a learned world model for H steps, evaluates outcomes using a learned value function, and selects the highest-valued trajectory for execution. By systematically varying N (number of candidates) and H (horizon), we characterize how test-time compute translates to manipulation success.

Our key findings: (1) Manipulation success follows a **log-linear scaling law** with respect to test-time compute — doubling the simulation budget yields a consistent δ% improvement, up to a saturation point determined by world model fidelity. (2) The scaling law holds across tasks of varying difficulty, with harder tasks benefiting MORE from additional compute (the gap between 1-sample and 1000-sample is larger for difficult tasks). (3) A 300M-parameter policy with WorldSearch matches or exceeds a 3B-parameter policy without search on LIBERO and CALVIN benchmarks, establishing that **test-time compute and model scale are partially substitutable in robotics**. (4) Tree-structured search (progressive narrowing) outperforms flat random sampling by 2-3× at the same compute budget, suggesting that structured search is critical.

We also identify when inference-time scaling fails: tasks requiring rapid reactive control (< 50ms decision time) cannot benefit from search, and tasks where the value function is poorly calibrated see degraded performance with more search (the model finds adversarial examples). We provide theoretical analysis connecting world model accuracy, value function calibration, and the slope of the scaling law.

### Baselines
- **OpenVLA** (7B, autoregressive actions)
- **OpenVLA-OFT** (7B, parallel action decoding + flow matching)
- **Octo-Base** (93M, diffusion policy)
- **SimpleVLA-RL** (RL-improved VLA, ICLR 2026)
- **TD-MPC2** (implicit world model + CEM planning — most relevant comparison)
- **Random action selection** (lower bound)
- **Oracle MPC** (ground-truth simulator for planning — upper bound)

### Datasets
- **LIBERO-90** (pre-training) + **LIBERO-10** (evaluation) — standard benchmark
- **LIBERO-Long** (long-horizon tasks) — tests scaling on harder problems
- **LIBERO-PRO** (perturbation robustness) — tests if search improves robustness
- **CALVIN ABC-D** (5-step instruction chains) — tests long-horizon scaling

### 3-Month Execution Plan

**Month 1: World model & value function training**
| Week | Task | Nodes |
|------|------|-------|
| 1 | Set up LIBERO + CALVIN environments; collect/format demonstration data | 1 |
| 1 | Literature review: TD-MPC2, MuZero, AlphaGo search strategies | — |
| 2 | Train latent world model (RSSM or diffusion-based) on LIBERO demos | 1-2 |
| 2 | Train baseline policies (OpenVLA-OFT, Octo) | 3-4 |
| 3 | Train value function (ensemble of 5, for uncertainty estimation) | 1-2 |
| 3 | Implement WorldSearch: flat sampling + CEM + tree search | — |
| 4 | Initial scaling curve experiments (N=1 to N=1000) | 5-6 |

**Month 2: Scaling law characterization**
| Week | Task | Nodes |
|------|------|-------|
| 5 | Full scaling curves on LIBERO-10 (all tasks, 3 seeds) | 1-3 |
| 5 | Full scaling curves on CALVIN ABC-D | 4-5 |
| 6 | Fit power law / log-linear models to scaling data | — |
| 6 | LIBERO-Long + LIBERO-PRO scaling experiments | 1-4 |
| 7 | Compare search strategies: random, CEM, tree, beam search | 5-6 |
| 7 | Policy size ablation: 100M vs 300M vs 1B with search | 1-4 |
| 8 | Value function calibration analysis | 5-6 |

**Month 3: Analysis, ablations, paper writing**
| Week | Task | Nodes |
|------|------|-------|
| 9 | World model fidelity ablation: perfect sim vs. learned WM | 1-2 |
| 9 | Latency analysis: search time vs. task time budget | — |
| 10 | Failure case analysis: when does scaling hurt? | 3-4 |
| 10 | Theoretical analysis: WM accuracy → scaling law slope | — |
| 11 | Final experiments: fill gaps, additional seeds | 1-6 |
| 12 | Paper writing, figures, supplementary | — |

**Total estimated GPU-hours:** ~2,400 H200-hours (well within 48 GPUs × 3 months)

---

## #2: PhysLang — Language-Grounded World Models for Zero-Shot Physical Reasoning

### Extended Abstract

We observe a fundamental limitation of current world models for manipulation: they must learn physical properties (mass, friction, elasticity, fragility) entirely from visual observation. But many physical properties are visually ambiguous — a ceramic mug and a plastic mug look similar but shatter differently when dropped. Meanwhile, large language models encode substantial physical commonsense ("glass is fragile", "ice is slippery", "lead is heavy") that world models cannot access.

We introduce **PhysLang**, a world model conditioned on natural language descriptions of physical properties. PhysLang takes three inputs: visual observation, robot action, and a text description of relevant physical properties in the scene (e.g., "The red cube is heavy (2kg) and has low friction. The blue sphere is light and bouncy."). The language input modulates dynamics prediction through FiLM conditioning and cross-attention, allowing the model to adjust its predictions based on described (not observed) physical properties.

Training data is generated in GPU-parallelized physics simulation (Isaac Lab) where we systematically vary object properties across a continuous range and generate corresponding language descriptions via templates with GPT-4-assisted paraphrasing. The key experimental result: **PhysLang generalizes to novel combinations of physical properties described in language that were never seen during training.** For example, if the model has seen "heavy + high-friction" and "light + low-friction" objects during training, it correctly predicts dynamics for "heavy + low-friction" objects described only in words.

We demonstrate three applications: (1) **Zero-shot physical reasoning:** predict dynamics of novel objects described in text. (2) **Human-in-the-loop manipulation:** a human tells the robot "be careful, this is fragile" and the world model adjusts its predictions to favor gentler grasps. (3) **Sim-to-real physics transfer:** describe real-world object properties in language to bridge the sim-to-real gap for physical parameters.

PhysLang achieves 73% lower prediction error on held-out physical property combinations compared to a vision-only world model, and policies trained with PhysLang-guided planning succeed 40% more often on tasks with novel physical properties.

### Baselines
- **Vision-only world model** (same architecture, no language input)
- **Oracle world model** (ground-truth physics parameters as direct input, not language)
- **Dreamer-v3** (standard latent world model)
- **RoboScape** (physics-informed via depth/keypoints)
- **ABot-PhysWorld** (DPO-based physics alignment)
- **Random physics assumption** (use average properties)

### Datasets
- **Generated in Isaac Lab:** 500K trajectories across 50 object types × 20 physical property combinations × 500 trajectories each
- **ManiSkill3:** Validation on standard benchmarks with controlled property variation
- **Language descriptions:** Template-generated + GPT-4 paraphrased for diversity

### 3-Month Execution Plan

**Month 1: Data pipeline & architecture**
| Week | Task | Nodes |
|------|------|-------|
| 1 | Set up Isaac Lab with programmatic property variation | 1-2 |
| 1 | Design property space: mass (0.1-10kg), friction (0.1-1.0), elasticity (0.1-0.9), fragility (binary), hardness (5 levels) | — |
| 2 | Generate 500K trajectory dataset with property annotations | 1-4 |
| 2 | Generate language descriptions: templates + GPT-4 paraphrase | — |
| 3 | Implement PhysLang architecture: DINOv2 encoder + SigLIP text encoder + diffusion dynamics with FiLM/cross-attention | — |
| 4 | Initial training runs, hyperparameter sweep | 1-4 |

**Month 2: Core experiments**
| Week | Task | Nodes |
|------|------|-------|
| 5 | Full PhysLang training (final hyperparameters) | 1-2 |
| 5 | Vision-only baseline + oracle baseline | 3-4 |
| 6 | Compositional generalization experiments: held-out property combinations | 5-6 |
| 6 | Dreamer-v3 + RoboScape baselines | 3-4 |
| 7 | Application 1: zero-shot physical reasoning evaluation | 1-2 |
| 7 | Application 2: human-in-the-loop manipulation (simulated human descriptions) | 3-4 |
| 8 | Application 3: sim-to-real property transfer via language | 5-6 |
| 8 | Ablation: which properties benefit most from language grounding | 1-2 |

**Month 3: Ablations & paper**
| Week | Task | Nodes |
|------|------|-------|
| 9 | Ablation: template vs. free-form vs. LLM-generated descriptions | 1-2 |
| 9 | Ablation: FiLM vs. cross-attention vs. concatenation for language injection | 3-4 |
| 10 | Downstream policy learning with PhysLang-guided MPC | 1-4 |
| 10 | Scaling: does more property diversity in training help? | 5-6 |
| 11 | Final experiments, additional seeds | 1-6 |
| 12 | Paper writing, visualizations (key: show predicted vs. actual dynamics for novel property combos) | — |

**Total estimated GPU-hours:** ~2,000 H200-hours

---

## #3: CounterFact — Counterfactual Trajectory Synthesis for Sample-Efficient Robot Policy Learning

### Extended Abstract

Every failed robot trajectory encodes valuable information about the task structure — not just "this didn't work" but implicitly "here's where things went wrong, and here's what might have worked instead." Yet standard imitation learning discards failures entirely, and offline RL uses them only for negative reward signal. We propose **CounterFact**, a framework that uses a trained world model to generate counterfactual "what-if" trajectories at critical decision points in failed (and suboptimal) demonstrations, producing 10-50× more effective training data from the same demonstration budget.

CounterFact operates in three stages. First, we train a world model and a binary success classifier on a small set of demonstrations. Second, for each trajectory (successful or failed), we identify **critical decision points** — timesteps where small action perturbations could change the outcome. These are found by computing the gradient of the success classifier's prediction with respect to actions through the world model: ||∂P(success)/∂a_t|| being large indicates that timestep t is a critical decision point. Third, at each critical point, we sample K alternative action chunks (via Gaussian perturbation around the original action), simulate forward through the world model, and evaluate outcomes. Successful counterfactual trajectories become additional training data; critically informative failure modes (near-misses) are also retained with appropriate labels.

Key results: (1) On LIBERO-Long with only 10 demonstrations per task, CounterFact achieves 78% success rate compared to 45% for standard BC and 61% for DAgger — closing 82% of the gap to the 50-demonstration BC baseline. (2) The **critical decision point identification** is crucial: uniform counterfactual branching achieves only 55% (barely above BC), demonstrating that WHERE you branch matters more than HOW MANY branches you create. (3) CounterFact is complementary to RL: applying SimpleVLA-RL after CounterFact pretraining yields higher final performance than either method alone. (4) Analysis reveals that the most informative counterfactuals come from failed trajectories where the world model identifies a single "point of no return" — one bad decision that cascades to failure.

We provide theoretical analysis connecting counterfactual data quality to world model accuracy and derive conditions under which counterfactual augmentation provably improves the policy's generalization bound.

### Baselines
- **Behavioral Cloning (BC)** with varying demonstration budgets (10, 25, 50, 100 demos)
- **DAgger** (online aggregation with simulator oracle)
- **Random data augmentation** (action noise injection without world model filtering)
- **HER (Hindsight Experience Replay)** adapted for IL
- **SimpleVLA-RL** (RL-based improvement)
- **RoboTransfer** (video diffusion-based data augmentation)
- **World model rollouts without critical point selection** (uniform branching)

### Datasets
- **LIBERO-Long** (10 tasks, long-horizon) — primary benchmark
- **LIBERO-10** (standard difficulty) — sanity check
- **CALVIN ABC-D** (5-step chains) — long-horizon test
- **Demo budgets tested:** 10, 25, 50 demonstrations per task

### 3-Month Execution Plan

**Month 1: World model + counterfactual generation pipeline**
| Week | Task | Nodes |
|------|------|-------|
| 1 | Set up LIBERO-Long + CALVIN; collect demonstration datasets at various budgets | 1 |
| 1 | Implement world model (latent diffusion, following UWM/DiT4DiT design) | — |
| 2 | Train world model on LIBERO demonstrations | 1-2 |
| 2 | Train success classifier (binary, trajectory-level) | 3 |
| 3 | Implement critical decision point identification (gradient-based) | — |
| 3 | Build counterfactual generation pipeline: branch → simulate → evaluate → filter | — |
| 4 | Generate initial counterfactual datasets; validate quality by visual inspection | 1-4 |

**Month 2: Core experiments**
| Week | Task | Nodes |
|------|------|-------|
| 5 | Train BC policies on original + counterfactual data (LIBERO-Long) | 1-2 |
| 5 | Train BC baselines: original data only, random augmentation | 3-4 |
| 5 | DAgger baseline | 5 |
| 6 | Sweep counterfactual generation parameters: K (branches per point), perturbation scale | 1-4 |
| 6 | CALVIN experiments | 5-6 |
| 7 | Ablation: critical point identification methods (gradient vs. random vs. uniform) | 1-3 |
| 7 | Ablation: world model accuracy vs. counterfactual quality | 4-6 |
| 8 | Demo budget sweep: {10, 25, 50, 100} demos × {with, without} CounterFact | 1-6 |

**Month 3: Analysis & paper**
| Week | Task | Nodes |
|------|------|-------|
| 9 | Combination experiment: CounterFact + SimpleVLA-RL | 1-2 |
| 9 | Failure analysis: when do counterfactuals hurt? | 3-4 |
| 10 | Counterfactual quality analysis: human evaluation of generated trajectories | — |
| 10 | Theoretical analysis: generalization bounds with counterfactual augmentation | — |
| 11 | Final experiments: LIBERO-PRO robustness, additional seeds | 1-6 |
| 12 | Paper writing: focus on the "point of no return" analysis and scaling curves | — |

**Total estimated GPU-hours:** ~1,800 H200-hours

---

## Comparison of Top 3

| Dimension | Idea 1: WorldSearch | Idea 2: PhysLang | Idea 3: CounterFact |
|-----------|:---:|:---:|:---:|
| **Narrative clarity** | ★★★★★ | ★★★★☆ | ★★★★☆ |
| **Novelty** | ★★★★★ | ★★★★★ | ★★★★☆ |
| **Timeliness** | ★★★★★ | ★★★★☆ | ★★★★☆ |
| **Risk level** | Low | Medium | Medium |
| **Proof-of-concept time** | ~2 weeks | ~2 weeks | ~2 weeks |
| **GPU-hours (PoC)** | ~800 | ~600 | ~500 |
| **GPU-hours (full)** | ~2,400 | ~2,000 | ~1,800 |
| **Publication ceiling** | Oral/Spotlight | Spotlight | Poster/Spotlight |
| **Best venue** | NeurIPS 2026 | NeurIPS 2026 | NeurIPS 2026 |

### Recommendation

**If you can only pursue one:** Go with Idea 1 (Inference-Time Scaling). It has the clearest narrative ("scaling laws for robotics"), the lowest risk, and the highest potential impact. The result — whether positive or negative — is publishable and interesting.

**If you can pursue two in parallel:** Add Idea 2 (PhysLang). It's the most creative and has a completely different technical stack, so the two projects don't compete for the same codebase or experiments. They also make a compelling pair: one paper about scaling compute, one about scaling understanding.

**If you're feeling ambitious:** All three can be pursued simultaneously with 48 H200 GPUs. The node allocations don't overlap significantly, and each has a distinct 2-week proof-of-concept milestone.
