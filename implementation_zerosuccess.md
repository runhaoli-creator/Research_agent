# Zero-Success Learning: Robot Manipulation from Failure Data Alone

## Comprehensive Implementation Specification (Sim-Only)

**Novelty:** Only Grollman & Billard (ICRA 2011) tried failure-only learning, in low-dimensional settings only. No modern work combines failure-only data + world model imagination for manipulation. Verified against 20+ papers. Gap confirmed.

**Sim-only strategy:** This idea is INHERENTLY sim-only — the entire point is that you don't need expert demonstrations (which are expensive in the real world). All experiments use ManiSkill3, LIBERO, CALVIN with oracle success checkers. The comparison "failure-only vs. N successful demos" is naturally simulation-based.

**Key NeurIPS argument without real robots:** The contribution is a FUNDAMENTAL FINDING about information content — "failures contain sufficient dynamics information for policy learning." This is a scientific claim about learning theory, not an engineering claim about real-world deployment. Simulation provides the controlled environment to test this claim rigorously.

---

### Core Hypothesis

Failed manipulation trajectories contain ALL the dynamics information needed for policy learning — objects still move, contacts still happen, friction still applies. The ONLY difference between a failed trajectory and a successful one is often a few critical actions near task completion. By training a world model on failures (which learns dynamics) and imagining corrective completions from near-success states, we can synthesize successful trajectories and train policies that achieve 70-85% success from ZERO successful demonstrations.

---

### Architecture

**Three-stage pipeline:**

#### Stage 1: World Model Training (on failure data only)

**Architecture:** Latent dynamics model (RSSM-based, following Dreamer-v3):
```
Visual encoder: DINOv2-ViT-B/14 (frozen) → Linear(768, 512)
RSSM:
  - Deterministic state: GRU(512)
  - Stochastic state: 32 categoricals × 32 classes = 1024-d
  - Transition: MLP(det + stoch + action → next_det)
  - Posterior: MLP(det + obs_embed → stoch_params)
  - Prior: MLP(det → stoch_params)
Decoder: ConvTranspose (512 → 3×64×64) for reconstruction
Reward predictor: MLP(det + stoch → reward) — predict TASK PROGRESS, not binary success
```

**Training data:** ONLY failed trajectories. No successful demonstrations. No reward signal (or: use task progress as a proxy signal — "how close did you get?").

**Key design:** The reward predictor learns "goal proximity" — how close the current state is to the goal, even in failures. This is trained using:
- Visual similarity to goal image (LPIPS distance)
- Object-to-target distance (from sim ground truth during training)

#### Stage 2: Near-Success State Identification

Define near-success states as timesteps where the goal-proximity function g(s) exceeds a threshold τ.

```python
def find_near_success_states(trajectory, world_model, goal_image, threshold=0.7):
    near_success = []
    for t, state in enumerate(trajectory):
        proximity = world_model.predict_goal_proximity(state, goal_image)
        if proximity > threshold:
            near_success.append((t, state, proximity))
    return near_success
```

**Threshold selection:** Sweep τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}. Report the fraction of failures that contain at least one near-success state at each threshold. Expected: at τ=0.7, ~30-50% of failures contain a near-success state.

#### Stage 3: Imagination-Based Completion

From each near-success state, search for action sequences that reach the goal:

```python
def imagine_completion(world_model, near_success_state, goal, N_candidates=256, horizon=10):
    best_actions = None
    best_score = -inf
    for _ in range(N_candidates):
        # Sample random action sequence
        action_seq = sample_action_sequence(horizon)
        # Simulate forward through world model
        predicted_states = world_model.rollout(near_success_state, action_seq)
        # Score by goal proximity at final state
        score = goal_proximity(predicted_states[-1], goal)
        if score > best_score:
            best_score = score
            best_actions = action_seq
    return best_actions, best_score
```

**Search methods (compare):**
- Random sampling (N=256, 1024, 4096)
- CEM (Cross-Entropy Method): 5 iterations, top 10%
- MPPI (Model Predictive Path Integral): temperature 1.0

**Trajectory stitching:** `synthetic_success = failure[0:t_near] + imagined_completion[t_near:t_near+H]`

Filter: keep only stitched trajectories where final goal proximity > 0.9 (high-confidence completions).

#### Stage 4: Policy Training on Synthetic Successes

Train a standard **Diffusion Policy** (Chi et al., 2023) on the synthetic successful trajectories:
```
Visual encoder: DINOv2-ViT-B/14 (frozen) → 256-d
Observation: last 2 frames concat + proprioception
Action: 16-step chunks, 7-DoF, DDPM 100 steps
```

---

### Data Collection (Failure Only)

**Failure generation strategies in ManiSkill3:**

1. **Random policy:** Uniform random actions. Produces diverse but uninformative failures. 5K trajectories per task.
2. **Scripted exploration:** Heuristic policies that approach the object but don't complete the task (e.g., reach toward object but grasp at wrong height). 5K trajectories per task.
3. **Noisy oracle:** Oracle policy with heavy action noise (σ=0.3). Produces near-success failures frequently. 5K trajectories per task.
4. **Mixed:** Equal mix of all three. 15K total per task.

**Tasks:**
- Push-to-Target (easiest — many near-success states in random play)
- Pick-and-Place (medium — requires correct grasp + placement)
- Stack (hard — requires sequential precision)
- Peg-Insert (hardest — requires sub-mm precision)
- Drawer-Open (medium — articulated object)
- Button-Press (easy-medium)

**Also evaluate on:**
- LIBERO-10 (10 tasks, collect failures with random policies in LIBERO)
- CALVIN (use random play data as failures)

---

### Baselines (8)

1. **BC with N successful demos:** N = {1, 5, 10, 25, 50, 100} demos per task. THE critical comparison. Show that Zero-Success at 0 demos approaches BC at N demos.
2. **BC with 0 demos (no training):** Random policy. Lower bound.
3. **Dreamer-v3 from scratch (online RL):** Standard model-based RL from failure + exploration. No demos.
4. **HER (Hindsight Experience Replay):** Relabel failed trajectories with achieved goals. Online RL.
5. **Plan2Explore → Fine-tune:** Exploration via world model disagreement, then RL.
6. **Random augmentation:** Perturb failure trajectories randomly, filter by success. No world model imagination.
7. **DAgger (sim oracle):** Interactive imitation with oracle corrections. Requires oracle but no human.
8. **Zero-Success without near-success filtering:** Imagine completions from ALL states, not just near-success. Tests value of near-success identification.

---

### Experiments (Sim-Only, Multi-Benchmark)

#### Experiment 1: Zero-Success vs. N-Success Learning Curve (Key Result)

Plot success rate vs. number of successful demonstrations for each method:

```
X-axis: number of SUCCESSFUL demonstrations (0, 1, 5, 10, 25, 50, 100)
Y-axis: task success rate (%)
Lines: Zero-Success (ours), BC, Dreamer-v3, HER, DAgger
```

**Expected killer result:** "Zero-Success (0 demos) achieves 75% success rate, matching BC with 25 successful demos."

Evaluate on: ManiSkill3 (6 tasks) + LIBERO-10 (10 tasks) + CALVIN (34 tasks).

#### Experiment 2: Failure Data Composition

Vary failure source: {random only, scripted only, noisy-oracle only, mixed}. Which type of failure is most informative? Expected: noisy-oracle produces most near-success states → most useful.

#### Experiment 3: Near-Success Analysis

Report statistics:
- What fraction of failures contain near-success states at various thresholds?
- How far (in timesteps) are near-success states from actual success?
- Visualize near-success states — what do they look like?

#### Experiment 4: World Model Quality from Failure Data

Compare world model prediction accuracy: (a) trained on failures only, (b) trained on successes only, (c) trained on mixed. Evaluate on same test set.

**Key finding:** World model quality from failures is comparable to successes for DYNAMICS prediction — because dynamics are the same in both. The difference is only in state coverage near success.

#### Experiment 5: Imagination Quality

Visualize imagined completions. Measure: (a) physical plausibility (penetration, gravity), (b) goal achievement rate, (c) diversity of imagined solutions.

#### Experiment 6: Scaling Failure Data

Vary amount of failure data: {1K, 5K, 15K, 50K} trajectories per task. Plot success rate. Is there a saturation point?

---

### Ablation Studies (8)

1. **Near-success threshold τ:** {0.5, 0.6, 0.7, 0.8, 0.9} — tradeoff between quantity and quality
2. **Imagination horizon H:** {3, 5, 10, 20} steps — short vs. long completions
3. **Search method:** Random (N=256/1024/4096) vs. CEM vs. MPPI
4. **Number of imagined completions per near-success state:** {1, 5, 10, 50}
5. **World model architecture:** RSSM (Dreamer) vs. Transformer vs. diffusion-based
6. **Goal proximity function:** LPIPS vs. SSIM vs. learned (MLP on latent states) vs. object distance
7. **Failure data quality:** random vs. scripted vs. noisy-oracle vs. mixed
8. **Iterative refinement:** 1 round vs. 2 rounds vs. 3 rounds (retrain WM on synthetic data, generate more)

---

### Analysis

1. **What do failures teach?** Visualize world model's learned dynamics from failure data. Show it understands pushing, sliding, grasping mechanics.
2. **Critical decision points:** Where in failures do near-success states occur? Early (approach phase) or late (manipulation phase)?
3. **Failure taxonomy:** Categorize failures and their utility: (a) too far from success (uninformative), (b) near-success but wrong action (most informative), (c) random flailing (moderate info for dynamics).
4. **Information theory analysis:** Estimate mutual information between failure data and task success. Theoretical justification for why failures are informative.
5. **Scaling law:** Does success rate scale log-linearly with failure data? Characterize the "failure scaling law."

---

### Node Allocation

| Node | Weeks 1-2 | Weeks 3-5 | Weeks 6-10 |
|------|-----------|-----------|------------|
| 1-2 | Failure data generation (random, scripted, noisy-oracle) across all tasks | World model training on failure data | Zero-Success pipeline end-to-end |
| 3 | — | Near-success identification + imagination pipeline | N-Success learning curve experiment |
| 4 | — | BC baselines (N=1,5,10,25,50,100) | LIBERO + CALVIN evaluation |
| 5 | — | Dreamer-v3 + HER + Plan2Explore baselines | Ablation studies |
| 6 | — | DAgger baseline | Analysis + visualization |

**Timeline:** 12 weeks. Core result (Zero-Success vs. BC learning curve on ManiSkill3) achievable in 5 weeks.

**Risk mitigation:** If Zero-Success at 0 demos produces < 50% success, relax to "Few-Success" (1-5 demos + failure data). The contribution becomes: "failures multiply the value of rare demonstrations by 10×."
