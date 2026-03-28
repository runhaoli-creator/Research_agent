# PhysBridge: From Physics Foundation Models to Robot Manipulation

## Comprehensive Implementation Specification (Sim-Only)

**Novelty:** PhysiX (4.5B, NeurIPS 2025), Walrus (1.3B), GPhyT (1.8TB) exist as physics foundation models but NONE has been connected to robot manipulation. Verified against 25+ papers. Gap confirmed.

**Sim-only NeurIPS strategy:** This paper's core claim is a COMPARISON (physics-first vs. video-first). All comparisons can be done entirely in simulation across LIBERO, CALVIN, ManiSkill3, and RLBench. The result ("physics FM transfers better than video FM") is a fundamental insight that doesn't require real-robot validation.

---

### Core Hypothesis

Adapting a pre-trained **physics foundation model** (PhysiX/GPhyT, trained on Newtonian dynamics data) to robot manipulation yields better data efficiency and physical reasoning than adapting a **video foundation model** (Cosmos/DreamZero, trained on internet video) — because physics understanding transfers directly while video pattern-matching does not.

---

### Architecture

**PhysBridge has three modules:**

#### Module 1: Visual Perception Adapter (Image → Physics State Tokens)

Converts RGB observations into physics-interpretable state tokens that the physics FM can process.

```
Input: RGB 224×224
Backbone: DINOv2-ViT-B/14 (frozen) → 256 patch tokens (768-d each)
Object Detection Head:
  - Learned object queries (N_obj=10 max objects)
  - 6-layer Transformer decoder (cross-attend to DINOv2 patch tokens)
  - Per-object output: position (3D), velocity (3D), shape embedding (32-d),
    orientation (4D quaternion), bounding box (4D)
  - Total per object: 46-d → project to physics FM token dim
  - Trained with supervision from sim ground-truth object states
```

This module is the main engineering challenge. It must produce physics-compatible state tokens from visual input.

#### Module 2: Physics Foundation Model Backbone (frozen or adapter-tuned)

Use one of:
- **GPhyT** (arXiv 2509.13805): hybrid neural-differentiator + numerical-integrator. Input: state sequences. Output: next-state predictions. Download pre-trained weights.
- **PhysiX** (arXiv 2506.17774, 4.5B): autoregressive next-token physics prediction. If weights are available, use directly. If not, train a smaller version (500M) on physics sim data from The Well dataset + custom rigid-body data.
- **Fallback (if neither is available):** Train our own GNN-based physics dynamics model (following GNS architecture from Sanchez-Gonzalez et al., 2020) on 10M+ diverse non-robot physics episodes from Isaac Lab. Architecture: 8-layer GNN with Transformer-style message passing, 512-d node features, 128-d edge features, ~200M params.

The physics backbone takes object state tokens (from Module 1) + action tokens (from Module 3) and predicts next-state tokens.

**Adapter tuning:** Insert LoRA adapters (rank 16) at each attention layer of the physics FM. Only LoRA params train during manipulation fine-tuning. Backbone weights stay frozen.

#### Module 3: Robot Action Adapter

```
Input: robot action (7-d: Δpos + Δrot + gripper)
MLP: 7 → 256 → physics_FM_dim (SiLU activation)
Output: action token compatible with physics FM input format
```

The action token is injected as a special "intervention node" connected to objects near the end-effector in the physics FM's graph/sequence.

#### Module 4: Action Prediction Head

For downstream policy use, add a head that predicts robot actions from physics FM hidden states:
```
Input: physics FM hidden states at current timestep
MLP: FM_dim → 512 → 256 → 7 (action output)
Or: Diffusion head for multi-modal action prediction (following Chi et al.)
```

---

### Data Generation

#### Physics Pre-training Data (if training own physics FM)

Generate in **Isaac Lab** (GPU-parallelized, PhysX backend):
- **Rigid body dynamics:** 2M episodes of objects falling, bouncing, sliding, colliding, stacking, toppling. 50+ object types, mass [0.01-50kg], friction [0.01-2.0], restitution [0-0.99]. No robot.
- **Multi-body interactions:** 500K episodes of 2-5 objects interacting (chain reactions, dominos, bowling-style collisions).
- **Contact dynamics:** 500K episodes focused on contact events (grasping geometry, insertion, surface contact forces).
- **Format:** Object states (position, velocity, orientation, angular velocity) at 100Hz, 200 timesteps per episode.
- **Total:** ~3M episodes, ~600M state transitions.

#### Manipulation Fine-tuning Data

Standard benchmarks with oracle policies:
- **ManiSkill3:** Push, PickAndPlace, Stack, PegInsert — 5K trajectories per task, physics varied
- **LIBERO-90:** Pre-training set, use provided 50 demos per task
- **LIBERO-10:** Evaluation set
- **CALVIN:** 24h play data + 20K annotated trajectories
- **RLBench-18:** 100 demos per task variation (from Mini Diffuser setup)

---

### Baselines (7 total)

1. **Cosmos Policy** (NVIDIA, 2601.16163): Video foundation model → manipulation policy. Use available weights or reproduce: fine-tune Cosmos-Predict2.5-2B on manipulation data with action head.

2. **DreamZero-style WAM:** Fine-tune a video diffusion model (Cosmos-2B or Stable Video Diffusion) jointly on video+action prediction. Represents the "video-first" paradigm.

3. **Dreamer-v3:** Standard latent world model. No foundation model pre-training. Represents training from scratch.

4. **TD-MPC2:** Implicit world model with CEM planning. No pre-training.

5. **SimDist** (2603.15759): Simulation distillation — pre-train world model in robot-specific sim, transfer to target tasks. Represents "robot-sim-first" (not general physics).

6. **PhysBridge-NoPretraining:** Same architecture as PhysBridge but physics backbone trained from scratch (no physics FM pre-training). Isolates the contribution of physics pre-training.

7. **PhysBridge-VideoInit:** Same architecture but initialize physics backbone from video FM weights instead of physics FM weights. Tests whether video features help physics prediction.

---

### Experiments (Sim-Only)

#### Experiment 1: Data Efficiency (Key Result)

Fix target benchmark (LIBERO-10). Vary fine-tuning data: {5%, 10%, 25%, 50%, 100%} of available demos.

Plot success rate vs. data fraction for: PhysBridge, Cosmos Policy, DreamZero-style, Dreamer-v3, SimDist.

**Expected killer result:** PhysBridge at 10% data matches DreamZero at 100%.

#### Experiment 2: Multi-Benchmark Evaluation

Full evaluation on ALL benchmarks with 100% data:

| Benchmark | Tasks | Metric |
|-----------|-------|--------|
| LIBERO-10 | 10 tasks | Success rate (100 eps × 3 seeds) |
| LIBERO-Long | 10 long-horizon tasks | Success rate |
| CALVIN ABC-D | 34 tasks, 5-step chains | Avg chain length |
| ManiSkill3 (physics-varying) | 4 tasks × varied physics | Success rate |
| RLBench-18 | 18 tasks | Success rate |

#### Experiment 3: Physics Generalization

Train on objects with standard physics. Test on objects with extreme physics (mass 10× heavier, friction 5× lower). Compare PhysBridge vs. video-based baselines.

**Expected:** PhysBridge degrades gracefully (physics FM understands Newton's laws). Video-based models collapse (never seen these dynamics in video).

#### Experiment 4: World Model Prediction Quality

Compare long-horizon prediction accuracy:
- Object position L2 error at t+{1,5,10,20,50}
- Physics violation rate (penetration, gravity violation)
- Contact prediction accuracy

#### Experiment 5: Cross-Simulator Transfer

Train PhysBridge on ManiSkill3. Evaluate on LIBERO (different simulator, different renderer). Compare transfer quality vs. video-based baselines.

---

### Ablation Studies (8)

1. **Physics FM backbone:** GPhyT vs. PhysiX vs. our GNN vs. random init
2. **Adapter type:** LoRA vs. full fine-tune vs. frozen backbone
3. **Perception adapter quality:** GT object states (oracle) vs. learned perception vs. DINOv2 features
4. **Physics pre-training data diversity:** rigid-only vs. rigid+multi-body vs. all
5. **Physics pre-training data scale:** {100K, 500K, 1M, 3M} episodes
6. **Robot action injection:** intervention node vs. concatenation vs. cross-attention
7. **Number of trainable params:** {1M, 5M, 20M, 50M} via LoRA rank
8. **Object detection accuracy:** How much perception error can PhysBridge tolerate?

---

### Analysis

1. **What transfers:** Probe physics FM features before/after manipulation fine-tuning. Which layers change? Which stay frozen? → Understanding what physics knowledge transfers.
2. **Physics vs. video representations:** CKA similarity between physics FM and video FM internal representations. Are they learning the same or different things?
3. **Failure analysis:** Where does PhysBridge fail? (likely: deformable objects, fluids — things not in rigid-body physics pre-training)
4. **Scaling analysis:** If we scale the physics FM (200M → 500M → 1B), does manipulation performance improve predictably?

---

### Node Allocation (48 H200 GPUs)

| Node | Weeks 1-2 | Weeks 3-5 | Weeks 6-10 |
|------|-----------|-----------|------------|
| 1-2 | Physics pre-training data gen (Isaac Lab) | Physics FM training (if needed) OR adapter tuning | Data efficiency experiments |
| 3 | ManiSkill3/LIBERO/CALVIN data prep | Cosmos Policy baseline | Multi-benchmark eval |
| 4 | — | DreamZero-style + SimDist baselines | Physics generalization |
| 5 | — | Dreamer-v3 + TD-MPC2 baselines | Cross-simulator transfer |
| 6 | — | Perception adapter training | Ablations + analysis |

**Timeline:** 12 weeks. Proof-of-concept (data efficiency curve on LIBERO) achievable in 3 weeks.
