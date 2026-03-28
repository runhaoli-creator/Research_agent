# PhysContext: One Interaction Is All You Need — In-Context Physical Property Learning for Zero-Shot Manipulation

## Comprehensive Implementation Specification for Coding Agent

---

### Project Overview

Build **PhysContext**, a Transformer-based manipulation world model that infers the physical properties of novel objects from watching a SINGLE diagnostic interaction — with zero fine-tuning, zero gradient updates, pure in-context learning. The core hypothesis: by pre-training a world model on diverse manipulation data with massive physical property variation and providing "physics context" (a short observation of a diagnostic action applied to the target object) within the model's context window, the model can immediately predict accurate dynamics for novel objects with unseen physical properties. This is the manipulation analog of how GPhyT (arXiv 2509.13805, 1.8TB physics data) achieves zero-shot generalization via in-context learning for PDE systems — but nobody has applied in-context physics learning to robot manipulation world models. AdaptiGraph (RSS 2024) does few-shot physics adaptation via gradient-based optimization; V-JEPA 2 uses the same dynamics model regardless of object physics; DreamZero requires 55+ demonstrations for adaptation. PhysContext requires ONE observation and ZERO gradient updates.

---

### Model Architecture

PhysContext is a causal Transformer that processes three types of token sequences within a single context window: (1) **physics context tokens** — observations from diagnostic interactions with the target object, (2) **current observation tokens** — the robot's current visual observation, and (3) **action tokens** — the robot's planned action. The model predicts the next observation tokens (future state).

#### Component 1: Visual Tokenizer (frozen)

Use a pre-trained visual encoder to convert RGB images into token sequences.

**Primary backbone: DINOv2-ViT-B/14** (86M params, `facebookresearch/dinov2`).
- Input: RGB 224×224
- Extract patch tokens from last layer: 16×16 = 256 tokens, each 768-dim
- Apply a learned linear projection: 768 → D (model dimension, D=512)
- Output: 256 visual tokens per frame, each D-dimensional

**Alternative backbones to compare (ablation):**
- DINOv2-ViT-L/14 (300M, 1024-dim → project to 512)
- SigLIP-ViT-B/16 (86M, `google/siglip-base-patch16-224`)
- CLIP-ViT-L/14 (304M, `openai/clip-vit-large-patch14`)

The visual encoder is **frozen** throughout — only the projection layers train.

#### Component 2: Action Tokenizer

A 2-layer MLP (512→256→D, SiLU activation) that encodes the robot action vector (7-dim: 3D end-effector delta position + 3D delta orientation + 1D gripper) into a single D-dimensional action token.

#### Component 3: Proprioception Tokenizer

A 2-layer MLP (input_dim→256→D, SiLU) encoding robot proprioceptive state (joint positions, velocities, gripper state — typically 14-20 dim depending on the robot) into a single D-dimensional proprioception token.

#### Component 4: Context-Conditioned Dynamics Transformer (the core model)

This is the main contribution — a causal Transformer that processes the physics context AND current state to predict the next state.

**Architecture:**
```
Model: GPT-2-style causal Transformer
- Layers: 12
- Hidden dim D: 512
- Attention heads: 8
- FFN dim: 2048
- Activation: SiLU (in FFN)
- Normalization: Pre-LayerNorm (RMSNorm)
- Position encoding: Rotary Position Embeddings (RoPE)
- Context length: 4096 tokens max
- Total parameters: ~85M (dynamics Transformer only)
```

**Input sequence structure (for a single prediction step):**

```
[CONTEXT_START] [ctx_frame_1_tokens...] [ctx_action_1] [ctx_frame_2_tokens...] ... [ctx_frame_K_tokens...] [CONTEXT_END] [obs_t_tokens...] [proprio_t] [action_t] → predict [obs_{t+1}_tokens...]
```

Where:
- `ctx_frame_i_tokens`: 256 visual tokens from frame i of the diagnostic interaction (the physics context)
- `ctx_action_i`: 1 action token from diagnostic interaction
- `obs_t_tokens`: 256 visual tokens from current observation
- `proprio_t`: 1 proprioception token
- `action_t`: 1 action token (the action to predict the outcome of)
- Total context tokens for K diagnostic frames: K × (256 + 1) + 256 + 1 + 1 = 257K + 258

For K=5 diagnostic frames: 257×5 + 258 = 1543 tokens (well within 4096 limit)
For K=10: 257×10 + 258 = 2828 tokens (still within limit)

**Prediction head:** A linear layer (D → 768) followed by an L2 loss against the visual encoder's output tokens for the ground-truth next frame. This predicts in the VISUAL FEATURE SPACE (not pixel space), following V-JEPA's approach. An auxiliary pixel decoder (4-layer ConvTranspose: 512→256→128→64→3, BN+SiLU) is used for visualization only.

**Special tokens:**
- `[CONTEXT_START]`, `[CONTEXT_END]`: learned embeddings marking diagnostic context boundaries
- `[OBS]`: marks start of current observation
- `[ACT]`: marks action token
- `[PRED]`: marks prediction target

**Segment embeddings:** Add a learned segment embedding to distinguish context tokens (segment 0) from current observation tokens (segment 1). This helps the model understand which observations are "reference physics" vs. "current state."

#### Component 5: Pixel Decoder (auxiliary, for visualization)

```
ConvTranspose2d decoder:
  Linear(D, 512*7*7) → Reshape(512,7,7) →
  ConvT(512,256,4,2,1) → BN → SiLU →
  ConvT(256,128,4,2,1) → BN → SiLU →
  ConvT(128,64,4,2,1) → BN → SiLU →
  ConvT(64,3,4,2,1) → Sigmoid
  Output: 3×112×112 (upscale to 224×224 via bilinear)
```

Used only for qualitative visualization and auxiliary reconstruction loss (weight 0.05).

---

### Data Generation Pipeline

#### Simulator: ManiSkill3

Use **ManiSkill3** (`pip install mani_skill`) as the primary simulator. GPU-parallelized via SAPIEN, supports programmatic physics variation, runs 1024+ parallel envs per GPU.

Secondary validation: **Isaac Lab** for photorealistic rendering ablation.

#### Physical Property Space

For each object in the scene, sample from:
- **Mass:** log-uniform [0.05, 10.0] kg → 8 discrete bins: {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}
- **Static friction:** uniform [0.05, 1.5] → 8 bins: {0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.5}
- **Restitution:** uniform [0.0, 0.95] → 6 bins: {0.0, 0.2, 0.4, 0.6, 0.8, 0.95}
- **Dynamic friction:** 0.8 × static (fixed ratio)
- **Damping:** uniform [0.0, 2.0]

Total unique property configurations per object: 8 × 8 × 6 = 384 combinations.

#### Object Set

Use ManiSkill3 YCB objects + additional assets — minimum **30 distinct geometries** spanning: cups, bowls, cans, boxes, bottles, tools, fruits. For each geometry, 3 texture variants. Total: 90 visual variants × 384 property configs = 34,560 object-property configurations.

#### Diagnostic Interaction Library

Define **5 standardized diagnostic actions** that reveal physical properties:
1. **Push-X:** Push object along +X with constant velocity (0.05 m/s) for 1 second → reveals mass + friction
2. **Push-Y:** Push along +Y → reveals mass + friction (orthogonal)
3. **Lift-and-Drop:** Grasp, lift 15cm, release → reveals mass + restitution
4. **Flick:** Quick lateral tap at object edge → reveals mass + friction + moment of inertia
5. **Press-Down:** Slowly press downward → reveals mass + surface friction

Each diagnostic action produces a sequence of K_diag = 10 frames at 5 Hz (2 seconds total). Record: RGB frames (224×224), robot proprioception, action commands, object ground-truth state (position, velocity, orientation).

#### Task Data Collection

For each object-property configuration, collect demonstrations on **6 manipulation tasks:**
1. **Push-to-Target:** Push object to a goal location on the table
2. **Pick-and-Place:** Pick up object, place at target
3. **Stack:** Stack object on top of another
4. **Slide:** Slide object across surface to target
5. **Toss-and-Catch:** Toss object to a target bin (tests mass + restitution)
6. **Careful-Place:** Place a fragile/bouncy object precisely (tests restitution)

Use ManiSkill3's scripted oracle policies for data collection. 20 trajectories per task per object-property config. Each trajectory: 50-150 timesteps at 10 Hz.

**Total dataset:**
- Diagnostic interactions: 34,560 configs × 5 diagnostic actions × 10 frames = 1.7M diagnostic frames
- Task trajectories: 34,560 configs × 6 tasks × 20 trajectories × ~100 timesteps = ~415M frames
- Storage: ~2TB in HDF5 (chunked, compressed)

#### Train/Val/Test Splits (Critical for Evaluation)

**Split by physical property combinations (compositional generalization):**
- **Train:** 70% of property combinations (269 out of 384)
- **Val:** 15% (58 combinations)
- **Test:** 15% (57 combinations — NOVEL property combinations never seen during training)

**Split by object geometry (object generalization):**
- Hold out 5 object geometries entirely for zero-shot object evaluation

**"Invisible Physics" test set:** 200 pairs of visually identical objects (same mesh, texture, viewpoint) with different physical properties. Used to test whether the model truly uses the diagnostic context.

---

### Training Procedure

#### Training Data Format

Each training example consists of:
1. **Physics context:** K_ctx diagnostic frames (randomly sampled from the 5 diagnostic actions) for the target object with its current physical properties. K_ctx is randomly sampled from {1, 3, 5, 10} during training (curriculum: start with K=10, gradually reduce to K=1).
2. **Trajectory segment:** A consecutive chunk of L=16 timesteps from a task trajectory with the SAME object and properties.
3. **Prediction target:** Next-frame visual tokens at each of the L timesteps.

#### Training Loss

**Primary loss — Latent prediction (following V-JEPA):**
```
L_latent = (1/L) Σ_{t=1}^{L} MSE(predicted_tokens_{t+1}, sg(encode(o_{t+1})))
```
where `sg` = stop-gradient, `encode` = frozen DINOv2 encoder.

**Auxiliary loss — Pixel reconstruction:**
```
L_pixel = (1/L) Σ_{t=1}^{L} MSE(decode(predicted_tokens_{t+1}), o_{t+1})
```

**Total loss:**
```
L = L_latent + 0.05 * L_pixel
```

#### Training Hyperparameters

```yaml
optimizer: AdamW
lr: 3e-4
weight_decay: 0.01
schedule: cosine_annealing
warmup_steps: 2000
total_steps: 500K
batch_size: 128 (across 8 GPUs, 16 per GPU)
gradient_clipping: 1.0
precision: bf16
context_curriculum:
  steps_0_100K: K_ctx=10 (easy — lots of context)
  steps_100K_300K: K_ctx=5
  steps_300K_500K: K_ctx=uniform(1,10) (random)
```

#### Multi-Step Training

For long-horizon prediction, train with teacher-forcing for steps 1-8 and autoregressive (use own predictions) for steps 9-16. This trains the model to handle compounding errors.

---

### Baselines (8 total)

Implement ALL baselines with matched compute budget:

1. **No-Context World Model:** Same Transformer architecture but WITHOUT the diagnostic context tokens. The model must predict dynamics from visual observation alone — it cannot know the object's physics. This is the most important ablation — the gap between this and PhysContext measures the VALUE of in-context physics learning.

2. **DreamZero-style (video diffusion baseline):** Use Cosmos-Predict2.5-2B (or a smaller video diffusion model) fine-tuned on our manipulation data. No physics context — just video prediction. Represents the "video-first" paradigm.

3. **Dreamer-v3:** RSSM-based world model (official implementation from `danijar/dreamerv3`). No physics context. Represents the standard latent world model.

4. **AdaptiGraph-style (gradient-based physics adaptation):** At test time, observe K diagnostic interactions, then run N gradient steps to adapt a physics-conditioned dynamics model. This tests whether in-context learning outperforms optimization-based adaptation.
   - Implementation: Same backbone + a physics parameter head. At test time, optimize the physics parameters via gradient descent on the diagnostic interaction prediction loss (N=10, 50, 100 steps).

5. **Oracle-Numerical:** Same PhysContext architecture but instead of diagnostic context tokens, directly input the ground-truth numerical physics parameters (mass, friction, restitution) as conditioning. This is the UPPER BOUND — if you know the exact physics, how well can you predict?

6. **Average-Context:** Same architecture but always provide the SAME diagnostic context (from an object with AVERAGE physical properties) regardless of the actual object. Tests whether the model truly adapts to the context or ignores it.

7. **Random-Context:** Same architecture but provide diagnostic context from a RANDOM object (different from the actual target). Tests whether the model uses physics-relevant information from the context or just uses the context as noise.

8. **TD-MPC2:** Implicit world model with CEM planning (reference implementation from `nicklashansen/tdmpc2`). No physics context. Represents the implicit model-based RL approach.

---

### Evaluation Experiments

#### Experiment 1: World Model Prediction Quality

**Metrics (computed on held-out test set with NOVEL property combinations):**
- Latent prediction MSE at horizons t+1, t+5, t+10, t+20
- Reconstruction SSIM and LPIPS at same horizons
- FVD over 16-frame predicted clips
- Object position/velocity L2 error (extracted via segmentation + centroid tracking)

**Key comparison:** PhysContext (K=1 context) vs. No-Context vs. Oracle-Numerical. Plot prediction error vs. horizon for each method. PhysContext should: (a) dramatically outperform No-Context, (b) approach Oracle-Numerical performance.

**Physics-specific evaluation:** For each physical property (mass, friction, restitution), test on objects where ONLY that property differs from training. Measure: does PhysContext correctly predict slower motion for heavy objects? Less sliding for high-friction objects? More bouncing for high-restitution objects?

#### Experiment 2: In-Context Scaling — How Many Diagnostic Interactions?

Vary K_ctx = {0, 1, 2, 3, 5, 10} diagnostic frames. Plot prediction error vs. K_ctx. Expected: sharp improvement from K=0 to K=1, diminishing returns beyond K=3-5.

Compare to AdaptiGraph-style baseline: vary optimization steps N = {0, 10, 50, 100, 500}. Plot adaptation quality vs. compute time. Expected: PhysContext at K=1 (instant) matches AdaptiGraph at N=100 (slow).

#### Experiment 3: Which Diagnostic Action Is Most Informative?

For K=1, test each diagnostic action individually: push-X, push-Y, lift-and-drop, flick, press-down. Measure prediction accuracy for each. Expected: lift-and-drop is most informative for mass; push is most informative for friction.

#### Experiment 4: Downstream Policy Learning

Train a **Diffusion Policy** (Cheng Chi et al., 2023) using PhysContext as the world model backbone for MPC-style planning:
1. At test time, perform ONE diagnostic interaction with the novel object
2. Feed diagnostic context into PhysContext
3. Sample N=100 candidate action chunks
4. Simulate each forward through PhysContext
5. Select the action chunk with the lowest predicted distance to the goal

**Benchmarks:**
- **LIBERO-10:** Standard 10-task benchmark, 50 demos/task. Report success rate (100 episodes × 3 seeds).
- **LIBERO-Long:** Long-horizon tasks.
- **Physics-Varying Benchmark (custom):** Same LIBERO tasks but with test objects having NOVEL physical properties (mass/friction outside training range). This is where PhysContext's advantage should be dramatic.
- **CALVIN ABC-D:** 5-step instruction chains.

**Baselines for policy:** All 8 world model baselines + direct BC (diffusion policy without world model planning).

#### Experiment 5: Zero-Shot Physics Inference

From the diagnostic context, extract PhysContext's implicit physics understanding:
1. After processing the diagnostic context, extract the model's hidden state at the [CONTEXT_END] token
2. Train a linear probe (frozen PhysContext) to predict mass, friction, restitution from this hidden state
3. Compare: PhysContext hidden state vs. DINOv2 features vs. CLIP features vs. random

Expected: PhysContext's [CONTEXT_END] hidden state encodes physics properties with R² > 0.8. Other encoders: R² < 0.2.

#### Experiment 6: Invisible Physics Test

Use the 200 visually identical object pairs with different physics:
1. Give PhysContext diagnostic context from each object in the pair
2. Predict dynamics for a standard task (push-to-target)
3. Measure: are the predictions DIFFERENT for the two objects?

**Metric:** KL divergence between predicted trajectory distributions for the two objects. PhysContext should show high KL (distinguishes physics). No-Context baseline should show KL ≈ 0 (cannot distinguish).

---

### Ablation Studies (8 ablations)

1. **Context length K_ctx:** {0, 1, 2, 3, 5, 10, 20}. Where is the sweet spot?

2. **Which diagnostic action as context:** Each of the 5 individually vs. combination of 2 vs. all 5. Which reveals the most physics?

3. **Visual backbone:** DINOv2-B vs. DINOv2-L vs. SigLIP-B vs. CLIP-L. Does the visual encoder matter for in-context physics?

4. **Model scale:** Transformer with {4, 8, 12, 16, 24} layers. Does in-context physics require a minimum model capacity?

5. **Context curriculum:** Always K=10 vs. always K=1 vs. curriculum (10→5→random) during training. Does gradually reducing context help?

6. **Training data diversity:** {10%, 25%, 50%, 100%} of physical property combinations during training. How much diversity is needed for generalization?

7. **Segment embeddings:** With vs. without segment embeddings distinguishing context from current observation. Are they necessary?

8. **Context from same vs. different task:** Diagnostic context always from push task, but evaluate on pick-and-place. Does the diagnostic action need to match the downstream task?

---

### Additional Analysis

1. **Attention map visualization:** Visualize attention patterns from the prediction tokens to the context tokens. Show that the model attends to physically relevant frames in the context (e.g., the moment of contact in a push, the bounce height in a drop). Use attention rollout.

2. **Physics embedding space:** Extract [CONTEXT_END] hidden states for all test objects. Visualize with t-SNE/UMAP colored by mass, friction, restitution. Show that the in-context physics encoding forms smooth, structured manifolds corresponding to physical properties.

3. **Failure case analysis:** Identify cases where in-context physics fails. Categorize: (a) diagnostic action doesn't reveal the relevant property, (b) property is outside training range (extrapolation), (c) complex multi-body interactions not captured by single-object diagnostics.

4. **Real-world qualitative evaluation:** Record ONE real-world push of a novel object. Feed as diagnostic context to PhysContext (trained in sim). Generate predicted dynamics. Qualitatively assess: does the model produce physically plausible predictions? Compare heavy vs. light real objects.

5. **Latency analysis:** Report wall-clock time for in-context adaptation: PhysContext (1 forward pass, ~10ms) vs. AdaptiGraph (100 gradient steps, ~5s). Show PhysContext is 500× faster.

---

### Hardware Allocation (48 H200 GPUs, 6 nodes × 8 GPUs)

| Node | Weeks 1-2 | Weeks 3-5 | Weeks 6-10 |
|------|-----------|-----------|------------|
| 1-2 | Data generation (ManiSkill3, 1024 envs/GPU, diagnostic + task trajectories) | PhysContext training (16 GPU DDP, 500K steps) | Downstream policy experiments (LIBERO, CALVIN) |
| 3 | Data processing + HDF5 storage | No-Context + Average-Context + Random-Context baselines | Physics-varying benchmark eval |
| 4 | — | AdaptiGraph-style + Oracle-Numerical baselines | Invisible physics test + zero-shot inference |
| 5 | — | Dreamer-v3 + TD-MPC2 baselines | Ablation studies (8 ablations, parallel) |
| 6 | — | Video diffusion baseline (Cosmos/SVD fine-tuning) | Analysis (attention maps, embeddings, failures) |

---

### Repository Structure

```
PhysContext/
├── configs/                      # Hydra configs
│   ├── data/                     # Data generation configs
│   ├── model/                    # Architecture configs (layers, dims, context length)
│   ├── train/                    # Training configs (lr, batch, curriculum)
│   ├── eval/                     # Evaluation configs
│   ├── baseline/                 # Baseline configs
│   └── ablation/                 # Ablation configs
├── physcontext/
│   ├── data/
│   │   ├── sim_generator.py      # ManiSkill3 data generation with physics variation
│   │   ├── diagnostic_actions.py # 5 standardized diagnostic action definitions
│   │   ├── dataset.py            # PyTorch Dataset: (context, trajectory, target) triples
│   │   ├── splits.py             # Train/val/test split by property combinations
│   │   └── invisible_physics.py  # Invisible physics test set generation
│   ├── models/
│   │   ├── visual_tokenizer.py   # DINOv2/SigLIP/CLIP backbone wrappers (frozen)
│   │   ├── action_tokenizer.py   # MLP action encoder
│   │   ├── proprio_tokenizer.py  # MLP proprioception encoder
│   │   ├── dynamics_transformer.py  # Core causal Transformer with context conditioning
│   │   ├── pixel_decoder.py      # Auxiliary CNN decoder for visualization
│   │   └── physcontext.py        # Full PhysContext model combining all components
│   ├── baselines/
│   │   ├── no_context.py         # PhysContext without diagnostic context
│   │   ├── dreamerv3.py          # Dreamer-v3 wrapper
│   │   ├── tdmpc2.py             # TD-MPC2 wrapper
│   │   ├── adaptigraph.py        # Gradient-based physics adaptation baseline
│   │   ├── oracle_numerical.py   # Ground-truth physics parameters as input
│   │   ├── avg_context.py        # Average-property context
│   │   ├── random_context.py     # Random-object context
│   │   └── video_diffusion.py    # Video diffusion world model baseline
│   ├── training/
│   │   ├── trainer.py            # Training loop with context curriculum
│   │   ├── losses.py             # Latent prediction + pixel reconstruction losses
│   │   └── curriculum.py         # Context length curriculum schedule
│   ├── evaluation/
│   │   ├── prediction.py         # World model prediction quality metrics
│   │   ├── context_scaling.py    # K_ctx scaling experiments
│   │   ├── diagnostic_info.py    # Which diagnostic action is most informative
│   │   ├── downstream_policy.py  # Diffusion policy + MPC planning
│   │   ├── physics_inference.py  # Linear probing of [CONTEXT_END] hidden state
│   │   ├── invisible_test.py     # Invisible physics test
│   │   └── metrics.py            # SSIM, LPIPS, FVD, position/velocity L2
│   ├── analysis/
│   │   ├── attention_maps.py     # Attention visualization
│   │   ├── physics_embeddings.py # t-SNE/UMAP of physics encoding
│   │   ├── failure_analysis.py   # Categorize failure cases
│   │   └── figures.py            # Generate all paper figures
│   └── utils/
│       ├── distributed.py        # DDP utilities
│       └── logging.py            # W&B logging
├── scripts/
│   ├── generate_data.sh          # Launch parallel data generation
│   ├── train.sh                  # Launch PhysContext training
│   ├── train_baselines.sh        # Launch all baselines
│   ├── evaluate.sh               # Full evaluation suite
│   ├── run_ablations.sh          # All ablation studies
│   └── make_figures.py           # Generate paper figures
├── requirements.txt
└── README.md
```

---

### Key Implementation Details

- **PyTorch 2.x** with `torch.compile`. **DistributedDataParallel** across 8-16 GPUs.
- **Hydra** for config management. **Weights & Biases** for tracking.
- ManiSkill3: `env = gym.make_vec("PushCube-v1", num_envs=1024)`. Customize physics via SAPIEN API: `actor.set_mass(m)`, `material.set_static_friction(μ)`, `material.set_restitution(e)`.
- DINOv2 weights: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`.
- For Diffusion Policy baseline: use `diffusion_policy` codebase from Cheng Chi.
- For Dreamer-v3: use official `danijar/dreamerv3` implementation.
- For TD-MPC2: use `nicklashansen/tdmpc2` implementation.
- RoPE implementation: use `rotary_embedding_torch` package or implement from LLaMA.
- Set random seeds everywhere. Run 3 seeds for main experiments, 1 for ablations.
- Save checkpoints every 25K steps. Best model by validation latent prediction MSE.
- FVD computation: use `pytorch-fid` adapted for video (compute features with I3D).

---

### Timeline (12 weeks)

| Weeks | Task | Deliverable |
|-------|------|-------------|
| 1-2 | Data generation pipeline + diagnostic action implementation | 2TB dataset in HDF5 |
| 3-4 | PhysContext model implementation + initial training | Trained PhysContext (500K steps) |
| 5-6 | All 8 baselines training | Trained baselines for comparison |
| 7-8 | Core experiments (prediction quality, context scaling, invisible physics) | Main results tables + figures |
| 9-10 | Downstream policy + ablation studies (all 8) | Policy results + ablation tables |
| 11-12 | Analysis + paper writing + figure generation | Camera-ready draft |
