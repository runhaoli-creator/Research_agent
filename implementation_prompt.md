# DynaCLIP: Physics-Grounded Visual Representations via Contrastive Dynamics Alignment

## Comprehensive Implementation Specification

---

### Core Thesis

Build **DynaCLIP**, a visual representation learning method for robotics where the contrastive similarity metric is **physical dynamics similarity**, not visual, semantic, or temporal similarity. Two observations are "positive pairs" if the physical systems depicted respond similarly to the same actions (produce similar trajectories, forces, contacts), regardless of visual appearance. Two observations are "hard negatives" if they look visually similar but have different physical dynamics (e.g., a ceramic mug vs. a plastic mug — identical appearance, different fragility and mass). DynaCLIP is pre-trained once on simulation data with programmatically varied physical properties, producing a frozen visual backbone that can replace DINOv2/CLIP/SigLIP for any downstream robotics task: world model prediction, policy learning, and zero-shot physics inference. The key finding: dynamics-aligned representations dramatically outperform semantics-aligned representations on physics-varying manipulation tasks, and uniquely enable the "Invisible Physics Test" — correctly handling visually identical objects with different physical properties that all existing encoders fail on.

---

### Phase 1: Simulation Data Generation Pipeline

**Simulator:** Use **ManiSkill3** (`pip install mani-skill`) as the primary data generation platform. ManiSkill3 is GPU-parallelized via SAPIEN, supports programmatic variation of physical properties (mass, friction, restitution, damping via the SAPIEN API: `actor.set_mass()`, `material.set_static_friction()`, `material.set_dynamic_friction()`, `material.set_restitution()`), and can run 1024+ parallel environments per GPU for efficient data collection.

**Object Set:** Use ManiSkill3's built-in YCB object models plus additional ShapeNet/Objaverse assets — at least **50 distinct object geometries** spanning categories: cups, bowls, cans, boxes, bottles, tools, fruits, kitchen items. For each object mesh, also create 3-5 **texture variants** (different colors/materials applied to the same geometry) — this ensures visual diversity independent of shape.

**Physical Property Space:** For each object geometry, sample **N_config = 100 physical property configurations** from:
- Mass: log-uniform in [0.05, 10.0] kg — bins: {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}
- Static friction coefficient: uniform in [0.05, 1.5] — bins: {0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.5}
- Restitution (bounciness): uniform in [0.0, 0.95] — bins: {0.0, 0.2, 0.4, 0.6, 0.8, 0.95}
- Dynamic friction: 0.8 × static friction (fixed ratio)
- Damping: uniform in [0.0, 2.0] for articulated objects

**Standardized Test Actions (for dynamics similarity computation):** For each object-property configuration, execute **K=5 standardized robot actions** using a Franka Panda arm and record the resulting object trajectories (position, orientation, linear velocity, angular velocity) for T=50 timesteps at 20Hz:
1. **Push-X:** End-effector pushes object along +X axis with fixed force for 0.5s
2. **Push-Y:** End-effector pushes object along +Y axis with fixed force for 0.5s
3. **Lift-and-Drop:** Grasp object, lift 20cm, release, record fall + bounce trajectory
4. **Flick:** Quick lateral contact at object edge, record spin/slide
5. **Press-Down:** Press object downward into surface, record compression/resistance

Each test action produces a trajectory τ_k ∈ R^(T × 13) (3D position + 4D quaternion + 3D linear vel + 3D angular vel). The **dynamics fingerprint** of an object-property configuration is D = [τ_1, τ_2, ..., τ_5] — the concatenation of all K test action trajectories.

**Dynamics Similarity Metric:** For two object-property configurations (i, j), compute:

```
sim_dyn(i, j) = -Σ_{k=1}^{K} DTW(τ_k^i, τ_k^j) / K
```

where DTW is Dynamic Time Warping distance between the trajectory pairs for each test action, averaged over all K actions. Normalize to [0, 1] via min-max over all pairs. Objects with similar mass, friction, and shape produce similar trajectories under the same actions → high dynamics similarity.

**Visual Observation Collection:** For each object-property configuration, render **M=20 RGB images** (224×224) from varied camera viewpoints (azimuth: 0-360°, elevation: 15-45°, distance: 0.4-0.8m) with the object placed on a table in 5 different poses. This gives 20 images per configuration, all paired with the same dynamics fingerprint.

**Total Dataset Size:**
- 50 objects × 5 texture variants × 100 property configs = 25,000 unique (appearance, physics) combinations
- 25,000 × 20 images = 500,000 images
- 25,000 × 5 test actions × 50 timesteps = dynamics fingerprints for all configurations
- Pre-compute all pairwise dynamics similarities (25,000² / 2 ≈ 312M pairs — sample 10M pairs for training)

**"Invisible Physics" Test Set (Critical):** Create a special held-out set of **500 pairs** where:
- Same object mesh, same texture, same viewpoint → **visually identical images**
- Different physical properties (e.g., mass 0.1kg vs. 5.0kg, or friction 0.1 vs. 1.5) → **different dynamics**
- DINOv2/CLIP/SigLIP embeddings for these pairs will be nearly identical (cosine similarity > 0.99)
- DynaCLIP embeddings must be distinguishable (cosine similarity < 0.7)

Store all data in HDF5: `images.h5` (images), `dynamics.h5` (trajectories + fingerprints), `pairs.h5` (pre-computed similarity scores).

---

### Phase 2: DynaCLIP Contrastive Pre-Training

**Architecture:**

```
Visual Encoder: DINOv2-ViT-B/14 (86M params)
  - Input: RGB 224×224
  - Extract: CLS token + mean-pooled patch tokens → concat → 1536-dim
  - Projection head: Linear(1536, 768) → LayerNorm → GELU → Linear(768, 512) → L2-normalize
  - Output: z ∈ R^512 (unit norm embedding)
```

The DINOv2 backbone weights are **unfrozen** during DynaCLIP pre-training — this is critical because we need to reshape the feature space from semantic similarity to dynamics similarity. The projection head is discarded after pre-training; the backbone is used for downstream tasks.

**Contrastive Loss — Soft InfoNCE with Dynamics Similarity:**

Standard InfoNCE uses binary positive/negative pairs. DynaCLIP uses **continuous dynamics similarity scores** as soft labels:

```python
def dynaclip_loss(z_i, z_j, sim_dyn_ij, temperature=0.07):
    """
    z_i, z_j: embeddings of images i, j (batch of B pairs)
    sim_dyn_ij: dynamics similarity scores in [0, 1] for all pairs
    """
    # Compute cosine similarity matrix between all embeddings
    logits = torch.mm(z_i, z_j.T) / temperature  # (B, B)

    # Soft labels: dynamics similarity scores (rows normalized to sum to 1)
    labels = sim_dyn_ij / sim_dyn_ij.sum(dim=1, keepdim=True)  # (B, B)

    # Cross-entropy between soft labels and logits
    loss = -torch.sum(labels * F.log_softmax(logits, dim=1)) / B
    return loss
```

**Hard Negative Mining:** Within each batch, prioritize pairs that are:
- **Hard negatives:** High visual similarity (cosine sim of DINOv2 embeddings > 0.9) but low dynamics similarity (sim_dyn < 0.3). These are the pairs that teach the model to distinguish physics.
- **Hard positives:** Low visual similarity (cosine sim < 0.5) but high dynamics similarity (sim_dyn > 0.7). These teach the model that different-looking objects can behave the same.

Implementation: pre-compute DINOv2 embeddings for all images. During training, sample batches with 30% hard negatives, 30% hard positives, 40% random pairs.

**Training Hyperparameters:**
- Optimizer: AdamW, lr=1e-5 (backbone) / lr=1e-3 (projection head), weight decay=0.05
- Schedule: cosine annealing, 500 warmup steps
- Batch size: 1024 pairs (across 8 GPUs, 128 per GPU)
- Temperature: 0.07 (learnable, initialized to 0.07)
- Training steps: 100K (expect convergence around 50-80K)
- Mixed precision: BF16
- Data augmentation on images: RandomResizedCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip, ColorJitter(0.1, 0.1, 0.1), no augmentations that could affect object identity

**Ablation: Other Backbone Options** (train all for comparison):
- DINOv2-ViT-L/14 (300M params) — larger backbone
- SigLIP-ViT-B/16 (86M params) — different pre-training
- ViT-B/14 from scratch (no pre-training) — does dynamics alignment alone suffice?
- DINOv2-ViT-B/14 frozen (only projection head trained) — how much fine-tuning is needed?

---

### Phase 3: Evaluation Suite

#### Experiment 1: Physics Property Probing (Representation Quality)

Train a **linear probe** (single linear layer) on frozen representations from each encoder to predict:
- Mass (regression, MSE loss)
- Friction coefficient (regression, MSE loss)
- Restitution (regression, MSE loss)
- Material category (classification, 10 classes: metal, wood, plastic, ceramic, glass, rubber, foam, stone, fabric, paper)

**Encoders to compare (8 total):**
1. **DynaCLIP** (ours)
2. DINOv2-ViT-B/14 (frozen, pre-trained)
3. DINOv2-ViT-L/14 (frozen, pre-trained)
4. SigLIP-ViT-B/16 (frozen, pre-trained)
5. CLIP-ViT-L/14 (frozen, pre-trained)
6. R3M (pre-trained on Ego4D, `r3m` package)
7. VIP (pre-trained, `vip` package)
8. MCR (pre-trained on DROID, from `mcr` repo if available, else re-implement)

**Metrics:** R² for regression, accuracy for classification. Report with 95% confidence intervals over 5 random seeds.

**Expected result:** DynaCLIP achieves R² > 0.8 for mass and friction. All other encoders achieve R² < 0.3 because their representations don't encode physics.

#### Experiment 2: The Invisible Physics Test

**Setup:** Use the 500 invisible physics pairs (visually identical, physically different).

**Protocol:**
1. Compute pairwise cosine similarity for each pair under each encoder
2. Plot distribution of similarities. DINOv2/CLIP should peak near 1.0 (can't distinguish). DynaCLIP should show bimodal distribution or spread below 0.8.
3. Binary classification task: given two images from an invisible pair, predict which has higher mass (or friction). Accuracy = 50% means the encoder can't distinguish.

**Downstream policy test:** Train a diffusion policy (see Experiment 4) on a "grasp-and-lift" task with objects of varying mass. Test on invisible physics pairs. DynaCLIP-backed policy should adjust grasp force appropriately for heavy vs. light objects. DINOv2-backed policy should fail on one of each pair (too gentle for heavy, or too aggressive for light).

**Expected result:** DynaCLIP binary classification accuracy > 85%. All other encoders ~50% (chance).

#### Experiment 3: Downstream World Model

Train a **Dreamer-v3-style world model** (RSSM with categorical latents) using each encoder as the frozen visual backbone. Architecture:
- Visual encoder: frozen encoder → learned linear projection to 512-dim
- RSSM: 512-dim stochastic state (32 categoricals × 32 classes), 512-dim deterministic state (GRU)
- Decoder: CNN decoder for reconstruction (auxiliary loss)
- Reward predictor: MLP (for RL experiments)

**Training data:** Demonstrations from ManiSkill3 tasks with physical property variation:
- Push (varying mass/friction): 10K trajectories
- Pick-and-place (varying mass): 10K trajectories
- Stack (varying mass/friction/restitution): 10K trajectories

**Metrics:**
- Latent prediction MSE at horizons t+1, t+5, t+10, t+20
- Reconstruction SSIM / LPIPS at same horizons
- FVD over 16-frame clips
- Physics violation rate (object penetration, gravity violation, momentum violation — measured against ground-truth sim state)

**Expected result:** DynaCLIP backbone reduces prediction error by 30-50% at horizon t+20 compared to DINOv2 backbone, because physics-relevant features are pre-encoded.

#### Experiment 4: Downstream Policy Learning

Train a **Diffusion Policy** (following Chi et al., 2023 — `diffusion_policy` codebase) using each encoder as the frozen visual backbone. Architecture:
- Visual encoder: frozen encoder → learned linear projection to 256-dim
- Observation: last 2 frames → 512-dim concatenated visual embedding + proprioception
- Action: 16-step action chunks, 7-DoF (end-effector delta pos + delta rot + gripper)
- Noise schedule: DDPM, 100 denoising steps for training, 10 for inference (DDIM)

**Benchmarks:**
1. **LIBERO-90/10:** Standard benchmark. 50 demos per task. Report average success rate over 10 tasks × 100 eval episodes × 3 seeds.
2. **LIBERO-Long:** Long-horizon tasks. 50 demos per task.
3. **Physics-Varying Benchmark (custom):** Pick-and-place and push tasks where test objects have DIFFERENT physical properties than training objects. Train on objects with mass 0.5-2.0 kg, test on mass 0.1 kg and 5.0 kg. Train on friction 0.5-1.0, test on friction 0.1 and 1.5.
4. **CALVIN ABC-D:** 5-step instruction chains.

**Expected result:** On standard LIBERO, DynaCLIP matches DINOv2 (±2%). On physics-varying benchmark, DynaCLIP outperforms DINOv2 by 25-40% because physics-aware features enable generalization to novel physical properties.

#### Experiment 5: Zero-Shot Physics Inference

Given a novel object image (not seen during pre-training), predict its physical properties using k-NN in DynaCLIP embedding space:
1. Build a library of 1000 reference objects with known physical properties
2. For a query image, find the 5 nearest neighbors in DynaCLIP embedding space
3. Predict properties as weighted average of neighbors' properties (weights = similarity)

**Compare:** DynaCLIP k-NN vs. DINOv2 k-NN vs. CLIP k-NN vs. random baseline.

**Expected result:** DynaCLIP achieves meaningful zero-shot predictions (R² > 0.5 for mass). Others near random (R² < 0.1).

---

### Phase 4: Ablation Studies (8 ablations)

1. **Dynamics similarity metric:** Compare DTW vs. endpoint L2 vs. full trajectory MSE vs. velocity-only DTW. Which captures dynamics similarity best?

2. **Number of test actions K:** K=1 (push only) vs. K=3 vs. K=5 vs. K=10. How many actions are needed to fingerprint an object's dynamics?

3. **Contrastive loss formulation:** Soft InfoNCE (ours) vs. standard InfoNCE (binary pos/neg) vs. triplet loss vs. BYOL-style (non-contrastive). Does soft labeling matter?

4. **Hard negative ratio:** 0%, 15%, 30%, 50% hard negatives in batch. Does targeted hard mining help?

5. **Backbone pre-training:** DINOv2-initialized vs. ImageNet-initialized vs. random-initialized backbone. How important is the starting point?

6. **Pre-training data scale:** 10K, 50K, 100K, 250K, 500K images. Log-linear scaling curve? Diminishing returns threshold?

7. **Physical property diversity:** Train on {mass only}, {friction only}, {mass + friction}, {all properties}. Which properties contribute most to representation quality?

8. **Fine-tuning depth:** Freeze all backbone, unfreeze last 2 layers, unfreeze last 4 layers, unfreeze all. Where is the sweet spot?

---

### Phase 5: Analysis

1. **t-SNE / UMAP Visualization:** Visualize DynaCLIP vs. DINOv2 embedding spaces colored by (a) object category, (b) mass, (c) friction. DINOv2 should cluster by category (appearance). DynaCLIP should show gradients along mass and friction axes.

2. **Dynamics Sensitivity Analysis:** Compute Jacobian of embedding with respect to physical property changes: ∂z/∂mass, ∂z/∂friction. Show that DynaCLIP embeddings are sensitive to physical property changes. Compare to DINOv2 (should be near-zero sensitivity).

3. **Cross-Domain Transfer:** Pre-train DynaCLIP on ManiSkill3 objects. Evaluate on Isaac Lab objects (different renderer, different meshes). Does the dynamics alignment transfer across simulation domains?

4. **Real-World Qualitative Evaluation:** Take real-world images from DROID or BridgeData V2. Compute DynaCLIP embeddings. Do metal objects cluster together (similar dynamics)? Do heavy objects cluster together? Qualitative analysis — no ground-truth physics available for real images.

5. **Computational Cost:** Report pre-training time, per-image inference time, memory footprint. DynaCLIP should add zero overhead at inference (same architecture as DINOv2, just different weights).

---

### Hardware Allocation (48 H200 GPUs, 6 nodes × 8 GPUs)

| Node | Phase 1 (Weeks 1-2) | Phase 2 (Weeks 3-4) | Phase 3-5 (Weeks 5-10) |
|------|---------------------|---------------------|------------------------|
| 1-2 | Data generation (ManiSkill3, 1024 envs/GPU) | DynaCLIP pre-training (16 GPU DDP) | Downstream world model experiments |
| 3 | Dynamics fingerprint computation | Backbone ablations (init, depth) | Downstream policy (LIBERO, CALVIN) |
| 4 | Pair mining + similarity computation | Loss/metric ablations | Physics-varying benchmark evaluation |
| 5 | Invisible physics test set creation | Data scale ablation | Probing + zero-shot inference |
| 6 | — | — | Analysis (t-SNE, sensitivity, transfer) |

---

### Repository Structure

```
DynaCLIP/
├── configs/                     # Hydra configs
│   ├── data/                    # Data generation configs
│   ├── pretrain/                # Contrastive pre-training configs
│   ├── eval/                    # Evaluation configs
│   └── ablation/                # Ablation study configs
├── dynaclip/
│   ├── data/
│   │   ├── sim_generator.py     # ManiSkill3 data generation with physics variation
│   │   ├── test_actions.py      # Standardized test action definitions
│   │   ├── dynamics_fingerprint.py  # Trajectory recording + DTW similarity
│   │   ├── pair_mining.py       # Hard positive/negative pair mining
│   │   ├── invisible_physics.py # Invisible physics test set generation
│   │   └── dataset.py           # PyTorch Dataset for contrastive pairs
│   ├── models/
│   │   ├── encoder.py           # DINOv2/SigLIP/CLIP backbone wrappers
│   │   ├── projection.py        # Projection head (MLP)
│   │   └── dynaclip.py          # Full DynaCLIP model (encoder + projection + loss)
│   ├── training/
│   │   ├── contrastive.py       # Soft InfoNCE loss implementation
│   │   ├── trainer.py           # Training loop with hard negative mining
│   │   └── scheduler.py         # Cosine annealing + warmup
│   ├── evaluation/
│   │   ├── probing.py           # Linear probing for physics properties
│   │   ├── invisible_test.py    # Invisible physics test evaluation
│   │   ├── world_model.py       # Dreamer-style world model with swappable backbone
│   │   ├── diffusion_policy.py  # Diffusion policy with swappable backbone
│   │   ├── zero_shot.py         # k-NN zero-shot physics inference
│   │   └── metrics.py           # All evaluation metrics
│   ├── analysis/
│   │   ├── tsne.py              # t-SNE / UMAP visualization
│   │   ├── sensitivity.py       # Jacobian sensitivity analysis
│   │   ├── transfer.py          # Cross-domain transfer evaluation
│   │   └── figures.py           # Paper figure generation
│   ├── baselines/
│   │   ├── r3m.py               # R3M baseline wrapper
│   │   ├── vip.py               # VIP baseline wrapper
│   │   ├── mcr.py               # MCR baseline wrapper
│   │   └── frozen_dinov2.py     # Frozen DINOv2 baseline
│   └── utils/
│       ├── distributed.py       # DDP / FSDP utilities
│       └── logging.py           # W&B logging
├── scripts/
│   ├── generate_data.sh         # Launch data generation
│   ├── pretrain.sh              # Launch DynaCLIP pre-training
│   ├── eval_all.sh              # Launch full evaluation suite
│   ├── run_ablations.sh         # Launch all ablations
│   └── make_figures.py          # Generate all paper figures
├── requirements.txt
└── README.md
```

---

### Key Implementation Details

- **PyTorch 2.x** with `torch.compile` for model compilation. Use **DistributedDataParallel** across 8-16 GPUs.
- **Hydra** for config management. Every experiment reproducible from config.
- **Weights & Biases** for tracking. Log: loss curves, probing accuracy, downstream success rates, t-SNE plots.
- ManiSkill3 data generation: use `gym.make_vec("PushCube-v1", num_envs=1024)` for parallel env creation. Customize physics via `env.unwrapped.scene.actors[obj].set_mass(m)` and material API.
- For dynamics fingerprints, use `tslearn.metrics.dtw` for DTW computation. Pre-compute all pairwise similarities in parallel (embarrassingly parallel).
- DINOv2 weights: load from `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`.
- R3M: `pip install r3m`, load with `r3m.load_r3m("resnet50")`.
- Diffusion Policy: clone `diffusion_policy` repo from Cheng Chi, swap visual encoder.
- Set random seeds everywhere. Run 3 seeds for all main experiments, 1 seed for ablations.
- Save checkpoints every 10K steps. Best model selected by validation probing accuracy.

---

### Timeline (12 weeks)

| Weeks | Task | Deliverable |
|-------|------|-------------|
| 1-2 | Data generation pipeline + dynamics fingerprinting | 500K images + similarity matrix |
| 3-4 | DynaCLIP pre-training + backbone ablations | Trained DynaCLIP encoder |
| 5-6 | Probing + invisible physics test + zero-shot | Core representation quality results |
| 7-8 | Downstream: world model + diffusion policy | Task performance comparison |
| 9-10 | Ablation studies (all 8) + analysis | Ablation tables + visualizations |
| 11-12 | Paper writing + figure generation | Camera-ready draft |
