# DynaCLIP: Physics-Grounded Visual Representations via Contrastive Dynamics Alignment

## Comprehensive Implementation Specification

**Novelty Status:** Verified novel against 25+ representation learning papers across 4 iteration cycles. NO prior work uses physical dynamics similarity as the contrastive metric for visual representation learning. Closest papers: MCR (ICLR 2025, temporal co-occurrence alignment), CLASS (CoRL 2025, action-sequence DTW), PSE (ICLR 2021, policy similarity), DynaMo (dynamics prediction pretext), R3M/VIP/AFRO/CLOUD (temporal/value/diffusion-based). All use fundamentally different similarity signals. Gap confirmed.

---

### Core Thesis

Build **DynaCLIP**, a visual representation learning method for robotics where the contrastive similarity metric is **physical dynamics similarity** — two observations are positive pairs if the physical systems respond similarly to the same actions (produce similar trajectories, forces, contacts), regardless of visual appearance. Hard negatives are visually similar but dynamically different pairs (e.g., ceramic vs. plastic mug — identical appearance, different fragility and mass). DynaCLIP is pre-trained once on GPU-parallelized simulation data with programmatically varied physical properties, producing a frozen visual backbone that replaces DINOv2/CLIP/SigLIP for any downstream robotics task: world model prediction, policy learning, and zero-shot physics inference. The key empirical result is the "Invisible Physics Test" — DynaCLIP correctly distinguishes and handles visually identical objects with different physical properties, which all existing visual encoders fundamentally cannot do. This addresses the critical finding from Physics-IQ (DeepMind, arXiv 2501.09038) that video/vision models score only 24.1% on physical understanding — DynaCLIP is the first visual backbone designed to encode physics.

---

### Phase 1: Simulation Data Generation

**Simulator:** ManiSkill3 (`pip install mani_skill`) — GPU-parallelized via SAPIEN, 1024+ parallel envs per GPU, programmatic physics property variation via `actor.set_mass()`, `material.set_static_friction()`, `material.set_dynamic_friction()`, `material.set_restitution()`.

**Object set:** 50 distinct YCB + ShapeNet/Objaverse geometries (cups, bowls, cans, boxes, bottles, tools, fruits, kitchen items). 5 texture variants per geometry (different colors/materials on same mesh). Total: 250 visual object variants.

**Physical property space per object:**
- Mass: log-uniform [0.05, 10.0] kg — 8 bins: {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}
- Static friction: uniform [0.05, 1.5] — 8 bins: {0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.5}
- Restitution: uniform [0.0, 0.95] — 6 bins: {0.0, 0.2, 0.4, 0.6, 0.8, 0.95}
- Dynamic friction: 0.8 × static (fixed)
- Total: 100 random property configs per geometry = 50 × 100 = 5,000 object-property configs

**Standardized test actions (dynamics fingerprinting):** For each config, execute K=5 diagnostic actions with a Franka Panda arm and record object trajectories (position, orientation, linear/angular velocity) for T=50 timesteps at 20Hz:
1. Push-X: end-effector pushes along +X, 0.05 m/s, 1s
2. Push-Y: along +Y
3. Lift-and-Drop: grasp, lift 20cm, release
4. Flick: quick lateral tap at object edge
5. Press-Down: slowly press downward

**Dynamics fingerprint:** D_i = [τ_1, τ_2, ..., τ_5] — concatenation of all K test trajectories, each τ_k ∈ R^(50×13).

**Dynamics similarity:** `sim_dyn(i,j) = -mean_k DTW(τ_k^i, τ_k^j)`, normalized to [0,1]. Use `tslearn.metrics.dtw`.

**Visual data:** For each config, render M=20 RGB images (224×224) from varied viewpoints (azimuth 0-360°, elevation 15-45°, distance 0.4-0.8m), 5 object poses. Total: 5,000 × 20 = 100,000 images.

**Pair mining:** Pre-compute DINOv2 embeddings for all images. Create:
- Hard negatives (30% of batch): cosine_sim(DINOv2) > 0.9 AND sim_dyn < 0.3
- Hard positives (30%): cosine_sim(DINOv2) < 0.5 AND sim_dyn > 0.7
- Random pairs (40%)
- Pre-compute 5M pairs, store in pairs.h5

**Invisible Physics test set:** 500 pairs where same mesh + same texture + same viewpoint (visually identical) but different physics. DINOv2 cosine sim > 0.99 for these pairs.

---

### Phase 2: DynaCLIP Model Architecture and Training

**Visual encoder:**
```
Backbone: DINOv2-ViT-B/14 (86M params)
  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
  Extract: CLS token (768-d) + mean-pooled patch tokens (768-d) → concat → 1536-d
  Projection head: Linear(1536, 768) → LayerNorm → GELU → Linear(768, 512) → L2-normalize
  Output: z ∈ R^512 (unit-norm embedding)
  Backbone: UNFROZEN during pre-training (reshape feature space)
  Projection head: discarded after pre-training
```

**Contrastive loss — Soft InfoNCE with dynamics similarity:**
```python
def dynaclip_loss(z_i, z_j, sim_dyn_matrix, temperature=0.07):
    # z_i, z_j: [B, 512] embeddings
    # sim_dyn_matrix: [B, B] pairwise dynamics similarity scores in [0,1]
    logits = z_i @ z_j.T / temperature  # [B, B]
    labels = F.softmax(sim_dyn_matrix / 0.1, dim=1)  # soft targets
    loss = -torch.sum(labels * F.log_softmax(logits, dim=1)) / B
    return loss
```

**Training hyperparameters:**
```yaml
optimizer: AdamW
lr_backbone: 1e-5
lr_projection: 1e-3
weight_decay: 0.05
schedule: cosine_annealing, 500 warmup steps
batch_size: 1024 pairs (128/GPU × 8 GPUs)
temperature: 0.07 (learnable, init 0.07)
steps: 100K
precision: bf16
augmentation: RandomResizedCrop(224, 0.8-1.0), RandomHorizontalFlip, ColorJitter(0.1)
hard_negative_ratio: 0.3
hard_positive_ratio: 0.3
```

**Ablation backbones (train all):**
- DINOv2-ViT-L/14 (300M)
- SigLIP-ViT-B/16 (86M)
- ViT-B/14 from scratch (no pre-training)
- DINOv2-ViT-B/14 frozen (only projection trains)

---

### Phase 3: Evaluation Suite

#### Experiment 1: Physics Property Linear Probing

Train a single linear layer (frozen encoder → property prediction) on held-out data:
- Mass regression (MSE, report R²)
- Friction regression (MSE, report R²)
- Restitution regression (MSE, report R²)
- Material classification (10 classes, report accuracy)

**8 encoders compared:**
1. **DynaCLIP** (ours)
2. DINOv2-ViT-B/14 (pre-trained, frozen)
3. DINOv2-ViT-L/14 (pre-trained, frozen)
4. SigLIP-ViT-B/16 (pre-trained, frozen)
5. CLIP-ViT-L/14 (pre-trained, frozen)
6. R3M (`pip install r3m`, resnet50)
7. VIP (`pip install vip`)
8. MCR (if weights available, else re-implement on DROID)

Report 95% CI over 5 seeds. Expected: DynaCLIP R² > 0.8 for mass/friction; others R² < 0.3.

#### Experiment 2: Invisible Physics Test

500 visually identical pairs with different physics:
1. Cosine similarity distribution: DINOv2 peaks at ~1.0; DynaCLIP spreads below 0.8
2. Binary classification (which object is heavier?): DynaCLIP > 85%; others ~50%
3. **Downstream policy test:** Train diffusion policy on grasp-and-lift with varying mass. Test on invisible pairs. DynaCLIP-backed policy adjusts grasp force; DINOv2-backed fails on one per pair.

#### Experiment 3: Downstream World Model

Train **Dreamer-v3-style RSSM** (512-dim stochastic, 512-dim deterministic GRU, categorical latents 32×32) with each encoder as frozen visual backbone.

Training data: ManiSkill3 push/pick-and-place/stack with physics variation, 10K trajectories each.

**Metrics:** Latent MSE at t+{1,5,10,20}, reconstruction SSIM/LPIPS, FVD (16-frame), object position L2 error, physics violation rate.

Expected: DynaCLIP backbone reduces t+20 prediction error by 30-50% vs. DINOv2.

#### Experiment 4: Downstream Diffusion Policy

Train **Diffusion Policy** (Chi et al., 2023) with each encoder as frozen backbone:
- Observation: last 2 frames → 512-d concat + proprioception
- Action: 16-step chunks, 7-DoF, DDPM 100 steps train / DDIM 10 steps inference

**Benchmarks:**
1. LIBERO-90/10: 50 demos/task, avg success rate over 10 tasks × 100 episodes × 3 seeds
2. LIBERO-Long: long-horizon tasks
3. **Physics-Varying Benchmark (custom):** train mass 0.5-2.0kg, test 0.1kg and 5.0kg; train friction 0.5-1.0, test 0.1 and 1.5
4. CALVIN ABC-D: 5-step instruction chains

Expected: Standard LIBERO: DynaCLIP ≈ DINOv2 (±2%). Physics-varying: DynaCLIP +25-40%.

#### Experiment 5: Zero-Shot Physics Inference

Build library of 1000 objects with known properties. For query image, find 5 nearest neighbors in DynaCLIP space, predict properties as weighted average. Compare k-NN: DynaCLIP vs. DINOv2 vs. CLIP vs. random. Expected: DynaCLIP R² > 0.5; others R² < 0.1.

---

### Ablation Studies (8 total)

1. **Dynamics similarity metric:** DTW vs. endpoint L2 vs. full trajectory MSE vs. velocity-only DTW
2. **Number of test actions K:** K={1,3,5,10} for fingerprinting
3. **Contrastive loss:** Soft InfoNCE (ours) vs. binary InfoNCE vs. triplet loss vs. BYOL (non-contrastive)
4. **Hard negative ratio:** {0%, 15%, 30%, 50%}
5. **Backbone initialization:** DINOv2 vs. ImageNet vs. random
6. **Data scale:** {10K, 25K, 50K, 100K} images — scaling curve
7. **Property diversity:** {mass only, friction only, mass+friction, all properties}
8. **Fine-tuning depth:** freeze all / unfreeze last 2 / last 4 / all layers

---

### Analysis

1. **t-SNE/UMAP:** DynaCLIP space colored by (a) object category, (b) mass, (c) friction. DINOv2 clusters by appearance; DynaCLIP shows physics gradients.

2. **Dynamics sensitivity Jacobian:** ∂z/∂mass, ∂z/∂friction. DynaCLIP should be sensitive to physics changes; DINOv2 near-zero.

3. **Cross-domain transfer:** Pre-train on ManiSkill3, evaluate on Isaac Lab objects. Does dynamics alignment transfer across simulators?

4. **Real-world qualitative:** Compute DynaCLIP embeddings on DROID/BridgeData V2 images. Do metal objects cluster? Do heavy objects cluster? Qualitative analysis.

5. **Computational cost:** DynaCLIP adds zero inference overhead (same architecture as DINOv2, different weights). Report: pre-training time, per-image ms, memory.

---

### Hardware Allocation (48 H200 GPUs, 6 nodes × 8 GPUs)

| Node | Weeks 1-2 | Weeks 3-4 | Weeks 5-10 |
|------|-----------|-----------|------------|
| 1-2 | Data gen (ManiSkill3, 1024 envs/GPU) + dynamics fingerprinting | DynaCLIP contrastive pre-training (16 GPU DDP) | Downstream world model experiments |
| 3 | DTW similarity computation (embarrassingly parallel) | Backbone ablations | Downstream policy (LIBERO, CALVIN) |
| 4 | Pair mining + Invisible Physics set | Loss/metric ablations | Physics-varying benchmark |
| 5 | — | Baseline representations (R3M, VIP, MCR fine-tune) | Probing + zero-shot inference |
| 6 | — | Data scale ablation | Analysis (t-SNE, sensitivity, cross-domain) |

---

### Repository Structure

```
DynaCLIP/
├── configs/                      # Hydra configs
│   ├── data/gen.yaml             # Data generation
│   ├── pretrain/dynaclip.yaml    # Contrastive pre-training
│   ├── eval/{probing,invisible,worldmodel,policy,zeroshot}.yaml
│   ├── ablation/{metric,actions,loss,negatives,backbone,scale,props,depth}.yaml
├── dynaclip/
│   ├── data/
│   │   ├── sim_generator.py      # ManiSkill3 parallel data gen with physics variation
│   │   ├── test_actions.py       # 5 standardized diagnostic actions
│   │   ├── dynamics_fingerprint.py  # DTW-based trajectory similarity
│   │   ├── pair_mining.py        # Hard pos/neg pair construction
│   │   ├── invisible_physics.py  # Invisible Physics test set
│   │   └── dataset.py            # PyTorch Dataset for contrastive pairs
│   ├── models/
│   │   ├── encoder.py            # DINOv2/SigLIP/CLIP backbone wrappers
│   │   ├── projection.py         # 2-layer MLP projection head
│   │   └── dynaclip.py           # Full model: encoder + projection + loss
│   ├── training/
│   │   ├── contrastive.py        # Soft InfoNCE implementation
│   │   ├── trainer.py            # Training loop with hard negative mining
│   │   └── scheduler.py          # Cosine annealing + warmup
│   ├── evaluation/
│   │   ├── probing.py            # Linear probing for physics properties
│   │   ├── invisible_test.py     # Invisible Physics Test
│   │   ├── world_model.py        # Dreamer-style WM with swappable backbone
│   │   ├── diffusion_policy.py   # Diffusion policy with swappable backbone
│   │   ├── zero_shot.py          # k-NN zero-shot physics inference
│   │   └── metrics.py            # SSIM, LPIPS, FVD, R², accuracy
│   ├── analysis/
│   │   ├── tsne.py               # t-SNE/UMAP visualization
│   │   ├── sensitivity.py        # Jacobian ∂z/∂property
│   │   ├── transfer.py           # Cross-domain transfer
│   │   └── figures.py            # Paper figure generation
│   ├── baselines/
│   │   ├── r3m.py, vip.py, mcr.py, frozen_dinov2.py
│   └── utils/
│       ├── distributed.py        # DDP
│       └── logging.py            # W&B
├── scripts/
│   ├── generate_data.sh, pretrain.sh, eval_all.sh, run_ablations.sh, make_figures.py
├── requirements.txt
└── README.md
```

### Key Implementation Notes

- PyTorch 2.x + `torch.compile` + DDP across 8-16 GPUs
- Hydra configs. W&B tracking. 3 seeds for main experiments, 1 for ablations.
- ManiSkill3: `gym.make_vec("PushCube-v1", num_envs=1024)`
- DINOv2: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`
- R3M: `pip install r3m` → `r3m.load_r3m("resnet50")`
- Diffusion Policy: clone `diffusion_policy` repo, swap visual encoder
- DTW: `tslearn.metrics.dtw` (pre-compute offline, embarrassingly parallel)
- FVD: I3D features + Fréchet distance
- Checkpoints every 10K steps. Best model by validation probing R².

### Timeline (12 weeks)

| Weeks | Task |
|-------|------|
| 1-2 | Data generation + dynamics fingerprinting + pair mining |
| 3-4 | DynaCLIP pre-training + backbone ablations |
| 5-6 | Probing + Invisible Physics Test + zero-shot inference |
| 7-8 | Downstream world model + diffusion policy on LIBERO/CALVIN |
| 9-10 | All 8 ablations + analysis (t-SNE, sensitivity, transfer) |
| 11-12 | Paper writing + figures |
