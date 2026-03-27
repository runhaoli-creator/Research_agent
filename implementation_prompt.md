# PhysLang: Language-Grounded World Models for Zero-Shot Physical Reasoning in Manipulation

## Implementation Specification for Coding Agent

---

### Project Overview and Core Hypothesis

Build **PhysLang**, a world model for robotic manipulation that is conditioned on natural language descriptions of object physical properties (mass, friction coefficient, elasticity, fragility, hardness) to predict future states. The core hypothesis: by conditioning dynamics predictions on language descriptions of physical properties that are visually ambiguous (e.g., "this cube is heavy and has a slippery surface"), the world model can generalize zero-shot to novel physical property combinations never seen during training, outperforming vision-only world models that must infer physics from pixels alone. This addresses a critical gap — AdaptiGraph (RSS 2024) conditions dynamics on numerical physical parameters but not language; DeliGrasp (CoRL 2024) uses LLMs to infer physical properties but feeds them to analytical controllers, not learned dynamics models; LED-WM (Nov 2025) conditions DreamerV3 on language but for game rules, not physical properties. Nobody has built a language-conditioned dynamics model for manipulation with physical property grounding.

---

### Model Architecture (PhysLang World Model)

The model has five components that must be implemented precisely:

**1. Visual Encoder (frozen, feature extractor):**
Implement support for three backbone options that will be compared as an ablation: (a) DINOv2-ViT-B/14 (86M params, patch size 14, output dim 768) — use the `facebookresearch/dinov2` pretrained weights from torch hub, extract CLS + patch tokens from the last layer, project via a learned linear layer to latent dim D=512; (b) DINOv2-ViT-L/14 (300M params, output dim 1024) — same extraction, project to D=512; (c) SigLIP-ViT-B/16 (86M params) from `google/siglip-base-patch16-224` — extract image features, project to D=512. The visual encoder is FROZEN during world model training (only the projection layer is trained). Input: RGB image 224×224. Output: visual embedding v ∈ R^(D) where D=512.

**2. Language Encoder (frozen, for physical property descriptions):**
Implement support for two options: (a) SigLIP text encoder (from the same `google/siglip-base-patch16-224` model) — encode the physical property description string, take the [EOS] token embedding, project via learned linear to D=512; (b) CLIP ViT-L/14 text encoder (`openai/clip-vit-large-patch14`) — same procedure. The text encoder is FROZEN; only the projection layer trains. Input: a natural language string like "The red cube has mass 2.3 kg, friction coefficient 0.2, and high elasticity. The blue cylinder is light at 0.3 kg with high friction and is fragile." Output: language embedding l ∈ R^(D).

**3. Action Encoder:**
A 2-layer MLP (hidden dim 256, SiLU activation) that encodes the robot action a_t (typically 7-DoF end-effector delta position + delta orientation + gripper command, so input dim = 7 or 8 depending on the environment) into action embedding a ∈ R^(D).

**4. Latent Dynamics Model (the core contribution):**
This is a Transformer-based dynamics model that predicts future latent states. Architecture: 8-layer Transformer decoder with D=512, 8 attention heads, feedforward dim 2048, pre-LayerNorm, SiLU activations. The input sequence at each prediction step is: [v_t; l; a_t] where v_t is the visual embedding of the current observation, l is the language embedding of physical property descriptions, and a_t is the action embedding. The language embedding l conditions the dynamics via **two mechanisms that will be ablated**: (a) **FiLM conditioning**: for each Transformer layer, compute scale γ and shift β from l via a learned linear layer, then apply γ * LayerNorm(x) + β; (b) **Cross-attention**: insert a cross-attention layer after each self-attention layer where the query is the dynamics state and the key/value come from the language embedding (expanded to a sequence of per-property tokens — split the description into per-object property sub-descriptions, encode each separately, and use the resulting sequence as cross-attention context); (c) **Concatenation baseline**: simply concatenate l to the input token sequence. The dynamics model predicts the next visual embedding: v̂_{t+1} = f(v_t, l, a_t). The model is autoregressive — for multi-step prediction, feed the predicted v̂_{t+1} back as v_{t+1}. Use **symlog prediction** (following DreamerV3) to handle varying prediction scales.

**5. Observation Decoder:**
A lightweight CNN decoder (4 ConvTranspose2d layers: 512→256→128→64→3, each with BatchNorm + SiLU, output resolution 224×224) that reconstructs the RGB observation from the predicted visual embedding. This is used for visualization and reconstruction loss, but the primary evaluation is on latent prediction accuracy and downstream policy success.

**Training Losses:**
- **Latent prediction loss** (primary): L_latent = MSE(v̂_{t+1}, sg(encode(o_{t+1}))) where sg is stop-gradient on the target encoding. This is the main loss.
- **Reconstruction loss** (auxiliary): L_recon = MSE(decode(v̂_{t+1}), o_{t+1}) + 0.1 * LPIPS(decode(v̂_{t+1}), o_{t+1}). Weight: 0.1× relative to latent loss.
- **Multi-step prediction loss**: compute L_latent at horizons t+1, t+2, ..., t+H with H=16, with exponential decay weighting (weight = 0.95^h for step h).
- Optimizer: AdamW, lr=3e-4, weight decay=0.01, cosine annealing schedule with 1000 step warmup. Batch size: 64 trajectories × 16 timesteps per trajectory. Train for 200K steps.

---

### Data Generation Pipeline

**Simulator:** Use **ManiSkill3** (pip install mani-skill) as the primary simulator because it is GPU-parallelized (can run 1000+ parallel environments on a single GPU), provides RGB + depth + segmentation + object state information, and allows programmatic variation of physical properties via SAPIEN. As a secondary validation simulator, also generate data in **Isaac Lab** for the photorealistic rendering ablation.

**Physical Property Space:**
Define a continuous property space for each object in the scene:
- Mass: [0.05, 10.0] kg, log-uniform sampling (5 discrete bins for evaluation: 0.05, 0.2, 1.0, 3.0, 10.0)
- Friction coefficient (static): [0.05, 1.5], uniform (5 bins: 0.05, 0.3, 0.6, 1.0, 1.5)
- Bounciness/restitution: [0.0, 0.95], uniform (4 bins: 0.0, 0.3, 0.6, 0.9)
- Density: derived from mass and mesh volume
- Damping: [0.0, 5.0], uniform (for articulated objects)

**Object Set:** Use ManiSkill3's built-in object assets (YCB objects: mug, bowl, can, box, banana, etc. — at least 20 distinct object geometries). For each object, sample N_prop=50 random physical property configurations. Each configuration gets a natural language description.

**Language Description Generation:**
For each object-property configuration, generate descriptions at three levels of detail:
- **Template-based (Level 1):** "The {color} {object} has mass {mass:.1f} kg, friction {friction:.2f}, and restitution {restitution:.2f}." — generate 5 paraphrases per template using simple rephrasing rules.
- **Natural language (Level 2):** Map numerical values to qualitative descriptors: mass < 0.2 → "very light", 0.2-1.0 → "light", 1.0-3.0 → "moderately heavy", 3.0-10.0 → "heavy", > 10.0 → "very heavy". Similarly for friction: < 0.2 → "very slippery/smooth", 0.2-0.5 → "somewhat smooth", 0.5-0.8 → "moderate friction", 0.8-1.2 → "rough/grippy", > 1.2 → "very rough". Example: "The red mug is heavy and has a very slippery surface."
- **Diverse paraphrases (Level 3):** Use an LLM (GPT-4 or Claude API) to generate 10 diverse paraphrases of each Level 2 description, varying sentence structure, vocabulary, and detail level. Cache these offline before training.

**Tasks for Data Collection (in ManiSkill3):**
Implement 6 manipulation tasks with diverse physical interactions:
1. **PickAndPlace** — pick up an object, place it at a target location. Physics relevance: mass affects required grasp force, friction affects slip.
2. **Push** — push an object to a target pose on a table. Physics relevance: mass and friction directly determine sliding dynamics.
3. **Stack** — stack 2-3 objects. Physics relevance: mass, friction, and restitution determine stability.
4. **TiltPour** — tilt a container to pour contents. Physics relevance: mass distribution changes during pouring.
5. **Slide** — slide an object across surfaces with varying friction. Physics relevance: friction is the dominant parameter.
6. **Drop** — drop an object and predict where it lands/bounces. Physics relevance: restitution and mass.

**Data Collection Protocol:**
Use scripted oracle policies (ManiSkill3 provides motion planning oracles for most tasks) to collect demonstrations. For each task × object × property configuration: collect 20 trajectories (demonstrations). Each trajectory: 100-200 timesteps at 20 Hz, recording (RGB 224×224, robot proprioception, action, object states, physical properties, language description). Total target: ~500K trajectories across all tasks, objects, and property configurations. Store in HDF5 format with efficient chunked compression.

**Train/Val/Test Split (Critical for Compositional Generalization):**
- **Training set:** 70% of property combinations (e.g., if we have mass bins [L, M, H] and friction bins [L, M, H], train on 6 out of 9 combinations)
- **Validation set:** 15% of property combinations (held-out combos)
- **Test set:** 15% completely held-out property combinations — these are the NOVEL combinations the model has never seen. This is the key evaluation: can the model predict dynamics for "heavy + slippery" if it has only seen "heavy + rough" and "light + slippery" during training?
- Additionally, hold out 5 object geometries entirely for zero-shot object generalization testing.

---

### Baselines to Implement

Implement ALL of the following baselines with the SAME compute budget and hyperparameter tuning effort:

1. **Vision-Only World Model (ablation):** Same architecture as PhysLang but with the language embedding l replaced by a zero vector. This isolates the contribution of language conditioning.

2. **Oracle-Numerical World Model:** Same architecture but replace the language encoder with a direct MLP encoding of the ground-truth numerical physical parameters [mass, friction, restitution] → R^D. This is the upper bound — if the model can access exact physics parameters, how well does it do?

3. **DreamerV3:** Implement the full DreamerV3 agent (RSSM with categorical latents, symlog predictions, actor-critic in imagination). Use the official implementation from `danijar/dreamerv3` as reference. Train on the same data. This represents the state-of-the-art latent world model baseline without any physics conditioning.

4. **AdaptiGraph-style (Numerical Property Conditioning):** Implement a GNN-based dynamics model conditioned on numerical property vectors (following AdaptiGraph, RSS 2024). Use the object point cloud as graph nodes, property vectors as node features, and predict next-state point clouds. This compares language conditioning vs. direct numerical conditioning.

5. **TD-MPC2:** Implement TD-MPC2 (implicit world model with latent dynamics, no decoder). Use the reference implementation from `nicklashansen/tdmpc2`. Fine-tune on our tasks. This compares explicit physics-conditioned world models vs. implicit task-oriented world models.

6. **Random Property Assumption:** Same world model architecture but always feed the AVERAGE property description ("The object has typical mass and friction") regardless of actual properties. This tests whether the model actually uses the language input or ignores it.

7. **CLIP-Retrieval Baseline:** At test time, use CLIP to find the most visually similar object from training data and use that object's dynamics predictions. This tests whether visual similarity is sufficient to infer physical properties.

---

### Evaluation Metrics

**World Model Prediction Quality:**
- **Latent MSE:** Mean squared error between predicted and actual visual embeddings at horizons t+1, t+5, t+10, t+20.
- **SSIM / LPIPS:** Structural similarity and perceptual similarity between decoded predicted frames and actual frames.
- **FVD (Fréchet Video Distance):** Computed over 16-frame predicted video clips vs. ground truth.
- **Physics Violation Rate:** For each predicted trajectory, check if objects penetrate each other, violate gravity, or exhibit impossible momentum changes. Use the simulator's ground-truth physics state as reference. Report % of timesteps with violations.
- **Object Position/Velocity Error:** L2 error between predicted and actual object position/velocity (extracted from the world model's implicit representation vs. ground truth simulator state).

**Compositional Generalization (Key Metric):**
- **Held-out Property Combination Accuracy:** Prediction quality metrics computed ONLY on the held-out test set (novel property combinations). Report the ratio: (held-out performance) / (in-distribution performance). A ratio close to 1.0 means good compositional generalization.
- **Property Interpolation vs. Extrapolation:** Separately evaluate on property values within the training range (interpolation) vs. outside it (extrapolation).

**Downstream Policy Learning:**
- Train a diffusion policy (following the Chi et al. "Diffusion Policy" architecture, 2023) using PhysLang's world model for two purposes: (1) as a data augmentation engine (generate synthetic trajectories with varied physical properties), (2) as an MPC planner (sample N action sequences, simulate forward through PhysLang, select highest-reward trajectory).
- Report task **success rate** (%) on all 6 tasks, averaged over 100 episodes per task, 3 random seeds.
- Report success rate specifically on **novel property combinations** (the held-out test set).

---

### Ablation Studies

Design and run the following ablations, each isolating one design choice:

1. **Language Conditioning Mechanism:** Compare FiLM vs. Cross-Attention vs. Concatenation (as described in the architecture section). Report prediction quality and downstream success. Hypothesis: Cross-attention will be best because it allows per-property attention, but FiLM may be competitive due to simplicity.

2. **Visual Backbone:** Compare DINOv2-ViT-B/14 vs. DINOv2-ViT-L/14 vs. SigLIP-ViT-B/16 as the frozen visual encoder. Report prediction quality. Hypothesis: DINOv2-L will have the best features but DINOv2-B may be sufficient.

3. **Language Encoder:** Compare SigLIP text encoder vs. CLIP ViT-L/14 text encoder. Hypothesis: minimal difference since physical property descriptions are simple text.

4. **Language Description Granularity:** Compare Level 1 (template) vs. Level 2 (qualitative natural language) vs. Level 3 (diverse paraphrases) descriptions. Hypothesis: Level 3 will generalize best to novel descriptions at test time; Level 1 will overfit to template structure.

5. **Number of Training Property Combinations:** Vary the number of property combinations seen during training: {10%, 25%, 50%, 70%} of all combinations. Plot compositional generalization accuracy vs. training diversity. Hypothesis: a log-linear relationship (more diversity → better generalization, with diminishing returns).

6. **Physical Properties Included:** Systematically remove each property from the language description (mass only, friction only, mass+friction, all properties). Measure which properties contribute most to prediction accuracy.

7. **Prediction Horizon:** Evaluate prediction quality at horizons 1, 5, 10, 20, 50 steps. Plot error vs. horizon for PhysLang vs. vision-only baseline. Hypothesis: PhysLang's advantage grows with horizon because physics-consistent predictions compound less error.

8. **With vs. Without Multi-Step Training Loss:** Compare training with H=1 (single-step prediction only) vs. H=16 (multi-step with decay weighting). Hypothesis: multi-step training is critical for long-horizon accuracy.

---

### Additional Analysis

1. **t-SNE / UMAP Visualization:** Visualize the learned dynamics latent space colored by physical properties. Show that PhysLang clusters by physics (heavy objects cluster together regardless of appearance) while the vision-only model clusters by appearance.

2. **Language Sensitivity Analysis:** At test time, systematically modify one word in the property description (e.g., change "heavy" to "light") and measure how the predicted dynamics change. Show that the model is sensitive to physically meaningful language changes and invariant to irrelevant language changes (e.g., changing "red" to "blue" in the property description shouldn't change dynamics).

3. **Failure Case Analysis:** Identify the property combinations and tasks where PhysLang fails. Categorize failures: (a) language ambiguity, (b) visual-language conflict, (c) out-of-distribution properties, (d) long-horizon error accumulation.

4. **Sim-to-Real Property Transfer Analysis:** Using a small set of real-world videos (from DROID or BridgeData V2), describe object properties in language and qualitatively evaluate whether PhysLang's predictions are more physically plausible than the vision-only baseline. This is qualitative, not quantitative, since we cannot programmatically vary real-world physics.

5. **Computational Cost Analysis:** Report training time (GPU-hours), inference time (ms per prediction step), and memory usage for PhysLang vs. all baselines. PhysLang should add minimal overhead since the language encoder is frozen and FiLM/cross-attention is lightweight.

---

### Hardware and Node Allocation (48 H200 GPUs, 6 nodes × 8 GPUs)

- **Nodes 1-2 (16 GPUs):** ManiSkill3 data generation — run 1000+ parallel environments per GPU to generate the 500K trajectory dataset. Then transition to PhysLang world model training (DDP across 8-16 GPUs, batch size 64 per GPU).
- **Node 3 (8 GPUs):** Baseline training — DreamerV3, TD-MPC2, vision-only world model (run sequentially or 2-3 in parallel since each needs only 2-4 GPUs).
- **Node 4 (8 GPUs):** AdaptiGraph-style baseline + Oracle-Numerical baseline + Random-Property baseline + CLIP-retrieval baseline.
- **Node 5 (8 GPUs):** Ablation studies — language conditioning mechanisms, visual backbones, language encoders, description granularity. Run in parallel (each ablation needs 1-2 GPUs).
- **Node 6 (8 GPUs):** Downstream policy learning (diffusion policy + MPC planning experiments) and evaluation. Also run the scaling/analysis experiments (t-SNE, sensitivity analysis, failure cases).

---

### Repository Structure

```
PhysLang/
├── configs/                    # Hydra configs for all experiments
│   ├── model/                  # PhysLang architecture configs
│   ├── data/                   # Dataset generation configs
│   ├── baseline/               # Baseline model configs
│   └── experiment/             # Full experiment configs (model + data + training)
├── physlang/
│   ├── models/
│   │   ├── visual_encoder.py   # DINOv2, SigLIP wrappers (frozen)
│   │   ├── language_encoder.py # SigLIP, CLIP text encoder wrappers (frozen)
│   │   ├── action_encoder.py   # MLP action encoder
│   │   ├── dynamics.py         # Transformer dynamics model with FiLM/cross-attn/concat
│   │   ├── decoder.py          # CNN observation decoder
│   │   └── physlang_model.py   # Full PhysLang model combining all components
│   ├── baselines/
│   │   ├── dreamerv3.py        # DreamerV3 baseline wrapper
│   │   ├── tdmpc2.py           # TD-MPC2 baseline wrapper
│   │   ├── adaptigraph.py      # GNN + numerical property conditioning
│   │   ├── vision_only.py      # PhysLang without language (zero vector)
│   │   ├── oracle_numerical.py # PhysLang with ground-truth numerical params
│   │   └── random_property.py  # PhysLang with fixed average description
│   ├── data/
│   │   ├── generate_data.py    # ManiSkill3 data generation with property variation
│   │   ├── language_gen.py     # Template + NL + paraphrase description generation
│   │   ├── dataset.py          # PyTorch Dataset/DataLoader for training
│   │   └── splits.py           # Train/val/test split logic for compositional generalization
│   ├── training/
│   │   ├── trainer.py          # Training loop with multi-step prediction loss
│   │   └── losses.py           # Latent MSE + reconstruction + LPIPS losses
│   ├── evaluation/
│   │   ├── metrics.py          # SSIM, LPIPS, FVD, physics violation rate
│   │   ├── compositional.py    # Compositional generalization evaluation
│   │   ├── downstream.py       # Diffusion policy + MPC downstream evaluation
│   │   └── analysis.py         # t-SNE, sensitivity analysis, failure cases
│   └── utils/
│       ├── visualization.py    # Video generation, comparison plots
│       └── logging.py          # W&B / TensorBoard logging
├── scripts/
│   ├── generate_dataset.sh     # Launch parallel data generation across nodes
│   ├── train_physlang.sh       # Launch PhysLang training
│   ├── train_baselines.sh      # Launch all baselines
│   ├── run_ablations.sh        # Launch all ablation studies
│   ├── evaluate_all.sh         # Run full evaluation suite
│   └── generate_figures.py     # Generate all paper figures
├── requirements.txt
└── README.md
```

---

### Key Implementation Details

- Use **PyTorch 2.x** with `torch.compile` for model compilation. Use **DeepSpeed ZeRO Stage 2** or **PyTorch FSDP** for distributed training across 8 GPUs per node.
- Use **Hydra** for configuration management — every experiment should be fully reproducible from its config.
- Use **Weights & Biases** for experiment tracking — log all metrics, hyperparameters, and generated videos.
- For ManiSkill3 data generation, use the `gym.make_vec` API with `num_envs=1024` per GPU for maximum throughput.
- For the diffusion policy downstream evaluation, use the `diffusion_policy` codebase from Cheng Chi et al. (2023) as reference.
- Set random seeds everywhere (torch, numpy, environment) and run 3 seeds per experiment for error bars.
- Save checkpoints every 10K steps; evaluate on validation set every 5K steps for early stopping.

---

### Expected Timeline

- **Week 1:** Data generation pipeline + dataset creation in ManiSkill3
- **Week 2:** PhysLang model implementation + initial training runs
- **Week 3:** Baseline implementations (DreamerV3, TD-MPC2, vision-only, oracle, AdaptiGraph)
- **Week 4:** Full training of PhysLang + all baselines
- **Week 5-6:** Ablation studies (all 8 ablations in parallel across nodes)
- **Week 7-8:** Downstream policy learning experiments + MPC planning
- **Week 9-10:** Analysis (t-SNE, sensitivity, failure cases, computational cost)
- **Week 11-12:** Paper writing, figure generation, supplementary material
