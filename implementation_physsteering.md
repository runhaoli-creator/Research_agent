# PhysSteering: Discovering and Steering Physics in Robot World Models

## Comprehensive Implementation Specification (Sim-Only)

**Novelty:** Physics Steering (Walrus paper, 2511.20798) discovered steerable physics concepts in physics FMs. SAE interpretability applied to VLAs (Stanford, 2603.19183). NOBODY has done activation-based physics discovery or steering for robot world models (DreamZero, Dreamer-4, Cosmos). Gap confirmed.

**Sim-only NeurIPS strategy:** This is fundamentally a DISCOVERY + ANALYSIS paper. All work is computational: extract activations from open-sourced models, train SAEs, probe for physics, test steerability. Evaluation on sim benchmarks (LIBERO, ManiSkill3). No real robot needed.

---

### Core Thesis

Robot world models (DreamZero, Dreamer-4, Cosmos) spontaneously develop internal representations that encode physical quantities (mass, friction, contact forces). These "physics features" are: (1) discoverable via Sparse Autoencoders, (2) interpretable (correlated with ground-truth physics), (3) steerable (modifying them changes predicted dynamics consistently), and (4) transferable (a mass concept from pushing transfers to grasping). Physics steering enables adapting world models to novel physics WITHOUT fine-tuning — just by adjusting activation vectors.

---

### Target Models (Open-Sourced)

1. **DreamZero** (NVIDIA, 14B WAM, open-sourced): Primary target. Joint video+action model. Extract activations from the DiT backbone during manipulation rollouts.

2. **Dreamer-4** (DeepMind, 2B): If weights are available. Extract activations from the RSSM dynamics model and the transformer backbone.

3. **Cosmos-Predict2.5-2B** (NVIDIA, open-sourced): Video world model. Extract activations from the flow-matching transformer during action-conditioned rollouts.

4. **Smaller reproduction models:** If large models are too expensive to run, train smaller versions (300M-500M) of each architecture on ManiSkill3 manipulation data WITH physics variation. This ensures we control the training data and know the ground-truth physics.

**Recommendation:** Start with smaller reproduction models (full control over data + physics ground truth), then validate findings on open-sourced DreamZero.

---

### Phase 1: Data Collection for Activation Analysis

**Simulation data with ground-truth physics:**

Generate rollouts in **ManiSkill3** with programmatic physics variation:
- 30 object types × 100 property configs (mass, friction, restitution) = 3,000 configs
- 6 tasks: Push, PickAndPlace, Stack, Slide, LiftAndDrop, PegInsert
- 50 trajectories per task per config = 900K trajectories
- Record: RGB frames, actions, proprioception, object states, AND ground-truth physics parameters

**Activation extraction:**

For each world model, run forward passes on all trajectories and extract:
- Hidden activations at EVERY layer, at EVERY timestep
- For DreamZero (DiT): extract activations after each DiT block (28 blocks in 14B model)
- For Dreamer-style: extract RSSM hidden states (both deterministic and stochastic)
- Store as memory-mapped arrays: ~50GB per model

---

### Phase 2: Sparse Autoencoder Training

Follow Anthropic/OpenAI mechanistic interpretability methodology:

**SAE architecture per layer:**
```
Input: activation vector h ∈ R^d (d = model hidden dim)
Encoder: W_enc ∈ R^(M×d), b_enc ∈ R^M → ReLU(W_enc · h + b_enc) = f ∈ R^M
Decoder: W_dec ∈ R^(d×M), b_dec ∈ R^d → W_dec · f + b_dec = ĥ ∈ R^d
Loss: MSE(h, ĥ) + λ * L1(f)  [reconstruction + sparsity]
```

**Hyperparameters:**
- Dictionary sizes M: {1K, 4K, 16K, 64K} (ablation)
- Sparsity λ: {1e-4, 3e-4, 1e-3} (sweep)
- Optimizer: Adam, lr=3e-4
- Training: 100K steps per layer, batch size 4096 activations
- Train one SAE per layer × per dictionary size

**Total SAEs:** ~28 layers × 4 dictionary sizes = 112 SAEs per model. Each trains in ~1 hour on 1 GPU.

---

### Phase 3: Physics Feature Discovery

For each SAE feature f_i, compute correlation with ground-truth physical quantities:

```python
for feature_idx in range(M):
    activations = sae.encode(all_hidden_states)[:, feature_idx]  # [N_samples]
    for prop in ['mass', 'friction', 'restitution', 'velocity', 'contact_force']:
        r, p = pearsonr(activations, ground_truth[prop])
        if abs(r) > 0.5 and p < 0.001:
            physics_features[prop].append((layer, feature_idx, r))
```

**Metrics:**
- **Pearson correlation** between feature activation and each physics quantity
- **Mutual information** for nonlinear relationships
- **Selectivity index:** does the feature activate ONLY for one property, or multiple?
- **Consistency:** does the same feature activate across different tasks (push, pick, stack)?

**Expected discovery:** "Layer 18, feature 2847 has r=0.92 correlation with object mass across all tasks. Layer 12, feature 1523 has r=0.87 correlation with friction coefficient."

---

### Phase 4: Steerability Tests

**Intervention protocol:**
1. Take a rollout with object of mass m1
2. At the [CONTEXT_END] or midway point, CLAMP the "mass feature" to the activation level corresponding to mass m2
3. Continue the rollout
4. Measure: does the predicted dynamics change as if the object's mass is m2?

**Steerability metric:**
```
steer_score(feature, prop) = corr(Δfeature_activation, Δpredicted_dynamics)
```

If clamping the mass feature HIGH makes the predicted object move SLOWER (consistent with heavier objects), steerability is confirmed.

**Control experiments:**
- Steer a RANDOM feature → dynamics should NOT change systematically
- Steer the mass feature → ONLY mass-related dynamics should change (not friction behavior)
- Steer on task A (push), evaluate on task B (pick) → does steering TRANSFER?

---

### Phase 5: Application — Physics Adaptation via Steering

**Use case:** Adapt DreamZero to a novel object with unknown physics.

1. Observe one interaction with novel object (push)
2. Extract the world model's prediction error
3. Search over physics feature activations (mass, friction) to minimize prediction error
4. Lock the optimal physics features
5. Use the steered world model for planning

**Compare:** Steering adaptation (0 gradient updates, ~10ms) vs. fine-tuning (100+ gradient steps, ~5s) vs. no adaptation.

---

### Baselines (6)

1. **Unsteered DreamZero:** No adaptation, use as-is on novel physics
2. **Fine-tuned DreamZero:** 100 gradient steps on novel object data
3. **DreamZero + LoRA:** Fine-tune LoRA adapters on novel object
4. **Random feature steering:** Steer random features (not physics-correlated)
5. **AdaptiGraph:** Gradient-based physics parameter optimization
6. **No world model (direct policy):** Diffusion policy without world model planning

---

### Evaluation (Sim-Only, Multi-Benchmark)

#### Experiment 1: Physics Feature Discovery (Core Contribution)

Report per-layer heatmaps: feature_idx × physical_property correlation.

**Benchmarks for discovery:**
- ManiSkill3 (primary — full physics ground truth)
- LIBERO (secondary — fixed physics but varied tasks)

Present: (a) number of physics features found per layer, (b) selectivity (single-property vs. entangled), (c) consistency across tasks, (d) comparison across model architectures.

#### Experiment 2: Steerability Quantification

For each discovered physics feature:
- Steer strength: {0.5×, 1×, 2×, 5×} of mean activation
- Measure dynamics change magnitude
- Plot: steering strength vs. dynamics change (should be monotonic)

#### Experiment 3: Steering for Novel Physics Adaptation

**ManiSkill3 physics-varying:** Train world model on standard physics. Test on 5× heavier objects, 5× lower friction.

| Method | Adaptation Time | Success Rate |
|--------|:-:|:-:|
| No adaptation | 0 | baseline |
| Physics steering (ours) | ~10ms | ? |
| Fine-tuning (100 steps) | ~5s | ? |
| LoRA (50 steps) | ~3s | ? |
| AdaptiGraph | ~5s | ? |

#### Experiment 4: Cross-Task Transfer of Physics Concepts

Discover mass feature from push task data. Test: does steering this feature correctly adapt dynamics on pick-and-place, stack, and peg-insert? Report per-task steerability score.

#### Experiment 5: Cross-Model Comparison

Compare physics feature emergence across architectures:
- DreamZero (DiT-based, 14B/2B)
- Dreamer-4/v3 (RSSM-based)
- Cosmos (flow-matching)

Which architecture develops the most/cleanest physics features?

---

### Ablation Studies (8)

1. **SAE dictionary size:** {1K, 4K, 16K, 64K} — minimum for physics discovery?
2. **Sparsity penalty λ:** {1e-4, 3e-4, 1e-3} — tradeoff between sparsity and reconstruction
3. **Training data diversity:** physics-varied data vs. fixed-physics data — are physics features emergent or only with varied training?
4. **Model scale:** 300M vs. 1B vs. 2B vs. 14B — minimum scale for physics emergence?
5. **Layer analysis:** which layers contain physics features (early/mid/late)?
6. **Feature type:** SAE features vs. PCA directions vs. random directions — are SAE features better than linear probes?
7. **Number of objects in training:** {5, 15, 30, 50} — minimum object diversity for physics features?
8. **Steering method:** activation clamping vs. activation addition vs. linear steering vector

---

### Analysis

1. **Emergence visualization:** t-SNE of activations colored by mass/friction. Show clustering by physics in mid-layers.
2. **Feature ablation:** zero out physics features → does world model lose physics understanding?
3. **Comparison to linear probing:** Are SAE features more interpretable/steerable than linear probe directions?
4. **Qualitative:** Render predicted videos with/without steering. Show: steering mass feature → object moves slower.

---

### Node Allocation

| Node | Weeks 1-2 | Weeks 3-5 | Weeks 6-10 |
|------|-----------|-----------|------------|
| 1-2 | Train reproduction models (DreamZero-300M, Dreamer-v3-300M) on ManiSkill3 | SAE training (112 SAEs per model, ~2 days per model) | Steering adaptation experiments |
| 3 | Physics-varying data generation | Physics feature correlation analysis | Cross-task transfer |
| 4 | Activation extraction pipeline | Steerability tests | Cross-model comparison |
| 5 | — | Baselines (fine-tuning, LoRA, AdaptiGraph) | Ablation studies |
| 6 | — | DreamZero-14B activation extraction (if feasible) | Analysis + visualization |

**Timeline:** 12 weeks. Core discovery (physics features exist) achievable in 4 weeks.
