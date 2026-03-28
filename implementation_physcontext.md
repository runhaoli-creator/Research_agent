# PhysContext: In-Context Physical Property Learning for Zero-Shot Manipulation

## Comprehensive Implementation Specification (Sim-Only)

**Novelty:** GPhyT (2509.13805) does in-context physics for PDEs, not manipulation. AdaptiGraph uses gradient optimization, not in-context. **⚠️ DALI (NeurIPS 2025, 2508.20294)** does context-conditioned world model adaptation without gradients on MetaWorld — 75% overlap. Differentiators: (1) explicit physics parameters vs. DALI's latent context, (2) deliberate diagnostic protocol, (3) compositional transfer.

**Sim-only strategy:** All experiments on LIBERO, CALVIN, ManiSkill3. Diagnostic interactions are simulated. The core contribution (in-context physics adaptation) is testable entirely in simulation with ground-truth physics.

**Key positioning vs. DALI:** DALI infers an opaque latent context vector. PhysContext's context encodes INTERPRETABLE physics (mass, friction) enabling COMPOSITIONAL transfer: "same friction as A + same mass as B → correct prediction for unseen C." This is impossible with DALI's latent approach. This compositional transfer experiment is the core differentiator.

---

### Architecture

**Causal Transformer dynamics model with physics context window:**

```
Layers: 12 Transformer decoder blocks
Hidden dim D: 512
Attention heads: 8
FFN dim: 2048
Activation: SiLU
Normalization: RMSNorm (pre-norm)
Position encoding: RoPE
Context length: 4096 tokens max
Total params: ~85M
```

**Visual tokenizer (frozen):** DINOv2-ViT-B/14 → 256 patch tokens (768-d) → Linear(768, 512) → 256 tokens per frame.

**Input sequence:**
```
[CTX_START] [diag_frame_1...256 tokens] [diag_action_1] ... [diag_frame_K...] [CTX_END] [obs_t...256 tokens] [proprio_t] [action_t] → predict [obs_{t+1}...256 tokens]
```

For K=5 diagnostic frames: 257×5 + 258 = 1543 tokens.

**Segment embeddings:** Learned embedding distinguishing context (segment 0) from current state (segment 1).

**Prediction:** L2 loss in DINOv2 feature space (not pixels). Auxiliary pixel decoder for visualization (weight 0.05).

**Training:** AdamW, lr=3e-4, cosine schedule, 2000 warmup, 500K steps, batch 128, bf16. Context curriculum: K=10 → K=5 → K=uniform(1,10).

---

### Data

**ManiSkill3** with physics variation. 30 objects × 100 property configs = 3,000 configs. 5 diagnostic actions (push-X, push-Y, lift-and-drop, flick, press-down) × 10 frames each. 6 tasks × 20 trajectories per config. Train/val/test split by property COMBINATION (70/15/15). Hold out 5 object geometries.

---

### Baselines (8)

1. **No-Context:** Same Transformer, no diagnostic context (zero vector). Measures context value.
2. **DALI** (NeurIPS 2025): Reproduce DALI — context-conditioned DreamerV3 with latent context inference. THE critical baseline to beat.
3. **AdaptiGraph-style:** Gradient-based physics optimization at test time (10/50/100 steps).
4. **Oracle-Numerical:** Ground-truth physics parameters as direct input. Upper bound.
5. **Dreamer-v3:** Standard latent world model. No adaptation.
6. **TD-MPC2:** Implicit world model.
7. **Average-Context:** Always same diagnostic context (average physics). Tests if model ignores context.
8. **Random-Context:** Diagnostic context from random different object.

---

### Experiments (Sim-Only)

#### Experiment 1: Prediction Quality on Novel Physics (vs. DALI)

Latent MSE, SSIM, LPIPS, FVD at horizons t+{1,5,10,20} on held-out property combinations.

**Key comparison:** PhysContext vs. DALI vs. No-Context vs. Oracle. Show PhysContext matches Oracle more closely than DALI on NOVEL property combinations.

#### Experiment 2: Compositional Transfer (Core Differentiator from DALI)

**The experiment DALI cannot do:**
- Object A: mass=2kg, friction=0.3
- Object B: mass=0.5kg, friction=1.0
- Novel object C: mass=2kg (from A), friction=1.0 (from B) — NEVER seen during training

PhysContext: provide diagnostic context from A (for mass) and B (for friction) separately, compose physics embeddings. DALI: cannot compose latent contexts because they're opaque.

Report prediction accuracy on compositional novel objects. PhysContext should handle this; DALI should fail.

#### Experiment 3: Context Scaling (K=0 to K=10)

Plot prediction error vs. number of diagnostic frames. Compare PhysContext curve vs. AdaptiGraph adaptation curves (gradient steps 0-500).

**Expected:** PhysContext at K=1 matches AdaptiGraph at N=100 gradient steps. PhysContext is 500× faster.

#### Experiment 4: Downstream Policy (Multi-Benchmark)

Diffusion Policy + MPC planning using PhysContext world model:

| Benchmark | Tasks | Metric |
|-----------|-------|--------|
| LIBERO-10 | 10 tasks | Success rate |
| LIBERO-Long | Long-horizon | Success rate |
| ManiSkill3 (physics-varying) | 4 tasks × novel physics | Success rate |
| CALVIN ABC-D | 5-step chains | Avg chain length |

#### Experiment 5: Zero-Shot Physics Inference

Linear probe on [CTX_END] hidden state → predict mass, friction, restitution. Compare to DALI's latent context. PhysContext should have higher R² (because physics is explicitly encoded, not latent).

#### Experiment 6: Invisible Physics Test

500 visually identical pairs. KL divergence between predicted dynamics. PhysContext > DALI > No-Context.

---

### Ablation Studies (8)

1. Context length K: {0,1,2,3,5,10,20}
2. Which diagnostic action: each individually vs. combinations
3. Visual backbone: DINOv2-B vs. DINOv2-L vs. SigLIP
4. Model scale: {4,8,12,16,24} layers
5. Context curriculum: always K=10 vs. always K=1 vs. curriculum
6. Training physics diversity: {10%,25%,50%,100%} combinations
7. Segment embeddings: with vs. without
8. Cross-task context: diagnostic from push, evaluate on pick-and-place

---

### Node Allocation

| Node | Weeks 1-2 | Weeks 3-5 | Weeks 6-10 |
|------|-----------|-----------|------------|
| 1-2 | Data gen (ManiSkill3) | PhysContext training (500K steps) | Downstream policy |
| 3 | — | DALI reproduction (critical baseline) | Compositional transfer |
| 4 | — | Other baselines | Invisible physics + inference |
| 5 | — | — | Ablations |
| 6 | — | — | Analysis |

**Timeline:** 12 weeks. Compositional transfer result (vs. DALI) is the make-or-break experiment — prioritize in weeks 5-6.
