# Daily Paper — March 28, 2026

Papers from the last 3 days (March 25-28) in robot learning, world models, VLA, and embodied AI.

---

## Highlights

- **MMaDA-VLA** (2603.25423) — first fully native pre-trained large diffusion VLA
- **Persistent Robot World Models** (2603.25685) — RL-based fix for world model rollout breakdown
- **Fast-dVLA** (2603.25661) — makes diffusion VLAs real-time via parameter decoupling
- **SoftMimicGen** (2603.25725) — scalable deformable object manipulation data generation
- **LaMP** (2603.25405) — 3D scene flow as motion prior for VLA

**Trends this week:**
1. VLA inference speed is THE focus (FASTER, Fast-dVLA, Mean-Flow One-Step, FODMP)
2. Reasoning-augmented VLAs keep growing (VLA-Thinker, DualCoT-VLA, Critic-in-the-Loop)
3. World models becoming structured planners, not just video generators (StructVLA, CoWVLA, PERSIST)
4. Dexterous manipulation scaling via human video (UniDex, MoDE-VLA)

---

## VLA Models

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25423](https://arxiv.org/abs/2603.25423) | **MMaDA-VLA: Large Diffusion VLA** | Fully native pre-trained large diffusion VLA unifying multi-modal understanding and generation | High |
| [2603.25661](https://arxiv.org/abs/2603.25661) | **Fast-dVLA: Real-Time Discrete Diffusion VLA** | Parameter decoupling fixes slow inference in diffusion VLAs | High |
| [2603.25038](https://arxiv.org/abs/2603.25038) | **Pi But Make It Fly: VLA for Aerial Manipulation** | Transfers manipulation VLAs to aerial pick-and-place via payload-aware guidance | High |
| [2603.25044](https://arxiv.org/abs/2603.25044) | **ThermoAct: Thermal-Aware VLA** | Integrates thermal sensor data for robot safety perception | Med |
| [2603.24935](https://arxiv.org/abs/2603.24935) | **SABER: Black-Box Attack on VLA** | Red-teaming VLAs via instruction-based adversarial edits | Med |
| [2603.24941](https://arxiv.org/abs/2603.24941) | **Beyond Attention Magnitude: Efficient VLAs** | TIES framework: +6% success, -78% tokens | High |
| [2603.23202](https://arxiv.org/abs/2603.23202) | **Gaze-Regularized VLA** | Human gaze as training-time regularizer for VLA attention | High |
| [2603.22280](https://arxiv.org/abs/2603.22280) | **DualCoT-VLA: Parallel Reasoning** | Dual visual+linguistic CoT; SOTA on LIBERO and RoboCasa | High |
| [2603.22760](https://arxiv.org/abs/2603.22760) | **SG-VLA: Spatially-Grounded Mobile Manipulation** | Spatial grounding co-training for mobile manipulation | High |
| [2603.22264](https://arxiv.org/abs/2603.22264) | **UniDex: Universal Dexterous Hand Control** | 50K+ traj across 8 hands; 81% task progress (CVPR 2026) | High |
| [2603.19199](https://arxiv.org/abs/2603.19199) | **FASTER: Real-Time Flow VLA** | 10x over Pi0.5 via horizon-aware adaptive sampling | High |
| [2603.19183](https://arxiv.org/abs/2603.19183) | **SAE for VLA Interpretability** | Sparse autoencoders reveal steerable features in Pi0.5 | Med |
| [2603.14523](https://arxiv.org/abs/2603.14523) | **VLA-Thinker: Thinking-with-Image** | Active visual info requests during reasoning; 97.5% LIBERO | High |
| [2603.12717](https://arxiv.org/abs/2603.12717) | **CoT Vulnerabilities in VLA** | VLA action decoders depend on entity references, not reasoning quality | Med |
| [2603.10052](https://arxiv.org/abs/2603.10052) | **OmniGuide: Universal Guidance Fields** | Differentiable energy: 24.2% → 92.4% without retraining | High |
| [2603.08361](https://arxiv.org/abs/2603.08361) | **DELTA-VLA: World Knowledge Variation** | Models world-knowledge variations instead of absolute future states | High |
| [2603.08122](https://arxiv.org/abs/2603.08122) | **MoDE-VLA: Mixture-of-Dexterous-Experts** | Force/tactile residual injection; 2x success on contact-rich | High |
| [2603.01469](https://arxiv.org/abs/2603.01469) | **Mean-Flow One-Step VLA** | 8.7x faster than SmolVLA, 83.9x faster than Diffusion Policy | High |
| [2603.00926](https://arxiv.org/abs/2603.00926) | **DAM-VLA: Dynamic Action Model** | Action routing + dual-scale weighting; SOTA on SIMPLER | High |
| [2603.05185](https://arxiv.org/abs/2603.05185) | **Critic in the Loop: Tri-System VLA** | VLM brain + VLA cerebellum + visual Critic | High |

---

## World Models

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25685](https://arxiv.org/abs/2603.25685) | **Persistent Robot World Models** | RL stabilizes multi-step autoregressive rollouts | High |
| [2603.23376](https://arxiv.org/abs/2603.23376) | **ABot-PhysWorld: Physics-Aligned 14B WFM** | DPO physics alignment; outperforms Veo 3.1 and Sora v2 Pro | High |
| [2603.17808](https://arxiv.org/abs/2603.17808) | **EVA: Executable Action Alignment** | Inverse dynamics rewards close the executability gap | High |
| [2603.16669](https://arxiv.org/abs/2603.16669) | **Kinema4D: Kinematic 4D World Modeling** | URDF-based 4D sim with 200K episodes | High |
| [2603.15759](https://arxiv.org/abs/2603.15759) | **Simulation Distillation** | Distills sim priors; freezes reward/value, adapts dynamics | High |
| [2603.12553](https://arxiv.org/abs/2603.12553) | **StructVLA: World Models as Structured Planners** | Sparse kinematic keyframes instead of dense video | High |
| [2603.12639](https://arxiv.org/abs/2603.12639) | **RoboStereo: Dual-Tower 4D WM** | RGB + 3D pointmaps; >97% improvement on fine-grained tasks | High |
| [2603.10448](https://arxiv.org/abs/2603.10448) | **DiT4DiT: Joint Video+Action Modeling** | Cascaded Video DiT → Action DiT; 98.6% LIBERO | High |
| [2603.09030](https://arxiv.org/abs/2603.09030) | **PlayWorld: WM from Autonomous Play** | First WM from unsupervised self-play | High |
| [2603.08546](https://arxiv.org/abs/2603.08546) | **Interactive World Simulator** | Consistency models; 10+ min stable at 15 FPS | High |
| [2603.03195](https://arxiv.org/abs/2603.03195) | **Chain of World (CoWVLA)** | Latent motion world model thinking | High |
| [2603.03482](https://arxiv.org/abs/2603.03482) | **PERSIST: Persistent 3D State World Model** | Latent 3D scene evolution for spatial memory | High |

---

## Diffusion Policy & Imitation Learning

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25725](https://arxiv.org/abs/2603.25725) | **SoftMimicGen: Deformable Manipulation Data** | Automated cloth/rope/tissue manipulation pipeline | High |
| [2603.24806](https://arxiv.org/abs/2603.24806) | **FODMP: Fast One-Step Diffusion** | Single-step decoder; 10x faster inference | High |
| [2603.25583](https://arxiv.org/abs/2603.25583) | **F-ACIL: Factorized Compositional IL** | 45% gains with 5-10x fewer demos | High |
| [2603.25405](https://arxiv.org/abs/2603.25405) | **LaMP: 3D Scene Flow as Motion Prior** | Dense 3D scene flow in VLA action generation | High |
| [2603.25481](https://arxiv.org/abs/2603.25481) | **LILAC: Language-Conditioned Optical Flow** | Flow-based trajectory from human/web video | High |
| [2603.07530](https://arxiv.org/abs/2603.07530) | **ICLR: In-Context IL with Visual Reasoning** | Few-shot + structured visual reasoning traces | High |
| [2603.05117](https://arxiv.org/abs/2603.05117) | **SeedPolicy: Self-Evolving Diffusion Policy** | 36.8% improvement clean, 169% randomized | High |
| [2603.05291](https://arxiv.org/abs/2603.05291) | **HD-ExpIt: Hierarchical Diffusion Policy** | Self-reinforcing diffusion planning + distillation | High |
| [2603.07691](https://arxiv.org/abs/2603.07691) | **RoboPCA: Pose-Centered Affordance** | Joint contact + pose via diffusion; +38.5% RLBench | High |

---

## RL & Dexterous Manipulation

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.16065](https://arxiv.org/abs/2603.16065) | **Large Reward Models** | Zero-shot VLM rewards; improves policy in 30 RL iterations | High |
| [2603.10451](https://arxiv.org/abs/2603.10451) | **FAR-Dex: Few-shot Dexterous Manipulation** | Online RL residual on top of base policy | High |
| [2603.25544](https://arxiv.org/abs/2603.25544) | **MuscleMimic: Musculoskeletal at Scale** | Order-of-magnitude speedups for musculoskeletal training | Med |
| [2603.12243](https://arxiv.org/abs/2603.12243) | **HandelBot: Real-World Piano Playing** | Sim + real-world residual RL for dexterous control | Med |

---

## Relevance to Our 5 Ideas

| Paper | Relevant To | Impact |
|-------|------------|--------|
| Persistent Robot WMs (2603.25685) | PhysBridge, PhysContext | Addresses rollout stability — could be combined |
| SAE for VLA (2603.19183) | **PhysSteering** | Same technique on VLAs; validates our approach for WMs |
| ABot-PhysWorld (2603.23376) | **DynaCLIP**, PhysBridge | DPO physics alignment — complementary, not competing |
| SoftMimicGen (2603.25725) | Zero-Success | Deformable data gen — could provide failure data |
| F-ACIL (2603.25583) | DynaCLIP | Factorized compositional learning — related representation idea |
| OmniGuide (2603.10052) | All ideas | Guidance fields could augment any of our methods |
| MMaDA-VLA (2603.25423) | All ideas | New SOTA VLA baseline to compare against |

**No paper today invalidates any of our 5 ideas.** The SAE for VLA paper (2603.19183) further validates the PhysSteering direction.
