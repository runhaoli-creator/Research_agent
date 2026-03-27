# World Model Landscape for Robotics & Embodied AI

*Literature survey as of March 2026*

---

## 1. Foundational Papers

### World Models (Ha & Schmidhuber, 2018) — NeurIPS 2018
- **arXiv:** 1803.10122 | **Citations:** ~1,547
- **Contribution:** Introduced the V-M-C architecture (VAE + MDN-RNN + controller) where an agent learns a compressed spatial-temporal model of the environment and trains a policy entirely inside the "dream."
- **Limitation:** Tested only on simple envs (VizDoom, CarRacing); dream-based training exploits model errors; linear controller.

### PlaNet (Hafner et al., 2019) — ICML 2019
- **arXiv:** 1811.04551 | **Citations:** ~1,739
- **Contribution:** Introduced the Recurrent State-Space Model (RSSM) combining deterministic + stochastic states. First purely model-based agent competitive from pixels using CEM planning in latent space.
- **Limitation:** CEM planning is expensive; no learned policy; short planning horizons.

### Dreamer v1/v2/v3 (Hafner et al., 2020–2025)
| Version | Venue | arXiv | Citations | Key Advance |
|---------|-------|-------|-----------|-------------|
| v1 | ICLR 2020 | 1912.01603 | ~1,777 | Actor-critic in imagination with analytic gradients |
| v2 | ICLR 2021 | 2010.02193 | ~1,135 | Discrete (categorical) latents; human-level Atari |
| v3 | Nature 2025 | 2301.04104 | ~971 | Fixed hyperparameters across 150+ tasks; symlog predictions |
- **Limitation (shared):** Single-task; high compute; real-world deployment gap.

### MuZero (Schrittwieser et al., 2020) — Nature 2020
- **Citations:** ~2,412
- **Contribution:** Learned a world model that plans without game rules. Predicts rewards, values, policies directly — no pixel reconstruction. Planning via MCTS in latent space.
- **Limitation:** Large compute for MCTS; discrete actions only (originally); non-interpretable latent model.

### TD-MPC / TD-MPC2 (Hansen et al., 2022/2024) — ICML 2022 / ICLR 2024
- **Citations:** ~374 / ~356
- **Contribution:** Task-oriented implicit world model (no decoder). TD-MPC2 scales to 104 tasks with single hyperparameter set and 317M multi-task agent.
- **Limitation:** Planning overhead at inference; continuous control only; multi-task agent underperforms specialists.

### IRIS (Micheli et al., 2023) — ICLR 2023
- **arXiv:** 2209.00588 | **Citations:** ~286
- **Contribution:** Transformer-based world model with discrete autoencoder. First strong results on Atari 100k with transformers.
- **Limitation:** Discrete tokenization loses visual detail; slow autoregressive generation.

### DayDreamer (Wu et al., 2023) — CoRL 2023
- **arXiv:** 2206.14176 | **Citations:** ~445
- **Contribution:** First demonstration of Dreamer learning on physical robots. Quadruped walks in 1 hour; arms learn pick-and-place from camera.
- **Limitation:** Simple tasks only; safety during exploration; requires reset mechanisms.

### DIAMOND (Alonso et al., 2024) — NeurIPS 2024 Spotlight
- **arXiv:** 2405.12399 | **Citations:** ~192
- **Contribution:** First diffusion-based world model for RL. Operates in pixel space (no tokenization). SOTA on Atari 100k (1.46 HNS).
- **Limitation:** Diffusion inference is slow; high compute.

---

## 2. Foundation World Models (Industry Scale)

### GAIA-1 (Wayve, 2023)
- **arXiv:** 2309.17080 | **Citations:** ~476
- **Contribution:** 9B-parameter generative world model for autonomous driving. Unsupervised next-token prediction over video + text + action tokens.
- **Limitation:** Driving-only; not reproducible; no formal peer review.

### UniSim (Yang et al., 2024) — ICLR 2024
- **arXiv:** 2310.06114 | **Citations:** ~375
- **Contribution:** Universal video diffusion simulator handling diverse action modalities. Zero-shot sim-to-real for planners and policies.
- **Limitation:** 512 TPU-v3 chips for training; hallucination in long-horizon; slow inference.

### Genie / Genie 2 / Genie 3 (DeepMind, 2024–2026)
| Version | Date | Key Achievement |
|---------|------|-----------------|
| Genie 1 | ICML 2024 (Best Paper) | 11B foundation world model from unlabeled video; latent action discovery |
| Genie 2 | Dec 2024 (blog) | Playable 3D environments from single image |
| Genie 3 | Aug 2025 (blog) | Real-time 24fps, 720p interactive 3D, consumer product |
- **Limitation:** No public papers for v2/v3; game-like environments only; consistency degrades after minutes.

### COSMOS / Cosmos-Predict2.5 (NVIDIA, 2025–2026)
- **arXiv:** 2501.03575 (platform), 2511.00062 (v2.5)
- **Contribution:** Open platform for Physical AI world models. v2.5 unifies Text/Image/Video2World at 2B-14B scale. Transfer2.5 adds sim-to-real translation.
- **Limitation:** Primarily video generation; action-conditioned prediction for closed-loop robotics still developing.

### V-JEPA 2 / V-JEPA 2.1 (Meta FAIR, 2025–2026)
- **arXiv:** 2506.09985
- **Contribution:** Self-supervised video world model pre-trained on 1M+ hours. V-JEPA 2-AC enables zero-shot robot control with only 62 hours of robot data. V-JEPA 2.1: 10× faster planning (10.6s → still not real-time).
- **Limitation:** Planning at 10.6s not suitable for fast manipulation; latent predictions hard to interpret.

---

## 3. Recent World Models for Robotics (Jan–Mar 2026)

### PointWorld (Huang et al., Jan 2026)
- **arXiv:** 2601.03782 | Stanford/NVIDIA
- **Contribution:** 3D world model representing state/action as 3D point flows. Trained on ~2M trajectories (500 hours). 0.1s inference with MPC.
- **Limitation:** Requires RGB-D input; generalization to deformable objects untested.

### DreamDojo (Gao et al., Feb 2026)
- **arXiv:** 2602.06949 | NVIDIA/Berkeley
- **Contribution:** Foundation world model pre-trained on 44k hours of egocentric human video. Continuous latent actions bridge unlabeled video and robot control.
- **Limitation:** Human-to-robot domain gap; 10.81 FPS may be insufficient for high-frequency control.

### MVISTA-4D (Wang et al., Feb 2026)
- **arXiv:** 2602.09878
- **Contribution:** 4D world model generating geometrically consistent multi-view RGBD from single view. Test-time action optimization via backprop through the generative model.
- **Limitation:** Computational cost of test-time optimization; scaling to cluttered scenes.

### RISE (Yang et al., Feb 2026)
- **arXiv:** 2602.11075
- **Contribution:** Compositional world model (separate dynamics + value) for policy self-improvement via imagination. +35–45% absolute on real-world tasks.
- **Limitation:** Compounding errors in long-horizon imagination.

### StructVLA (Jin et al., Mar 2026)
- **arXiv:** 2603.12553
- **Contribution:** World model as structured planner — predicts sparse kinematic keyframes instead of dense video. 75.0% SimplerEnv-WidowX, 94.8% LIBERO.
- **Limitation:** Relies on gripper-transition structure; may not generalize to non-gripper end-effectors.

### RoboStereo (Zhang et al., Mar 2026)
- **arXiv:** 2603.12639
- **Contribution:** Dual-tower 4D world model with bidirectional cross-modal enhancement. Three policy optimization modes (test-time augmentation, imitative-evolutionary, open-exploration).
- **Limitation:** Very high reported improvements (97%+) need independent verification.

### ABot-PhysWorld (Chen et al., Mar 2026)
- **arXiv:** 2603.23376
- **Contribution:** 14B DiT for manipulation video with DPO-based physics alignment. Introduces EZSbench (training-independent zero-shot benchmark). Outperforms Veo 3.1 and Sora v2 Pro.
- **Limitation:** 14B model is expensive; DPO may not capture all physical constraints.

### MIND-V (Zhang et al., Dec 2025 → Mar 2026 revision)
- **arXiv:** 2512.06628
- **Contribution:** Hierarchical world model: semantic reasoning (VLM) → behavioral translator → video generator. RL post-training using V-JEPA2 as physics validator.
- **Limitation:** Relies on V-JEPA2's physics understanding; degrades on very complex multi-step tasks.

### Ctrl-World (Guo et al., Oct 2025 → Mar 2026 revision)
- **arXiv:** 2510.10125 | Stanford (Chelsea Finn)
- **Contribution:** Controllable multi-view world model for policy-in-the-loop interaction. Pose-conditioned memory for 20+ second consistent trajectories. +44.7% policy improvement via imagined fine-tuning.
- **Limitation:** DROID data covers primarily tabletop manipulation.

---

## 4. Vision-Language-Action (VLA) Models

### π0 / π0.5 / π*0.6 (Physical Intelligence, 2024–2025)
| Model | arXiv | Key Advance |
|-------|-------|-------------|
| π0 | 2410.24164 | Flow matching VLA on pre-trained VLM; 50 Hz continuous actions |
| π0.5 | 2504.16054 | Open-world generalization via multi-modal co-training |
| π*0.6 | 2511.14759 | RECAP: RL self-improvement for VLAs from deployment data |

### OpenVLA / OpenVLA-OFT (Stanford/Berkeley, 2024–2025)
- **arXiv:** 2406.09246 / 2502.19645
- **Contribution:** Open-source 7B VLA outperforming RT-2-X. OFT adds parallel decoding + action chunking → LIBERO 76.5% → 97.1%, 26× faster.

### GR00T N1 (NVIDIA, Mar 2025)
- **arXiv:** 2503.14734
- **Contribution:** Dual-system VLA for humanoid robots. VLM (System 2) + diffusion transformer (System 1). 2.2B params, 63.9ms inference.

### X-VLA (ICLR 2026)
- **arXiv:** 2510.10274
- **Contribution:** Soft prompts for cross-embodiment flow-matching VLA. 0.9B params, SOTA cross-embodiment. IROS 2025 AgiBot Challenge winner.

### Octo (Berkeley, RSS 2024)
- **arXiv:** 2405.12213
- **Contribution:** Generalist diffusion policy on 800K OXE trajectories. 27M–93M params. Fine-tunable to 9 robot platforms.

### HPT (MIT/Meta, NeurIPS 2024)
- **arXiv:** 2409.20537
- **Contribution:** Stem-trunk-head architecture for heterogeneous pre-training. 50+ datasets, 200K+ trajectories.

### VLA-JEPA (Sun et al., Feb 2026)
- **arXiv:** 2602.10098
- **Contribution:** JEPA-style pretraining for VLAs. Leakage-free latent state prediction from video.

### EgoScale (Zheng et al., Feb 2026)
- **arXiv:** 2602.16710
- **Contribution:** Log-linear scaling law between egocentric video volume and VLA loss. 20,854 hours of action-labeled egocentric video.

### SimpleVLA-RL (ICLR 2026)
- **arXiv:** 2509.09674
- **Contribution:** Efficient RL framework for VLAs. LIBERO-Long: 17.1% → 91.7% with single demo per task. Discovers novel "pushcut" phenomenon.

---

## 5. Video Diffusion for Robotics

### DiT4DiT (Mar 2026)
- **arXiv:** 2603.10448
- **Contribution:** Cascaded Video DiT → Action DiT. 98.6% LIBERO, 10×+ sample efficiency. Built on Cosmos-Predict2.5-2B.

### UWM — Unified World Models (Apr 2025)
- **arXiv:** 2504.02792
- **Contribution:** Multimodal diffusion transformer unifying video + action diffusion with independent per-modality timesteps.

### mimic-video (Dec 2025)
- **arXiv:** 2512.15692
- **Contribution:** Video-Action Models (VAMs) with pretrained video model + flow matching IDM. 10× sample efficiency, 2× convergence speed over VLAs.

### Vidarc (Dec 2025)
- **arXiv:** 2512.17661
- **Contribution:** Autoregressive video diffusion with KV caching. 91% latency reduction. Action-relevant masking for embodiment-critical regions.

### VPP — Video Prediction Policy (Dec 2024, ICML 2025 Spotlight)
- **arXiv:** 2412.14803
- **Contribution:** Implicit inverse dynamics from video diffusion internal features. +18.6% on CALVIN ABC-D.

### Vid2World (May 2025)
- **arXiv:** 2505.14357
- **Contribution:** General approach for converting pretrained video diffusion models to interactive world models via causalization.

### ViPRA (Nov 2025, NeurIPS 2025)
- **arXiv:** 2511.07732 | CMU
- **Contribution:** Motion-centric latent actions from actionless video via perceptual + optical flow losses. 22 Hz, 100–200 demos.

---

## 6. Evaluation & Benchmarks

### WoW-World-Eval (Fan et al., Jan 2026)
- **arXiv:** 2601.04137
- **Contribution:** 609-example Turing test for video foundation models as world models. 22 metrics. Most models score 17.27 on extended planning; real-world success near 0%.

### LIBERO-PRO (Oct 2025)
- **arXiv:** 2510.03827
- **Contribution:** Exposed brittleness of OpenVLA, π0, π0.5 under viewpoint, initial state, and language perturbations.

### EZSbench (Mar 2026, part of ABot-PhysWorld)
- First training-independent embodied zero-shot benchmark.

---

## 7. Key Trends

1. **2D video → 3D/4D world models:** PointWorld, MVISTA-4D, RoboStereo, 4D Latent WM move beyond flat video.
2. **Physics alignment as first-class concern:** ABot-PhysWorld (DPO), MIND-V (V-JEPA2 validator), RoboScape (depth+keypoints).
3. **Imagination-based policy improvement:** RISE, Ctrl-World, DreamDojo use world models for data-free policy optimization.
4. **Scaling from human video:** DreamDojo (44k hrs), V-JEPA 2 (1M+ hrs), EgoScale (20k hrs) demonstrate transfer.
5. **Flow matching > autoregressive tokens** for continuous robot actions (π0, X-VLA, UWM).
6. **RL self-improvement of VLAs:** π*0.6, SimpleVLA-RL, VLA-RL show early inference-time scaling signals.
7. **Video diffusion models encode policies:** DiT4DiT, mimic-video, VPP establish video models as rich physical priors.
8. **Sparse/structured prediction:** StructVLA predicts keyframes, not dense video — faster and often better.
9. **Real-time gap persists:** V-JEPA 2.1 at 10.6s planning; most video world models cannot close the loop at manipulation frequencies (10–50 Hz).
