# Gap Analysis: World Models for Robotics

*Based on landscape survey of 50+ papers, March 2026*

---

## Gap 1: No Real-Time Closed-Loop World Model for Manipulation

**Evidence:**
- V-JEPA 2.1 reduced planning from 100+ seconds to 10.6 seconds — still 100–500× too slow for 10–50 Hz control.
- PointWorld achieves 0.1s inference but only for 3D point flow predictions, not full scene dynamics.
- Vidarc achieved 91% latency reduction but still cannot match real-time control rates for dexterous tasks.
- Video diffusion models (DiT4DiT, UWM, VPP) require multiple denoising steps per frame.

**Why it matters:** Closed-loop control with world model feedback is essential for contact-rich manipulation where open-loop execution fails. The field has fast policies (π0 at 50 Hz) and accurate-but-slow world models — bridging this gap would unlock model-predictive control at scale.

**Contradiction:** DIAMOND argues pixel-space fidelity matters for world models (better RL performance), but pixel-space diffusion is inherently slow. Latent world models (Dreamer, TD-MPC2) are fast but lose visual detail. The field hasn't resolved whether speed or fidelity should be prioritized for manipulation.

---

## Gap 2: World Models Are Single-Embodiment

**Evidence:**
- Every major world model paper trains and evaluates on a single robot: PointWorld (Franka), Ctrl-World (Franka/DROID), DreamDojo (Franka+humanoid but separate), RISE (single arm), TD-MPC2 (DM Control).
- VLA/policy papers DO cross-embodiment (Octo: 9 platforms, X-VLA: 6 sims + 3 real, OpenVLA: OXE multi-robot) — but the world models they use are not shared across embodiments.
- No paper has demonstrated a single world model that transfers dynamics predictions across robot morphologies.

**Why it matters:** Environment dynamics (gravity, friction, object interactions) are embodiment-independent. A world model that captures these universal physics should transfer — only the robot-specific kinematics and contact geometry should need adaptation. This would dramatically reduce the cost of deploying world models to new platforms.

---

## Gap 3: Physics Hallucination Remains Unsolved

**Evidence:**
- WoW-World-Eval (Jan 2026): models score only 68.02/100 on physical consistency; real-world success near 0%.
- ABot-PhysWorld uses DPO post-training to suppress impossible physics — a patch, not a solution.
- MIND-V uses V-JEPA2 as external physics validator — outsourcing the problem rather than solving it.
- RoboScape uses depth + keypoints as physics proxies — helpful but doesn't capture forces/contact.
- Video Generation Models survey (Jan 2026) identifies physics hallucination as a fundamental unsolved challenge.

**Why it matters:** A world model that predicts physically impossible outcomes (object penetration, gravity violations, impossible contact) cannot be trusted for planning. This is the #1 barrier to deploying video world models for robot decision-making.

**Underexplored approach:** No paper directly constrains latent dynamics to follow Newtonian physics via differentiable physics losses. ABot-PhysWorld's DPO and MIND-V's validator are post-hoc — the dynamics representation itself doesn't encode physical laws.

---

## Gap 4: No Compositional Generalization in World Models

**Evidence:**
- All world models are trained and tested on the same object/scene distributions. No paper demonstrates generalization to novel object combinations, novel arrangements, or novel material properties.
- LIBERO-PRO (Oct 2025) showed VLAs collapse under even mild distribution shift (viewpoint, language).
- Object-centric world models exist (SlotFormer, C-SWM) but are limited to toy scenes with simple shapes.
- No modern world model uses object-centric representations with realistic visual backbones.

**Why it matters:** Real manipulation requires handling novel objects and arrangements. A world model that only predicts dynamics for objects it was trained on has limited practical value. Compositional generalization — predicting dynamics for novel combinations of known components — is essential.

---

## Gap 5: Inference-Time Scaling Laws Don't Exist for Robotics

**Evidence:**
- LLMs show clear inference-time scaling (chain-of-thought, tree search, reasoning tokens).
- VLA-RL reports "early signal" of inference scaling in robotics but it's preliminary.
- SimpleVLA-RL hints at inference-time benefits but doesn't characterize a scaling law.
- No paper has systematically studied how more test-time compute (via world model simulation) improves manipulation performance.

**Why it matters:** If a reliable inference-time scaling law exists for robotics, it would transform the field: instead of training ever-larger policies, you could use moderate-sized policies with world model "thinking." This is the most impactful open question at the intersection of scaling laws and embodied AI.

---

## Gap 6: Counterfactual Reasoning Is Missing from Robot World Models

**Evidence:**
- All world models predict FORWARD: given current state + action → future state.
- No world model supports COUNTERFACTUAL reasoning: "what would have happened if I had taken a different action at timestep t?"
- Offline RL methods (CQL, IQL, Decision Transformer) work with fixed trajectories. World models could generate synthetic counterfactual trajectories for exponentially more training signal.
- Hindsight Experience Replay is a primitive form of this — but it doesn't use a world model to generate physically plausible alternatives.

**Why it matters:** Failures are highly informative — but only if you can reason about what went wrong and what would have worked instead. Counterfactual world models could turn every failed trajectory into dozens of successful synthetic trajectories.

---

## Gap 7: World Models Don't Leverage Language for Physical Understanding

**Evidence:**
- VLAs use language for task specification ("pick up the red cup").
- World models either have no language input (Dreamer, TD-MPC) or use language only for visual generation conditioning (COSMOS, Genie).
- No world model uses language descriptions of PHYSICAL PROPERTIES ("this object is heavy and fragile", "the surface is slippery") to modulate dynamics predictions.
- LLMs encode substantial physical commonsense, but world models don't access this knowledge.

**Why it matters:** Language could serve as a powerful interface for injecting physical priors into world models — both from LLMs and from human instruction. A world model that understands "glass breaks when dropped" and "ice is slippery" could generalize to novel scenarios described in language.

---

## Gap 8: Temporal Abstraction in World Models Is Primitive

**Evidence:**
- Most world models predict at fixed temporal resolution (every frame or every timestep).
- StructVLA predicts gripper-transition keyframes — a step toward temporal abstraction but hardcoded to gripper events.
- MIND-V has a hierarchy but the temporal levels are manually designed.
- No world model LEARNS the appropriate temporal abstraction level for different task phases.

**Why it matters:** Transit motions (moving arm through free space) need less temporal resolution than contact events (grasping, inserting). Adaptive temporal resolution would:
- Enable long-horizon prediction (predict in coarse steps, refine at contacts)
- Reduce compute (skip uninteresting timesteps)
- Better handle the multi-scale nature of manipulation tasks

---

## Gap 9: World Models as Data Engines — Targeted Not Random

**Evidence:**
- Data augmentation for robot learning is random: random crops, color jitter, camera randomization.
- RoboTransfer generates diverse synthetic data but without targeting specific gaps in the training distribution.
- No paper uses a world model to IDENTIFY what data is missing and GENERATE targeted demonstrations to fill those gaps.
- Policy learning is known to fail at distribution boundaries — states encountered during deployment but absent from training.

**Why it matters:** Targeted data generation (as opposed to random augmentation) could solve the long tail of manipulation failures. A world model that identifies "the policy has never seen this type of grasp approach angle" and generates training data for it would be fundamentally more sample-efficient.

---

## Summary: Most Promising Research Directions

| Priority | Gap | Estimated Impact | Feasibility |
|----------|-----|------------------|-------------|
| 1 | Inference-time scaling laws | Field-defining | High (sim-only) |
| 2 | Language-grounded physical understanding | Novel paradigm | High (sim-only) |
| 3 | Cross-embodiment world models | Major practical impact | Medium |
| 4 | Counterfactual reasoning | 10× sample efficiency | Medium |
| 5 | Physics-constrained dynamics | Trust/deployment | Medium |
| 6 | Compositional generalization | Fundamental capability | Medium |
| 7 | Targeted data generation | Practical scaling | High |
| 8 | Temporal abstraction | Long-horizon enabling | Medium |
| 9 | Real-time inference | Deployment critical | Hard |
