# Daily Paper — March 28, 2026

Papers from the last 3 days (March 25-28) in robot learning, world models, VLA, embodied AI, **LLM agents, and reasoning**.

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

---
---

# Part 2: LLM Agents & Reasoning

## Highlights

- **ARC-AGI-3** (2603.24621) — humans 100%, best frontier model (Gemini 3.1 Pro) 0.37%, GPT-5.4 0.26%
- **HyperAgents** (2603.19461) — self-referential agents with metacognitive self-improvement
- **AVO** (2603.24517) — coding agents evolve attention kernels beating cuDNN by 3.5% and FlashAttention-4 by 10.5%
- **Internal Safety Collapse** (2603.23509) — 95.3% safety failure rate across GPT-5.2, Claude Sonnet 4.5
- **Mamba-3** (2603.15569) — major SSM architecture advance
- **PRM Hackability** (2603.06621) — process reward models are exploitable fluency detectors, not reasoning verifiers

**Trends this week:**
1. **Self-improving/self-evolving agents** (HyperAgents, Memento-Skills, MetaClaw, AgentFactory)
2. **RL infrastructure for agents** going production-ready (ProRL, OpenClaw-RL, PivotRL)
3. **Multi-tool orchestration** replacing single tool calls (ToolTree, graduated rewards)
4. **Memory coordination** across agents (MemMA, MemCollab, MSA 100M tokens)
5. **Safety alarms** — Internal Safety Collapse, PRM hackability, agent attack surfaces

---

## LLM Agent Frameworks & Self-Improvement

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.19461](https://arxiv.org/abs/2603.19461) | **HyperAgents: Self-Referential Agents** | Meta-agent edits task-agent AND itself; outperforms AI-Scientist-v2 | High |
| [2603.18743](https://arxiv.org/abs/2603.18743) | **Memento-Skills: Let Agents Design Agents** | Autonomous agent design via memory-based RL; +116% on Humanity's Last Exam | High |
| [2603.17187](https://arxiv.org/abs/2603.17187) | **MetaClaw: Meta-Learning in the Wild** | Continual meta-learning evolving base policy + skills; +32% accuracy | High |
| [2603.18000](https://arxiv.org/abs/2603.18000) | **AgentFactory: Self-Evolving via Subagent Reuse** | Preserves solutions as executable code; -57% orchestration cost | High |
| [2603.24639](https://arxiv.org/abs/2603.24639) | **Experiential Reflective Learning** | Self-improvement via heuristic generation; +7.8% on Gaia2 | High |
| [2603.08127](https://arxiv.org/abs/2603.08127) | **EvoScientist: Evolving AI Scientists** | Multi-agent AI scientist with persistent memory; SOTA novelty | High |

---

## Tool Use & Orchestration

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.22862](https://arxiv.org/abs/2603.22862) | **Evolution of Tool Use in LLM Agents (Survey)** | From single-tool to multi-tool orchestration | High |
| [2603.24709](https://arxiv.org/abs/2603.24709) | **Multi-Step Tool Orchestration with RL** | Graduated rewards + real API cache; gains on ComplexFuncBench | High |
| [2603.19896](https://arxiv.org/abs/2603.19896) | **Utility-Guided Agent Orchestration** | Quality-cost trade-off via utility estimation | High |
| [2603.12740](https://arxiv.org/abs/2603.12740) | **ToolTree: MCTS for Tool Planning** | Dual-feedback MCTS with bidirectional pruning | High |

---

## Benchmarks

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.24621](https://arxiv.org/abs/2603.24621) | **ARC-AGI-3** | Humans 100%, Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Opus 4.6 0.25% | High |
| [2603.24755](https://arxiv.org/abs/2603.24755) | **SlopCodeBench: Coding Agent Degradation** | Measures verbosity/structural erosion over long sessions | High |
| [2603.24943](https://arxiv.org/abs/2603.24943) | **FinMCP-Bench: Financial Tool Use** | 613 samples, 65 real financial MCPs | High |
| [2603.01152](https://arxiv.org/abs/2603.01152) | **DeepResearch-9K** | 9K challenging deep research tasks | High |
| [2603.02297](https://arxiv.org/abs/2603.02297) | **ZeroDayBench: 0-Day CVE Detection** | LLM agents find/patch 22 novel CVEs (ICLR 2026) | High |
| [2602.24173](https://arxiv.org/abs/2602.24173) | **LemmaBench: Research-Level Math** | 358 lemmas; GPT-5 only 15% correct | High |

---

## Multi-Agent Systems

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.15371](https://arxiv.org/abs/2603.15371) | **BIGMAS: Brain-Inspired Graph Multi-Agent** | Global workspace theory; improves reasoning across 6 LLMs | High |
| [2603.04474](https://arxiv.org/abs/2603.04474) | **Error Cascades in Multi-Agent Collaboration** | Models error propagation + mitigation | High |
| [2603.22651](https://arxiv.org/abs/2603.22651) | **Multi-Agent for Financial Docs** | Sequential vs parallel vs hierarchical vs reflexive | High |
| [2603.23875](https://arxiv.org/abs/2603.23875) | **SEMA: Self-Evolving Multi-Agent RTS** | Low-latency multi-agent decision-making | High |

---

## Memory Systems

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.18718](https://arxiv.org/abs/2603.18718) | **MemMA: Memory Cycle Coordination** | Forward + backward memory paths with self-repair | High |
| [2603.23234](https://arxiv.org/abs/2603.23234) | **MemCollab: Cross-Agent Memory** | Contrastive trajectory distillation for shared memory | High |
| [2603.23516](https://arxiv.org/abs/2603.23516) | **MSA: Memory Sparse Attention** | Scales memory models to 100M tokens | High |

---

## Reasoning & Chain-of-Thought

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.01326](https://arxiv.org/abs/2603.01326) | **Truth as a Trajectory** | Geometric invariants distinguish valid vs spurious reasoning | High |
| [2603.21301](https://arxiv.org/abs/2603.21301) | **Enhancing Reasoning at Inference Time** | Self-consistency + nucleus sampling: +9-15% accuracy | High |
| [2603.22492](https://arxiv.org/abs/2603.22492) | **Tiny Inference-Time Scaling** | -63.3% time, -51% FLOPs via latent verifiers | High |
| [2603.21162](https://arxiv.org/abs/2603.21162) | **ReSCALE: Fix MCTS for LLMs** | Sequential Halving + Gumbel restores monotonic scaling | High |
| [2602.13517](https://arxiv.org/abs/2602.13517) | **Deep-Thinking Tokens** | Identifies tokens where predictions undergo significant revision | High |
| [2603.00306](https://arxiv.org/abs/2603.00306) | **When Does CoT Help: Markovian Perspective** | Theoretical analysis of CoT effectiveness | High |

---

## RL for LLMs

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.18815](https://arxiv.org/abs/2603.18815) | **ProRL Agent (NVIDIA)** | Rollout-as-a-Service for multi-turn RL; NeMo Gym | High |
| [2603.10165](https://arxiv.org/abs/2603.10165) | **OpenClaw-RL: Train by Talking** | Every next-state signal as RL reward; zero coordination | High |
| [2603.12109](https://arxiv.org/abs/2603.12109) | **Information Self-Locking in RL** | Agents stuck in low-info patterns; +60% with fix | High |
| [2603.22446](https://arxiv.org/abs/2603.22446) | **Sparse but Critical RLVR Tokens** | Only few token distributions change; inserting them recovers RL gains (ICLR 2026) | High |
| [2603.10535](https://arxiv.org/abs/2603.10535) | **GR3: Fix Length Inflation** | Multiplicative rescaling controls verbosity | High |
| [2603.09117](https://arxiv.org/abs/2603.09117) | **DCPO: Fix Calibration Degeneration** | Decouples reasoning and confidence in RLVR | High |
| [2603.06621](https://arxiv.org/abs/2603.06621) | **PRM Hackability** | PRMs are exploitable fluency detectors, not reasoning verifiers | High |

---

## Code Agents

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.24517](https://arxiv.org/abs/2603.24517) | **AVO: Agentic Variation Operators** | 7 days autonomous evolution; beats cuDNN 3.5%, FlashAttn-4 10.5% | High |
| [2603.24755](https://arxiv.org/abs/2603.24755) | **SlopCodeBench** | Measures coding agent degradation over long sessions | High |

---

## Safety & Security

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.23509](https://arxiv.org/abs/2603.23509) | **Internal Safety Collapse** | 95.3% safety failure rate across frontier LLMs | High |
| [2603.11088](https://arxiv.org/abs/2603.11088) | **Attack & Defense Landscape of Agentic AI** | First systematic agent security survey (Berkeley/UIUC) | High |
| [2603.24857](https://arxiv.org/abs/2603.24857) | **AI Security in FM Era** | Unified threat taxonomy for foundation models | High |

---

## Architecture & Efficiency

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.15569](https://arxiv.org/abs/2603.15569) | **Mamba-3** | Improved SSM: complex dynamics, MIMO, better scaling | High |
| [2603.11021](https://arxiv.org/abs/2603.11021) | **LLVQ: Leech Lattice Quantization** | Optimal 24D sphere packing for SOTA LLM compression | High |
| [2603.19133](https://arxiv.org/abs/2603.19133) | **PicoSpec: Edge-Cloud Speculative Decoding** | Training-free; 2.9x speedup | High |
| [2603.25040](https://arxiv.org/abs/2603.25040) | **Intern-S1-Pro: 1T Scientific Model** | 512 experts, 22B activated, 100+ scientific tasks | High |

---

## Major Lab Updates (as of March 28, 2026)

| Lab | Update |
|-----|--------|
| OpenAI | GPT-5.4 launched Mar 11 (1.05M input, 58.7% SWE-bench). "Spud" finished pre-training ~Mar 24. |
| Anthropic | Claude Opus 4.6 + Sonnet 4.6; 1M context; 80.8% SWE-Bench |
| Google DeepMind | Gemini 2.5 Pro (late Feb); Gemini 3.1 Pro scores highest on ARC-AGI-3 (0.37%) |
| Meta | Llama 4 Maverick (open-source, agentic); HyperAgents framework |
| NVIDIA | ProRL Agent for multi-turn RL training |
