# Daily Paper — March 29, 2026

Papers from the last 2 days (March 28-29). Covers VLA/World Model/Robotics AND LLM/VLM/Agent.

---

## Major Announcement

**Google Gemini 3 Deep Think** released March 28 — gold-medal IPhO/IChO, 50.5% on CMT-Benchmark (theoretical physics), identified a flaw in a peer-reviewed math paper. Available via API.

---

# Part 1: VLA / World Model / Robotics

## Highlights

- **Fast-WAM** (2603.16666) — surprising finding: test-time video imagination is NOT needed, training-time video modeling provides the key benefit
- **LeWorldModel** (2603.19312) — stable end-to-end JEPA world model from raw pixels
- **Persistent Robot WMs** (2603.25685) — RL post-training stabilizes multi-step rollouts
- **MMaDA-VLA** (2603.25406) — 98.0% LIBERO via unified discrete diffusion

## VLA Models

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25406](https://arxiv.org/abs/2603.25406) | **MMaDA-VLA** | Unified discrete diffusion for language+vision+action; 98.0% LIBERO | High |
| [2603.25661](https://arxiv.org/abs/2603.25661) | **Fast-dVLA** | Parameter decoupling for real-time discrete diffusion VLA | High |
| [2603.25038](https://arxiv.org/abs/2603.25038) | **AirVLA: Pi But Make It Fly** | VLA transfer to aerial manipulation; 62% compositional success | High |
| [2603.25481](https://arxiv.org/abs/2603.25481) | **LILAC** | Language-conditioned object-centric optical flow for 6-DoF | High |
| [2603.24806](https://arxiv.org/abs/2603.24806) | **FODMP** | One-step diffusion movement primitives; 10x faster | High |
| [2603.22003](https://arxiv.org/abs/2603.22003) | **VP-VLA** | Visual prompting (crosshairs, bboxes) as VLA interface | High |
| [2603.22280](https://arxiv.org/abs/2603.22280) | **DualCoT-VLA** | Parallel visual-linguistic chain-of-thought | High |
| [2603.19199](https://arxiv.org/abs/2603.19199) | **FASTER** | 10x over Pi0.5 via horizon-aware scheduling | High |
| [2603.14523](https://arxiv.org/abs/2603.14523) | **VLA-Thinker** | Thinking-with-image reasoning; 97.5% LIBERO | High |
| [2603.25044](https://arxiv.org/abs/2603.25044) | **ThermoAct** | Thermal-aware VLA for safety | Med |
| [2603.24935](https://arxiv.org/abs/2603.24935) | **SABER** | Black-box adversarial attacks on VLA | Med |
| [2603.24941](https://arxiv.org/abs/2603.24941) | **TIES** | Token rank consistency; +6% success, -78% tokens | Med |

## World Models

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25685](https://arxiv.org/abs/2603.25685) | **Persistent Robot WMs** | RL stabilizes autoregressive rollouts; 14% LPIPS improvement | High |
| [2603.23376](https://arxiv.org/abs/2603.23376) | **ABot-PhysWorld** | 14B DiT with DPO physics alignment; beats Veo 3.1 | High |
| [2603.17808](https://arxiv.org/abs/2603.17808) | **EVA** | Inverse dynamics rewards for executable WM alignment | High |
| [2603.16669](https://arxiv.org/abs/2603.16669) | **Kinema4D** | URDF-based 4D kinematic world simulation | High |
| [2603.16666](https://arxiv.org/abs/2603.16666) | **Fast-WAM** | Test-time imagination NOT needed; training-time video modeling is key | High |
| [2603.19312](https://arxiv.org/abs/2603.19312) | **LeWorldModel** | Stable end-to-end JEPA from pixels; encodes physical structures | High |
| [2603.03195](https://arxiv.org/abs/2603.03195) | **Chain of World** | Latent motion WM thinking; outperforms baselines | High |

## Manipulation & Diffusion Policy

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25725](https://arxiv.org/abs/2603.25725) | **SoftMimicGen** | Automated deformable object manipulation data gen | High |
| [2603.25583](https://arxiv.org/abs/2603.25583) | **F-ACIL** | Factorized compositional IL; +45% with 5-10x fewer demos | High |
| [2603.22263](https://arxiv.org/abs/2603.22263) | **DexDrummer** | Bimanual drumming with sim-to-real dexterous control | High |
| [2603.16065](https://arxiv.org/abs/2603.16065) | **Large Reward Models** | Zero-shot VLM rewards for online RL | High |
| [2603.05117](https://arxiv.org/abs/2603.05117) | **SeedPolicy** | Self-evolving diffusion policy; +169% on RoboTwin | High |
| [2603.08546](https://arxiv.org/abs/2603.08546) | **Interactive World Simulator** | Consistency models; 10+ min stable at 15 FPS | High |
| [2603.10340](https://arxiv.org/abs/2603.10340) | **CGVD** | Suppress visual clutter in VLA; 77.5% vs 43.0% | High |

---

# Part 2: LLM / VLM / Agent

## Highlights

- **Gemini 3 Deep Think** — gold-medal physics/chemistry olympiad, found flaw in peer-reviewed math paper
- **Qwen3-Coder-Next** (2603.00729) — 80B MoE (3B active) open-weight coding agent with agentic RL
- **Intern-S1-Pro** (2603.25040) — first 1T-param scientific multimodal model (512 experts)
- **MSA** (2603.23516) — memory sparse attention scaling to 100M tokens, linear complexity
- **ARRoL** (2603.24840) — online rollout pruning for GRPO; +2.99 accuracy, 1.7x speedup
- **Multi-Agent Scaling Laws** (2603.24676) — derives scaling laws for LLM population consensus

## Agent Frameworks & Self-Improvement

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.24639](https://arxiv.org/abs/2603.24639) | **ERL: Experiential Reflective Learning** | Heuristic generation from trajectories; +7.8% on Gaia2 | High |
| [2603.25681](https://arxiv.org/abs/2603.25681) | **Self-Improvement Overview** | Comprehensive lifecycle: data acq → selection → optim → inference | High |
| [2603.25158](https://arxiv.org/abs/2603.25158) | **Trace2Skill** | Distill trajectory lessons into transferable skills; +57.65pp cross-model | High |
| [2603.00026](https://arxiv.org/abs/2603.00026) | **ActMem** | Bridges memory retrieval and reasoning | High |
| [2603.00030](https://arxiv.org/abs/2603.00030) | **SimpleTool** | Parallel decoding for real-time function calling | High |
| [2603.00718](https://arxiv.org/abs/2603.00718) | **SkillCraft** | Can LLM agents learn skillful tool use? | High |
| [2603.00829](https://arxiv.org/abs/2603.00829) | **Constitutional Monitoring** | Black-box monitoring for agent scheming | High |

## Multi-Agent Systems

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25268](https://arxiv.org/abs/2603.25268) | **CRAFT** | Multi-agent 3D construction; reasoning ≠ better teamwork | High |
| [2603.24676](https://arxiv.org/abs/2603.24676) | **Multi-Agent Scaling Laws** | Memetic drift scaling laws for LLM population consensus | High |
| [2603.25001](https://arxiv.org/abs/2603.25001) | **MP-Bench** | Rethinking failure attribution in multi-agent systems | High |
| [2603.00623](https://arxiv.org/abs/2603.00623) | **TraceSIR** | Structured analysis of execution traces | Med |

## Reasoning & Inference Scaling

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.24840](https://arxiv.org/abs/2603.24840) | **ARRoL** | Online rollout pruning for GRPO/DAPO; +2.99, 1.7x speedup | High |
| [2603.00578](https://arxiv.org/abs/2603.00578) | **Draft-Thinking** | Efficient long CoT reasoning | High |
| [2603.00296](https://arxiv.org/abs/2603.00296) | **Stepwise Penalization** | Length-efficient CoT without accuracy loss | High |
| [2603.22492](https://arxiv.org/abs/2603.22492) | **Tiny Inference-Time Scaling** | Latent verifiers: -63% time, -51% FLOPs | High |
| [2603.00306](https://arxiv.org/abs/2603.00306) | **When Does CoT Help** | Markovian theoretical analysis | High |
| [2603.01070](https://arxiv.org/abs/2603.01070) | **RL Unlocks Aha Moment** | RL for geometric interleaved reasoning | Med |

## Memory Systems

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.23516](https://arxiv.org/abs/2603.23516) | **MSA: 100M Token Memory** | Linear-complexity sparse attention; surpasses frontier LLMs+RAG | High |
| [2603.19935](https://arxiv.org/abs/2603.19935) | **Memori** | 81.95% accuracy using only 5% of context; 20x cost savings | High |
| [2603.18718](https://arxiv.org/abs/2603.18718) | **MemMA** | Multi-agent memory with self-evolving construction | High |
| [2603.00680](https://arxiv.org/abs/2603.00680) | **MemPO** | Self-memory policy optimization for long-horizon | Med |

## Code Agents

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.00729](https://arxiv.org/abs/2603.00729) | **Qwen3-Coder-Next** | 80B MoE (3B active) coding agent; agentic RL; open-weight | High |
| [2603.00575](https://arxiv.org/abs/2603.00575) | **SWE-Hub** | Unified production system for SE tasks | High |
| [2603.24517](https://arxiv.org/abs/2603.24517) | **AVO** | Agent-evolved attention beats cuDNN 3.5%, FlashAttn-4 10.5% | High |

## VLM & Scientific Models

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25040](https://arxiv.org/abs/2603.25040) | **Intern-S1-Pro** | 1T-param, 512 experts, 22B active; 100+ scientific tasks | High |
| [2603.00842](https://arxiv.org/abs/2603.00842) | **MedGPT-oss** | General-purpose VLM for biomedicine | High |
| [2603.25075](https://arxiv.org/abs/2603.25075) | **Sparse Visual Thought Circuits** | SAE composability breaks in VLMs; mid-decoder task concentration | Med |
| [2603.24866](https://arxiv.org/abs/2603.24866) | **Physical Generative Reasoning** | Benchmark for VLM physical world construction | Med |

## RL for LLMs / Alignment

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.00025](https://arxiv.org/abs/2603.00025) | **TAB-PO** | Token-level adaptive barrier for preference optimization | High |
| [2603.25201](https://arxiv.org/abs/2603.25201) | **SafeMath** | Safety alignment IMPROVES math accuracy (not hurts) | High |
| [2603.25412](https://arxiv.org/abs/2603.25412) | **Reasoning Safety Monitor** | 9 unsafe reasoning types; ~85% detection accuracy | High |
| [2603.25326](https://arxiv.org/abs/2603.25326) | **LLM Manipulation** | 10K-participant study: models produce manipulative behaviors | High |

## Efficiency & Architecture

| arXiv | Title | Summary | Rel |
|-------|-------|---------|:---:|
| [2603.25702](https://arxiv.org/abs/2603.25702) | **S2D2** | Training-free self-speculative decoding for diffusion LLMs | High |
| [2603.25284](https://arxiv.org/abs/2603.25284) | **SliderQuant** | Adaptive sliding quantization for Llama/Qwen/DeepSeek-R1 | High |
| [2603.00040](https://arxiv.org/abs/2603.00040) | **Attn-QAT** | 4-bit attention quantization-aware training | High |
| [2603.00042](https://arxiv.org/abs/2603.00042) | **Sub-1-Bit LLMs** | Extreme compression via latent geometry alignment | Med |
