# Diffusion Language Models: Comprehensive Research Landscape (2025--2026)

*Generated 2026-03-31. Covers arXiv, ICLR 2025/2026, NeurIPS 2025, ICML 2025, ACL 2025, NAACL 2025.*

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Taxonomy of Approaches](#taxonomy-of-approaches)
3. [Paper-by-Paper Catalog](#paper-by-paper-catalog)
4. [Saturation Map: What's Done](#saturation-map)
5. [Gap Analysis: What's Unsolved](#gap-analysis)
6. [Failure Modes & Negative Results](#failure-modes)
7. [Highest-Leverage Open Problems](#highest-leverage-open-problems)

---

## Executive Summary

Diffusion language models (DLMs) have gone from academic curiosity to industry deployment in 18 months. The field coalesced around **masked discrete diffusion** (absorbing-state) as the dominant paradigm after MDLM (NeurIPS 2024) and SEDD (ICML 2024 Best Paper) showed it was far more capable than previously believed. In Feb 2025 LLaDA proved 8B-scale masked diffusion matches LLaMA3-8B; by mid-2025, Google (Gemini Diffusion, 1,479 tok/s), Inception Labs (Mercury, 1,100 tok/s), and ByteDance (Seed Diffusion, 2,146 tok/s) had commercial-grade systems.

**Current consensus:**
- Discrete masked diffusion > continuous diffusion > uniform-state diffusion for text
- DLMs match AR on perplexity and downstream tasks at 8B scale
- DLMs are 3-10x faster at inference via parallel decoding
- DLMs are 2-5x more data-hungry (Quokka scaling laws)
- DLMs are dramatically better at diversity (93.4% unique openings vs AR's 3.3%)
- DLMs **fail badly** at: agentic multi-turn, tool calling, long-horizon reasoning, strict JSON/structured output

**The field is saturated on:** basic masked diffusion training, AR-to-diffusion adaptation, simple benchmark matching, speed records.

**The field has critical gaps in:** reasoning under arbitrary order, alignment/RLHF for DLMs, agentic capabilities, variable-length generation, long-context, multimodal unification, and diffusion-native architectures.

---

## Taxonomy of Approaches

### A. Discrete Masked Diffusion (Absorbing State)
The dominant paradigm. Forward: progressively mask tokens to [MASK]. Reverse: iteratively unmask.
- **Models:** MDLM, SEDD, LLaDA, Dream, Mercury, Gemini Diffusion, Seed Diffusion
- **Status:** SATURATED for basic training. Active frontier: alignment, reasoning, efficiency.

### B. Discrete Uniform-State Diffusion
Forward: corrupt tokens to random vocabulary items. More general but harder to train.
- **Models:** D3PM, Duo
- **Status:** Largely superseded by masked diffusion. Duo showed it can work via Gaussian lifting but not adopted at scale.

### C. Continuous/Latent Diffusion for Text
Map tokens to embeddings, diffuse in continuous space, round back to tokens.
- **Models:** Diffusion-LM, GENIE, CoDAR, NeoDiff, CANDI
- **Status:** Persistently worse than discrete due to **rounding bottleneck**. CoDAR (Mar 2026) partially addresses this. NOT saturated -- the rounding problem is still open.

### D. Discrete Flow Matching
Continuous-time formulation of discrete diffusion via CTMC or flow paths.
- **Models:** SEDD (CTMC), FS-DFM (few-step flow matching)
- **Status:** FS-DFM (ICLR 2026) achieves 128x speedup with 8 steps. Active frontier.

### E. Block/Hybrid Diffusion
Autoregressive over blocks, diffusion within blocks. Gets KV-cache + parallel decoding.
- **Models:** BD3LM (ICLR 2025 Oral), ReFusion, DFlash
- **Status:** Most practical architecture for deployment. Active frontier.

### F. AR-to-Diffusion Adaptation
Convert pretrained AR models to diffusion via continued pretraining.
- **Models:** DiffuGPT, DiffuLLaMA (ICLR 2025), DiffusionVL, dLLM framework
- **Status:** Well-understood recipe. <200B tokens to convert. Saturating.

---

## Paper-by-Paper Catalog

### Foundational / Scaling

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **LLaDA: Large Language Diffusion Models** | 2502.09992 | Feb 2025 | First 8B masked diffusion LM trained from scratch; matches LLaMA3-8B on ICL and instruction-following; solves reversal curse | **Landmark** -- proved DLMs scale | No alignment pipeline; no long-context; fixed-length generation |
| **LLaDA 1.5 (VRPO)** | 2505.19223 | May 2025 | First DPO-style alignment for masked diffusion; VRPO reduces gradient variance via antithetic sampling and timestep-stratified ELBOs | **Novel** -- first principled alignment for DLMs | Only tested at 8B; withdrew from ICLR 2026; variance reduction is necessary but not sufficient for competitive alignment |
| **LLaDA-MoE** | (GitHub) | Sep 2025 | First MoE diffusion LM; 7B total, ~1B active; beats LLaDA 1.5 dense 8B | Novel architecture | Not published as full paper; limited evaluation |
| **Dream 7B** | 2508.15487 | Aug 2025 | Context-adaptive token-level noise rescheduling; AR initialization; strongest open DLM | **Novel** training recipe | Still relies on fixed-length generation; noise rescheduling adds complexity |
| **Mercury** | 2506.17298 | Jun 2025 | Commercial DLM from Inception Labs; 1,100 tok/s; 88% HumanEval | Industry **landmark** | Closed-source; no paper on training details; code-focused only |
| **Gemini Diffusion** | (blog) | May 2025 | Google DeepMind; 1,479 tok/s; first >100B DLM; matches Gemini 2.0 Flash-Lite on code | Industry **landmark** | Weak on complex reasoning (GPQA 40.4% vs 56.5%); weak on general knowledge (MMLU 69.1% vs 79.0%); closed experimental |
| **Seed Diffusion** | 2508.02193 | Aug 2025 | ByteDance; 2,146 tok/s on H20; two-stage curriculum distilling optimal trajectories | Speed **record** | Code-only evaluation; specific to structured generation |
| **Quokka (Training Optimal DLMs)** | 2510.03280 | Sep 2025 | First scaling laws for DLMs; N_opt ~ C^0.5, D_opt ~ C^0.5; DLMs 2-5x more data-hungry than AR | **Landmark** for DLM scaling | Higher irreducible loss (2.41 vs 1.69 for AR) -- fundamental ELBO gap |
| **Super Data Learners** | 2511.03276 | Nov 2025 | DLMs have much higher data potential; 8B params, 1.5T tokens, 480 epochs without degradation | **Novel** scaling insight | Doesn't close the per-sample efficiency gap |

### Architecture & Efficiency

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **Block Diffusion (BD3LM)** | 2503.09573 | Mar 2025 | Interpolates AR and diffusion via block-wise generation; enables KV-caching + variable-length | **Novel** paradigm (ICLR 2025 Oral) | Block size is a fixed hyperparameter; quality-speed tradeoff not fully characterized |
| **ReFusion** | 2512.13586 | Dec 2025 | Slot-level parallel AR decoding; 34% perf gain + 18x speedup over prior MDMs; 78.66% HumanEval | **Novel** slot reorganization | Complex training; not yet scaled beyond 7B |
| **DID (Deletion-Insertion Diffusion)** | 2603.23507 | Mar 2026 | Replaces mask/unmask with delete/insert; variable-length native; eliminates MASK token overhead | **Novel** paradigm shift | Very new; limited benchmarks so far; insertion ordering is hard to learn |
| **FS-DFM** | 2509.20624 | Sep 2025 | 8-step discrete flow matching for 1024-token generation; 128x fewer steps than baseline | **Novel** (ICLR 2026) | Requires step-aware training; not yet at 8B scale |
| **DiffuGPT/DiffuLLaMA** | 2410.17891 | Oct 2024 (ICLR 2025) | Recipe to convert AR models (127M-7B) to DLMs with <200B tokens | **Novel** adaptation recipe | Ceiling limited by source AR model quality |
| **Soft-Masked Diffusion** | 2510.17206 | Oct 2025 | Augments binary mask with intermediate context from previous denoising step | Incremental | Marginal gains over standard MDLM |
| **Progressive Token Evolution** | 2601.07351 | Jan 2026 | Beyond hard masks -- progressive soft evolution of tokens during denoising | Incremental | Similar direction to Soft-Masked |

### Reasoning & Code

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **d1: Scaling Reasoning in DLMs** | 2504.12216 | Apr 2025 | First RL (diffu-GRPO) for masked DLMs; SFT + RL pipeline; "aha moments" in self-correction | **Landmark** (NeurIPS 2025 Spotlight) | GSM8K still lags strong AR; planning near-doubled but from low base |
| **The Flexibility Trap** | 2601.15165 | Jan 2026 | Arbitrary-order generation NARROWS reasoning -- DLMs skip high-entropy "reasoning spark" tokens | **Critical negative result** | JustGRPO fix constrains to AR order during training, partly defeating the purpose of diffusion |
| **LogicDiff** | (follow-up) | 2026 | Logic-role-guided unmasking; LLaDA-8B: 22% -> 60.7% GSM8K without retraining | **Novel** inference method | Requires a lightweight classifier head; doesn't solve the fundamental problem |
| **TraceRL / TraDo** | (GitHub, ICLR 2026) | Sep 2025 | Trajectory-aware RL for DLMs; SOTA reasoning: TraDo-8B-Thinking with long-CoT | **Novel** RL framework | Complex training pipeline; requires value model for variance reduction |
| **Dream-Coder 7B** | 2509.01142 | Sep 2025 | Emergent any-order code generation: sketch-first, left-to-right, or interleaved | **Novel** emergent behaviors | Only 7B; limited to code domain |
| **Stable-DiffCoder** | 2601.15892 | Jan 2026 | Pushes DLM code frontier; random masking improves reasoning on low-resource languages | Incremental over Dream-Coder | Still behind frontier AR coders |
| **On Reasoning Abilities of MDMs** | 2510.13117 | Oct 2025 | Theory: MDMs equivalent to poly-padded PLTs; can solve all CoT problems; more efficient for regular languages | **Novel** theory | Gap between theoretical expressiveness and practical performance |
| **Prophet: Fast Decoding** | (OpenReview) | Oct 2025 | Early answer convergence: correct answer identified at ~50% steps; 3.4x step reduction | **Novel** observation | Works for reasoning tasks; less applicable to open-ended generation |

### Alignment & Post-Training

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **LLaDA 1.5 (VRPO)** | 2505.19223 | May 2025 | See above | Novel | Withdrawn from ICLR 2026 |
| **TCSM (Target Concrete Score Matching)** | 2504.16431 | Apr 2025 | Unifies pre-training and post-training (reward, preference, distillation) under one framework | **Novel** (ICML 2025, Apple) | Theoretical elegance > practical gains demonstrated so far |
| **dLLM-RL / TraceRL** | (ICLR 2026) | Sep 2025 | Comprehensive RL framework supporting all open-source DLMs (LLaDA, Dream, MMaDA, etc.) | **Novel** infrastructure | Complex; value model adds overhead |
| **JustGRPO** | 2601.15165 | Jan 2026 | Standard GRPO applied to DLMs with AR-order constraint during training | Surprisingly simple | Constrains to AR order in training, losing key DLM advantage |

### Speculative & Parallel Decoding

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **Self Speculative Decoding (SSD)** | 2510.04147 | Oct 2025 | DLM as both drafter and verifier; 3.46x speedup; lossless | **Novel** | Requires hierarchical verification trees |
| **SpecDiff** | (NAACL 2025) | 2024 | Uses discrete diffusion as drafter for AR verification; 7.2x speedup | **Novel** | Drafter quality limits gains |
| **BlockSpec** | (ICLR 2026 sub) | 2025 | Block-level speculation; 7-14x speedup; first block-level spec decoding for DLMs | **Novel** | Complex trajectory exploration |
| **DART** | 2601.19278 | Jan 2026 | Argues dLLMs are fundamentally unsuitable as drop-in drafters for AR speculative decoding | **Critical negative result** | The bidirectional-vs-causal mismatch is architectural |
| **Adaptive Parallel Decoding** | (NeurIPS 2025 Spotlight) | 2025 | Multiplicative mixture of DLM marginals + small AR model; dynamic parallel token count | **Novel** | Requires auxiliary AR model |
| **DFlash** | (Feb 2026) | Feb 2026 | Block diffusion for flash speculative decoding | Incremental | Builds on BD3LM + BlockSpec |

### Multimodal

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **MMaDA** | (NeurIPS 2025) | 2025 | Unified diffusion for text reasoning + multimodal understanding + image generation; block diffusion + mixed CoT | **Novel** unified model | Jack-of-all-trades; doesn't excel at any single modality vs. specialists |
| **LaViDa** | 2505.16839 | May 2025 | First competitive diffusion VLM; complementary masking + prefix KV cache + timestep shifting | **Novel** (NeurIPS 2025 Spotlight) | Still behind best AR VLMs on some benchmarks |
| **DiffusionVL** | 2512.15713 | Dec 2025 | Converts any AR VLM to diffusion VLM; DiffusionVL-3B > LaViDa-8B | **Novel** efficiency | Relies on AR model quality |
| **UniDisc** | 2503.20853 | Mar 2025 | First unified multimodal discrete diffusion (text + image) | **Novel** | Early stage; limited benchmarks |
| **LLaDA-V** | (May 2025) | May 2025 | Vision extension of LLaDA | Incremental | Outperformed by later DiffusionVL |

### Theory & Analysis

| Paper | ID | Date | Contribution | Novel vs Incremental | Key Limitation |
|---|---|---|---|---|---|
| **Theoretical Benefit and Limitation of DLMs** | 2502.09622 | Feb 2025 | MDMs achieve near-optimal perplexity in O(1) steps; but CANNOT handle high-accuracy tasks efficiently | **Critical** theory (NeurIPS 2025) | Shows fundamental accuracy-vs-efficiency tradeoff |
| **AR vs MDLM Controlled Comparison** | 2603.22075 | Mar 2026 | Identical data/compute comparison; MDLM 4.7% slower training; 93.4% vs 3.3% diversity; AR overfits earlier | **Novel** controlled study | Small scale (not 8B) |
| **The Diffusion Duality (Duo)** | (ICML 2025) | 2025 | Uniform-state diffusion = underlying Gaussian; discrete consistency distillation; 100x fewer steps | **Novel** theoretical bridge | Only tested at smaller scales |
| **Scaling Behavior of Discrete DLMs** | 2512.10858 | Dec 2025 | Scaling depends heavily on noise type; masked > uniform at scale | **Novel** empirical scaling | Limited to smaller scales |
| **Absorbing = Conditional Probs** | (ICLR 2025) | 2025 | Concrete score in absorbing diffusion = conditional probability x time scalar; unifies absorbing diffusion and AO-ARMs | **Novel** theoretical unification | |
| **Diffusion Beats AR in Data-Constrained Settings** | (CMU blog) | Sep 2025 | Rule of thumb: compute-constrained -> AR; data-constrained -> diffusion | **Useful** heuristic | Approximate; depends on task |

### Surveys

| Paper | ID | Date |
|---|---|---|
| **A Survey on Diffusion Language Models** | 2508.10875 | Aug 2025 |
| **Discrete Diffusion in LLMs & Multimodal: A Survey** | 2506.13759 | Jun 2025 |
| **Parallel Text Generation: From Parallel Decoding to DLMs** | 2508.08712 | Aug 2025 |
| **Top 10 Open Challenges for DLMs** | 2601.14041 | Jan 2026 |
| **The Bitter Lesson of DLMs for Agentic Workflows** | 2601.12979 | Jan 2026 |

### Frameworks & Tools

| Project | Link | What it does |
|---|---|---|
| **dLLM** | 2602.22661 | Unified framework: train, finetune, deploy, evaluate LLaDA/Dream/BD3LM/custom |
| **dLLM-RL** | github.com/Gen-Verse/dLLM-RL | RL framework for all open-source DLMs |
| **MegaDLMs** | github.com/JinjieNi/MegaDLMs | GPU-optimized training at any scale |
| **Awesome-DLMs** | github.com/VILA-Lab/Awesome-DLMs | Curated paper list (200+ papers) |

---

## Saturation Map

### SATURATED (diminishing returns on new papers)

1. **Basic masked diffusion training at 8B scale** -- LLaDA, Dream, Mercury all proved it works. Another "we trained an 8B masked diffusion model" paper adds little.

2. **AR-to-diffusion adaptation** -- DiffuGPT/DiffuLLaMA established the recipe (ICLR 2025). DiffusionVL, dLLM framework automated it. The conversion is well-understood.

3. **Simple benchmark matching** -- Showing DLMs match AR on HumanEval, MBPP, MMLU is table stakes now. Mercury, Gemini Diffusion, Seed Diffusion all do this.

4. **Inference speed records** -- 1,000-2,000 tok/s is established. Another "our DLM is fast" paper needs a genuinely new mechanism, not just engineering.

5. **Basic multimodal extension** -- LaViDa, LLaDA-V, DiffusionVL, MMaDA all showed DLMs can do vision-language. Straightforward extensions are incremental.

6. **Score matching theory for discrete diffusion** -- CSM -> SEDD -> TCSM pipeline is quite mature. Incremental theoretical refinements have diminishing impact.

### ACTIVE BUT CROWDED (many groups working; competitive)

1. **Speculative/parallel decoding** -- SSD, SpecDiff, BlockSpec, DFlash, DART, APD all in 2025-2026. Expect continued papers but the design space is getting mapped out.

2. **Block/hybrid diffusion** -- BD3LM opened this; ReFusion, DFlash, and others are iterating. The optimal block strategy is still debated.

3. **RL for reasoning** -- d1, TraceRL/TraDo, JustGRPO, LogicDiff are all concurrent. Many groups racing on this.

4. **Noise schedule optimization** -- Dream's context-adaptive rescheduling, various curriculum learning approaches.

---

## Gap Analysis: What's Unsolved

### GAP 1: Long-Context Diffusion (WIDE OPEN)
**Status:** Nearly zero work on DLMs beyond 4K-8K context. All major DLMs (LLaDA, Dream) were trained with short contexts. Gemini Diffusion context length is undisclosed.

**Why it's hard:** Masked diffusion requires full bidirectional attention over the entire sequence at every denoising step. No KV-cache reuse (except block diffusion). Compute scales quadratically with context length per step, multiplied by number of denoising steps.

**What's been tried:** Block diffusion partially helps but doesn't solve the within-block attention scaling. Nothing equivalent to RoPE extension, ALiBi, or ring attention for DLMs.

**Opportunity:** A DLM that handles 32K-128K context with reasonable efficiency would be a major contribution.

### GAP 2: Alignment / RLHF / Preference Optimization (EARLY STAGE)
**Status:** Only LLaDA 1.5 (VRPO) and d1 (diffu-GRPO) have attempted this. VRPO was withdrawn from ICLR 2026. The fundamental problem: ELBO-based likelihood estimation has high variance, making DPO/PPO unstable.

**What's missing:**
- No equivalent of "constitutional AI" or RLAIF for DLMs
- No online RLHF at scale (d1 is small-scale)
- TCSM provides a theoretical framework for reward-based finetuning but lacks large-scale validation
- No DLM equivalent of instruction-tuned models that match ChatGPT/Claude quality

**Opportunity:** A principled, low-variance alignment method that doesn't degrade to "just use AR order" would be transformative.

### GAP 3: Reasoning Under Arbitrary Order (FUNDAMENTAL PROBLEM)
**Status:** The Flexibility Trap (Jan 2026) showed that arbitrary-order generation HURTS reasoning by skipping high-entropy "reasoning spark" tokens. Current fixes (JustGRPO, LogicDiff) either constrain to AR order or add inference-time heuristics.

**The dilemma:** DLMs' key advantage (arbitrary order) is also their key weakness for reasoning. No one has shown how to get the benefits of parallel generation without sacrificing sequential reasoning quality.

**What's needed:** A generation order that is neither fully arbitrary (flexibility trap) nor fully left-to-right (no DLM advantage), but informed by problem structure.

### GAP 4: Variable-Length Generation (STRUCTURAL LIMITATION)
**Status:** All MDLMs require pre-specifying output length. Block diffusion partially helps but still has fixed block sizes. DID (Mar 2026) proposes deletion-insertion as an alternative but is very new.

**Why it matters:** Real tasks (chat, code, writing) have variable output length. Current practice: pad to max length and hope EOS prediction works. This wastes 30-50% compute on padding.

**Opportunity:** DID opened a new direction. Adaptive block sizing, or a fundamentally variable-length discrete diffusion process, would be high-impact.

### GAP 5: Agentic / Tool-Use / Multi-Turn (TOTAL FAILURE)
**Status:** The Bitter Lesson paper (Jan 2026) showed dLLMs score <7.5% success on AgentBoard (vs 45% for Qwen-8B) and 0% on multi-turn tool calling. Root cause: non-causal parallel decoding can't maintain symbolic precision (strict JSON schemas) or causal decision chains.

**What's needed:**
- DLMs that can reliably produce structured output (JSON, function calls)
- Multi-turn memory and state tracking
- Ability to commit to partial plans (contradicts the "refine everything" diffusion philosophy)

**Opportunity:** DiffuAgent showed DLMs can be effective in non-causal auxiliary roles (memory summarization, tool selection). A hybrid architecture where DLMs handle non-causal components and AR handles causal decisions is unexplored.

### GAP 6: Diffusion-Native Architecture (NOT STARTED)
**Status:** All current DLMs use vanilla Transformer backbones designed for AR. The "Top 10 Challenges" paper (Jan 2026) argues DLMs are "trapped" in AR-legacy infrastructure. No one has designed an attention mechanism specifically for iterative denoising.

**What's needed:**
- Attention that handles partially-masked sequences efficiently
- Mechanisms for re-using computation across denoising steps (not just KV-cache)
- Noise-schedule-aware architecture (different capacity at different noise levels)

**Opportunity:** The equivalent of "what U-Net was to image diffusion" for text diffusion -- a purpose-built architecture -- does not yet exist.

### GAP 7: Closing the Irreducible Loss Gap (THEORETICAL)
**Status:** Quokka showed DLMs have higher irreducible loss (2.41 vs 1.69 for AR). This is fundamental: the ELBO is a variational bound, not the true likelihood. The forward noising process and discretization introduce a non-vanishing gap.

**Approaches tried:** Energy-based models (EDLM) improve the approximation by modeling sequence-level correlations. CoDAR addresses the rounding bottleneck for continuous diffusion. Neither fully closes the gap.

**Opportunity:** Tightening the ELBO or finding a non-ELBO training objective that doesn't sacrifice scalability.

### GAP 8: Continuous Diffusion for Text (PERSISTENT UNDERPERFORMANCE)
**Status:** Continuous diffusion consistently underperforms discrete. The rounding bottleneck (embedding -> token) loses information. CoDAR (Mar 2026) partially addresses this with a learned AR discretizer, but adds complexity.

**Why it matters:** Continuous diffusion has theoretical advantages (smooth optimization landscape, natural guidance mechanisms). If the rounding problem were solved, it could unlock better controllability.

**Opportunity:** A genuinely good continuous-to-discrete bridge. Or: working entirely in a continuous latent space with a powerful decoder (the autoencoder approach from image diffusion, applied to text).

### GAP 9: One/Few-Step Generation (VERY EARLY)
**Status:** FS-DFM achieves 8 steps for 1024 tokens. "One-step Language Modeling via Continuous Denoising" (Feb 2026) explores single-step. Consistency distillation (CD4LM) is being explored. But quality degrades significantly at very few steps.

**Why it matters:** True single-step generation would make DLMs categorically faster than AR (one forward pass for an entire sequence).

**Opportunity:** Consistency distillation and progressive distillation applied to discrete diffusion are underexplored at scale.

### GAP 10: Data Engineering for Bidirectional Learning (UNEXPLORED)
**Status:** All DLMs are trained on standard web corpora curated for AR models. No one has created data specifically designed for bidirectional denoising (e.g., annotated with structural landmarks, dependency graphs, or anchor tokens).

**Why it matters:** The "Top 10 Challenges" paper argues this is fundamental -- current data emphasizes sequential continuity, which is what AR excels at. Bidirectional models need data that rewards global coherence.

**Opportunity:** Low-hanging fruit. Curating training data with explicit structural annotations could improve DLM quality without algorithmic changes.

---

## Failure Modes & Negative Results

### 1. Agentic Total Failure
- **Paper:** Bitter Lesson of DLMs for Agentic Workflows (2601.12979)
- **Finding:** 0% success on multi-turn tool calling across ALL tested dLLMs. Retry loops in embodied settings. Malformed JSON.
- **Root cause:** Non-causal parallel decoding weakens causal dependency and produces "fuzzy intermediate states."

### 2. The Flexibility Trap
- **Paper:** 2601.15165
- **Finding:** Arbitrary-order generation REDUCES reasoning ability. DLMs skip high-entropy tokens (logical connectives like "Therefore", "Since") that are critical reasoning branching points.
- **Root cause:** Confidence-based unmasking is adversarial to reasoning -- it solves easy tokens first and defers hard ones.

### 3. Accuracy-Efficiency Tradeoff is Fundamental
- **Paper:** Theoretical Benefit and Limitation (2502.09622, NeurIPS 2025)
- **Finding:** MDMs can generate low-perplexity text in O(1) steps but CANNOT handle high-accuracy tasks efficiently. On GSM8K and MBPP, MDMs are "significantly worse" than AR across all step counts.
- **Root cause:** The independence assumption in token-wise denoising distributions ignores inter-token dependencies that matter for exact-match tasks.

### 4. Gemini Diffusion's Reasoning Gap
- **Finding:** GPQA Diamond 40.4% vs 56.5% for comparable AR model. MMLU 69.1% vs 79.0%.
- **Root cause:** Complex multi-step reasoning benefits from sequential token generation; parallel refinement can't substitute for causal chains.

### 5. DART's Drafter Impossibility
- **Paper:** 2601.19278
- **Finding:** dLLMs are "fundamentally unsuitable as drop-in draft models" for AR speculative decoding. The bidirectional context modeling objective mismatches the strictly prefix-conditioned draft requirement.

### 6. Higher Irreducible Loss
- **Paper:** Quokka (2510.03280)
- **Finding:** DLM irreducible loss = 2.41 vs AR = 1.69. This is structural (ELBO gap), not a training issue.

---

## Highest-Leverage Open Problems

Ranked by estimated impact x feasibility:

### Tier 1: High Impact, Feasible (12-18 month horizon)

1. **Reasoning-Aware Denoising Order**
   - Problem: How to unmask tokens in an order that respects logical dependencies, without collapsing to pure AR?
   - Why now: Flexibility Trap + LogicDiff showed the problem and a partial fix. A learned, differentiable unmasking scheduler that balances parallel speed with logical ordering is the next step.
   - Not yet tried: End-to-end learned denoising order that maximizes downstream reasoning accuracy.

2. **Efficient Long-Context DLMs**
   - Problem: Scale masked diffusion to 32K+ context without quadratic blowup.
   - Possible approach: Hierarchical block diffusion with cross-block sparse attention; or sliding-window denoising with global summary tokens.
   - Not yet tried: Any serious attempt at long-context DLMs beyond block diffusion.

3. **Structured Output Guarantees for DLMs**
   - Problem: DLMs can't reliably produce valid JSON, function calls, or code that compiles.
   - Possible approach: Grammar-constrained decoding adapted to diffusion (constraint the denoising to valid parse trees); or hybrid AR-for-structure, diffusion-for-content.
   - Not yet tried: Formal grammar constraints during denoising.

### Tier 2: High Impact, Hard (2+ year horizon)

4. **Diffusion-Native Transformer Architecture**
   - Problem: Current architectures waste compute on MASK tokens, can't reuse computation across steps, have no noise-level conditioning in attention.
   - Possible approach: Multi-resolution attention that is sparse at high noise and dense at low noise; or a recurrent denoiser that maintains state across steps.

5. **Closing the ELBO Gap**
   - Problem: 2.41 vs 1.69 irreducible loss is fundamental.
   - Possible approach: Non-ELBO objectives (score matching avoids this but has its own issues); variational annealing; or tighter bounds via auxiliary variables.

6. **True Variable-Length Discrete Diffusion**
   - Problem: Fixed-length is a fundamental constraint of the masked diffusion framework.
   - Most promising direction: DID (deletion-insertion) is the first serious attempt. Needs scaling validation.

### Tier 3: Speculative / Moonshot

7. **Diffusion Models for Agentic Reasoning**
   - Current evidence strongly negative. Would require a fundamentally different approach to how DLMs interact with tools and environments.

8. **One-Step Text Diffusion at Scale**
   - Consistency distillation for discrete spaces is theoretically possible but quality at 1 step is far below usable.

9. **Unified Understanding + Generation Architecture**
   - MMaDA is the first attempt. A model that genuinely excels at all modalities (not jack-of-all-trades) is a moonshot.

---

## Key Relationships & Citation Graph

```
D3PM (2021) --> MDLM (NeurIPS 2024) --> LLaDA (Feb 2025) --> LLaDA 1.5 (May 2025) --> LLaDA-MoE (Sep 2025)
                                     \-> Dream (Aug 2025) --> Dream-Coder (Sep 2025)
                                     \-> Block Diffusion (ICLR 2025) --> ReFusion (Dec 2025) --> DFlash (Feb 2026)
SEDD (ICML 2024) --> TCSM (ICML 2025)
                 \-> Absorbing=ConditionalProbs (ICLR 2025)
DiffuLLaMA (ICLR 2025) --> DiffusionVL (Dec 2025)
                        \-> dLLM framework (Feb 2026)
LLaDA --> d1 (NeurIPS 2025) --> Flexibility Trap (Jan 2026) --> JustGRPO + LogicDiff (2026)
      \-> LaViDa (NeurIPS 2025) --> Sparse-LaViDa (Dec 2025)
      \-> MMaDA (NeurIPS 2025) --> TraDo/dLLM-RL (ICLR 2026)
      \-> Bitter Lesson for Agents (Jan 2026)
Mercury (Jun 2025)
Gemini Diffusion (May 2025)
Seed Diffusion (Aug 2025)
```

---

## Sources

### Surveys
- [A Survey on Diffusion Language Models](https://arxiv.org/abs/2508.10875)
- [Discrete Diffusion in LLMs & Multimodal Models: A Survey](https://arxiv.org/abs/2506.13759)
- [Parallel Text Generation: From Parallel Decoding to DLMs](https://arxiv.org/abs/2508.08712)
- [Top 10 Open Challenges for DLMs](https://arxiv.org/abs/2601.14041)

### Core Models
- [LLaDA](https://arxiv.org/abs/2502.09992)
- [LLaDA 1.5 (VRPO)](https://arxiv.org/abs/2505.19223)
- [Dream 7B](https://arxiv.org/abs/2508.15487)
- [Mercury](https://arxiv.org/abs/2506.17298)
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [Seed Diffusion](https://arxiv.org/html/2508.02193v1)
- [Block Diffusion (BD3LM)](https://arxiv.org/abs/2503.09573)
- [MDLM](https://arxiv.org/abs/2406.07524)
- [SEDD](https://arxiv.org/abs/2310.16834)
- [DiffuGPT/DiffuLLaMA](https://arxiv.org/abs/2410.17891)

### Reasoning & Alignment
- [d1: Scaling Reasoning in DLMs](https://arxiv.org/abs/2504.12216)
- [The Flexibility Trap](https://arxiv.org/abs/2601.15165)
- [TCSM (Apple)](https://arxiv.org/abs/2504.16431)
- [dLLM-RL / TraDo](https://github.com/Gen-Verse/dLLM-RL)

### Efficiency & Decoding
- [ReFusion](https://arxiv.org/html/2512.13586v1)
- [FS-DFM](https://arxiv.org/html/2509.20624v1)
- [Self Speculative Decoding](https://arxiv.org/abs/2510.04147)
- [BlockSpec](https://openreview.net/forum?id=hmAviop5rm)
- [Adaptive Parallel Decoding](https://openreview.net/forum?id=xwqTt26NJf)

### Analysis & Negative Results
- [The Bitter Lesson for Agentic Workflows](https://arxiv.org/abs/2601.12979)
- [Theoretical Benefit and Limitation](https://arxiv.org/abs/2502.09622)
- [AR vs MDLM Controlled Comparison](https://arxiv.org/html/2603.22075v1)
- [Quokka (Scaling Laws)](https://arxiv.org/abs/2510.03280)
- [Diffusion Beats AR Data-Constrained](https://blog.ml.cmu.edu/2025/09/22/diffusion-beats-autoregressive-in-data-constrained-settings/)
- [AR vs Diffusion: Text Embedding](https://arxiv.org/abs/2505.15045)

### Multimodal
- [MMaDA](https://openreview.net/forum?id=wczmXLuLGd)
- [LaViDa](https://arxiv.org/abs/2505.16839)
- [DiffusionVL](https://arxiv.org/html/2512.15713)
- [UniDisc](https://arxiv.org/abs/2503.20853)

### Novel Paradigms
- [DID (Deletion-Insertion Diffusion)](https://arxiv.org/abs/2603.23507)
- [CoDAR (Continuous Diffusion)](https://arxiv.org/abs/2603.xxxx)
- [NeoDiff](https://arxiv.org/abs/2505.22165)
- [Duo (Diffusion Duality)](https://openreview.net/forum?id=9P9Y8FOSOk)
- [EDLM (Energy-Based)](https://arxiv.org/abs/2410.21357)
- [DART](https://arxiv.org/html/2601.19278v1)

### Frameworks
- [dLLM framework](https://arxiv.org/abs/2602.22661)
- [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs)
- [MegaDLMs](https://github.com/JinjieNi/MegaDLMs)
