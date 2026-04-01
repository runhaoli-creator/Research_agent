# LLM/Agent Research Ideas: 8-Stage Output

---

## Stage 1: Frontier Map

### Theme 1: Test-Time Scaling / Inference-Time Search

**Saturated:** The core result (more inference compute helps) is established (Snell et al., 2408.03314). Multiple surveys exist (2503.24235, 2511.14772). Best-of-N and majority voting scaling laws are characterized. MCTS+LLM for math is crowded.

**Unsolved:** (1) No Chinchilla-equivalent law for inference compute. (2) Non-math/non-code domains have almost zero TTS work. (3) MCTS scaling is pathological for LLMs — ReSCALE (2603.21162) patched it with Gumbel+Sequential Halving but only tested on GSM8K/Game24. (4) Value functions for LLM tree search are terrible, causing the pathology. (5) Latent reasoning (Coconut, 2412.06769) replaces CoT with continuous hidden states but hasn't scaled.

**Weak points:** "Think Deep Not Long" (2602.13517) shows only ~5% of thinking tokens undergo meaningful internal revision — the rest are filler. Overthinking tax is severe. Current TTS treats all tokens equally.

**Room for new work:** Adaptive per-token compute allocation based on predicted difficulty; latent reasoning at scale; TTS for non-verifiable domains; joint optimization of pre-training, post-training, and inference compute.

---

### Theme 2: Agent Training with RL

**Saturated:** Basic GRPO/PPO on math benchmarks. DeepSeek-R1 recipe widely replicated.

**Unsolved:** (1) Credit assignment for multi-turn agent trajectories — THE bottleneck. HCAPO (2603.08754), HiPER (2602.16165), EMPG (2509.09265) are early attempts. (2) Entropy collapse during training (DAPO 2503.14476). (3) Echo Trap — agents overfit to locally rewarded patterns in multi-turn RL (RAGEN 2504.20073). (4) Generalization tax — RL improves in-domain, degrades OOD (2601.18217).

**Weak points:** Exploration is almost entirely on-policy. Safe exploration during agent RL is unaddressed. Cost of online interaction is intractable for complex environments.

**Room for new work:** Principled credit assignment via counterfactual reasoning; exploration strategies for LLM agents; generalization-preserving RL; RL in learned world models before real environments.

---

### Theme 3: Self-Correction / Verification

**Saturated:** The negative result (LLMs can't self-correct without external feedback) is established (Huang et al., 2310.01798). PRMs for math are well-studied.

**Unsolved:** (1) Accuracy-Correction Paradox — stronger models have LOWER correction rates (2601.00828). (2) Self-Correction Blind Spot — LLMs can correct identical errors from external sources but not their own; "Wait" prompt fixes 89.3% of blind spots (2507.02778). (3) PRMs are hackable fluency detectors, not reasoning verifiers (2603.06621). (4) Verification for non-formal domains doesn't exist.

**Weak points:** All verification relies on outcome correctness (math answer right/wrong). No method verifies reasoning QUALITY independent of outcome.

**Room for new work:** RL-trained self-correction leveraging the blind spot finding; adversarial training for verification robustness; consistency-based verification for open-ended domains; integrating formal sketches into reasoning.

---

### Theme 4: Long-Horizon Planning and Memory

**Saturated:** Documenting that LLMs fail at long-horizon tasks. RAG-based memory. Simple knowledge graph stores.

**Unsolved:** (1) Myopic step-wise reasoning is fundamentally greedy — early errors compound (FLARE, 2601.22311). (2) Self-conditioning on past errors creates degenerative loops (2509.09677). (3) Plan repair when execution diverges from plan. (4) RL-trained memory management — learning WHEN and WHAT to store/retrieve.

**Weak points:** Context windows don't replace purpose-built memory (2603.07670). Long-horizon agents condition on their own errors with no recovery mechanism.

**Room for new work:** Future-aware planning with learned value estimation; error-recovery mechanisms for agents; memory compression optimized for task utility rather than retrieval accuracy.

---

### Theme 5: Synthetic Data Generation

**Saturated:** Distillation from stronger models for math/code.

**Unsolved:** (1) Model collapse under recursive synthetic training (2510.01631). (2) Quality-Diversity-Complexity joint optimization — current methods optimize quality alone (2412.02980). (3) Synthetic data for multi-turn agent TRAJECTORIES is much harder than single-turn QA.

**Weak points:** LLMs produce monotonous synthetic data even at high temperature (2406.15126). Benefits plateau at ~300B tokens (2503.19551).

**Room for new work:** QDC-aware data generation; trajectory-level synthetic data for agents; detecting harmful synthetic data before training.

---

### Theme 6: Multi-Agent Collaboration

**Saturated:** Multi-agent debate for math. Role-based frameworks.

**Unsolved:** (1) LLMs are not trained for multi-agent interaction — pre-training is single-agent (2505.21298). (2) Conformity bias (2509.11035). (3) No principled framework distinguishing genuine collaboration from resource accumulation. (4) Multi-agent scaling laws only just derived (2603.24676).

**Room for new work:** Training models for cooperation from the ground up; adversarial debate training; stable MARL for LLMs.

---

### Theme 7: Code Agents

**Saturated:** SWE-Bench Verified approaching ceiling (75%+), data contamination concerns (94% pre-dates cutoffs, 2512.10218).

**Unsolved:** (1) Futility detection — agents in expensive unproductive loops (2509.09853). (2) Long-horizon software evolution beyond discrete issue fixing (SWE-EVO, 2512.18470). (3) Context overflow as primary failure mode (2509.16941). (4) Live self-evolution of agent scaffold (Live-SWE-agent, 2511.13646).

**Room for new work:** Futility detection with strategy switching; codebase-level (not issue-level) agent training; efficiency optimization for code agent trajectories.

---

### Theme 8: World Models for Language Agents

**Saturated:** LLMs as implicit world models for text games.

**Unsolved:** (1) LLM world models are unreliable for stochastic/complex environments. (2) Irreversible actions break tree search (WebDreamer, 2411.06559). (3) The world-model-then-policy paradigm for LLM agents is nascent (RWML, 2602.05842). (4) Using LLMs simultaneously as policy+value+world model (SYMPHONY, 2601.22623).

**Room for new work:** RL-trained world models for LLM agents before policy RL; grounded world models connecting language prediction to environment dynamics.

---

### Theme 9: Calibration and Uncertainty

**Saturated:** Post-training destroys calibration is well-documented.

**Unsolved:** (1) Trajectory-level uncertainty for agents — formalized only in 2602.05073. (2) Confidence-first training (CoCA, 2603.05881) is brand new. (3) Uncertainty-driven action selection for agents — knowing when to ask vs. assume.

**Room for new work:** Agents that express calibrated uncertainty about actions; RL training that jointly optimizes accuracy and calibration.

---

## Stage 2: 10 Candidate Ideas

### Idea 1: Adversarial Self-Correction via Self-Play RL

**Core thesis:** Train LLMs to self-correct by adversarial self-play — the model alternates between generating plausible errors and detecting/fixing them, bootstrapping self-correction from the demonstrated capability to correct external errors.

**Why now:** The Self-Correction Blind Spot paper (2507.02778) proved LLMs CAN correct errors identical to their own — just not when they know it's their own output. The "Wait" prompt recovers 89.3% of blind spots. This dormant capability can be activated via targeted RL training.

**Technical mechanism:** Two-phase RL training. Phase 1 (Error Injection): given a correct solution, the model generates a subtly corrupted version — reward = the corruption is plausible (accepted by a weak verifier) but actually wrong (rejected by a strong verifier). Phase 2 (Error Correction): given a corrupted solution, the model identifies and fixes the error — reward = the correction restores correctness. The adversary becomes harder as the corrector improves, creating a natural curriculum. Key insight: by framing self-correction as correction of "someone else's" work (the adversary phase), we bypass the blind spot.

**Not trivial combination:** This isn't "adversarial training + LLMs." The key novelty is the mechanism for bypassing the self-correction blind spot by externalizing the error source during training, grounded in the specific empirical finding about blind spots.

**Hypothesis:** Adversarially self-play-trained models achieve >50% correction rate on their own deep errors, vs. <20% for standard models.

**Algorithmic sketch:** (1) Generate correct solution S via standard CoT. (2) Phase 1 rollout: model generates S' = corrupt(S), reward_adv = verifier_weak(S') * (1 - verifier_strong(S')). (3) Phase 2 rollout: model generates S'' = correct(S'), reward_cor = verifier_strong(S''). (4) Train both phases jointly with GRPO, alternating adversary/corrector roles.

**Experimental setup:** Train on Llama-3.1-8B or Qwen2.5-7B. Evaluate on MATH, GSM8K, AIME (where self-correction is measurable). Compare correction rates: baseline (prompt-based), Reflexion, SCoRe, our method. Ablate: adversary difficulty, error types, training phases.

**Risks:** The adversary might produce only trivial errors. The correction phase might memorize specific error patterns rather than learning general self-correction.

**Novelty score:** 8/10
**Top-conference potential:** Strong

---

### Idea 2: Adaptive Depth Reasoning — Per-Token Compute Based on Deep-Thinking Prediction

**Core thesis:** Not all reasoning tokens need equal compute. Identify "deep-thinking tokens" (where internal predictions undergo significant revision) BEFORE generation, and allocate extra transformer depth/width only at those positions.

**Why now:** "Think Deep Not Long" (2602.13517) showed only ~5% of tokens undergo significant layer-wise revision, and these are the tokens that actually matter for reasoning. This finding enables targeted compute allocation.

**Technical mechanism:** Add a lightweight difficulty router (2-layer MLP on first-layer hidden states) that predicts, for each token position, whether it will require deep thinking. Easy tokens use early exit (skip top layers). Hard tokens use full depth + optional additional recurrent processing. The router is trained jointly with the main model via a combined loss: task performance + compute penalty.

**Not trivial combination:** This is not Mixture of Depths (MoD) or early exit. MoD skips tokens from processing entirely. Our method routes tokens to different computational DEPTHS based on predicted reasoning difficulty, specifically informed by the deep-thinking token phenomenon. The difficulty prediction is trained on the actual internal revision patterns, not just token perplexity.

**Hypothesis:** 2-3x fewer FLOPs at inference with <2% accuracy loss on reasoning benchmarks, by concentrating compute on the ~5% of tokens that matter.

**Algorithmic sketch:** (1) Forward pass through first L/4 layers. (2) Router predicts difficulty d_i for each token. (3) If d_i < threshold: early-exit at L/2 layers. (4) If d_i > threshold: full L layers + K additional recurrent refinement steps. (5) Loss = task_loss + λ * mean(layers_used). (6) Router trained via straight-through estimator on the gradient of task_loss w.r.t. layers_used.

**Experimental setup:** Modify Llama-3.1-8B or Qwen2.5-7B architecture. Evaluate on MATH, GSM8K, AIME, GPQA, HumanEval. Compare: standard model, MoD, early-exit baselines, our method. Measure: accuracy vs. FLOPs curve.

**Risks:** The router might not be accurate enough, leading to accuracy loss. The additional recurrent steps might not help if the model architecture isn't designed for iterative refinement.

**Novelty score:** 7/10
**Top-conference potential:** Strong

---

### Idea 3: Causal Credit Assignment for Multi-Turn Agent RL

**Core thesis:** In multi-turn agent RL, assign credit to individual actions by estimating their causal effect on the final outcome — not via heuristic decomposition but via learned counterfactual reasoning.

**Why now:** Credit assignment is THE bottleneck for agent RL (identified across every sub-field). Current approaches use trajectory-level bandit rewards or heuristic step-level decomposition. No principled causal method exists.

**Technical mechanism:** Train a counterfactual world model C that, given (trajectory, step t, alternative action a'), predicts what would have happened. The causal effect of action a_t = V(trajectory_actual) - E[V(trajectory_counterfactual)]. Use these causal effects as per-step advantages in GRPO. The world model C is trained on the agent's own trajectories with random action perturbations.

**Not trivial combination:** This is not hindsight relabeling (HER) or basic advantage estimation. The key is a LEARNED counterfactual estimator that produces step-level causal attributions from trajectory-level outcomes. Existing credit assignment methods (HCAPO, HiPER) use temporal distance or attention-based heuristics, not causal counterfactuals.

**Hypothesis:** Causal credit assignment produces 2x faster convergence and 20%+ higher final performance on multi-turn agent benchmarks vs. trajectory-level rewards.

**Algorithmic sketch:** (1) Collect trajectories (s_1, a_1, ..., s_T, r_T). (2) For each step t, sample K counterfactual actions a'_t. (3) Use world model C to predict counterfactual outcomes. (4) Compute causal advantage: A_t = r_T - mean(C(trajectory_t→a'_t)). (5) Train policy with GRPO using per-step advantages A_t instead of trajectory-level reward r_T.

**Experimental setup:** Train on Llama-3.1-8B. Evaluate on WebArena, OSWorld, SWE-Bench-Lite, ALFWorld. Compare: trajectory-level GRPO, HCAPO, HiPER, EMPG, our method. Ablate: number of counterfactuals K, world model quality, action perturbation strategy.

**Risks:** The counterfactual world model might be inaccurate, producing misleading credit assignments. Training the world model itself requires significant data. The counterfactual computation is expensive.

**Novelty score:** 8/10
**Top-conference potential:** Strong

---

### Idea 4: Consistency-Based Verification for Non-Formal Domains

**Core thesis:** For domains without formal verifiers (science, ethics, open-ended reasoning), verify reasoning quality by checking INTERNAL CONSISTENCY — does each step logically follow from the previous? — rather than outcome correctness.

**Why now:** RLVR works only because math/code have deterministic verifiers. Extending RL-based reasoning improvement to non-verifiable domains is the fundamental frontier. Internal consistency is a domain-agnostic proxy for reasoning quality that doesn't require ground-truth answers.

**Technical mechanism:** Given a reasoning chain (s_1, s_2, ..., s_n), train a consistency verifier V(s_i | s_{i-1}, ..., s_1) that scores whether step s_i logically follows from the preceding steps. V is trained on synthetic consistency/inconsistency pairs generated by deliberately shuffling, contradicting, or non-sequituring reasoning steps. At inference, use V as the reward in RLVR — reward = product of per-step consistency scores.

**Not trivial combination:** This is not "NLI applied to reasoning steps." NLI checks entailment between two statements. Consistency verification checks whether a reasoning step is a valid logical continuation of a CHAIN of prior steps in context. The training data generation (targeted inconsistency injection) is the methodological contribution.

**Hypothesis:** RL training with consistency rewards improves reasoning quality on non-math domains (science QA, ethical reasoning, commonsense) by 10-15% without any ground-truth labels.

**Algorithmic sketch:** (1) Collect reasoning chains from base model. (2) Generate inconsistency pairs: for each chain, create 5 corrupted versions (step swap, contradiction injection, non-sequitur insertion, premise dropping, circular reasoning). (3) Train consistency verifier V on (chain, label) pairs via binary classification. (4) Use V as reward in GRPO: reward = min(V(s_i | s_{<i})) over all steps. (5) Train policy to maximize consistency reward.

**Experimental setup:** Train on Llama-3.1-8B or Qwen2.5-7B. Evaluate on ARC, GPQA, OpenBookQA, ETHICS benchmark, StrategyQA. Compare: standard CoT, self-consistency, RefGPT (self-correction), PRM-trained (on math then transferred), our method. Ablate: inconsistency types, verifier size, reward aggregation (min vs. product vs. mean).

**Risks:** Consistency doesn't guarantee correctness — a chain of steps can be internally consistent but factually wrong. The consistency verifier might learn surface patterns (grammar, formatting) rather than logical structure.

**Novelty score:** 8/10
**Top-conference potential:** Strong

---

### Idea 5: Futility-Aware Agents with Learned Strategy Switching

**Core thesis:** Train LLM agents to detect when they're stuck in unproductive loops and SWITCH to a qualitatively different strategy, rather than persisting or giving up.

**Why now:** Code agents stuck in loops waste massive compute — "token snowball" (2509.09853). Sonnet 4's primary failure mode on SWE-Bench Pro is endless file reading (17%) and context overflow (35.6%), not reasoning errors. No agent has a mechanism to recognize and escape futility.

**Technical mechanism:** Train a futility detector F(trajectory) that monitors rolling trajectory features (action diversity, state revisitation, progress rate, error repetition) and outputs (continue, switch_strategy, abandon). When F outputs switch_strategy, the agent receives a high-level "strategy prompt" from a learned strategy library (e.g., "try a different file," "simplify the approach," "read documentation instead"). F and the strategy library are trained jointly via RL where reward = task completion - compute cost.

**Not trivial combination:** This is not just "add a timeout." The key contribution is the LEARNED strategy switching mechanism — the agent doesn't just stop; it switches to a qualitatively different approach based on what type of futility it's in (circular reasoning vs. wrong direction vs. missing information). The strategy library is learned, not hand-coded.

**Hypothesis:** Futility-aware agents solve 15-25% more tasks on SWE-Bench Verified within the same compute budget, by reallocating wasted compute to productive strategies.

**Algorithmic sketch:** (1) Collect trajectories from a base code agent, labeling: success, failure-stuck-in-loop, failure-wrong-direction, failure-missing-info. (2) Train futility detector F on trajectory features (action n-gram entropy, state embedding similarity over sliding window, error message repetition count). (3) Train strategy library: K=5 meta-strategies, each a prompt template. (4) RL training: when F triggers, try each strategy, reward = task completion - compute_penalty. (5) Joint optimization: F's threshold, strategy selection, and base policy.

**Experimental setup:** Base agent: Qwen3-Coder-Next or Claude Code or similar open agent. Evaluate on SWE-Bench Verified, SWE-Bench Pro, WebArena, OSWorld. Compare: base agent, base + timeout, base + random restart, base + oracle strategy switch, our method. Ablate: futility detection features, number of strategies, detection threshold, compute budget.

**Risks:** The futility detector might trigger too early (false positives) or too late (wasted compute). The strategy library might not contain the right alternative for a given stuck state. Hard to get training data for strategy switching.

**Novelty score:** 7/10
**Top-conference potential:** Strong

---

### Idea 6: World Model Pre-Training for Agent Policy RL

**Core thesis:** Before training an LLM agent's policy via RL, first train it to predict environment state transitions (world model). This world model phase provides the agent with a prior about how environments work, dramatically improving sample efficiency of subsequent policy RL.

**Why now:** RWML (2602.05842) is the first paper to train LLM world models before policy RL, but uses simple self-supervised state prediction. The world-model-then-policy paradigm is the natural analog of model-based RL (Dreamer) for LLM agents.

**Technical mechanism:** Two-phase training: Phase 1 (World Model): given (state_t, action_t), predict state_{t+1} using GRPO where reward = how close the prediction is to the actual next state. Phase 2 (Policy RL): use the world model for imagined rollouts during policy training — the agent "imagines" the consequences of actions in the world model before committing to them in the real environment.

**Not trivial combination:** RWML exists but uses basic next-state prediction loss. Our contribution is (1) RL-based world model training with environment-fidelity reward (not just prediction loss), and (2) using the world model for imagined rollouts during policy RL, creating a Dyna-style architecture for LLM agents.

**Hypothesis:** World model pre-training reduces the number of real environment interactions needed for policy convergence by 5-10x.

**Algorithmic sketch:** (1) Collect diverse trajectory data from random/heuristic policies in the target environment. (2) Phase 1: train LLM as world model via GRPO — given (context, action), predict next observation; reward = ROUGE/BERTScore between predicted and actual next state. (3) Phase 2: Dyna-style policy RL — alternate between real environment rollouts and imagined rollouts through the world model. Real rollouts update both world model and policy; imagined rollouts update only policy.

**Experimental setup:** Train on Llama-3.1-8B. Evaluate on WebArena, ALFWorld, ScienceWorld. Compare: direct policy RL (no world model), RWML, BC + RL, our method. Ablate: world model training data size, ratio of real-to-imagined rollouts, world model update frequency.

**Risks:** The LLM world model might be inaccurate enough to mislead policy RL ("model exploitation" in model-based RL). Stochastic environments may be particularly hard to model.

**Novelty score:** 7/10
**Top-conference potential:** Moderate

---

### Idea 7: Generalization-Preserving RL via Diverse Trajectory Anchoring

**Core thesis:** RL training improves in-domain performance but consistently degrades OOD capabilities (the "generalization tax"). Fix this by anchoring the RL-trained model to diverse pre-training behavior using trajectory-level KL regularization on a held-out diverse dataset.

**Why now:** The generalization tax is documented (2601.18217, 2603.12011) but no solution exists. Standard KL regularization (PPO) doesn't work because it regularizes on the RL training distribution, not diverse capabilities.

**Technical mechanism:** During RL training, maintain a "diversity buffer" of trajectories from diverse domains (code, math, writing, reasoning, conversation). At each RL step, compute a trajectory-level KL divergence between the current policy and the pre-trained policy ON THE DIVERSITY BUFFER (not the RL data). Add this as a regularization term: loss = RL_loss + β * KL_diverse. The key: β is adaptive — it increases when diversity buffer performance drops and decreases when it's stable.

**Not trivial:** Standard PPO KL regularization doesn't prevent generalization tax because it operates on the RL data distribution. Our method explicitly regularizes on a DIVERSE held-out distribution, with adaptive weighting to prevent capability loss.

**Hypothesis:** Eliminates >80% of the generalization tax (OOD degradation) while preserving >90% of the RL in-domain gains.

**Algorithmic sketch:** (1) Prepare diversity buffer: 10K trajectories across 10 domains from the pre-trained model. (2) Standard GRPO RL on target domain. (3) Every N steps, evaluate policy on diversity buffer. (4) Compute KL_diverse = mean over diversity buffer of KL(policy || reference). (5) Adaptive β: if diversity_performance drops >2%, increase β by 1.5x; if stable, decrease by 0.9x.

**Experimental setup:** Train on Llama-3.1-8B. RL on MATH (in-domain). Evaluate OOD on: HumanEval, MMLU, ARC, HellaSwag, TriviaQA. Compare: standard GRPO, GRPO+standard KL, DPO, our method. Ablate: diversity buffer composition, β schedule, buffer size.

**Risks:** The adaptive β might oscillate. The diversity buffer might be too different from the RL domain, causing optimization conflict. The overhead of evaluating on the diversity buffer may be significant.

**Novelty score:** 7/10
**Top-conference potential:** Moderate

---

### Idea 8: Latent Process Rewards — Scoring Internal Reasoning Quality, Not Surface Steps

**Core thesis:** Instead of scoring visible reasoning steps (surface PRMs), score internal hidden-state transitions. A step that "looks" correct but involves poor internal processing (low deep-thinking signal) should get a low process reward.

**Why now:** PRMs are hackable because they evaluate surface text, which models can game with fluent-but-wrong reasoning. "Think Deep Not Long" showed internal revision signals (layer-wise prediction changes) correlate with genuine reasoning effort. Scoring these internal signals is harder to hack.

**Technical mechanism:** Train a latent PRM (L-PRM) that takes as input the hidden states at each reasoning step (not the text tokens) and predicts step correctness. The hidden states capture internal processing quality that surface text doesn't reveal. L-PRM is trained on (hidden_states, correctness_label) pairs from math/code where correctness is verifiable. At inference, L-PRM scores steps based on internal computation quality.

**Not trivial:** Standard PRMs score text. L-PRM scores hidden states. This is not "just a deeper PRM." The key insight is that the internal representation captures whether the model "actually thought hard" vs. "generated fluent text," which surface PRMs cannot distinguish.

**Hypothesis:** L-PRM is significantly harder to hack than surface PRMs — models cannot game internal computation patterns as easily as surface text.

**Algorithmic sketch:** (1) Generate solutions with intermediate hidden states saved. (2) Label each step's correctness (using math/code verifiers). (3) Train L-PRM: input = hidden states at step boundary, output = correctness score. (4) Use L-PRM as reward in RLVR. (5) Evaluate hack-resistance: train a policy to maximize L-PRM reward and check if ground-truth accuracy also improves (unlike surface PRMs where maximizing PRM reward reduces accuracy).

**Experimental setup:** Base model: Llama-3.1-8B. Train L-PRM on MATH/GSM8K hidden states. Evaluate: L-PRM accuracy vs. surface PRM accuracy, and crucially, L-PRM hack-resistance (does RL against L-PRM improve or degrade real accuracy?). Compare: ORM, surface PRM, Math-Shepherd PRM, L-PRM. Ablate: which hidden layers to use, step boundary detection, PRM architecture.

**Risks:** Hidden states might not carry more information about reasoning quality than surface text. The L-PRM might learn to detect hidden-state artifacts rather than reasoning quality. Training data (hidden states + correctness labels) requires saving model internals during generation, which is expensive.

**Novelty score:** 8/10
**Top-conference potential:** Strong

---

### Idea 9: Uncertainty-Gated Agent Actions — Knowing When to Act vs. When to Verify

**Core thesis:** Train agents with a calibrated uncertainty output that gates their actions — high-confidence actions are executed directly, low-confidence actions trigger additional verification (re-reasoning, tool use, or human clarification) before execution.

**Why now:** CoCA (2603.05881) showed GRPO can jointly train accuracy and calibration. Post-training destroys calibration (2602.05073). Agents that know when they don't know would dramatically reduce failure rates.

**Technical mechanism:** Add an uncertainty head to the agent model: at each action step, the model outputs both an action AND a confidence score. Train via modified GRPO where reward = accuracy * calibration — the agent is rewarded for correct actions AND for correctly predicting when it will fail. When confidence < threshold, the agent enters a verification sub-routine: re-sample K solutions and check consistency, or invoke a tool to verify, or ask for clarification.

**Not trivial:** This is not "just add a confidence head." The key is the RL training objective that JOINTLY optimizes for accuracy and calibration, and the learned gating policy that decides WHAT TO DO when uncertain (not just "flag as uncertain"). The verification sub-routine is learned, not hard-coded.

**Hypothesis:** Uncertainty-gated agents achieve 30%+ fewer failures on hard tasks by selectively verifying uncertain actions, with minimal overhead on easy tasks.

**Algorithmic sketch:** (1) Add confidence head: Linear(hidden_dim, 1) with sigmoid. (2) Modified GRPO reward: r = accuracy + α * ECE_bonus (Expected Calibration Error bonus — reward low ECE). (3) Gating policy: if confidence < τ, enter verification mode. (4) Verification mode: re-sample 3 solutions, majority vote, or call tool. (5) Train jointly: main policy + confidence head + gating threshold τ + verification policy.

**Experimental setup:** Train on Llama-3.1-8B. Evaluate on WebArena, HotpotQA, MATH (where some problems are easy, some hard). Compare: standard agent, agent + self-consistency (always verify), agent + random verification, our uncertainty-gated agent. Ablate: calibration training objective, gating threshold, verification strategy.

**Risks:** Calibration might degrade as the policy improves (moving target). The verification sub-routine adds latency. Hard to get good calibration training signal for open-ended tasks.

**Novelty score:** 7/10
**Top-conference potential:** Moderate

---

### Idea 10: Cooperative Self-Play for LLM Reasoning — Solver + Challenger Training

**Core thesis:** Train a single LLM to be both a solver and a challenger via self-play. The solver generates solutions; the challenger finds flaws. Both improve through competition, producing a model that generates more robust reasoning AND can self-verify.

**Why now:** Multi-agent debate shows modest gains over self-consistency. The problem: both sides use the same model with same biases (conformity bias, 2509.11035). Training a single model to play BOTH roles via RL avoids the need for multiple models and naturally creates adversarial pressure.

**Technical mechanism:** Single model, two roles. Solver role: generate solution, reward = correctness. Challenger role: given a solution, generate a critique — reward = critique identifies a real error (verified by outcome) OR solution is actually correct and critique agrees. Self-play loop: solver generates solution → challenger critiques → if critique identifies real error, solver revises → repeat. Both roles trained with GRPO on the same model using role-specific prompts.

**Not trivial:** This is not multi-agent debate (which uses separate models or same model without adversarial training). The key is that both roles improve through RL self-play: the solver learns to avoid errors the challenger finds, and the challenger learns to find increasingly subtle errors.

**Hypothesis:** Self-play-trained models outperform standard RL models by 10-15% on hard reasoning benchmarks AND show genuine self-verification capability (challenger role catches >40% of errors at test time).

**Algorithmic sketch:** (1) Initialize roles with the same pre-trained model + role-specific system prompts. (2) Solver generates solution S. (3) Challenger generates critique C of S. (4) Reward_solver = correctness(S) - caught_errors(C). (5) Reward_challenger = caught_real_errors(C) + correctly_approved_solutions(C). (6) Train both roles via GRPO on same parameters with role-specific rewards. (7) Iterate.

**Experimental setup:** Train on Qwen2.5-7B. Evaluate on MATH, AIME, GPQA, ARC-Challenge. Compare: standard GRPO, standard + self-consistency, standard + separate verifier, multi-agent debate, our self-play. Ablate: number of self-play rounds, solver/challenger reward balance, role specialization depth.

**Risks:** Role collapse — the model might converge to trivial challenger behavior (always approve) or trivial solver behavior (overly cautious). The self-play dynamic might be unstable.

**Novelty score:** 8/10
**Top-conference potential:** Strong

---

## Stage 3: Hard Filtering

| # | Idea | Incremental? | Crowded? | Easy to imitate? | Eval clear? | Engineering? | Algorithmic novelty? | Stand out? | Keep? |
|---|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Adversarial Self-Correction | No | No | No | Yes | No | **Yes** — self-play for correction | Yes | **YES** |
| 2 | Adaptive Depth Reasoning | Somewhat | MoD is related | Moderate | Yes | Partially | Moderate — extends MoD | Maybe | **NO** — too close to MoD/early-exit |
| 3 | Causal Credit Assignment | No | Moderately | No | Yes | No | **Yes** — counterfactual estimator | Yes | **YES** |
| 4 | Consistency Verification | No | No | Moderate | Somewhat | No | **Yes** — consistency as RLVR reward | Yes | **YES** |
| 5 | Futility Detection | Somewhat | No | Yes (simple) | Yes | Partially | Moderate — learned switching | Maybe | **BORDERLINE** |
| 6 | World Model Pre-Training | Somewhat | RWML exists | Moderate | Yes | Partially | Moderate — Dyna for LLMs | Maybe | **NO** — too close to RWML |
| 7 | Generalization-Preserving RL | Somewhat | No | Yes (simple) | Yes | Yes | Weak — just KL on diverse set | No | **NO** — mechanism too simple |
| 8 | Latent Process Rewards | No | No | No | Yes | No | **Yes** — hidden-state PRM | Yes | **YES** |
| 9 | Uncertainty-Gated Actions | Somewhat | CoCA adjacent | Moderate | Somewhat | Partially | Moderate | Maybe | **NO** — CoCA too close |
| 10 | Cooperative Self-Play | No | Somewhat (debate) | No | Yes | No | **Yes** — single-model self-play RL | Yes | **YES** |

**Top 5:**
1. **Adversarial Self-Correction** (#1) — strongest gap-mechanism match, surprising insight foundation
2. **Latent Process Rewards** (#8) — attacks the PRM hackability problem at its root
3. **Cooperative Self-Play** (#10) — elegant mechanism, clear evaluation, broad impact
4. **Consistency Verification** (#4) — opens RLVR to non-verifiable domains (huge impact)
5. **Causal Credit Assignment** (#3) — addresses THE bottleneck for agent RL

**Eliminated:** #2 (too close to MoD), #5 (mechanism too simple for oral), #6 (too close to RWML), #7 (mechanism too simple), #9 (too close to CoCA).

---

## Stage 4: Literature-Grounded Novelty Review

*(Searching latest literature for each of the top 5)*

### Idea 1: Adversarial Self-Correction

**Closest works:**
- SCoRe (2409.12917): RL for self-correction, but NOT adversarial. Uses two-stage RL where the model first generates, then self-corrects. No error injection training.
- Self-Correction Blind Spot (2507.02778): Identifies the phenomenon but proposes no training method.
- Accuracy-Correction Paradox (2601.00828): Characterizes the problem but offers no solution.
- "Learning from Mistakes" (RL for self-correction on math): exists but uses OUTCOME-BASED correction, not ADVERSARIAL error injection.

**Difference:** No paper trains self-correction via adversarial self-play with explicit error injection as Phase 1. The blind-spot-bypassing mechanism (externalizing error source) is novel.

**Status: Adjacent but still open.** The blind spot finding is used as motivation but nobody has built the adversarial training pipeline on top of it.

**Likely reviewer attack:** "How is this different from SCoRe?" Defense: SCoRe uses outcome-based self-correction RL without adversarial error injection. We explicitly train the model to generate and detect errors as separate adversarial roles.

**Revision:** Add a Phase 0 (warm-start) where the model is first trained on externally generated error-correction pairs (from a teacher model) before self-play. This provides curriculum from easy to hard errors.

---

### Idea 4: Consistency Verification for Non-Formal Domains

**Closest works:**
- VPRMs (2601.17223): Rule-based step-level verification, but only for math.
- NLI-based verification: NLI models check entailment between pairs of statements, not chain consistency.
- NSVIF (2601.17789): Neuro-symbolic verification but uses external symbolic checkers, not learned consistency.
- "Self-Consistency" (Wang et al., 2203.11171): Consistency of multiple solutions, not internal chain consistency.

**Difference:** No paper uses internal chain consistency as an RLVR reward for non-verifiable domains. The inconsistency injection training methodology is novel.

**Status: Genuinely new.** The specific idea of training a consistency verifier on synthetic inconsistency pairs and using it as RLVR reward for non-formal domains has no direct precedent.

**Likely reviewer attack:** "Consistency ≠ correctness. A chain can be consistent but wrong." Defense: We don't claim consistency implies correctness. We claim it improves reasoning QUALITY and is better than no verification signal. Show empirically that consistency-trained models also improve accuracy on benchmarks with ground truth.

---

### Idea 8: Latent Process Rewards

**Closest works:**
- "Think Deep Not Long" (2602.13517): Identifies deep-thinking tokens via hidden state analysis but doesn't use them as reward.
- Sparse RLVR Tokens (2603.22446): Shows only few token distributions change meaningfully in RL, but doesn't propose hidden-state-based rewards.
- Standard PRMs: All operate on surface text, not hidden states.

**Difference:** No paper trains a PRM on hidden states instead of text. The hack-resistance hypothesis is novel and testable.

**Status: Very promising and likely novel.**

**Likely reviewer attack:** "Maybe hidden states don't carry more signal than surface text." Defense: Ablation comparing L-PRM vs. surface PRM accuracy AND hack-resistance. The hack-resistance experiment is the killer defense.

---

### Idea 10: Cooperative Self-Play

**Closest works:**
- Multi-agent debate (Du et al., 2023): Uses separate models or same model without adversarial RL training.
- "Training a single model for debate" (2601.22297): Mentioned as a direction but no concrete method.
- STILL (2408.05284): Self-Taught Improvement via RL — uses self-evaluation but not adversarial solver-challenger dynamics.
- Constitutional AI: Model self-critiques, but through prompting, not RL-trained roles.

**Difference:** No paper trains solver + challenger roles within a single model via RL self-play with role-specific rewards.

**Status: Promising and plausibly novel.**

**Likely reviewer attack:** "Role collapse will make this degenerate." Defense: Design anti-collapse mechanisms (minimum challenge rate, diversity bonus) and show they prevent collapse in ablations.

---

### Idea 3: Causal Credit Assignment

**Closest works:**
- HCAPO (2603.08754): Hierarchical credit assignment but uses heuristic decomposition, not counterfactuals.
- HiPER (2602.16165): Hierarchical process reward, temporal distance based.
- EMPG (2509.09265): Efficient credit via privileged information, not counterfactual.
- TARL (2509.14480): Tool-augmented RL with per-tool rewards, not causal counterfactual.
- Counterfactual credit in RL: well-studied in tabular/small-state RL but not for LLM agents.

**Difference:** Counterfactual world model for LLM agent credit assignment is novel. The specific mechanism (learned counterfactual simulator for agent trajectories) doesn't exist.

**Status: Adjacent but still open.** Credit assignment is hot, but the counterfactual approach for LLM agents is new.

**Likely reviewer attack:** "The counterfactual world model will be too inaccurate." Defense: Show that even noisy counterfactuals produce better credit than trajectory-level rewards. Ablate world model quality vs. credit assignment quality.

---

## Stage 5: Iterative Refinement (3 Rounds)

### Round 1

**Idea 1 (Adversarial Self-Correction):** Strengthen by adding the Phase 0 curriculum and testing on non-math domains where self-correction is most needed (open-ended QA, code debugging). Add evaluation: measure correction depth (how deep are the errors the model can now fix?) vs. baseline. Confidence UP.

**Idea 4 (Consistency Verification):** Strengthen by defining 7 specific inconsistency types: logical contradiction, unsupported claim, circular reasoning, scope shift, dropped premise, strawman, and false equivalence. Train separate detectors for each. The structured taxonomy makes the contribution more concrete. Confidence UP.

**Idea 8 (Latent Process Rewards):** Strengthen by proposing "L-PRM hacking test" as a standard evaluation: train a policy to maximize the PRM, then check real accuracy. If L-PRM maintains accuracy while surface PRM doesn't, that's the paper's core result. Confidence STABLE.

**Idea 10 (Cooperative Self-Play):** Strengthen by adding a "debate transcript" output that provides interpretable verification at test time. The model generates solution + challenger critique + response. This gives users a way to assess solution quality. Add anti-collapse reward bonus: reward challenger for finding errors + reward for novel critique strategies. Confidence UP.

**Idea 3 (Causal Credit):** Weaken slightly — the counterfactual world model is expensive and might not work well for complex environments like WebArena. Restrict scope to simpler environments (ALFWorld, InterCode) where the world model can be accurate. Confidence DOWN slightly.

### Round 2

**Idea 1:** Add comparison to "Wait" prompt baseline (from blind spot paper). If our trained model outperforms "Wait" significantly, that proves the training adds value beyond prompt engineering. Changed Phase 2 to also train on the model's OWN errors (not just adversary errors) to directly attack the blind spot. Confidence UP.

**Idea 4:** Realize that the inconsistency injection approach is similar to contrastive learning. Reframe: this is "contrastive RL for reasoning quality" — positive examples are consistent chains, negatives are inconsistent chains. The RL learns to distinguish and prefer consistent reasoning. Cleaner framing. Confidence UP.

**Idea 8:** Add experiment: L-PRM trained on math hidden states → evaluate as reward on non-math domains. If hidden-state reasoning quality signals transfer across domains (while surface PRMs don't), that's a much stronger result. Confidence UP.

**Idea 10:** Concern: self-play in LLMs might converge too quickly due to limited model capacity. Add ensemble self-play — use 3 different LoRA adaptors as 3 "personalities" to increase diversity. This prevents convergence to a single strategy. Confidence STABLE.

**Idea 3:** Replace full counterfactual world model with a lighter mechanism: action influence estimation via attention attribution. Instead of simulating counterfactuals, estimate each action's influence by computing how much the model's internal attention at the final step depends on that action's hidden states. Much cheaper, still principled. Confidence UP (now more practical).

### Round 3

Final refinements:

**Idea 1:** Final title: "Adversarial Self-Play for LLM Self-Correction: Activating Dormant Correction Capabilities." Core result: first model achieving >50% deep error correction rate (vs. <20% for baselines). Remains in top 5. Confidence: HIGH.

**Idea 4:** Final title: "Consistency Is All You Need: RLVR Without Verifiers for Open-Ended Reasoning." Core result: first demonstration of meaningful reasoning improvement via RL in non-verifiable domains. Remains in top 5. Confidence: HIGH.

**Idea 8:** Final title: "Latent Process Rewards: Hack-Resistant Verification via Internal Computation Quality." Core result: first PRM that resists reward hacking, demonstrated via the L-PRM hacking test. Remains in top 5. Confidence: HIGH.

**Idea 10:** Final title: "Solve, Challenge, Revise: Cooperative Self-Play for Robust LLM Reasoning." Core result: single-model self-play outperforms multi-agent debate and standard RL. Remains in top 5. Confidence: MODERATE-HIGH.

**Idea 3:** Final title: "Attention-Based Causal Credit for Multi-Turn Agent RL." Simplified mechanism improves practicality. Remains in top 5. Confidence: MODERATE-HIGH.

---

## Stage 6: Final 5 Ideas

### A. Adversarial Self-Play for LLM Self-Correction

**B. Core insight:** LLMs have dormant self-correction capabilities (they can correct identical errors from external sources but not their own). Adversarial self-play training activates these capabilities by having the model alternate between generating plausible errors (adversary) and detecting/fixing them (corrector), bypassing the self-correction blind spot by externalizing the error source.

**C. Why this could matter:** Self-correction is the most wanted and least achieved capability in LLMs. Every deployed system would benefit from models that can catch and fix their own mistakes. The blind spot finding provides a specific, testable mechanism for unlocking this capability.

**D. Technical novelty:** Three-phase RL training: Phase 0 (warm-start on external error-correction pairs), Phase 1 (adversary generates plausible errors, rewarded for subtle corruptions), Phase 2 (corrector fixes errors, rewarded for restoring correctness). The adversary-corrector dynamic creates a natural curriculum. The key insight is that Phase 1 externalizes the error source, bypassing the blind spot.

**E. Why existing work doesn't solve it:** SCoRe (2409.12917) uses outcome-based self-correction RL without adversarial error injection. The blind spot paper (2507.02778) identifies the phenomenon but proposes no training method. Constitutional AI uses prompting, not RL-trained adversarial roles. No paper trains error generation as Phase 1 to bootstrap self-correction.

**F. Research plan:**
- Model: Qwen2.5-7B or Llama-3.1-8B
- Training: Phase 0: SFT on 10K externally generated error-correction pairs. Phase 1+2: GRPO self-play, alternating roles, 50K RL steps
- Data: MATH, GSM8K (verifiable), GPQA, ARC (semi-verifiable for harder evaluation)
- Evaluation: Correction rate (% of own errors successfully fixed), correction depth (depth of errors fixable), overall accuracy improvement
- Baselines: Standard CoT, Self-consistency, SCoRe, "Wait" prompt, Reflexion, standard RL
- Ablations: Phase 0 warm-start size, adversary difficulty, error type distribution, self-play rounds
- Failure analysis: What error types resist correction? When does adversary generate unhelpful errors?
- Compute: 8×H100 for ~48 hours
- Expected: >50% correction rate on own deep errors (vs. <20% baseline)

**G. Reviewer attacks:** (1) "The adversary might generate trivial errors." (2) "Correction might not transfer to genuinely novel errors." (3) "How is this different from SCoRe?"

**H. Defense:** (1) Show adversary difficulty increases over self-play rounds (curriculum effect). (2) Evaluate on held-out error types not seen during training. (3) Direct ablation removing Phase 1 (adversary training) — show it's essential.

**I. Final novelty judgment:** Very promising and likely novel.

**J. Final top-conference potential:** Strong.

---

### B. Consistency Is All You Need: RLVR Without Verifiers

**B. Core insight:** For non-verifiable domains (science, ethics, open-ended reasoning), internal chain consistency — does each step logically follow from the previous? — is a domain-agnostic proxy for reasoning quality that can replace formal verifiers in RLVR.

**C. Why this could matter:** RLVR (DeepSeek-R1 style training) only works on math/code because they have deterministic verifiers. Extending RL-based reasoning improvement to ALL domains is the single most important open problem. Consistency verification provides a universal reward signal.

**D. Technical novelty:** (1) Taxonomy of 7 inconsistency types with targeted injection: logical contradiction, unsupported claim, circular reasoning, scope shift, dropped premise, strawman, false equivalence. (2) Consistency verifier trained on synthetic positive/negative reasoning chains. (3) Consistency score as RLVR reward: reward = min per-step consistency score. (4) RL training that improves reasoning quality on non-verifiable domains without ground-truth labels.

**E. Why existing work doesn't solve it:** VPRMs (2601.17223) are rule-based and math-only. NLI checks pairwise entailment, not chain consistency. Self-consistency (Wang et al.) checks agreement across solutions, not within a solution. No paper uses trained consistency verification as RLVR reward.

**F. Research plan:**
- Model: Qwen2.5-7B or Llama-3.1-8B
- Training: (1) Generate 100K reasoning chains on diverse topics. (2) Create 500K inconsistency-injected negative examples (7 types). (3) Train consistency verifier (DeBERTa-v3-large or small LLM). (4) RLVR with consistency reward using GRPO, 100K steps.
- Data construction: Source chains from diverse QA datasets (TriviaQA, StrategyQA, OpenBookQA, ETHICS, SciQ)
- Evaluation: Accuracy on ARC, GPQA, OpenBookQA, MMLU (held-out domains). Human evaluation of reasoning quality (blind A/B test). Consistency rate measured by independent verifier.
- Baselines: Standard CoT, Self-consistency voting, Self-refinement, Math-PRM transferred to open domains, no-RL
- Ablations: Which inconsistency types matter most, verifier size, reward aggregation (min vs. product vs. mean), RLVR training scale
- Compute: 8×H100 for ~72 hours
- Expected: 10-15% accuracy improvement on non-math reasoning benchmarks without ground-truth labels

**G. Reviewer attacks:** (1) "Consistency ≠ correctness." (2) "The verifier might learn surface patterns." (3) "How do you evaluate on domains without ground truth?"

**H. Defense:** (1) Show on benchmarks WITH ground truth that consistency-trained models also improve accuracy. (2) Ablate by showing performance with individual inconsistency types — logical ones help most, proving it's not surface-level. (3) Use human evaluation for truly open-ended domains.

**I. Final novelty judgment:** Very promising and likely novel.

**J. Final top-conference potential:** Strong (potentially exceptional if results are dramatic).

---

### C. Latent Process Rewards: Hack-Resistant Verification

**B. Core insight:** Standard PRMs evaluate surface text and are hackable fluency detectors. Scoring hidden-state transitions instead of text captures whether the model "actually reasoned" vs. "generated plausible text," making the reward signal much harder to game.

**C. Why this could matter:** PRM hackability (2603.06621) is a fundamental problem — RL training against PRMs improves PRM scores while degrading real accuracy. If L-PRMs resist hacking, they enable reliable RLVR at scale.

**D. Technical novelty:** (1) Process reward model trained on hidden states (layer activations at step boundaries) instead of text tokens. (2) L-PRM hacking test: train a policy to maximize the reward model, then check if real accuracy increases (L-PRM) or decreases (surface PRM). (3) Cross-domain transfer experiment: L-PRM trained on math hidden states used as reward for non-math domains.

**E. Why existing work doesn't solve it:** All existing PRMs (Math-Shepherd, OmegaPRM, etc.) operate on surface text. "Think Deep Not Long" analyzes hidden states but doesn't build a reward model. The PRM hackability paper documents the problem but proposes no hidden-state-based solution.

**F. Research plan:**
- Model: Llama-3.1-8B (policy) + separate L-PRM (Llama-3.1-3B head on frozen policy hidden states)
- Training: (1) Generate 50K solutions with hidden states saved at each reasoning step. (2) Label step correctness via math verifiers. (3) Train L-PRM on (hidden_states, label) pairs. (4) RLVR with L-PRM reward.
- Evaluation: (a) L-PRM accuracy on step correctness prediction vs. surface PRM accuracy. (b) THE KEY: L-PRM hacking test — train policy to max L-PRM reward for 10K steps, measure real accuracy change. Do same for surface PRM. L-PRM should maintain/improve real accuracy. Surface PRM should degrade it. (c) Cross-domain: L-PRM from math → reward on GPQA, ARC.
- Baselines: ORM, surface PRM (Math-Shepherd), PQM (ICLR 2026), outcome-only RLVR
- Ablations: Which hidden layers to extract, step boundary detection method, L-PRM architecture (MLP vs. small transformer), number of training solutions
- Compute: 8×H100 for ~96 hours (hidden state extraction is expensive)
- Expected: L-PRM maintains real accuracy under RL hacking pressure; surface PRM degrades by 10-20%

**G. Reviewer attacks:** (1) "Hidden states might not contain more signal than text." (2) "Extracting hidden states is expensive." (3) "The L-PRM might just be a better PRM, not fundamentally different."

**H. Defense:** (1) Direct comparison: train both L-PRM and surface PRM on same data, show L-PRM is harder to hack. (2) Report computational overhead — show it's a one-time cost. (3) The hacking test IS the fundamental difference — this is not about accuracy but about robustness.

**I. Final novelty judgment:** Very promising and likely novel.

**J. Final top-conference potential:** Strong.

---

### D. Solve, Challenge, Revise: Cooperative Self-Play

**B. Core insight:** Train a single LLM to play both solver and challenger roles via RL self-play. The solver generates solutions; the challenger finds flaws. Both improve through competition within a single model, producing robust reasoning AND built-in self-verification.

**C. Why this could matter:** Multi-agent debate shows modest gains because all agents share the same biases. Adversarial self-play within a single model creates genuine adversarial pressure while being practical (one model, not multiple). The challenger role gives the model built-in verification at test time.

**D. Technical novelty:** (1) Single-model dual-role RL self-play with role-specific GRPO rewards. (2) Anti-collapse mechanisms: minimum challenge rate, critique diversity bonus, ensemble LoRA for multiple "personalities." (3) Test-time usage: model generates solution + self-challenge + revision, providing interpretable verification chain.

**E. Why existing work doesn't solve it:** Multi-agent debate uses separate models or same model without adversarial RL training (Du et al., 2023). STILL (2408.05284) uses self-evaluation without adversarial dynamics. No paper trains solver+challenger as RL self-play in a single model with role-specific rewards.

**F. Research plan:**
- Model: Qwen2.5-7B with 3 LoRA adaptors (solver, challenger, revision)
- Training: GRPO self-play, 100K steps. Solver reward = correctness(solution). Challenger reward = identified_real_error(critique) + correctly_approved(critique). Revision reward = improved_correctness(revision).
- Data: MATH, GSM8K, AIME, GPQA
- Evaluation: Final accuracy, correction rate (challenger catches what % of errors), revision improvement rate, comparison to multi-agent debate
- Baselines: Standard GRPO, Self-consistency, Multi-agent debate (separate models), Multi-agent debate (same model, prompt-based), our self-play
- Ablations: Number of self-play rounds, LoRA ensemble vs. single model, role reward balance, minimum challenge rate
- Compute: 8×H100 for ~72 hours
- Expected: 10-15% over standard GRPO; challenger catches >40% of solver errors

**G. Reviewer attacks:** (1) "Role collapse." (2) "How is this different from multi-agent debate?" (3) "The challenger might become a rubber stamp."

**H. Defense:** (1) Anti-collapse mechanisms + ablation showing they're needed. (2) Direct comparison showing self-play RL outperforms prompt-based debate. (3) Report challenger diversity metrics (unique critique patterns over time).

**I. Final novelty judgment:** Promising and plausibly novel.

**J. Final top-conference potential:** Strong.

---

### E. Attention-Based Causal Credit for Multi-Turn Agent RL

**B. Core insight:** In multi-turn agent RL, use attention attribution to estimate each action's causal influence on the final outcome, providing principled per-step credit assignment that's much cheaper than full counterfactual simulation.

**C. Why this could matter:** Credit assignment is THE bottleneck for agent RL. Trajectory-level rewards give no signal about which actions mattered. Better credit = faster convergence + higher final performance.

**D. Technical novelty:** (1) Compute influence of action at step t on the model's final-step hidden state via attention rollout/attribution. (2) Use influence scores as per-step advantage weights in GRPO: A_t = influence_t * trajectory_reward. (3) The influence scores are free (computed from the model's own attention patterns), requiring no separate world model.

**E. Why existing work doesn't solve it:** HCAPO uses temporal distance heuristics. HiPER uses hierarchical decomposition. EMPG uses privileged information. None use the model's own attention patterns as a principled influence estimator. Attention attribution exists for interpretability but hasn't been used for credit assignment in agent RL.

**F. Research plan:**
- Model: Llama-3.1-8B
- Training: Modified GRPO with attention-weighted per-step advantages, 50K steps
- Environments: ALFWorld, InterCode-SQL, SWE-Bench-Lite, WebArena-Lite
- Evaluation: Convergence speed (steps to X% success), final success rate, credit assignment quality (correlation between estimated influence and actual importance via ablation)
- Baselines: Trajectory-level GRPO, HCAPO, HiPER, uniform per-step reward, random credit
- Ablations: Attention layer selection, attribution method (rollout vs. gradient vs. last-layer), influence normalization, threshold for credit assignment
- Compute: 8×H100 for ~48 hours (attention attribution is cheap)
- Expected: 2x faster convergence, 15-20% higher final performance on WebArena

**G. Reviewer attacks:** (1) "Attention ≠ causation." (2) "This is just attention-based interpretability applied to RL." (3) "Works on simple environments but will it scale?"

**H. Defense:** (1) Show empirically that attention-attributed credit outperforms alternative heuristics — if it works, the theory question is secondary. (2) The APPLICATION to credit assignment in agent RL is the novelty, not the attribution method itself. (3) Test on WebArena (complex enough to be convincing).

**I. Final novelty judgment:** Promising and plausibly novel.

**J. Final top-conference potential:** Strong.

---

## Stage 7: Coding-Agent Paragraphs

### Idea A: Adversarial Self-Correction

Implement a three-phase RL training pipeline for LLM self-correction using adversarial self-play. Start with Qwen2.5-7B-Instruct as the base model, using the transformers library with DeepSpeed ZeRO Stage 2 for distributed training. Phase 0 (warm-start): generate 10K error-correction pairs using GPT-4 or Claude — take correct MATH/GSM8K solutions, ask the teacher to introduce subtle errors, then fix them — and SFT the base model on these pairs for 3 epochs. Phase 1 (adversary training): given a correct solution, the model generates a subtly corrupted version using a system prompt like "You are an error injector. Introduce a plausible but incorrect modification." Train via GRPO where the reward is: weak_verifier(corrupted_solution) * (1 - strong_verifier(corrupted_solution)) — the corruption must fool a weak check (format, surface plausibility via a small LM judge) but fail a strong check (math outcome verification). Phase 2 (corrector training): given a corrupted solution (from Phase 1), the model identifies and fixes the error. Reward = strong_verifier(corrected_solution). Alternate Phase 1 and Phase 2 every 1000 steps for 50K total RL steps. Use GRPO with group size 16, clip ratio 0.2, KL coefficient 0.01. Evaluate on MATH-500, GSM8K test, AIME 2024, and GPQA Diamond. Measure: correction rate (% of own errors fixed when prompted to self-correct), correction depth (categorize errors as surface/medium/deep and report per-category correction rates), and overall accuracy. Compare against 6 baselines: base model without correction, base model with "Wait, let me reconsider" prompt (from blind spot paper), self-consistency (k=16), SCoRe (2409.12917) if reproducible, Reflexion with 3 rounds, and standard GRPO without adversarial phases. Run ablations on: Phase 0 warm-start size (0, 1K, 5K, 10K), adversary difficulty (how corrupted the errors are, controlled by corruption budget), error type distribution (arithmetic vs. logical vs. conceptual), and self-play rounds (1, 3, 5, 10). Also run the "externalization test": at evaluation time, present the model with errors labeled as "from another model" vs. "from yourself" and measure whether the trained model shows reduced blind-spot effect. The key result that validates the hypothesis is correction rate >50% on own deep errors vs. <20% for all baselines. Implementation pitfalls: adversary collapsing to trivial errors (fix: minimum corruption distance from original), corrector memorizing specific error patterns (fix: ensure diverse error types in Phase 1), and reward hacking in Phase 1 where adversary produces incoherent text that technically satisfies the reward (fix: add fluency penalty). For publication quality, include human evaluation of correction quality on 200 randomly sampled correction attempts.

### Idea B: Consistency Verification (RLVR Without Verifiers)

Implement a consistency-based RLVR system that improves reasoning quality on non-verifiable domains without ground-truth labels. Start with Qwen2.5-7B-Instruct as the policy model and DeBERTa-v3-large as the consistency verifier backbone. Step 1: generate 100K reasoning chains across diverse domains using the base model on TriviaQA, StrategyQA, OpenBookQA, ETHICS benchmark, SciQ, and ARC-Challenge — use CoT prompting with diverse few-shot examples. Step 2: create inconsistency-injected negatives — for each chain, produce 5 corrupted versions via targeted injection of 7 types: logical contradiction (negate a key claim), unsupported claim (insert a fact not derivable from premises), circular reasoning (make step N reference step N+2), scope shift (change the subject mid-argument), dropped premise (remove a key assumption used later), strawman (misrepresent a prior step), and false equivalence (equate unrelated concepts). Implement each corruption type as a separate function using string manipulation + LLM-assisted generation. Step 3: train the consistency verifier as a binary classifier on (chain, label) pairs — 100K positive + 500K negative, using standard cross-entropy loss, batch size 64, learning rate 2e-5, 5 epochs. Step 4: RLVR with consistency reward — use GRPO with the consistency verifier as reward. For each generated reasoning chain, the reward is the minimum per-step consistency score: r = min_i V(s_i | s_1,...,s_{i-1}) where V is the consistency verifier applied to each step conditioned on prior steps. GRPO settings: group size 8, 100K training steps, KL coefficient 0.02. Evaluate on ARC-Challenge (has ground truth for accuracy), GPQA Diamond, OpenBookQA test, MMLU (selected reasoning-heavy subsets), and ETHICS (hard test). Also run human evaluation: 500 randomly sampled reasoning chains rated blind by 3 annotators on logical coherence (1-5 scale). Compare against: base model, self-consistency (k=8), self-refinement (3 rounds), math-PRM transferred to open domains (train PRM on MATH then apply to non-math), and DPO trained on preference pairs from the same domains. Ablations: which inconsistency types contribute most (remove each type one at a time), verifier size (DeBERTa-base vs. DeBERTa-large vs. Llama-3.1-3B), reward aggregation (min vs. mean vs. product), RLVR training duration (10K vs. 50K vs. 100K steps). Critical analysis: measure correlation between consistency scores and human-judged reasoning quality — if r > 0.6, the proxy is validated. Measure the "consistency-accuracy gap": on benchmarks with ground truth, does higher consistency reliably predict higher accuracy? Pitfalls: the verifier overfitting to surface patterns (fix: include hard negatives that are grammatically correct but logically wrong), reward hacking where the model produces trivially consistent but uninformative chains (fix: add a length/informativeness penalty), and domain shift between training and evaluation (fix: train verifier on diverse domains). For publication quality, show the method works on at least 3 genuinely non-verifiable domains where no prior RLVR method has shown improvement.

### Idea C: Latent Process Rewards

Implement a process reward model that operates on transformer hidden states instead of surface text tokens, and demonstrate it resists reward hacking. Use Llama-3.1-8B-Instruct as the base policy model. Step 1: generate 50K solutions to MATH and GSM8K problems with intermediate hidden states saved — at each reasoning step boundary (detected by newline + "Step" pattern or end of sentence), extract the hidden state vector from layer L/2 and layer L (last layer), concatenate them into a 2*hidden_dim vector. This requires modifying the generation loop to hook into the model's forward pass and save intermediate activations, storing them as memory-mapped numpy arrays (expect ~200GB for 50K solutions). Step 2: label each step's correctness by running the remaining steps through a math verifier — a step is "correct" if removing it causes the final answer to become wrong, "incorrect" if the step itself contains a mathematical error. Step 3: train the L-PRM — architecture is a 3-layer MLP (2*4096 → 2048 → 512 → 1 with GELU activations and LayerNorm) that takes the hidden state vector at each step and predicts correctness probability. Train with binary cross-entropy, batch size 256, learning rate 1e-4, 20 epochs. Step 4: THE KEY EXPERIMENT (L-PRM Hacking Test) — train two separate RL runs of the policy model, each for 10K GRPO steps: Run A maximizes L-PRM reward, Run B maximizes a standard surface PRM reward (use Math-Shepherd or train your own surface PRM on the same data). After RL, evaluate BOTH models on held-out MATH-500 with ground-truth verification. The hypothesis: Run A (L-PRM) maintains or improves real accuracy; Run B (surface PRM) improves PRM score but degrades real accuracy. This is the paper's core result. Step 5: cross-domain transfer — take the L-PRM trained on math hidden states and use it as reward for RLVR on GPQA and ARC-Challenge. If hidden-state quality signals transfer across domains better than surface-text signals, that's a major finding. Compare against 5 baselines: outcome reward model (ORM), Math-Shepherd PRM (surface), PQM from ICLR 2026 if available, an L-PRM trained on random labels (control), and RLVR with no PRM (outcome-only). Ablations: which layers to extract (L/4, L/2, 3L/4, L), step boundary detection (newline vs. sentence vs. learned), L-PRM architecture (MLP vs. small transformer), training set size (5K, 10K, 25K, 50K solutions). Pitfalls: hidden state extraction is slow (fix: batch extraction, use activation hooks not re-running forward passes), hidden states might be too model-specific to transfer (fix: test on both Llama and Qwen to check generality), and step boundary detection affects everything (fix: ablate and report sensitivity). For publication quality, the hacking test graph (PRM score vs. real accuracy over RL training steps, for both L-PRM and surface PRM) is THE key figure — it must be clean and convincing.

### Idea D: Cooperative Self-Play

Implement single-model cooperative self-play for reasoning via dual-role GRPO training. Use Qwen2.5-7B-Instruct as the base model with 3 LoRA adaptors (rank 16): solver_lora, challenger_lora, and revision_lora. Training loop (each iteration): (1) Solver generates solution S using solver_lora + base — standard CoT on a MATH/AIME problem. (2) Challenger generates critique C of S using challenger_lora + base — system prompt "Find errors in this solution. If correct, explain why." (3) If C claims error: Revision generates S' using revision_lora + base, given S and C. (4) Compute rewards: solver_reward = correctness(S), challenger_reward = found_real_error(C)*0.5 + correctly_approved_correct_solution(C)*0.3 + critique_novelty(C)*0.2, revision_reward = correctness(S') - correctness(S). found_real_error is 1 if C identifies an error AND the solution is actually wrong (verified by math checker). correctly_approved is 1 if C says "correct" AND the solution is actually correct. critique_novelty is measured by semantic distance from previous critiques (using embedding similarity). (5) GRPO update each LoRA separately with its role-specific reward. Use group size 8, 100K total iterations, KL coefficient 0.01 per LoRA. Anti-collapse mechanisms: (a) minimum challenge rate — if challenger approves >90% of solutions in a window, add penalty to challenger_reward; (b) critique diversity bonus — reward critiques that are semantically different from the last 10 critiques; (c) every 10K steps, shuffle LoRA assignments to prevent over-specialization. Evaluate on MATH-500, AIME 2024, GPQA Diamond, ARC-Challenge. At test time, run the full pipeline: solver generates → challenger critiques → revision improves. Report: final accuracy, correction rate (challenger catches what % of solver errors), revision improvement rate, and "debate transcript quality" (human evaluation of critique usefulness). Compare against: base model, standard GRPO (single role), self-consistency (k=16), multi-agent debate (3 copies of base model, prompt-based roles), Reflexion (3 rounds), and SCoRe. Ablations: number of self-play iterations (10K, 50K, 100K), LoRA rank (4, 16, 64), with vs. without anti-collapse, solver/challenger reward balance, number of LoRA personas (1, 3, 5). Pitfalls: LoRA interference when training 3 adaptors on the same base (fix: use separate optimizers per LoRA, update one at a time), challenger generating vacuous critiques (fix: the novelty bonus + minimum challenge rate), and training instability from competing objectives (fix: gradient clipping per role, monitor each role's reward separately). For publication quality, include: a figure showing the "arms race" between solver and challenger accuracy over training (both should improve), examples of increasingly sophisticated challenger critiques over training time, and ablation proving that self-play training is essential (removing any phase degrades final performance).

### Idea E: Attention-Based Causal Credit

Implement attention-based causal credit assignment for multi-turn agent RL. Use Llama-3.1-8B-Instruct as the agent model. The key modification to standard GRPO: instead of using trajectory-level reward R for all steps, compute per-step credit A_t = influence_t * R, where influence_t measures how much action at step t affected the model's final-step hidden state. Influence computation: at the end of a trajectory, compute attention rollout from the final token's attention to all previous tokens. For each action step t, influence_t = sum of attention rollout weights on all tokens belonging to action t's output, normalized to sum to 1 across all steps. Use attention rollout (the product of attention matrices across layers, following Abnar & Zuidema 2020) from the last layer only for efficiency, or from layers L/2 to L for more accuracy. This is computed in a single additional forward pass with attention weights saved. Modified GRPO: advantage_t = influence_t * (R - baseline) instead of advantage = R - baseline for all steps. Everything else in GRPO remains standard. Train for 50K steps with group size 8, learning rate 1e-6, KL coefficient 0.02. Evaluate on 4 multi-turn agent benchmarks: ALFWorld (text game, 134 tasks), InterCode-SQL (database querying, 153 tasks), SWE-Bench-Lite (25 issues), and a subset of WebArena (50 tasks — pick tasks completable in <20 steps). Measure: success rate at convergence, steps to reach 50% success rate (convergence speed), and credit assignment quality (compute correlation between our influence scores and "ground-truth importance" estimated by leave-one-out: how much does success rate drop if step t is replaced with random action). Compare against 6 baselines: trajectory-level GRPO (all steps get same credit), uniform per-step (credit = 1/T for all steps), temporal decay (credit decays with distance from reward), HCAPO hierarchical credit (2603.08754) if reproducible, attention-weighted but RANDOM attention (control for attention structure), and an oracle per-step reward (hand-labeled for a subset). Ablations: attention layer selection (last only vs. last 6 vs. all), attribution method (rollout vs. gradient-based vs. raw last-layer attention), influence normalization (softmax vs. linear vs. top-k), and training on environments of different complexity (ALFWorld=simple → WebArena=complex). Pitfalls: attention rollout can be noisy for very long trajectories (fix: use sliding-window attention rollout, last 50 tokens only), influence scores might be uniform for models with broad attention patterns (fix: report attention entropy and filter runs where attribution is degenerate), and the method might not help on simple environments where all steps matter equally (fix: evaluate across environments of varying complexity to show the method shines on complex tasks). For publication quality, include: a visualization of influence scores on a sample trajectory showing they align with intuitively important actions, a convergence curve showing 2x speedup over trajectory-level GRPO, and the correlation plot between attention-based influence and leave-one-out importance.

---

## Stage 8: Final Honesty Check

**Most likely to be secretly already explored:** Idea D (Cooperative Self-Play). Self-play for reasoning is a natural idea that multiple groups may be working on. The specific RL self-play mechanism with dual roles is novel, but someone at DeepMind or Meta could easily have an unpublished version.

**Most likely to fail empirically:** Idea E (Attention-Based Credit). Attention ≠ causation is a real objection. The influence scores might not actually correlate with action importance, especially in complex environments.

**Most likely to be too much engineering:** Idea A (Adversarial Self-Correction). The three-phase pipeline has many moving parts. If the adversary is poorly calibrated, the whole system fails. Getting the balance right requires significant tuning.

**Best risk-reward tradeoff:** Idea B (Consistency Verification). If it works, it opens RLVR to all non-verifiable domains — enormous impact. The method is clean and easy to implement. The risk (consistency ≠ correctness) is addressable with the right experiments. Even partial success is publishable.

**Personal bet for top conference:** Idea C (Latent Process Rewards). The L-PRM hacking test is a KILLER experiment — it directly, cleanly, and dramatically shows the difference between L-PRM and surface PRM. If the graph shows L-PRM maintaining accuracy while surface PRM degrades, that single figure sells the paper. The contribution is deep (challenges the entire PRM paradigm) and the experiment is elegant.

| Idea | Novelty | Risk | Impact | Executability | Overall Bet |
|------|:---:|:---:|:---:|:---:|:---:|
| A. Adversarial Self-Correction | 8 | Med | High | Med | Strong |
| B. Consistency Verification | 9 | Med | Very High | High | **Strongest** |
| C. Latent Process Rewards | 9 | Med | High | Med | **Strongest** |
| D. Cooperative Self-Play | 7 | Med-High | Med-High | Med | Moderate-Strong |
| E. Attention-Based Credit | 7 | High | High | High | Moderate |
