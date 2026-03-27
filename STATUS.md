# Research_agent — Status

## Current Phase: Ideation → Implementation

## Selected Idea: PhysLang
Language-grounded world models conditioned on physical property descriptions for zero-shot dynamics generalization in robot manipulation. Targeting NeurIPS 2026.

## Research Question
Can conditioning a world model on natural language descriptions of physical properties (mass, friction, elasticity) enable zero-shot generalization to novel physical property combinations in manipulation?

## Key Milestones
- [x] Literature review (50+ papers surveyed)
- [x] Idea formulation (8 ideas generated, top 3 with extended abstracts)
- [x] Novelty verification (all 3 top ideas confirmed novel via web search)
- [ ] Prototype implementation
- [ ] Data generation (ManiSkill3, 500K trajectories)
- [ ] PhysLang model training
- [ ] Baseline experiments (DreamerV3, TD-MPC2, AdaptiGraph, vision-only, oracle)
- [ ] Ablation studies (8 ablations)
- [ ] Analysis (t-SNE, sensitivity, failure cases)
- [ ] Paper writing
- [ ] Submission

## Decisions
- **2026-03-27:** Selected PhysLang over WorldSearch (inference-time scaling) and CounterFact (counterfactual trajectories) due to strongest novelty moat. WorldSearch risks being scooped by Stanford Pavone lab (RoboMonkey/CoVer line).
- **2026-03-27:** ManiSkill3 chosen as primary simulator (GPU-parallelized, programmable physics). Isaac Lab as secondary for photorealistic rendering ablation.
