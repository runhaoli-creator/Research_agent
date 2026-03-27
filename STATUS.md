# Research_agent — Status

## Current Phase: Ideation Complete → Ready for Implementation

## Selected Idea: DynaCLIP (Cycle 2 winner, replaces PhysLang)
Physics-grounded visual representations via contrastive dynamics alignment. "CLIP aligned vision with language. DynaCLIP aligns vision with physics." Targeting NeurIPS 2026.

## Research Question
Can visual representations aligned with physical dynamics similarity (instead of semantic/visual similarity) dramatically improve physics-aware manipulation and enable zero-shot physical property inference?

## Key Milestones
- [x] Literature review (70+ papers surveyed across 2 cycles)
- [x] Idea generation Cycle 1 (8 ideas, all poster-level)
- [x] Brutal self-critique + novelty verification (6 ideas checked against 100+ papers)
- [x] Idea generation Cycle 2 (2 new ideas, DynaCLIP reaches oral potential)
- [x] Implementation spec written (comprehensive coding agent prompt)
- [ ] Data generation (ManiSkill3, 500K images + dynamics fingerprints)
- [ ] DynaCLIP pre-training (contrastive learning, 100K steps)
- [ ] Core evaluation (probing, invisible physics test, zero-shot)
- [ ] Downstream experiments (world model, diffusion policy, LIBERO, CALVIN)
- [ ] Ablation studies (8 ablations)
- [ ] Analysis (t-SNE, sensitivity, cross-domain transfer)
- [ ] Paper writing

## Decisions
- **2026-03-27 (Cycle 1):** Initially selected PhysLang. After brutal self-critique, recognized it as poster-level due to Oracle baseline problem.
- **2026-03-27 (Cycle 2):** Selected DynaCLIP over PhysLang, WorldSearch, CounterFact, and Zero-Success Learning. DynaCLIP has strongest novelty moat (confirmed novel against 25+ representation learning papers), cleanest narrative ("CLIP for physics"), and highest oral potential (8/10).
- **Why DynaCLIP > PhysLang:** PhysLang conditions a world model on language descriptions of physics — but "why not just use the numbers?" is a devastating critique. DynaCLIP learns physics-grounded REPRESENTATIONS that benefit ALL downstream tasks, and the invisible physics test is a killer experiment.
