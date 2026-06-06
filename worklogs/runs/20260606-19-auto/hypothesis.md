# 20260606-19-auto -- empty-handed

reason: After reading exemplars.md, prior_attempts.md (including the new
Substrate budget constraint section), the recent ledger entries, and the
hypothesis + curator records for runs 13-18, I cannot produce a probe
that simultaneously satisfies all of (a) novelty boundary clear of dead
families A-H and the disqualifier list, (b) load-bearing primitive that
fires at random initialization (no reward-bootstrap dependency), (c)
ablation that produces visibly different *training dynamics* on a logged
scalar rather than tied final scores, and (d) reasonable expectation of
clearing both the Reviewer schema/novelty bar and producing measurable
signal within the 120s quick or sparse stage budget. The fresh regions
attempted in this turn (replicator dynamics on latent state embeddings;
maximum-entropy-production-rate policies; Frank-Wolfe on per-state
simplex driven by transition-conditional kernel density; Stein
variational policy ensembles in distribution space; Pólya-urn pursuit
on action-conditional path-recurrence times; Sinkhorn iterative
proportional fitting on the empirical transition matrix; Kantorovich
potential ascent on consecutive state-occupancy distributions) all
collapse to one of: (i) reward-free curiosity / count-based / RND
disqualifiers (any "maximize change in state distribution" or
"transition-novelty" signal), (ii) skill-discovery / mutual-information
rebadges (DIAYN/VIC structurally), (iii) Stein Variational Policy
Gradient (Liu-Wang 2017, published), (iv) intrinsic-motivation
behaviour-cloning of self (k-NN action imitation), (v) replicator
dynamics with reward fitness (a known framing of multi-agent learning
gradients), or (vi) family C "geometric within-trajectory statistic"
when the fitness is reduced to an inner-product on local features. The
substrate-budget constraint compounds this: even a structurally novel
score-function probe will land at the DST=99 / RG=0.011 floor in 120s,
making the discriminating observable invisible from panel output alone.
The honest call is empty-handed rather than another null-result run
that consumes panel budget without exercising a load-bearing primitive
or producing a clean falsifier on a logged training scalar.
