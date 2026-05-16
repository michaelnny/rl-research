---
thread: <thread-slug>
primary_benchmark: <env_id>            # one of: ALE/MontezumaRevenge-v5 | humanoid.run | minecart-v0
sanity_envs: [CartPole-v1, Pendulum-v1]   # default; override only with justification (and explain why below)
pillar: <sparse-long-horizon | long-horizon-dense | multi-signal>
seeds: [42, 43]                            # >=2 in early phase
---

# Hypothesis: <short title>

## Claim

2-4 sentences. What is the algorithmic mechanism, and why might it address
the named pillar? Be specific about the mechanism — vague claims are
`needs-sharpening`.

## How it differs structurally from PPO and Q-learning

Be specific. What mathematical object is the optimization target? What signal
flows where? Why is this *not* `∇ log π · A` or a Bellman update?

If your method has a gradient anywhere, say what it is the gradient *of* and
what it is the gradient *with respect to*.

## Implementation sketch

```
# 5-15 lines of pseudocode.
# Identify the novel primitives by name.
# This must match the structure of the eventual train.py.
```

## What success would look like

Qualitative. NOT "X% more return."

Acceptable examples:
- "Training curve is non-monotone, consistent with the multi-modal credit signal."
- "Compute scales sub-linearly in episode length, unlike PPO."
- "On multi-signal envs, learns Pareto-distinct policies under different scalarizations."

## Falsification

What observation would convince you this idea is wrong? If you cannot answer
this, your hypothesis is not a hypothesis.
