# Baselines

Frozen baseline results live here. Do not regenerate baselines as part of the
loop — they are the yardstick.

- `random.json` — random-policy returns (mean over 100 episodes) for every env
  in the suite. Built once; consumed by the sanity gate to compute the
  "strictly above random" pass criterion.
- `ppo/<env_id>/` — frozen PPO baseline runs (one subdir per benchmark) with
  full TensorBoard logs and `result.json` per seed. The hyperparameters that
  produced these are documented inline in `src/rl_research/baselines/ppo.py`.

PPO baselines are run once with frozen hyperparameters per domain. They are
NOT re-tuned per candidate.
