import numpy as np
from rlh_bench.envs.capacity_scheduling import (
    RecoverableCapacitySchedulingEnv, CapacitySchedulingConfig
)

cfg = CapacitySchedulingConfig(
    horizon=500, num_projects=16, num_modes=4, num_products=4,
    action_dim=32, n_bundles=2,
)
N_SEEDS = 20

policies = [
    ('zero', lambda r, o: np.zeros(32, dtype=np.float32)),
    ('ones', lambda r, o: np.ones(32, dtype=np.float32)),
    ('half_pos', lambda r, o: 0.5 * np.ones(32, dtype=np.float32)),
    ('rand_pm', lambda r, o: r.uniform(-1, 1, size=32).astype(np.float32)),
    ('rand_pos', lambda r, o: r.uniform(0, 1, size=32).astype(np.float32)),
]

for name, pol in policies:
    fills, mand, success = [], [], []
    wears, heats, churn = [], [], []
    energy, late, inv = [], [], []
    for seed in range(N_SEEDS):
        env = RecoverableCapacitySchedulingEnv(cfg, reward_mode='vector')
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        for _ in range(500):
            obs, r, term, trunc, info = env.step(pol(rng, obs))
            if term: break
        rv = info['reward_vector']
        fills.append(rv[1]); mand.append(rv[2])
        success.append(int(info['is_success']))
        wears.append(info['mean_wear']); heats.append(info['mean_heat'])
        churn.append(info['total_setup_churn'])
        energy.append(info['total_energy']); late.append(info['total_lateness']); inv.append(info['total_inventory_waste'])
    print(f'{name:12s} succ={np.mean(success):.2f} fill={np.mean(fills):.2f} '
          f'mand={np.mean(mand):.2f} wear={np.mean(wears):.2f} '
          f'heat={np.mean(heats):.2f} churn={np.mean(churn):.1f} '
          f'energy={np.mean(energy):.0f} late={np.mean(late):.2f} inv={np.mean(inv):.2f}')
