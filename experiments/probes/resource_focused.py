"""Tiny stateless policies for resource-allocation ablations.

Used in session 0001 to probe what the heuristic ResourceGreedyPolicy
leaves on the table. Kept here so a future session can reuse them
without retyping. Not part of the substrate.
"""

from __future__ import annotations

import numpy as np


def focused(obs: np.ndarray, env) -> np.ndarray:
    """Allocate the full budget to the leftmost incomplete project.

    Saturates safe_allocation (so it accrues quadratic safety violation),
    but on every resource env this trades success-and-service against the
    safety component and wins under the default scalarization weights.
    """
    k = env.config.num_projects
    ratios = obs[:k]
    a = np.zeros(k, dtype=np.float32)
    for i in range(k):
        if ratios[i] < 1.0:
            a[i] = float(env.config.budget)
            break
    return a


def safe_focused(obs: np.ndarray, env) -> np.ndarray:
    """Allocate exactly safe_allocation to the leftmost incomplete project;
    leave the remaining budget *unallocated* (action sums to safe_allocation).

    Zero safety violation, but typically too slow to hit success=1
    inside the horizon. Useful as a control that isolates the
    "concentration" axis from the "amount" axis.
    """
    k = env.config.num_projects
    ratios = obs[:k]
    a = np.zeros(k, dtype=np.float32)
    sa = float(env.config.safe_allocation)
    for i in range(k):
        if ratios[i] < 1.0:
            a[i] = sa
            break
    return a
