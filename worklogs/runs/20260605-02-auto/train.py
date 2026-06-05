"""Suffix-Inheritance Trie (SIT).

Run 20260605-02-auto.

Primitive: an observation-keyed action-suffix trie. Each node stores a
hash of the observation reached at its prefix and a small bag of outcome
vectors collected by every recorded suffix passing through that node.

Improvement operator: Pareto-dominant suffix inheritance with verification.
After every episode, scan node pairs (n, n') sharing an observation hash;
if a suffix from n' Pareto-dominates n's best, run ONE verification rollout
that executes (path-to-n) ⧺ donor-suffix in a fresh env. Commit the graft
if the rollout replicates dominance, else mark (h(n), donor-suffix) non-
fungible so the same losing graft is never retried.

Side information: transition geometry (observation-equivalence as the
graft-point rule) + vector diagnostics (Pareto dominance read directly
from info["vector"] on vector envs).

Contract:
    train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn

For vector envs the operator reads info["vector"]; the dominance order is
componentwise Pareto. For scalar envs, dominance is `>` with a shorter-
suffix structural tiebreak.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import harness

# ---------------------------------------------------------------------------
# observation hashing
# ---------------------------------------------------------------------------


def obs_hash(obs) -> bytes:
    """Lightweight, collision-resistant observation hash (8 bytes)."""
    arr = np.asarray(obs)
    if arr.dtype.kind in "fc":
        # quantize floats lightly so near-equal floats collide on purpose
        arr = np.round(arr.astype(np.float32), 3)
    b = arr.tobytes()
    # SHA-1 then truncate to 8 bytes; cheap and good enough as a key.
    return hashlib.sha1(b).digest()[:8]


# ---------------------------------------------------------------------------
# trie data structure
# ---------------------------------------------------------------------------


@dataclass
class TrieNode:
    """Trie node keyed by the action that led here from its parent."""

    obs_hash: bytes | None = None
    # children: action -> child node
    children: dict[int, TrieNode] = field(default_factory=dict)
    # bag of (terminal_outcome_vector, suffix_length_from_here)
    # outcome vector: shape (k,) for vector envs, shape (1,) for scalar envs
    outcomes: list[tuple[np.ndarray, int]] = field(default_factory=list)
    # visit count, used in softmax sampling fallback
    visits: int = 0


@dataclass
class GraftCandidate:
    """A proposed donor suffix to graft under a recipient node."""

    recipient_path: tuple[int, ...]
    donor_suffix: tuple[int, ...]
    predicted_outcome: np.ndarray


class Trie:
    def __init__(self, n_actions: int, k_outcome: int, vector: bool):
        self.root = TrieNode()
        self.n_actions = n_actions
        self.k = k_outcome
        self.vector = vector
        # observation-hash -> list of (path-tuple, node)
        self.hash_index: dict[bytes, list[tuple[tuple[int, ...], TrieNode]]] = defaultdict(list)
        # set of (recipient_obs_hash, donor_suffix) pairs that failed verification
        self.non_fungible: set[tuple[bytes, tuple[int, ...]]] = set()

    def descend(self, path: tuple[int, ...]) -> TrieNode | None:
        node = self.root
        for a in path:
            if a not in node.children:
                return None
            node = node.children[a]
        return node

    def insert_trajectory(
        self,
        actions: list[int],
        obs_hashes: list[bytes],
        terminal_outcome: np.ndarray,
    ) -> None:
        """Insert a full trajectory into the trie.

        actions[i] is the action taken at step i.
        obs_hashes[i] is the hash of the observation reached AFTER actions[:i]
        (so obs_hashes[0] = hash of initial obs; len = len(actions)+1).
        terminal_outcome: shape (k,) np.float64 vector accumulated over episode.
        """
        node = self.root
        # tag root with starting-obs hash
        if node.obs_hash is None:
            node.obs_hash = obs_hashes[0]
            self.hash_index[obs_hashes[0]].append(((), node))
        node.visits += 1
        node.outcomes.append((terminal_outcome.copy(), len(actions)))
        path: list[int] = []
        for i, a in enumerate(actions):
            path.append(a)
            child = node.children.get(a)
            new_child = child is None
            if new_child:
                child = TrieNode()
                node.children[a] = child
            node = child
            # observation reached AFTER taking a is obs_hashes[i+1]
            h = obs_hashes[i + 1] if i + 1 < len(obs_hashes) else None
            if (new_child and h is not None) or (h is not None and node.obs_hash is None):
                node.obs_hash = h
                self.hash_index[h].append((tuple(path), node))
            node.visits += 1
            remaining_len = len(actions) - (i + 1)
            node.outcomes.append((terminal_outcome.copy(), remaining_len))
            # cap bag size to keep memory bounded
            if len(node.outcomes) > 8:
                # drop the oldest dominated entry
                node.outcomes = node.outcomes[-8:]

    # ------------------------- dominance helpers -------------------------

    def _dominates(self, a: np.ndarray, b: np.ndarray, len_a: int, len_b: int) -> bool:
        """Strict Pareto dominance with structural tiebreak for scalar envs."""
        if self.vector:
            return bool(np.all(a >= b) and np.any(a > b))
        # scalar: a strictly greater, or equal-but-shorter.
        if a[0] > b[0]:
            return True
        if a[0] == b[0] and len_a < len_b:
            return True
        return False

    def _best_outcome(self, node: TrieNode) -> tuple[np.ndarray, int] | None:
        if not node.outcomes:
            return None
        best = node.outcomes[0]
        for o, ln in node.outcomes[1:]:
            if self._dominates(o, best[0], ln, best[1]):
                best = (o, ln)
        return best

    # ------------------------- sampling policy ---------------------------

    def sample_action(
        self,
        node: TrieNode,
        rng: np.random.Generator,
        eps: float = 0.2,
    ) -> int:
        """Sample next action via softmax over child best-outcome scalar score.

        For unexplored actions we maintain a uniform mixture (eps); this is
        the only stochastic exploration knob the algorithm uses.
        """
        unexplored = [a for a in range(self.n_actions) if a not in node.children]
        if unexplored and rng.random() < eps:
            return int(rng.choice(unexplored))
        if not node.children:
            return int(rng.integers(self.n_actions))
        # collapse Pareto outcome to a scalar score for sampling-only purposes
        # (this is NOT scalarization of the reward signal — outcomes stored in
        # the trie remain vectors; this is only the softmax temperature input).
        actions = list(node.children.keys())
        scores = []
        for a in actions:
            child = node.children[a]
            best = self._best_outcome(child)
            if best is None:
                scores.append(0.0)
            else:
                # use componentwise mean as a sampling proxy; ties broken by
                # negative suffix length (prefer shorter winning suffixes).
                scores.append(float(best[0].sum()) - 0.001 * best[1])
        scores = np.asarray(scores, dtype=np.float64)
        scores -= scores.max()
        probs = np.exp(scores / max(1.0, scores.std() + 1e-6))
        probs /= probs.sum()
        # ε-mixture with unexplored actions
        if unexplored:
            mix = np.zeros(self.n_actions, dtype=np.float64)
            for a, p in zip(actions, probs):
                mix[a] = (1.0 - eps) * p
            for a in unexplored:
                mix[a] = eps / len(unexplored)
            return int(rng.choice(self.n_actions, p=mix / mix.sum()))
        return int(rng.choice(actions, p=probs))

    # ------------------------- graft proposal ----------------------------

    def propose_grafts(self, max_proposals: int = 8) -> list[GraftCandidate]:
        """For each obs hash with multiple paths, find dominant donor suffixes.

        Returns at most max_proposals candidates.
        """
        proposals: list[GraftCandidate] = []
        # iterate hash buckets with at least 2 paths
        for h, paths in self.hash_index.items():
            if len(paths) < 2 or len(proposals) >= max_proposals:
                continue
            # for each unordered pair (recipient, donor), check best-suffix
            # outcome at the donor against the recipient's best continuation.
            for i in range(len(paths)):
                for j in range(len(paths)):
                    if i == j:
                        continue
                    recip_path, recip_node = paths[i]
                    donor_path, donor_node = paths[j]
                    if len(proposals) >= max_proposals:
                        break
                    recip_best = self._best_outcome(recip_node)
                    donor_best = self._best_outcome(donor_node)
                    if recip_best is None or donor_best is None:
                        continue
                    if not self._dominates(
                        donor_best[0],
                        recip_best[0],
                        donor_best[1],
                        recip_best[1],
                    ):
                        continue
                    # extract donor suffix: action sequence from donor_node
                    # following best-outcome children until terminal.
                    suffix = self._extract_best_suffix(donor_node, max_len=64)
                    if not suffix:
                        continue
                    key = (h, tuple(suffix))
                    if key in self.non_fungible:
                        continue
                    proposals.append(
                        GraftCandidate(
                            recipient_path=recip_path,
                            donor_suffix=tuple(suffix),
                            predicted_outcome=donor_best[0].copy(),
                        )
                    )
        return proposals

    def _extract_best_suffix(self, node: TrieNode, max_len: int) -> list[int]:
        """Walk best-outcome children from node, returning the action sequence."""
        out: list[int] = []
        cur = node
        for _ in range(max_len):
            if not cur.children:
                break
            best_action = None
            best_score: tuple[np.ndarray, int] | None = None
            for a, ch in cur.children.items():
                ch_best = self._best_outcome(ch)
                if ch_best is None:
                    continue
                if best_score is None or self._dominates(
                    ch_best[0], best_score[0], ch_best[1], best_score[1]
                ):
                    best_action = a
                    best_score = ch_best
            if best_action is None:
                break
            out.append(best_action)
            cur = cur.children[best_action]
        return out

    def best_root_path(self, max_len: int = 256) -> list[int]:
        """Deterministic best-from-root walk for deployment."""
        return self._extract_best_suffix(self.root, max_len=max_len)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------


def rollout_with_policy(
    env,
    sample_fn,
    rng: np.random.Generator,
    is_vector: bool,
    k: int,
    max_steps: int = 1000,
    seed: int | None = None,
) -> tuple[list[int], list[bytes], np.ndarray]:
    """Run one episode using sample_fn(node_history, last_obs) -> action.

    Returns (actions, obs_hashes, terminal_outcome_vector).
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    actions: list[int] = []
    obs_hashes: list[bytes] = [obs_hash(obs)]
    outcome = np.zeros(k, dtype=np.float64)
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = sample_fn(steps, obs, obs_hashes)
        obs, reward, term, trunc, info = env.step(int(action))
        actions.append(int(action))
        obs_hashes.append(obs_hash(obs))
        if is_vector:
            vec = np.asarray(info["vector"], dtype=np.float64)
            outcome += vec
        else:
            outcome[0] += float(reward)
        done = bool(term) or bool(trunc)
        steps += 1
    return actions, obs_hashes, outcome


def execute_fixed_actions(
    env,
    actions: list[int],
    is_vector: bool,
    k: int,
    seed: int | None = None,
) -> tuple[list[bytes], np.ndarray, bool]:
    """Execute a fixed action sequence to verify a graft.

    Returns (obs_hashes, terminal_outcome, finished_naturally). If the env
    terminates before all actions are consumed, we stop.
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    obs_hashes: list[bytes] = [obs_hash(obs)]
    outcome = np.zeros(k, dtype=np.float64)
    for a in actions:
        obs, reward, term, trunc, info = env.step(int(a))
        obs_hashes.append(obs_hash(obs))
        if is_vector:
            vec = np.asarray(info["vector"], dtype=np.float64)
            outcome += vec
        else:
            outcome[0] += float(reward)
        if bool(term) or bool(trunc):
            return obs_hashes, outcome, True
    return obs_hashes, outcome, False


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(env_id: str, seed: int, time_budget_s: int) -> harness.PolicyFn:
    t0 = time.monotonic()
    is_vector = harness.ENV_TYPE[env_id] == "vector"

    # discover action-space size and outcome dim
    env = harness.make_env(env_id, seed)
    n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 0
    if n_actions <= 0:
        env.close()
        # fall back: random policy if action space is continuous (shouldn't be)
        return _random_policy(env_id, seed)

    if is_vector:
        # probe one step to learn vector dim
        obs, _ = env.reset(seed=seed)
        try:
            _o, _r, _t, _tr, info = env.step(0)
            vec = np.asarray(info["vector"], dtype=np.float64)
            k = int(vec.shape[0])
        except Exception:
            k = int(harness.HV_REF[env_id].shape[0])
        env.close()
        env = harness.make_env(env_id, seed)
    else:
        k = 1

    trie = Trie(n_actions=n_actions, k_outcome=k, vector=is_vector)
    rng = np.random.default_rng(seed + 12345)

    n_episodes = 0
    n_proposals = 0
    n_grafts_committed = 0
    n_grafts_rejected = 0

    # --- main loop: alternate exploration episodes with graft passes ---

    deadline = t0 + max(5, int(time_budget_s) - 5)
    eps = 0.35  # exploration rate; decays slightly over time
    max_episode_steps = min(harness.MAX_EPISODE_STEPS, 400)

    while time.monotonic() < deadline:
        # exploration episode driven by trie sampling
        def sample_fn(_step_idx, _obs, hashes_so_far, _eps=eps):
            # walk the trie to the corresponding node from the actions taken
            # so far; then sample the next action.
            # We track the node by descending action history in trie space.
            # We compute it from the hashes_so_far position (== depth so far).
            # Action history equals current actions; we descend on actions.
            # Simpler approach: pass actions taken so far via closure.
            return trie.sample_action(_node[0], rng, eps=_eps)

        _node = [trie.root]

        actions: list[int] = []
        obs_hashes_local: list[bytes] = []
        outcome_local = np.zeros(k, dtype=np.float64)

        # rolled out manually so we can update _node[0] each step
        ep_seed = int(seed + 7919 * (n_episodes + 1))
        obs, _ = env.reset(seed=ep_seed)
        obs_hashes_local.append(obs_hash(obs))
        # tag root once
        if trie.root.obs_hash is None:
            trie.root.obs_hash = obs_hashes_local[0]
            trie.hash_index[obs_hashes_local[0]].append(((), trie.root))
        done = False
        steps = 0
        while not done and steps < max_episode_steps:
            a = trie.sample_action(_node[0], rng, eps=eps)
            obs, reward, term, trunc, info = env.step(int(a))
            actions.append(int(a))
            obs_hashes_local.append(obs_hash(obs))
            if is_vector:
                vec = np.asarray(info["vector"], dtype=np.float64)
                outcome_local += vec
            else:
                outcome_local[0] += float(reward)
            done = bool(term) or bool(trunc)
            steps += 1
            # advance trie cursor
            if a in _node[0].children:
                _node[0] = _node[0].children[a]
            else:
                # node will be created at insert; but for sampling cursor
                # within this episode we only need to descend along this
                # action; create a transient empty node if absent.
                _node[0] = TrieNode()  # transient; not added to trie

            if time.monotonic() >= deadline:
                break

        trie.insert_trajectory(actions, obs_hashes_local, outcome_local)
        n_episodes += 1

        # --- graft pass ---
        # propose a small number of grafts; verify each in a fresh env.
        if time.monotonic() >= deadline:
            break
        proposals = trie.propose_grafts(max_proposals=4)
        for cand in proposals:
            if time.monotonic() >= deadline:
                break
            n_proposals += 1
            full_actions = list(cand.recipient_path) + list(cand.donor_suffix)
            # verify in a fresh env instance with the SAME initial seed bucket
            # so the dynamics are deterministic-comparable for envs that are
            # seed-deterministic. We use a small offset to vary the trial.
            verify_seed = int(seed + 17 * (n_proposals + 1))
            verify_env = harness.make_env(env_id, verify_seed)
            try:
                v_hashes, v_outcome, _finished = execute_fixed_actions(
                    verify_env,
                    full_actions,
                    is_vector,
                    k,
                    seed=verify_seed,
                )
            finally:
                verify_env.close()

            # determine the recipient node's current best outcome
            recip_node = trie.descend(cand.recipient_path)
            if recip_node is None:
                continue
            recip_best = trie._best_outcome(recip_node)
            verified_dominates = recip_best is None or trie._dominates(
                v_outcome, recip_best[0], len(cand.donor_suffix), recip_best[1]
            )

            if verified_dominates:
                # commit: insert verified trajectory into trie
                trie.insert_trajectory(full_actions, v_hashes, v_outcome)
                n_grafts_committed += 1
            else:
                trie.non_fungible.add((recip_node.obs_hash or b"", cand.donor_suffix))
                n_grafts_rejected += 1

        # gentle eps decay so late exploration follows the trie's best signal
        eps = max(0.05, eps * 0.995)

    env.close()
    train_s = time.monotonic() - t0
    print(
        f"[train] env={env_id} seed={seed} eps_episodes={n_episodes} "
        f"grafts_proposed={n_proposals} committed={n_grafts_committed} "
        f"rejected={n_grafts_rejected} "
        f"trie_nodes={_count_nodes(trie.root)} "
        f"hash_buckets={len(trie.hash_index)} "
        f"non_fungible={len(trie.non_fungible)} "
        f"train_s={train_s:.1f} budget_s={time_budget_s}",
        flush=True,
    )

    # --- build deployment policy ---
    # The deployment policy walks best-from-root; if the eval episode goes
    # off-trie (e.g. the eval seed produces a different observation sequence
    # than any recorded trajectory), we fall back to trie-softmax sampling
    # with eps=0 (deterministic best child).
    best_path = trie.best_root_path(max_len=512)
    rng_eval = np.random.default_rng(seed + 99_991)
    state = {"step": 0, "node": trie.root, "obs_seen": None}

    def policy_fn(obs):
        h = obs_hash(obs)
        # if at the start of an episode (heuristic: obs is initial), reset
        if state["obs_seen"] is None or h == (trie.root.obs_hash or b""):
            # detect new episode by obs equality with root hash
            if state["obs_seen"] is not None and h != (trie.root.obs_hash or b""):
                pass  # not actually new
            else:
                state["step"] = 0
                state["node"] = trie.root
        state["obs_seen"] = h
        # play recorded best path while it lasts
        if state["step"] < len(best_path):
            a = best_path[state["step"]]
            state["step"] += 1
            if a in state["node"].children:
                state["node"] = state["node"].children[a]
            return int(a)
        # off-path: deterministic descent by best-child outcome
        node = state["node"]
        if not node.children:
            return int(rng_eval.integers(n_actions))
        best_action = None
        best_score: tuple[np.ndarray, int] | None = None
        for act, ch in node.children.items():
            cb = trie._best_outcome(ch)
            if cb is None:
                continue
            if best_score is None or trie._dominates(cb[0], best_score[0], cb[1], best_score[1]):
                best_action = act
                best_score = cb
        if best_action is None:
            best_action = int(rng_eval.integers(n_actions))
        if best_action in node.children:
            state["node"] = node.children[best_action]
        state["step"] += 1
        return int(best_action)

    return policy_fn


def _count_nodes(node: TrieNode) -> int:
    n = 1
    for c in node.children.values():
        n += _count_nodes(c)
    return n


def _random_policy(env_id: str, seed: int) -> harness.PolicyFn:
    env = harness.make_env(env_id, seed)
    action_space = env.action_space
    env.close()
    rng = np.random.default_rng(seed + 99_999)

    def policy_fn(_obs):
        if hasattr(action_space, "n"):
            return int(rng.integers(int(action_space.n)))
        return action_space.sample()

    return policy_fn


# ---------------------------------------------------------------------------
# CLI shell (matches repo train.py)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=harness.ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-s", type=int, default=harness.TIME_BUDGET_S)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()
    policy = train(args.env, args.seed, args.time_budget_s)
    score = harness.evaluate(policy, args.env, seed=args.seed)
    print("---", flush=True)
    print(f"env:           {args.env}", flush=True)
    print(f"seed:          {args.seed}", flush=True)
    print(f"env_type:      {harness.ENV_TYPE[args.env]}", flush=True)
    print(f"wallclock_s:   {time.monotonic() - t0:.1f}", flush=True)
    print(f"final_score:   {score:.6f}", flush=True)


if __name__ == "__main__":
    main()
