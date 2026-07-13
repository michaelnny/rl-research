"""Evaluator-owned FactorLab runner for process-isolated candidate algorithms."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rlx_bench.budget import BudgetLedger, BudgetLimits, BudgetedEnv
from rlx_bench.factorlab import FactorLabConfig, FactorLabEnv, ObjectiveProtocol
from rlx_bench.metrics import normalize_returns
from rlx_bench.suite import EvaluatorWorldSuite, WorldBand, WorldSuiteSpec

from .ipc import CandidateClient, CandidateProcessLimits, CandidateProtocolError


PROTOCOL_VERSION = "rlx-candidate-jsonl-v1"


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    horizon: int = 64
    n_factors: int = 4
    max_causal_lag: int = 64
    memory_lag: int = 0
    reward_events: int = 1
    conflict_strength: float = 1.0
    training_episodes: int = 64
    training_trials: int = 3
    public_worlds: int = 4
    heldout_worlds: int = 8
    wall_seconds: float = 900.0
    response_seconds: float = 5.0
    preference: tuple[float, ...] = (1.0, 0.0)
    suite_namespace: str = "factorlab-v0-candidate-eval"
    suite_version: int = 0

    def __post_init__(self) -> None:
        if (
            self.training_episodes < 1
            or self.training_trials < 1
            or self.public_worlds < 1
            or self.heldout_worlds < 1
        ):
            raise ValueError("episode and world counts must be positive")
        if self.wall_seconds <= 0.0 or self.response_seconds <= 0.0:
            raise ValueError("time limits must be positive")
        if len(self.preference) != 2:
            raise ValueError("v1 evaluator currently supports a two-objective preference")

    def factor_config(self) -> FactorLabConfig:
        return FactorLabConfig(
            horizon=self.horizon,
            n_objectives=2,
            n_factors=self.n_factors,
            action_mode="factored_discrete",
            levels_per_factor=(2,),
            max_causal_lag=self.max_causal_lag,
            memory_lag=self.memory_lag,
            reward_events=self.reward_events,
            conflict_strength=self.conflict_strength,
            protocol=ObjectiveProtocol.PREFERENCE_CONDITIONED,
        )


def _initialize(
    client: CandidateClient,
    *,
    phase: str,
    checkpoint: Any,
    public_manifest: dict[str, Any],
    preference: tuple[float, ...],
    trial_index: int,
    trial_seed: int,
) -> None:
    response = client.request(
        {
            "type": "init",
            "protocol": PROTOCOL_VERSION,
            "phase": phase,
            "checkpoint": checkpoint,
            "public_suite_manifest": public_manifest,
            "preference": preference,
            "trial_index": trial_index,
            "trial_seed": trial_seed,
        }
    )
    if response != {"type": "ready"}:
        raise CandidateProtocolError("candidate init response must be {'type': 'ready'}")


def _episode(
    client: CandidateClient,
    env: BudgetedEnv,
    *,
    phase: str,
    preference: tuple[float, ...],
) -> tuple[float, ...]:
    observation, info = env.reset(preference=preference)
    client.notify(
        {
            "type": "reset",
            "phase": phase,
            "observation": observation,
            "public_info": info,
        }
    )
    rewards: list[tuple[float, ...]] = []
    terminated = False
    while not terminated:
        if observation["action_required"]:
            response = client.request(
                {"type": "act", "phase": phase, "observation": observation}
            )
            if response.get("type") != "action" or set(response) != {"type", "action"}:
                raise CandidateProtocolError("act response must contain only type and action")
            action = response["action"]
        else:
            action = None
        try:
            next_observation, reward, terminated, truncated, step_info = env.step(action)
        except (TypeError, ValueError) as exc:
            raise CandidateProtocolError(f"candidate action was rejected: {exc}") from exc
        rewards.append(reward)
        client.notify(
            {
                "type": "transition",
                "phase": phase,
                "reward_vector": reward,
                "observation": next_observation,
                "terminated": terminated,
                "truncated": truncated,
                "public_info": step_info,
            }
        )
        observation = next_observation
    returns = tuple(float(value) for value in np.sum(np.asarray(rewards), axis=0))
    client.notify(
        {
            "type": "episode_end",
            "phase": phase,
            "return_vector": returns,
        }
    )
    return returns


def _checkpoint(client: CandidateClient, max_bytes: int) -> tuple[Any, str]:
    response = client.request({"type": "checkpoint"})
    if response.get("type") != "checkpoint" or set(response) != {"type", "state"}:
        raise CandidateProtocolError("checkpoint response must contain only type and state")
    try:
        encoded = json.dumps(response["state"], sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise CandidateProtocolError("candidate checkpoint is not finite JSON") from exc
    if len(encoded) > max_bytes:
        raise CandidateProtocolError("candidate checkpoint exceeds message limit")
    return response["state"], hashlib.sha256(encoded).hexdigest()


def _analytic_references(conflict_strength: float, weights: np.ndarray) -> tuple[float, float]:
    """Return random-policy utility and the best cue-aware per-factor utility."""

    first_weight, second_weight = (float(value) for value in weights)
    aligned = first_weight + second_weight * (1.0 - conflict_strength**2)
    opposed = second_weight * (2.0 * conflict_strength - conflict_strength**2)
    random_utility = 0.5 * (aligned + opposed)
    return random_utility, max(aligned, opposed)


def evaluate_candidate(
    candidate_argv: tuple[str, ...] | list[str],
    *,
    cwd: Path,
    master_key: bytes,
    config: CandidateEvaluationConfig = CandidateEvaluationConfig(),
    unreadable_roots: tuple[Path, ...] = (),
    unwritable_roots: tuple[Path, ...] = (),
    sandbox: bool | None = None,
) -> dict[str, Any]:
    factor_config = config.factor_config()
    suite = EvaluatorWorldSuite(
        factor_config,
        WorldSuiteSpec(
            namespace=config.suite_namespace,
            version=config.suite_version,
            master_key=master_key,
            public_worlds=config.public_worlds,
            tune_worlds=1,
            heldout_worlds=config.heldout_worlds,
            audit_worlds=1,
        ),
    )
    total_episodes = config.training_trials * (
        config.training_episodes + config.heldout_worlds
    )
    ledger = BudgetLedger(
        BudgetLimits(
            transitions=total_episodes * config.horizon,
            episodes=total_episodes,
            wall_seconds=config.wall_seconds,
            policies=config.training_trials,
        )
    )
    process_limits = CandidateProcessLimits(response_seconds=config.response_seconds)
    public_manifest = suite.public_manifest().to_dict()
    stderr_records: list[dict[str, Any]] = []
    checkpoint_digests: list[str] = []
    try:
        heldout_returns: list[tuple[float, ...]] = []
        for trial_index in range(config.training_trials):
            ledger.register_policy()
            trial_seed = int.from_bytes(
                hmac.new(
                    master_key,
                    f"rlx-candidate-trial-v1|{config.suite_namespace}|{trial_index}".encode(),
                    hashlib.sha256,
                ).digest()[:8],
                "big",
            ) & (2**63 - 1)
            with CandidateClient(
                candidate_argv,
                cwd=cwd,
                limits=process_limits,
                unreadable_roots=unreadable_roots,
                unwritable_roots=unwritable_roots,
                sandbox=sandbox,
            ) as training_client:
                _initialize(
                    training_client,
                    phase="training",
                    checkpoint=None,
                    public_manifest=public_manifest,
                    preference=config.preference,
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                )
                for episode in range(config.training_episodes):
                    world = suite.world(WorldBand.PUBLIC, episode % config.public_worlds)
                    _episode(
                        training_client,
                        BudgetedEnv(FactorLabEnv(world), ledger),
                        phase="training",
                        preference=config.preference,
                    )
                checkpoint, checkpoint_sha256 = _checkpoint(
                    training_client, process_limits.max_message_bytes
                )
                checkpoint_digests.append(checkpoint_sha256)
                training_client.close()
                digest, size = training_client.stderr_digest()
                stderr_records.append(
                    {
                        "phase": "training",
                        "trial_index": trial_index,
                        "sha256": digest,
                        "bytes": size,
                    }
                )

            for index in range(config.heldout_worlds):
                with CandidateClient(
                    candidate_argv,
                    cwd=cwd,
                    limits=process_limits,
                    unreadable_roots=unreadable_roots,
                    unwritable_roots=unwritable_roots,
                    sandbox=sandbox,
                ) as evaluation_client:
                    _initialize(
                        evaluation_client,
                        phase="evaluation",
                        checkpoint=checkpoint,
                        public_manifest=public_manifest,
                        preference=config.preference,
                        trial_index=trial_index,
                        trial_seed=trial_seed,
                    )
                    world = suite.world(WorldBand.HELDOUT, index)
                    heldout_returns.append(
                        _episode(
                            evaluation_client,
                            BudgetedEnv(FactorLabEnv(world), ledger),
                            phase="evaluation",
                            preference=config.preference,
                        )
                    )
                    evaluation_client.close()
                    digest, size = evaluation_client.stderr_digest()
                    stderr_records.append(
                        {
                            "phase": "evaluation",
                            "trial_index": trial_index,
                            "world_index": index,
                            "sha256": digest,
                            "bytes": size,
                        }
                    )
        returns = np.asarray(heldout_returns, dtype=np.float64)
        world_bound = suite.world(WorldBand.HELDOUT, 0).return_upper_bound
        normalized = normalize_returns(
            returns,
            lower=(0.0,) * factor_config.n_objectives,
            upper=world_bound,
        )
        weights = np.asarray(config.preference, dtype=np.float64)
        weights /= weights.sum()
        utilities = normalized @ weights
        random_utility, analytic_ceiling = _analytic_references(
            config.conflict_strength, weights
        )
        usage = ledger.snapshot()
        return {
            "status": "complete",
            "protocol": PROTOCOL_VERSION,
            "task_id": factor_config.task_id,
            "suite_id": suite.suite_id,
            "objective_protocol": factor_config.protocol.value,
            "training_episodes": config.training_episodes,
            "training_trials": config.training_trials,
            "heldout_worlds": config.heldout_worlds,
            "heldout_evaluations": len(heldout_returns),
            "normalized_return_mean": [float(value) for value in normalized.mean(axis=0)],
            "normalized_return_std": [float(value) for value in normalized.std(axis=0)],
            "normalized_utility_mean": float(utilities.mean()),
            "normalized_utility_std": float(utilities.std()),
            "random_policy_expected_utility": random_utility,
            "analytic_cue_policy_ceiling": analytic_ceiling,
            "improvement_over_random": float(utilities.mean()) - random_utility,
            "regret_to_ceiling": analytic_ceiling - float(utilities.mean()),
            "checkpoint_sha256": checkpoint_digests,
            "budget_usage": asdict(usage),
            "candidate_stderr": stderr_records,
            "heldout_identifiers_exposed": False,
        }
    except Exception as exc:
        usage = ledger.snapshot()
        return {
            "status": "candidate_error",
            "protocol": PROTOCOL_VERSION,
            "task_id": factor_config.task_id,
            "suite_id": suite.suite_id,
            "objective_protocol": factor_config.protocol.value,
            "error_class": type(exc).__name__,
            "error_detail": str(exc)[:1000],
            "budget_usage": asdict(usage),
            "candidate_stderr": stderr_records,
            "heldout_identifiers_exposed": False,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlx-evaluate-candidate")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--n-factors", type=int, default=4)
    parser.add_argument("--max-causal-lag", type=int, default=64)
    parser.add_argument("--memory-lag", type=int, default=0)
    parser.add_argument("--reward-events", type=int, default=1)
    parser.add_argument("--conflict-strength", type=float, default=1.0)
    parser.add_argument("--training-episodes", type=int, default=64)
    parser.add_argument("--training-trials", type=int, default=3)
    parser.add_argument("--public-worlds", type=int, default=4)
    parser.add_argument("--heldout-worlds", type=int, default=8)
    parser.add_argument("--wall-seconds", type=float, default=900.0)
    parser.add_argument("--response-seconds", type=float, default=5.0)
    parser.add_argument("--suite-namespace", default="factorlab-v0-candidate-eval")
    parser.add_argument("--suite-version", type=int, default=0)
    parser.add_argument("--key-file-env", default="RLX_FACTORLAB_SUITE_KEY_FILE")
    parser.add_argument("--no-sandbox", action="store_true")
    parser.add_argument("--candidate", nargs=argparse.REMAINDER, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate = list(args.candidate)
    if candidate and candidate[0] == "--":
        candidate.pop(0)
    if not candidate:
        raise SystemExit("--candidate requires an executable argv")
    key_file = os.environ.get(args.key_file_env)
    if key_file is None:
        raise SystemExit(
            f"missing evaluator key-file environment variable {args.key_file_env}"
        )
    key_path = Path(key_file)
    try:
        metadata = key_path.stat()
        master_key = key_path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"cannot read evaluator key file {key_path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise SystemExit("evaluator key file must be regular and have owner-only permissions")
    if len(master_key) != 32:
        raise SystemExit("evaluator key file must contain exactly 32 bytes")
    unreadable = tuple(
        Path(path)
        for path in os.environ.get("RLX_UNREADABLE_ROOTS", "").split(os.pathsep)
        if path
    )
    unwritable = tuple(
        Path(path)
        for path in os.environ.get("RLX_UNWRITABLE_ROOTS", "").split(os.pathsep)
        if path
    )
    report = evaluate_candidate(
        candidate,
        cwd=Path.cwd(),
        master_key=master_key,
        config=CandidateEvaluationConfig(
            horizon=args.horizon,
            n_factors=args.n_factors,
            max_causal_lag=args.max_causal_lag,
            memory_lag=args.memory_lag,
            reward_events=args.reward_events,
            conflict_strength=args.conflict_strength,
            training_episodes=args.training_episodes,
            training_trials=args.training_trials,
            public_worlds=args.public_worlds,
            heldout_worlds=args.heldout_worlds,
            wall_seconds=args.wall_seconds,
            response_seconds=args.response_seconds,
            suite_namespace=args.suite_namespace,
            suite_version=args.suite_version,
        ),
        unreadable_roots=unreadable,
        unwritable_roots=unwritable,
        sandbox=False if args.no_sandbox else None,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
