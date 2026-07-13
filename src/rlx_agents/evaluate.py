"""Evaluator-owned Neural FactorLab runner for isolated neural candidates."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rlx_bench.budget import BudgetLedger, BudgetLimits, BudgetedEnv
from rlx_bench.factorlab import FactorLabConfig, FactorLabEnv, ObjectiveProtocol
from rlx_bench.metrics import normalize_returns
from rlx_bench.suite import EvaluatorWorldSuite, WorldBand, WorldSuiteSpec

from .ipc import CandidateClient, CandidateProcessLimits, CandidateProtocolError


PROTOCOL_VERSION = "rlx-neural-candidate-jsonl-v2"


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    horizon: int = 5000
    n_factors: int = 12
    levels_per_factor: int = 10
    signal_dim: int = 16
    context_dim: int = 8
    state_dim: int = 8
    teacher_hidden_dim: int = 16
    max_causal_lag: int = 5000
    memory_lag: int = 0
    reward_events: int = 1
    conflict_strength: float = 0.75
    terminal_state_weight: float = 1.0
    training_episodes: int = 256
    training_batch_size: int = 64
    training_trials: int = 3
    public_worlds: int = 32
    heldout_worlds: int = 16
    wall_seconds: float = 14_400.0
    response_seconds: float = 30.0
    max_parameters: int = 2_000_000
    max_checkpoint_bytes: int = 64 * 1024 * 1024
    preference: tuple[float, ...] = (1.0, 0.0)
    suite_namespace: str = "factorlab-long-v1-neural-eval"
    suite_version: int = 1

    def __post_init__(self) -> None:
        counts = (
            self.training_episodes,
            self.training_batch_size,
            self.training_trials,
            self.public_worlds,
            self.heldout_worlds,
        )
        if any(value < 1 for value in counts):
            raise ValueError("episode, batch, trial, and world counts must be positive")
        if self.wall_seconds <= 0.0 or self.response_seconds <= 0.0:
            raise ValueError("time limits must be positive")
        if self.max_parameters < 1 or self.max_checkpoint_bytes < 1024:
            raise ValueError("model and checkpoint limits must be positive")
        if len(self.preference) != 2:
            raise ValueError("v2 evaluator currently supports a two-objective preference")

    def factor_config(self) -> FactorLabConfig:
        return FactorLabConfig(
            horizon=self.horizon,
            n_objectives=2,
            n_factors=self.n_factors,
            action_mode="factored_discrete",
            levels_per_factor=(self.levels_per_factor,),
            signal_dim=self.signal_dim,
            context_dim=self.context_dim,
            state_dim=self.state_dim,
            teacher_hidden_dim=self.teacher_hidden_dim,
            max_causal_lag=self.max_causal_lag,
            memory_lag=self.memory_lag,
            reward_events=self.reward_events,
            conflict_strength=self.conflict_strength,
            terminal_state_weight=self.terminal_state_weight,
            protocol=ObjectiveProtocol.PREFERENCE_CONDITIONED,
        )


def _public_task_spec(suite: EvaluatorWorldSuite, preference: tuple[float, ...]) -> dict[str, Any]:
    _, info = FactorLabEnv(suite.world(WorldBand.PUBLIC, 0)).reset(preference=preference)
    return info


def _validate_model_manifest(value: Any, max_parameters: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateProtocolError("ready response needs a model_manifest object")
    required = {
        "model_family",
        "architecture",
        "framework",
        "trainable_parameters",
        "recurrent",
        "device",
    }
    if set(value) != required:
        raise CandidateProtocolError(f"model_manifest fields must be exactly {sorted(required)}")
    if value["model_family"] != "neural_policy":
        raise CandidateProtocolError("candidate must declare model_family=neural_policy")
    if not isinstance(value["architecture"], str) or not value["architecture"]:
        raise CandidateProtocolError("candidate architecture must be a non-empty string")
    if not isinstance(value["framework"], str) or not value["framework"]:
        raise CandidateProtocolError("candidate framework must be a non-empty string")
    parameters = value["trainable_parameters"]
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters < 1:
        raise CandidateProtocolError("trainable_parameters must be a positive integer")
    if parameters > max_parameters:
        raise CandidateProtocolError(
            f"candidate declares {parameters} parameters, above cap {max_parameters}"
        )
    if not isinstance(value["recurrent"], bool):
        raise CandidateProtocolError("recurrent must be boolean")
    if value["device"] not in {"cpu", "cuda", "mps"}:
        raise CandidateProtocolError("device must be cpu, cuda, or mps")
    return dict(value)


def _initialize(
    client: CandidateClient,
    *,
    phase: str,
    checkpoint: dict[str, Any] | None,
    public_manifest: dict[str, Any],
    task_spec: dict[str, Any],
    preference: tuple[float, ...],
    trial_index: int,
    trial_seed: int,
    max_parameters: int,
) -> dict[str, Any]:
    response = client.request(
        {
            "type": "init",
            "protocol": PROTOCOL_VERSION,
            "phase": phase,
            "checkpoint": checkpoint,
            "public_suite_manifest": public_manifest,
            "task_spec": task_spec,
            "preference": preference,
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "model_budget": {"max_trainable_parameters": max_parameters},
        }
    )
    if response.get("type") != "ready" or set(response) != {"type", "model_manifest"}:
        raise CandidateProtocolError("init response must contain type=ready and model_manifest")
    return _validate_model_manifest(response["model_manifest"], max_parameters)


def _episode_batch(
    client: CandidateClient,
    environments: Sequence[BudgetedEnv],
    *,
    phase: str,
    preference: tuple[float, ...],
    batch_index: int,
) -> tuple[tuple[float, ...], ...]:
    reset_values = [env.reset(preference=preference) for env in environments]
    observations = [value[0] for value in reset_values]
    public_info = [value[1] for value in reset_values]
    client.notify(
        {
            "type": "reset_batch",
            "phase": phase,
            "batch_index": batch_index,
            "observations": observations,
            "public_info": public_info,
        }
    )
    returns = np.zeros((len(environments), len(preference)), dtype=np.float64)
    for step in range(public_info[0]["horizon"]):
        required = [bool(observation["action_required"]) for observation in observations]
        if any(required):
            response = client.request(
                {
                    "type": "act_batch",
                    "phase": phase,
                    "batch_index": batch_index,
                    "step": step,
                    "observations": observations,
                    "action_required": required,
                }
            )
            if response.get("type") != "actions" or set(response) != {"type", "actions"}:
                raise CandidateProtocolError("act_batch response must contain type and actions")
            actions = response["actions"]
            if not isinstance(actions, list) or len(actions) != len(environments):
                raise CandidateProtocolError("candidate returned the wrong action batch size")
        else:
            actions = [None] * len(environments)
        next_observations: list[dict[str, Any]] = []
        rewards: list[tuple[float, ...]] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        step_info: list[dict[str, Any]] = []
        for index, (env, action) in enumerate(zip(environments, actions, strict=True)):
            if not required[index] and action is not None:
                raise CandidateProtocolError("candidate supplied an action for a warm-up slot")
            try:
                observation, reward, done, cut, info = env.step(action)
            except (TypeError, ValueError) as exc:
                raise CandidateProtocolError(f"candidate action was rejected: {exc}") from exc
            next_observations.append(observation)
            rewards.append(reward)
            terminated.append(done)
            truncated.append(cut)
            step_info.append(info)
            returns[index] += np.asarray(reward)
        client.notify(
            {
                "type": "transition_batch",
                "phase": phase,
                "batch_index": batch_index,
                "step": step,
                "reward_vectors": rewards,
                "observations": next_observations,
                "terminated": terminated,
                "truncated": truncated,
                "public_info": step_info,
            }
        )
        observations = next_observations
    result = tuple(tuple(float(value) for value in row) for row in returns)
    client.notify(
        {
            "type": "episode_end_batch",
            "phase": phase,
            "batch_index": batch_index,
            "return_vectors": result,
        }
    )
    return result


def _checkpoint(client: CandidateClient, max_bytes: int) -> tuple[bytes, str]:
    response = client.request(
        {
            "type": "checkpoint",
            "format": "binary_file",
            "max_bytes": max_bytes,
        }
    )
    if response.get("type") != "checkpoint" or set(response) != {
        "type",
        "artifact",
        "sha256",
    }:
        raise CandidateProtocolError(
            "checkpoint response must contain type, artifact, and sha256"
        )
    content = client.read_artifact(response["artifact"], max_bytes=max_bytes)
    digest = hashlib.sha256(content).hexdigest()
    if response["sha256"] != digest:
        raise CandidateProtocolError("candidate checkpoint digest does not match artifact")
    return content, digest


def _random_reference(
    suite: EvaluatorWorldSuite,
    preference: tuple[float, ...],
    *,
    policies: int = 8,
) -> tuple[float, float]:
    weights = np.asarray(preference, dtype=np.float64)
    weights /= weights.sum()
    values: list[float] = []
    for policy_seed in range(policies):
        rng = np.random.default_rng(policy_seed)
        for index in range(suite.spec.heldout_worlds):
            world = suite.world(WorldBand.HELDOUT, index)
            env = FactorLabEnv(world)
            observation, _ = env.reset(preference=tuple(float(value) for value in weights))
            rewards = np.zeros(world.config.n_objectives)
            for _ in range(world.config.horizon):
                action = world.action_spec.sample(rng) if observation["action_required"] else None
                observation, reward, _, _, _ = env.step(action)
                rewards += np.asarray(reward)
            values.append(float(np.dot(weights, rewards / np.asarray(world.return_upper_bound))))
    return float(np.mean(values)), float(np.std(values))


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
            tune_worlds=max(2, config.public_worlds // 4),
            heldout_worlds=config.heldout_worlds,
            audit_worlds=max(2, config.public_worlds // 4),
        ),
    )
    total_episodes = config.training_trials * (config.training_episodes + config.heldout_worlds)
    ledger = BudgetLedger(
        BudgetLimits(
            transitions=total_episodes * config.horizon,
            episodes=total_episodes,
            wall_seconds=config.wall_seconds,
            policies=config.training_trials,
        )
    )
    limits = CandidateProcessLimits(
        response_seconds=config.response_seconds,
        max_message_bytes=64 * 1024 * 1024,
    )
    public_manifest = suite.public_manifest().to_dict()
    task_spec = _public_task_spec(suite, config.preference)
    stderr_records: list[dict[str, Any]] = []
    checkpoint_digests: list[str] = []
    model_manifests: list[dict[str, Any]] = []
    training_wall_seconds = 0.0
    try:
        heldout_returns: list[tuple[float, ...]] = []
        for trial_index in range(config.training_trials):
            ledger.register_policy()
            trial_seed = int.from_bytes(
                hmac.new(
                    master_key,
                    f"rlx-neural-candidate-trial-v2|{config.suite_namespace}|{trial_index}".encode(),
                    hashlib.sha256,
                ).digest()[:8],
                "big",
            ) & (2**63 - 1)
            started = time.monotonic()
            with CandidateClient(
                candidate_argv,
                cwd=cwd,
                limits=limits,
                unreadable_roots=unreadable_roots,
                unwritable_roots=unwritable_roots,
                sandbox=sandbox,
            ) as training_client:
                manifest = _initialize(
                    training_client,
                    phase="training",
                    checkpoint=None,
                    public_manifest=public_manifest,
                    task_spec=task_spec,
                    preference=config.preference,
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                    max_parameters=config.max_parameters,
                )
                model_manifests.append(manifest)
                for batch_start in range(0, config.training_episodes, config.training_batch_size):
                    batch_size = min(
                        config.training_batch_size,
                        config.training_episodes - batch_start,
                    )
                    worlds = [
                        suite.world(
                            WorldBand.PUBLIC,
                            (batch_start + offset) % config.public_worlds,
                        )
                        for offset in range(batch_size)
                    ]
                    _episode_batch(
                        training_client,
                        [BudgetedEnv(FactorLabEnv(world), ledger) for world in worlds],
                        phase="training",
                        preference=config.preference,
                        batch_index=batch_start // config.training_batch_size,
                    )
                checkpoint, digest = _checkpoint(
                    training_client, config.max_checkpoint_bytes
                )
                checkpoint_digests.append(digest)
                training_client.close()
                stderr_digest, size = training_client.stderr_digest()
                stderr_records.append(
                    {
                        "phase": "training",
                        "trial_index": trial_index,
                        "sha256": stderr_digest,
                        "bytes": size,
                    }
                )
            training_wall_seconds += time.monotonic() - started

            checkpoint_descriptor = {
                "format": "binary_file",
                "artifact": "checkpoint.bin",
                "sha256": checkpoint_digests[-1],
                "bytes": len(checkpoint),
            }
            for index in range(config.heldout_worlds):
                with CandidateClient(
                    candidate_argv,
                    cwd=cwd,
                    limits=limits,
                    unreadable_roots=unreadable_roots,
                    unwritable_roots=unwritable_roots,
                    seed_artifacts={"checkpoint.bin": checkpoint},
                    sandbox=sandbox,
                ) as evaluation_client:
                    evaluation_manifest = _initialize(
                        evaluation_client,
                        phase="evaluation",
                        checkpoint=checkpoint_descriptor,
                        public_manifest=public_manifest,
                        task_spec=task_spec,
                        preference=config.preference,
                        trial_index=trial_index,
                        trial_seed=trial_seed,
                        max_parameters=config.max_parameters,
                    )
                    comparable_fields = {
                        "model_family",
                        "architecture",
                        "framework",
                        "trainable_parameters",
                        "recurrent",
                    }
                    if any(evaluation_manifest[field] != manifest[field] for field in comparable_fields):
                        raise CandidateProtocolError(
                            "evaluation model manifest does not match training manifest"
                        )
                    values = _episode_batch(
                        evaluation_client,
                        [BudgetedEnv(FactorLabEnv(suite.world(WorldBand.HELDOUT, index)), ledger)],
                        phase="evaluation",
                        preference=config.preference,
                        batch_index=index,
                    )
                    heldout_returns.append(values[0])
                    evaluation_client.close()
                    stderr_digest, size = evaluation_client.stderr_digest()
                    stderr_records.append(
                        {
                            "phase": "evaluation",
                            "trial_index": trial_index,
                            "world_index": index,
                            "sha256": stderr_digest,
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
        random_mean, random_std = _random_reference(suite, config.preference)
        usage = ledger.snapshot()
        accelerator_upper = training_wall_seconds if any(
            manifest["device"] in {"cuda", "mps"} for manifest in model_manifests
        ) else 0.0
        return {
            "status": "complete",
            "protocol": PROTOCOL_VERSION,
            "task_id": factor_config.task_id,
            "suite_id": suite.suite_id,
            "objective_protocol": factor_config.protocol.value,
            "training_episodes": config.training_episodes,
            "training_batch_size": config.training_batch_size,
            "training_trials": config.training_trials,
            "heldout_worlds": config.heldout_worlds,
            "heldout_evaluations": len(heldout_returns),
            "normalized_return_mean": [float(value) for value in normalized.mean(axis=0)],
            "normalized_return_std": [float(value) for value in normalized.std(axis=0)],
            "normalized_utility_mean": float(utilities.mean()),
            "normalized_utility_std": float(utilities.std()),
            "random_policy_utility_mean": random_mean,
            "random_policy_utility_std": random_std,
            "improvement_over_random": float(utilities.mean()) - random_mean,
            "regret_to_declared_upper_bound": 1.0 - float(utilities.mean()),
            "model_manifests": model_manifests,
            "checkpoint_sha256": checkpoint_digests,
            "budget_usage": asdict(usage),
            "training_wall_seconds": training_wall_seconds,
            "accelerator_seconds_upper_bound": accelerator_upper,
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
            "model_manifests": model_manifests,
            "budget_usage": asdict(usage),
            "candidate_stderr": stderr_records,
            "heldout_identifiers_exposed": False,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlx-evaluate-candidate")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--n-factors", type=int, default=12)
    parser.add_argument("--levels-per-factor", type=int, default=10)
    parser.add_argument("--signal-dim", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--teacher-hidden-dim", type=int, default=16)
    parser.add_argument("--max-causal-lag", type=int, default=5000)
    parser.add_argument("--memory-lag", type=int, default=0)
    parser.add_argument("--reward-events", type=int, default=1)
    parser.add_argument("--conflict-strength", type=float, default=0.75)
    parser.add_argument("--terminal-state-weight", type=float, default=1.0)
    parser.add_argument("--training-episodes", type=int, default=256)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument("--training-trials", type=int, default=3)
    parser.add_argument("--public-worlds", type=int, default=32)
    parser.add_argument("--heldout-worlds", type=int, default=16)
    parser.add_argument("--wall-seconds", type=float, default=14_400.0)
    parser.add_argument("--response-seconds", type=float, default=30.0)
    parser.add_argument("--max-parameters", type=int, default=2_000_000)
    parser.add_argument("--max-checkpoint-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--suite-namespace", default="factorlab-long-v1-neural-eval")
    parser.add_argument("--suite-version", type=int, default=1)
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
        raise SystemExit(f"missing evaluator key-file environment variable {args.key_file_env}")
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
            levels_per_factor=args.levels_per_factor,
            signal_dim=args.signal_dim,
            context_dim=args.context_dim,
            state_dim=args.state_dim,
            teacher_hidden_dim=args.teacher_hidden_dim,
            max_causal_lag=args.max_causal_lag,
            memory_lag=args.memory_lag,
            reward_events=args.reward_events,
            conflict_strength=args.conflict_strength,
            terminal_state_weight=args.terminal_state_weight,
            training_episodes=args.training_episodes,
            training_batch_size=args.training_batch_size,
            training_trials=args.training_trials,
            public_worlds=args.public_worlds,
            heldout_worlds=args.heldout_worlds,
            wall_seconds=args.wall_seconds,
            response_seconds=args.response_seconds,
            max_parameters=args.max_parameters,
            max_checkpoint_bytes=args.max_checkpoint_bytes,
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
