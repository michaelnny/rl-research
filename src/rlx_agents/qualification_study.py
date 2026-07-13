"""Preregistered qualification study for a named Neural FactorLab tier."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rlx_bench.audit import audit_causal_contract
from rlx_bench.factorlab import (
    BENCHMARK_REVISION,
    FactorLabConfig,
    FactorLabEnv,
    FactorLabInspector,
    generate_world,
)
from rlx_bench.independent_audit import nonlinear_midpoint_residual, run_independent_audit
from rlx_bench.oracle import exact_weighted_solution
from rlx_bench.qualification import (
    CheckStatus,
    QualificationCheck,
    QualificationReport,
    make_qualification_report,
)
from rlx_bench.suite import EvaluatorWorldSuite, WorldBand, WorldSuiteSpec

from .evaluate import _random_reference
from .neural import NeuralReferenceConfig, evaluate_actor_critic, train_actor_critic


@dataclass(frozen=True)
class QualificationStudyResult:
    report: QualificationReport
    evidence_bundle: dict[str, Any]
    evidence_sha256: str


def _progress(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("qualification protocol root must be an object")
    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return value, digest


def _config(value: Mapping[str, Any]) -> FactorLabConfig:
    fields = {field.name for field in dataclasses.fields(FactorLabConfig)}
    unknown = set(value) - fields
    if unknown:
        raise ValueError(f"unknown FactorLab config fields: {sorted(unknown)}")
    payload = dict(value)
    if "levels_per_factor" in payload:
        levels = payload["levels_per_factor"]
        payload["levels_per_factor"] = (
            (int(levels),) if isinstance(levels, int) else tuple(int(item) for item in levels)
        )
    if "effects" in payload:
        payload["effects"] = tuple(payload["effects"])
    return FactorLabConfig(**payload)


def _suite(
    config: FactorLabConfig,
    protocol: Mapping[str, Any],
    master_key: bytes,
    *,
    namespace_suffix: str = "",
) -> EvaluatorWorldSuite:
    suite = protocol["suite"]
    return EvaluatorWorldSuite(
        config,
        WorldSuiteSpec(
            namespace=str(suite["namespace"]) + namespace_suffix,
            version=int(suite["version"]),
            master_key=master_key,
            public_worlds=int(suite["public_worlds"]),
            tune_worlds=int(suite["tune_worlds"]),
            heldout_worlds=int(suite["heldout_worlds"]),
            audit_worlds=int(suite["audit_worlds"]),
        ),
    )


def _worlds(suite: EvaluatorWorldSuite, band: WorldBand) -> tuple[Any, ...]:
    count = suite.public_manifest().band_counts[band.value]
    return tuple(suite.world(band, index) for index in range(count))


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size < 2:
        return float(samples.mean()), float(samples.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, samples.size, size=(draws, samples.size))
    means = samples[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def _train_variant(
    suite: EvaluatorWorldSuite,
    reference: Mapping[str, Any],
    architecture: str,
    seeds: Sequence[int],
    preference: tuple[float, ...],
    device: str,
) -> tuple[list[dict[str, Any]], int]:
    public_worlds = _worlds(suite, WorldBand.PUBLIC)
    heldout_worlds = _worlds(suite, WorldBand.HELDOUT)
    records: list[dict[str, Any]] = []
    failures = 0
    for seed in seeds:
        _progress("reference_seed_started", architecture=architecture, seed=int(seed))
        started = time.monotonic()
        try:
            trained = train_actor_critic(
                public_worlds,
                preference,
                config=NeuralReferenceConfig(
                    hidden_size=int(reference["hidden_size"]),
                    architecture=architecture,
                    residual_blocks=int(reference["residual_blocks"]),
                    transformer_layers=int(reference["transformer_layers"]),
                    transformer_heads=int(reference["transformer_heads"]),
                    context_window=int(reference["context_window"]),
                    learning_rate=float(reference["learning_rate"]),
                    entropy_coefficient=float(reference["entropy_coefficient"]),
                    value_coefficient=float(reference["value_coefficient"]),
                    episodes=int(reference["episodes"]),
                    batch_episodes=int(reference["batch_episodes"]),
                    optimization_batch_size=int(reference["optimization_batch_size"]),
                    optimization_epochs=int(reference["optimization_epochs"]),
                    ppo_clip=float(reference["ppo_clip"]),
                    truncated_bptt_steps=int(reference["truncated_bptt_steps"]),
                    max_parameters=int(reference["max_parameters"]),
                    device=device,
                ),
                seed=int(seed),
            )
            evaluation_device = trained.manifest.device
            public_results = evaluate_actor_critic(
                trained.model, public_worlds, preference, device=evaluation_device
            )
            heldout_results = evaluate_actor_critic(
                trained.model, heldout_worlds, preference, device=evaluation_device
            )
            records.append(
                {
                    "training_seed": int(seed),
                    "architecture": architecture,
                    "manifest": asdict(trained.manifest),
                    "transitions": trained.transitions,
                    "wall_seconds": time.monotonic() - started,
                    "training_tail_utility": float(
                        np.mean(
                            trained.episode_utilities[
                                -min(64, len(trained.episode_utilities)) :
                            ]
                        )
                    ),
                    "public_utility_mean": float(
                        np.mean([result.normalized_utility for result in public_results])
                    ),
                    "heldout_utility_mean": float(
                        np.mean([result.normalized_utility for result in heldout_results])
                    ),
                    "heldout_utility_std": float(
                        np.std([result.normalized_utility for result in heldout_results])
                    ),
                }
            )
            _progress(
                "reference_seed_completed",
                architecture=architecture,
                seed=int(seed),
                wall_seconds=time.monotonic() - started,
            )
        except Exception as exc:  # a failed seed is scientific evidence, not a retry
            failures += 1
            records.append(
                {
                    "training_seed": int(seed),
                    "architecture": architecture,
                    "failure_class": type(exc).__name__,
                    "failure_detail": str(exc)[:500],
                    "wall_seconds": time.monotonic() - started,
                }
            )
            _progress(
                "reference_seed_failed",
                architecture=architecture,
                seed=int(seed),
                failure_class=type(exc).__name__,
            )
    return records, failures


def _successful(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [record for record in records if "heldout_utility_mean" in record]


def _matched_reward_timing_specificity(
    config: FactorLabConfig,
    kernel_key: bytes,
) -> float:
    terminal = generate_world(dataclasses.replace(config, reward_events=1), 8101, kernel_key=kernel_key)
    sparse = generate_world(dataclasses.replace(config, reward_events=4), 8101, kernel_key=kernel_key)
    rng = np.random.default_rng(404)
    actions = [terminal.action_spec.sample(rng) for _ in range(terminal.config.decision_count)]
    first = np.asarray(
        FactorLabInspector(terminal).simulate(actions, preference=(1.0, 0.0)).return_vector
    )
    second = np.asarray(
        FactorLabInspector(sparse).simulate(actions, preference=(1.0, 0.0)).return_vector
    )
    return float(np.max(np.abs(first - second)))


def _mechanics(
    suite: EvaluatorWorldSuite,
    thresholds: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    public = _worlds(suite, WorldBand.PUBLIC)
    heldout = _worlds(suite, WorldBand.HELDOUT)
    public_signals = {
        hashlib.sha256(np.asarray(signal, dtype=np.float64).tobytes()).hexdigest()
        for world in public
        for signal in world.signals
    }
    heldout_signals = {
        hashlib.sha256(np.asarray(signal, dtype=np.float64).tobytes()).hexdigest()
        for world in heldout
        for signal in world.signals
    }
    rng = np.random.default_rng(17)
    transitions = 0
    started = time.monotonic()
    for world in public[: min(8, len(public))]:
        env = FactorLabEnv(world)
        observation, _ = env.reset(preference=(1.0, 0.0))
        for _ in range(world.config.horizon):
            action = world.action_spec.sample(rng) if observation["action_required"] else None
            observation, _, _, _, _ = env.step(action)
            transitions += 1
    elapsed = max(time.monotonic() - started, 1e-9)
    residual = nonlinear_midpoint_residual(public[0])
    return {
        "continuous_public_signal_count": len(public_signals),
        "continuous_heldout_signal_count": len(heldout_signals),
        "public_heldout_signal_overlap": len(public_signals & heldout_signals),
        "nonlinear_midpoint_residual": residual,
        "transitions_per_second": transitions / elapsed,
        "joint_action_choices": suite.config.joint_discrete_choices,
        "horizon": suite.config.horizon,
        "observation_width": suite.config.observation_width,
        "selected_device": device,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "passed": (
            not (public_signals & heldout_signals)
            and suite.config.horizon >= int(thresholds["minimum_anchor_horizon"])
            and suite.config.joint_discrete_choices
            >= int(thresholds["minimum_joint_choices"])
            and residual >= float(thresholds["minimum_nonlinear_residual"])
            and transitions / elapsed
            >= float(thresholds["minimum_env_transitions_per_second"])
        ),
    }


def _scaling_contrast(
    anchor_world: Any,
    protocol: Mapping[str, Any],
    master_key: bytes,
) -> dict[str, Any]:
    contrast = protocol["scaling_contrast"]
    long_horizon = int(contrast["horizon"])
    long_config = dataclasses.replace(
        anchor_world.config,
        horizon=long_horizon,
        max_causal_lag=long_horizon,
    )
    world = generate_world(long_config, 20_000, kernel_key=master_key)
    environment = FactorLabEnv(world)
    observation, _ = environment.reset(preference=(1.0, 0.0))
    rng = np.random.default_rng(20_000)
    started = time.monotonic()
    for _ in range(long_horizon):
        action = world.action_spec.sample(rng) if observation["action_required"] else None
        observation, _, _, _, _ = environment.step(action)
    elapsed = max(time.monotonic() - started, 1e-9)
    edges = FactorLabInspector(world).influence_edges()
    return {
        "anchor_horizon": anchor_world.config.horizon,
        "long_horizon": long_horizon,
        "horizon_ratio": long_horizon / anchor_world.config.horizon,
        "long_transitions_per_second": long_horizon / elapsed,
        "long_rollout_seconds": elapsed,
        "signal_storage_bytes": int(world.signals.nbytes),
        "lag_storage_bytes": int(world.intrinsic_lags.nbytes),
        "storage_bytes_per_step": (
            world.signals.nbytes + world.intrinsic_lags.nbytes
        )
        / long_horizon,
        "maximum_declared_causal_span": max(
            (edge.reward_time - edge.action_time for edge in edges), default=0
        ),
        "joint_action_choices": long_config.joint_discrete_choices,
        "branch_logits": sum(long_config.levels_per_factor),
    }


def run_qualification_study(
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    master_key: bytes,
    device_override: str | None = None,
) -> QualificationStudyResult:
    if len(master_key) != 32:
        raise ValueError("qualification suite key must contain exactly 32 bytes")
    anchor = _config(protocol["anchor_configuration"])
    suite = _suite(anchor, protocol, master_key)
    reference = protocol["reference"]
    thresholds = protocol["thresholds"]
    statistics = protocol["statistics"]
    preference = tuple(float(value) for value in protocol["preference"])
    requested_device = device_override or str(reference["device"])
    device = requested_device
    main_seeds = tuple(int(value) for value in reference["training_seeds"])

    _progress("qualification_phase_started", phase="mechanics")
    mechanics = _mechanics(suite, thresholds, device)
    _progress("qualification_phase_completed", phase="mechanics")
    _progress("qualification_phase_started", phase="causal_audit")
    causal_records = []
    for world in _worlds(suite, WorldBand.AUDIT):
        result = audit_causal_contract(world, min_detection_rate=0.95)
        causal_records.append(
            {
                "passed": result.passed,
                "detection_rate": result.detection_rate,
                "declared_edges": len(result.declared_edges),
                "recovered_edges": len(result.recovered_edges),
                "unexpected_edges": len(result.unexpected_edges),
                "interventions": result.interventions,
            }
        )
    _progress("qualification_phase_completed", phase="causal_audit")

    tiny_config = FactorLabConfig(
        horizon=3,
        n_objectives=2,
        n_factors=1,
        levels_per_factor=(2,),
        signal_dim=2,
        context_dim=2,
        state_dim=2,
        teacher_hidden_dim=4,
        max_causal_lag=3,
        terminal_state_weight=1.0,
    )
    tiny_world = generate_world(tiny_config, 90210, kernel_key=master_key)
    exact = exact_weighted_solution(tiny_world, preference, max_sequences=1000)
    independent = run_independent_audit(tiny_world)
    feasibility = {
        "exact_sequences": exact.sequences_evaluated,
        "exact_scalar_value": exact.scalar_value,
        "independent_audit": asdict(independent),
    }

    _progress("qualification_phase_started", phase="learnability")
    main_records, main_failures = _train_variant(
        suite,
        reference,
        str(reference["anchor_architecture"]),
        main_seeds,
        preference,
        device,
    )
    _progress("qualification_phase_completed", phase="learnability")
    successful_main = _successful(main_records)
    random_mean, random_std = _random_reference(
        suite, preference, policies=int(reference["random_policies"])
    )
    improvements = [
        float(record["heldout_utility_mean"]) - random_mean
        for record in successful_main
    ]
    heldout_means = [float(record["heldout_utility_mean"]) for record in successful_main]
    public_means = [float(record["public_utility_mean"]) for record in successful_main]
    ci = _bootstrap_mean_ci(
        improvements,
        draws=int(statistics["bootstrap_draws"]),
        seed=int(statistics["bootstrap_seed"]),
        confidence=float(statistics["confidence"]),
    ) if improvements else (-1.0, -1.0)

    _progress("qualification_phase_started", phase="scaling_and_specificity")
    scaling = _scaling_contrast(suite.world(WorldBand.PUBLIC, 0), protocol, master_key)
    timing_error = _matched_reward_timing_specificity(anchor, master_key)
    _progress("qualification_phase_completed", phase="scaling_and_specificity")

    generalization_gaps = [
        abs(float(record["public_utility_mean"]) - float(record["heldout_utility_mean"]))
        for record in successful_main
    ]
    mean_improvement = float(np.mean(improvements)) if improvements else -1.0
    mean_heldout = float(np.mean(heldout_means)) if heldout_means else -1.0
    measurements = {
        "mechanics": mechanics,
        "causal_audit": causal_records,
        "feasibility": feasibility,
        "reference": {
            "random_utility_mean": random_mean,
            "random_utility_std": random_std,
            "anchor_runs": main_records,
            "anchor_failures": main_failures,
            "mean_improvement_over_random": mean_improvement,
            "improvement_bootstrap_ci": list(ci),
            "mean_heldout_utility": mean_heldout,
        },
        "factor_sensitivity": {
            "scaling_contrast": scaling,
        },
        "specificity": {"matched_reward_timing_return_max_abs_error": timing_error},
        "generalization": {
            "public_means": public_means,
            "heldout_means": heldout_means,
            "absolute_gaps": generalization_gaps,
        },
    }
    bundle = {
        "evidence_format": "rlx-neural-qualification-evidence-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "tier_id": protocol["tier_id"],
        "task_id": anchor.task_id,
        "suite_id": suite.suite_id,
        "benchmark_revision": BENCHMARK_REVISION,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
        },
        "measurements": measurements,
        "heldout_identifiers_exposed": False,
    }
    evidence_sha = hashlib.sha256(_canonical_bytes(bundle)).hexdigest()
    refs = (f"sha256:{evidence_sha}", f"protocol-sha256:{protocol_sha256}")

    def check(name: str, passed: bool, summary: dict[str, Any]) -> QualificationCheck:
        return QualificationCheck(
            name,
            CheckStatus.VERIFIED if passed else CheckStatus.FAILED,
            summary,
            refs if passed else (),
        )

    all_causal = bool(causal_records) and all(record["passed"] for record in causal_records)
    learnability_pass = (
        not main_failures
        and len(successful_main) == len(main_seeds)
        and all(
            float(record["wall_seconds"]) <= float(reference["max_wall_seconds_per_seed"])
            for record in successful_main
        )
        and mean_improvement >= float(thresholds["minimum_mean_improvement"])
        and ci[0] >= float(thresholds["minimum_improvement_ci_lower"])
    )
    headroom_pass = (
        mean_heldout <= float(thresholds["maximum_reference_utility"])
        and 1.0 - mean_heldout >= float(thresholds["minimum_headroom"])
    )
    sensitivity_pass = (
        scaling["long_horizon"] >= int(thresholds["minimum_long_horizon"])
        and scaling["long_transitions_per_second"]
        >= float(thresholds["minimum_long_transitions_per_second"])
        and scaling["storage_bytes_per_step"]
        <= float(thresholds["maximum_storage_bytes_per_step"])
        and scaling["maximum_declared_causal_span"] >= scaling["long_horizon"] - 1
        and scaling["joint_action_choices"] >= int(thresholds["minimum_joint_choices"])
    )
    specificity_pass = timing_error <= float(thresholds["maximum_specificity_error"])
    generalization_pass = (
        bool(generalization_gaps)
        and max(generalization_gaps) <= float(thresholds["maximum_public_heldout_gap"])
        and mean_heldout > random_mean
    )
    statistics_pass = (
        len(successful_main) >= int(statistics["minimum_training_seeds"])
        and not main_failures
        and np.isfinite(ci).all()
    )
    checks = [
        check(
            "mechanics",
            bool(mechanics["passed"]),
            {
                **mechanics,
                "neural_admissibility": True,
                "parameter_budget": int(reference["max_parameters"]),
            },
        ),
        check(
            "causal_audit",
            all_causal,
            {
                "worlds": len(causal_records),
                "minimum_detection_rate": min(
                    (record["detection_rate"] for record in causal_records), default=0.0
                ),
                "unexpected_edges": sum(
                    record["unexpected_edges"] for record in causal_records
                ),
            },
        ),
        check("feasibility", exact.sequences_evaluated > 0, feasibility),
        check(
            "learnability",
            learnability_pass,
            {
                "neural_architecture": reference["anchor_architecture"],
                "training_seeds": len(successful_main),
                "mean_improvement_over_random": mean_improvement,
                "improvement_bootstrap_ci": list(ci),
                "maximum_seed_wall_seconds": max(
                    (float(record["wall_seconds"]) for record in successful_main),
                    default=0.0,
                ),
                "failures": main_failures,
            },
        ),
        check(
            "headroom",
            headroom_pass,
            {"mean_reference_utility": mean_heldout, "regret_to_upper_bound": 1.0 - mean_heldout},
        ),
        check(
            "factor_sensitivity",
            sensitivity_pass,
            {
                **scaling,
            },
        ),
        check(
            "specificity",
            specificity_pass,
            {"matched_reward_timing_return_max_abs_error": timing_error},
        ),
        check(
            "generalization",
            generalization_pass,
            {
                "heldout_worlds": suite.spec.heldout_worlds,
                "maximum_public_heldout_gap": max(generalization_gaps, default=1.0),
                "heldout_identifiers_exposed": False,
            },
        ),
        check(
            "statistics",
            statistics_pass,
            {
                "independent_training_seeds": len(successful_main),
                "bootstrap_draws": int(statistics["bootstrap_draws"]),
                "confidence": float(statistics["confidence"]),
                "improvement_ci": list(ci),
                "failure_count": main_failures,
            },
        ),
        check("independent_audit", independent.passed, asdict(independent)),
    ]
    report = make_qualification_report(
        task_id=anchor.task_id,
        suite_id=suite.suite_id,
        benchmark_revision=BENCHMARK_REVISION,
        checks=checks,
    )
    return QualificationStudyResult(report, bundle, evidence_sha)
