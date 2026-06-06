"""Validate a probe candidate.json file.

Usage:
    uv run python scripts/validate_candidate.py worklogs/runs/<run_id>/candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402

REQUIRED_STRING_FIELDS = {
    "run_id",
    "algorithm_name",
    "principle",
    "primitive_name",
    "primitive_type",
    "update_family",
    "memory",
    "feedback_signal",
    "claimed_stage",
    "nearest_disqualifier",
    "novelty_boundary",
    "empirical_claim",
    "falsifier",
    "ablation_plan",
    "proof_debt",
}

REQUIRED_BOOL_FIELDS = {"uses_reward", "uses_vector_reward"}

UPDATE_FAMILIES = {
    "direct_policy_update",
    "trajectory_rewrite",
    "population_update",
    "state_action_operator",
    "memory_relabeling",
    "model_update",
    "planning_update",
    "other",
}

MEMORY_TYPES = {
    "none",
    "episode",
    "replay",
    "table",
    "network",
    "graph",
    "population",
    "model",
    "other",
}

NEAREST_DISQUALIFIERS = {
    "none",
    "q_learning",
    "policy_gradient",
    "actor_critic",
    "mcts",
    "sac",
    "scalarization",
    "count_based",
    "rnd",
    "go_explore",
    "her",
    "options",
    "reward_machine",
    "successor_features",
    "distributional_rl",
    "decision_transformer",
    "cem_es",
    "topk_cloning",
    "baseline_modification",
    "dead_family",
    "published_method",
    "other",
}


def validate_candidate(data: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["candidate must be a JSON object"]

    for field in sorted(REQUIRED_STRING_FIELDS):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    for field in sorted(REQUIRED_BOOL_FIELDS):
        if not isinstance(data.get(field), bool):
            errors.append(f"{field} must be a boolean")

    stage = data.get("claimed_stage")
    if isinstance(stage, str) and stage not in harness.STAGES:
        errors.append(f"claimed_stage must be one of {sorted(harness.STAGES)}")

    update_family = data.get("update_family")
    if isinstance(update_family, str) and update_family not in UPDATE_FAMILIES:
        errors.append(f"update_family must be one of {sorted(UPDATE_FAMILIES)}")

    memory = data.get("memory")
    if isinstance(memory, str) and memory not in MEMORY_TYPES:
        errors.append(f"memory must be one of {sorted(MEMORY_TYPES)}")

    nearest = data.get("nearest_disqualifier")
    if isinstance(nearest, str) and nearest not in NEAREST_DISQUALIFIERS:
        errors.append(f"nearest_disqualifier must be one of {sorted(NEAREST_DISQUALIFIERS)}")

    claimed_stage = data.get("claimed_stage")
    uses_vector = data.get("uses_vector_reward")
    if isinstance(claimed_stage, str) and isinstance(uses_vector, bool):
        stage_envs = harness.STAGES.get(claimed_stage, [])
        has_vector_env = any(harness.ENV_TYPE[env] == "vector" for env in stage_envs)
        if uses_vector and not has_vector_env:
            errors.append("uses_vector_reward=true requires a stage containing vector envs")

    if data.get("nearest_disqualifier") == "scalarization" and data.get("uses_vector_reward"):
        boundary = data.get("novelty_boundary")
        if isinstance(boundary, str) and "scalar" not in boundary.lower():
            errors.append("scalarization-risk probes must explain the scalarization boundary")

    return errors


def load_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"candidate file does not exist: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_candidate(load_json(args.path))
    if errors:
        for error in errors:
            print(f"candidate.json: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("candidate.json: ok")


if __name__ == "__main__":
    main()
