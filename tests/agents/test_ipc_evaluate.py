from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest

from rlx_agents.evaluate import CandidateEvaluationConfig, evaluate_candidate
from rlx_agents.ipc import CandidateClient, CandidateProcessLimits, CandidateProtocolError


GOOD_CANDIDATE = r"""
import itertools
import json
import os
import sys

phase = None
hypotheses = None
episode = 0
scores = {}
current = None
best = None
for line in sys.stdin:
    message = json.loads(line)
    kind = message["type"]
    if kind == "init":
        if "RLX_FACTORLAB_SUITE_KEY_FILE" in os.environ:
            print(json.dumps({"type": "secret_leaked"}), flush=True)
            continue
        phase = message["phase"]
        if phase == "evaluation":
            best = message["checkpoint"]["best"]
        print(json.dumps({"type": "ready"}), flush=True)
    elif kind == "reset":
        pass
    elif kind == "act":
        cue = message["observation"]["revealed_cue"]
        if hypotheses is None:
            width = len(cue)
            hypotheses = [
                (permutation, signs)
                for permutation in itertools.permutations(range(width))
                for signs in itertools.product((-1, 1), repeat=width)
            ]
        if phase == "training":
            public_worlds = message.get("public_worlds", 1)
            current = hypotheses[min(episode, len(hypotheses) - 1)]
        else:
            current = (tuple(best[0]), tuple(best[1]))
        permutation, signs = current
        target = [signs[index] * cue[source] for index, source in enumerate(permutation)]
        action = [1 if value > 0 else 0 for value in target]
        print(json.dumps({"type": "action", "action": action}), flush=True)
    elif kind in {"transition", "episode_end"}:
        if kind == "episode_end" and phase == "training":
            score = message["return_vector"][0]
            scores[episode] = score
            episode += 1
    elif kind == "checkpoint":
        winner = max(scores, key=scores.get)
        print(json.dumps({"type": "checkpoint", "state": {"best": hypotheses[winner]}}), flush=True)
    elif kind == "close":
        print(json.dumps({"type": "closed"}), flush=True)
        break
"""


def _script(tmp_path, content: str, name: str = "candidate.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(content))
    return path


def test_candidate_client_strips_evaluator_environment_and_round_trips_json(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RLX_FACTORLAB_SUITE_KEY_FILE", "/secret/path")
    script = _script(tmp_path, GOOD_CANDIDATE)

    with CandidateClient((sys.executable, str(script)), cwd=tmp_path, sandbox=False) as client:
        response = client.request(
            {
                "type": "init",
                "phase": "training",
                "checkpoint": None,
                "protocol": "test",
            }
        )
        assert response == {"type": "ready"}


def test_candidate_client_times_out_and_terminates_process(tmp_path) -> None:
    script = _script(
        tmp_path,
        """
        import json, sys, time
        for line in sys.stdin:
            time.sleep(30)
        """,
    )
    client = CandidateClient(
        (sys.executable, str(script)),
        cwd=tmp_path,
        sandbox=False,
        limits=CandidateProcessLimits(response_seconds=0.05),
    )

    started = time.monotonic()
    with pytest.raises(CandidateProtocolError, match="did not respond"):
        client.request({"type": "init"})
    assert time.monotonic() - started < 3.0
    assert client.process.poll() is not None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox contract")
def test_candidate_sandbox_denies_runtime_reads(tmp_path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    secret = runtime / "campaign.key"
    secret.write_text("must-not-be-readable")
    script = _script(
        tmp_path,
        f"""
        import json, sys
        for line in sys.stdin:
            message = json.loads(line)
            if message["type"] == "init":
                try:
                    open({str(secret)!r}).read()
                    leaked = True
                except OSError:
                    leaked = False
                print(json.dumps({{"type": "leaked" if leaked else "ready"}}), flush=True)
            elif message["type"] == "close":
                print(json.dumps({{"type": "closed"}}), flush=True)
                break
        """,
    )

    with CandidateClient(
        (sys.executable, str(script)),
        cwd=tmp_path,
        unreadable_roots=(runtime,),
        sandbox=True,
    ) as client:
        assert client.request({"type": "init"}) == {"type": "ready"}


def test_candidate_evaluator_hides_worlds_and_restarts_from_checkpoint(tmp_path) -> None:
    script = _script(tmp_path, GOOD_CANDIDATE)
    config = CandidateEvaluationConfig(
        horizon=8,
        n_factors=2,
        max_causal_lag=8,
        training_episodes=8,
        training_trials=2,
        public_worlds=1,
        heldout_worlds=2,
        wall_seconds=20,
        response_seconds=2,
    )

    report = evaluate_candidate(
        (sys.executable, str(script)),
        cwd=tmp_path,
        master_key=b"k" * 32,
        config=config,
        sandbox=False,
    )

    assert report["status"] == "complete"
    assert report["normalized_return_mean"] == pytest.approx([1.0, 0.0])
    assert report["normalized_utility_mean"] == pytest.approx(1.0)
    assert report["random_policy_expected_utility"] == pytest.approx(0.5)
    assert report["improvement_over_random"] == pytest.approx(0.5)
    assert report["regret_to_ceiling"] == pytest.approx(0.0)
    assert report["training_trials"] == 2
    assert report["budget_usage"]["episodes"] == 20
    assert report["budget_usage"]["transitions"] == 160
    assert report["budget_usage"]["policies"] == 2
    assert len(report["candidate_stderr"]) == 6
    serialized = json.dumps(report)
    assert "world_id" not in serialized
    assert "master_seed" not in serialized
    assert report["heldout_identifiers_exposed"] is False


def test_invalid_candidate_action_becomes_scientific_error_record(tmp_path) -> None:
    script = _script(
        tmp_path,
        """
        import json, sys
        for line in sys.stdin:
            message = json.loads(line)
            if message["type"] == "init":
                print(json.dumps({"type": "ready"}), flush=True)
            elif message["type"] == "act":
                print(json.dumps({"type": "action", "action": [99]}), flush=True)
            elif message["type"] == "close":
                print(json.dumps({"type": "closed"}), flush=True)
                break
        """,
    )
    config = CandidateEvaluationConfig(
        horizon=3,
        n_factors=1,
        max_causal_lag=3,
        training_episodes=1,
        training_trials=1,
        public_worlds=1,
        heldout_worlds=1,
        wall_seconds=10,
    )

    report = evaluate_candidate(
        (sys.executable, str(script)),
        cwd=tmp_path,
        master_key=b"z" * 32,
        config=config,
        sandbox=False,
    )

    assert report["status"] == "candidate_error"
    assert report["error_class"] == "CandidateProtocolError"
    assert "rejected" in report["error_detail"]
