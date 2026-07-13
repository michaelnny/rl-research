from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest

from rlx_agents.evaluate import CandidateEvaluationConfig, evaluate_candidate
from rlx_agents.ipc import CandidateClient, CandidateProcessLimits, CandidateProtocolError


PROTOCOL_FIXTURE = r"""
import hashlib
import json
import os
import sys

factors = 1
for line in sys.stdin:
    message = json.loads(line)
    kind = message["type"]
    if kind == "init":
        if "RLX_FACTORLAB_SUITE_KEY_FILE" in os.environ:
            print(json.dumps({"type": "secret_leaked"}), flush=True)
            continue
        factors = len(message["task_spec"]["action_spec"]["factors"])
        checkpoint = message.get("checkpoint")
        if checkpoint:
            path = os.path.join(os.environ["RLX_CANDIDATE_SCRATCH"], checkpoint["artifact"])
            content = open(path, "rb").read()
            assert hashlib.sha256(content).hexdigest() == checkpoint["sha256"]
        manifest = {
            "model_family": "neural_policy",
            "architecture": "test_residual_policy",
            "framework": "test",
            "trainable_parameters": 32,
            "recurrent": False,
            "device": "cpu",
        }
        print(json.dumps({"type": "ready", "model_manifest": manifest}), flush=True)
    elif kind == "act_batch":
        actions = [[0] * factors for _ in message["observations"]]
        print(json.dumps({"type": "actions", "actions": actions}), flush=True)
    elif kind == "checkpoint":
        content = b"bounded-neural-checkpoint-fixture"
        name = "model.bin"
        path = os.path.join(os.environ["RLX_CANDIDATE_SCRATCH"], name)
        open(path, "wb").write(content)
        print(json.dumps({
            "type": "checkpoint",
            "artifact": name,
            "sha256": hashlib.sha256(content).hexdigest(),
        }), flush=True)
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
    script = _script(
        tmp_path,
        """
        import json, os, sys
        for line in sys.stdin:
            message = json.loads(line)
            if message["type"] == "init":
                print(json.dumps({"type": "clean", "leaked": "RLX_FACTORLAB_SUITE_KEY_FILE" in os.environ}), flush=True)
            elif message["type"] == "close":
                print(json.dumps({"type": "closed"}), flush=True)
                break
        """,
    )
    with CandidateClient((sys.executable, str(script)), cwd=tmp_path, sandbox=False) as client:
        assert client.request({"type": "init"}) == {"type": "clean", "leaked": False}


def test_candidate_client_reads_bounded_scratch_artifact(tmp_path) -> None:
    script = _script(
        tmp_path,
        """
        import json, os, sys
        for line in sys.stdin:
            message = json.loads(line)
            if message["type"] == "write":
                open(os.path.join(os.environ["RLX_CANDIDATE_SCRATCH"], "state.bin"), "wb").write(b"state")
                print(json.dumps({"type": "written"}), flush=True)
            elif message["type"] == "close":
                print(json.dumps({"type": "closed"}), flush=True)
                break
        """,
    )
    with CandidateClient((sys.executable, str(script)), cwd=tmp_path, sandbox=False) as client:
        assert client.request({"type": "write"}) == {"type": "written"}
        assert client.read_artifact("state.bin", max_bytes=10) == b"state"


def test_candidate_client_times_out_and_terminates_process(tmp_path) -> None:
    script = _script(tmp_path, "import sys, time\nfor line in sys.stdin: time.sleep(30)\n")
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
                    open({str(secret)!r}).read(); leaked = True
                except OSError:
                    leaked = False
                print(json.dumps({{"type": "leaked" if leaked else "ready"}}), flush=True)
            elif message["type"] == "close":
                print(json.dumps({{"type": "closed"}}), flush=True); break
        """,
    )
    with CandidateClient(
        (sys.executable, str(script)),
        cwd=tmp_path,
        unreadable_roots=(runtime,),
        sandbox=True,
    ) as client:
        assert client.request({"type": "init"}) == {"type": "ready"}


def test_neural_candidate_evaluator_batches_training_and_restarts_from_binary_checkpoint(
    tmp_path,
) -> None:
    script = _script(tmp_path, PROTOCOL_FIXTURE)
    config = CandidateEvaluationConfig(
        horizon=6,
        n_factors=2,
        levels_per_factor=3,
        signal_dim=3,
        context_dim=2,
        state_dim=2,
        teacher_hidden_dim=4,
        max_causal_lag=6,
        training_episodes=4,
        training_batch_size=2,
        training_trials=2,
        public_worlds=2,
        heldout_worlds=2,
        wall_seconds_total=20,
        response_seconds=2,
        max_parameters=1000,
    )
    report = evaluate_candidate(
        (sys.executable, str(script)),
        cwd=tmp_path,
        master_key=b"k" * 32,
        config=config,
        sandbox=False,
    )

    assert report["status"] == "complete"
    assert report["protocol"] == "rlx-neural-candidate-jsonl-v2"
    assert report["training_batch_size"] == 2
    assert report["budget_usage"]["episodes"] == 12
    assert report["budget_usage"]["transitions"] == 72
    assert report["model_manifests"][0]["model_family"] == "neural_policy"
    assert len(report["checkpoint_sha256"]) == 2
    assert len(report["candidate_stderr"]) == 6
    assert report["heldout_identifiers_exposed"] is False
    serialized = json.dumps(report)
    assert "world_id" not in serialized and "master_seed" not in serialized


def test_parameter_budget_violation_becomes_scientific_error_record(tmp_path) -> None:
    script = _script(
        tmp_path,
        PROTOCOL_FIXTURE.replace('"trainable_parameters": 32', '"trainable_parameters": 32000'),
    )
    report = evaluate_candidate(
        (sys.executable, str(script)),
        cwd=tmp_path,
        master_key=b"z" * 32,
        config=CandidateEvaluationConfig(
            horizon=3,
            n_factors=1,
            levels_per_factor=2,
            signal_dim=2,
            context_dim=2,
            state_dim=2,
            teacher_hidden_dim=4,
            max_causal_lag=3,
            training_episodes=1,
            training_batch_size=1,
            training_trials=1,
            public_worlds=1,
            heldout_worlds=1,
            wall_seconds_total=10,
            max_parameters=100,
        ),
        sandbox=False,
    )
    assert report["status"] == "candidate_error"
    assert report["error_class"] == "CandidateProtocolError"
    assert "above cap" in report["error_detail"]
