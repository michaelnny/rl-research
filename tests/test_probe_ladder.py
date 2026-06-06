from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from scripts.run_probe_ladder import parse_panel_text, score_delta, should_confirm  # noqa: E402

PANEL_TEXT = """[run] stage=quick envs=deep-sea-treasure-concave-v0 budget=120s workers=1 train_path=/tmp/train.py
[env] deep-sea-treasure-concave-v0       score=    2.500000 random=1.000000 strong=3.000000 margin_random=    1.500000 margin_strong=   -0.500000 beat_random=1 beat_strong=0
---
stage:           quick
n_envs:          1
n_beat_random:   1
n_beat_strong:   0
wallclock_s:     4.2
"""


def test_parse_panel_text() -> None:
    result = parse_panel_text(PANEL_TEXT)

    assert result.status == "completed"
    assert result.envs == ["deep-sea-treasure-concave-v0"]
    assert result.scores == {"deep-sea-treasure-concave-v0": 2.5}
    assert result.beat_random == 1
    assert result.beat_strong == 0
    assert result.wallclock_s == 4.2


def test_parse_panel_timeout_marks_budget() -> None:
    result = parse_panel_text(PANEL_TEXT + "\n[run_panel] timeout after 150s\n")

    assert result.status == "killed-budget"


def test_score_delta_handles_missing_scores() -> None:
    assert score_delta({"a": 2.0, "b": None}, {"a": 1.5, "b": 1.0}) == {
        "a": 0.5,
        "b": None,
    }


def test_should_confirm_requires_candidate_ablation_lift() -> None:
    candidate = parse_panel_text(PANEL_TEXT)
    ablation = parse_panel_text(PANEL_TEXT.replace("2.500000", "2.000000"))

    assert should_confirm(candidate, ablation)


def test_should_confirm_skips_when_ablation_matches() -> None:
    candidate = parse_panel_text(PANEL_TEXT)
    ablation = parse_panel_text(PANEL_TEXT)

    assert not should_confirm(candidate, ablation)
