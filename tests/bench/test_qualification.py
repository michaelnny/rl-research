from __future__ import annotations

import pytest

from rlx_bench.qualification import (
    REQUIRED_QUALIFICATION_CHECKS,
    CheckStatus,
    QualificationCheck,
    make_qualification_report,
)


def test_verified_check_requires_immutable_evidence_reference() -> None:
    with pytest.raises(ValueError, match="evidence"):
        QualificationCheck("mechanics", CheckStatus.VERIFIED, {"passed": True})


def test_observations_cannot_accidentally_mark_benchmark_qualified() -> None:
    checks = [
        QualificationCheck(name, CheckStatus.OBSERVED, {"passed": True})
        for name in REQUIRED_QUALIFICATION_CHECKS
    ]

    report = make_qualification_report(
        task_id="task", suite_id="suite", benchmark_revision="rev", checks=checks
    )

    assert report.qualified is False
    assert report.to_dict()["qualified"] is False


def test_only_complete_verified_evidence_set_can_be_qualified() -> None:
    checks = [
        QualificationCheck(
            name,
            CheckStatus.VERIFIED,
            {"passed": True},
            evidence_refs=(f"artifact:sha256:{index:064x}",),
        )
        for index, name in enumerate(REQUIRED_QUALIFICATION_CHECKS)
    ]

    report = make_qualification_report(
        task_id="task", suite_id="suite", benchmark_revision="rev", checks=checks
    )

    assert report.qualified is True
    assert report.report_id.startswith("flq-")


def test_report_identity_is_content_addressed_and_order_independent() -> None:
    checks = [
        QualificationCheck(name, CheckStatus.NOT_RUN, {})
        for name in REQUIRED_QUALIFICATION_CHECKS
    ]
    forward = make_qualification_report(
        task_id="task", suite_id="suite", benchmark_revision="rev", checks=checks
    )
    reverse = make_qualification_report(
        task_id="task", suite_id="suite", benchmark_revision="rev", checks=list(reversed(checks))
    )

    assert forward.report_id == reverse.report_id
