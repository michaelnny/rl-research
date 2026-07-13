"""Evidence-gated benchmark qualification records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


REQUIRED_QUALIFICATION_CHECKS = (
    "mechanics",
    "causal_audit",
    "feasibility",
    "learnability",
    "headroom",
    "factor_sensitivity",
    "specificity",
    "generalization",
    "statistics",
    "independent_audit",
)


class CheckStatus(str, Enum):
    NOT_RUN = "not_run"
    OBSERVED = "observed"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True)
class QualificationCheck:
    name: str
    status: CheckStatus | str
    measurements: dict[str, Any]
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_QUALIFICATION_CHECKS:
            raise ValueError(f"unknown qualification check: {self.name}")
        status = CheckStatus(self.status)
        object.__setattr__(self, "status", status)
        if status is CheckStatus.VERIFIED and not self.evidence_refs:
            raise ValueError("verified checks require immutable evidence references")
        json.dumps(self.measurements, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class QualificationReport:
    task_id: str
    suite_id: str
    benchmark_revision: str
    checks: tuple[QualificationCheck, ...]
    report_id: str

    @property
    def qualified(self) -> bool:
        by_name = {check.name: check for check in self.checks}
        return set(by_name) == set(REQUIRED_QUALIFICATION_CHECKS) and all(
            by_name[name].status is CheckStatus.VERIFIED
            and bool(by_name[name].evidence_refs)
            for name in REQUIRED_QUALIFICATION_CHECKS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "task_id": self.task_id,
            "suite_id": self.suite_id,
            "benchmark_revision": self.benchmark_revision,
            "qualified": self.qualified,
            "checks": [
                {
                    **asdict(check),
                    "status": check.status.value,
                    "evidence_refs": list(check.evidence_refs),
                }
                for check in self.checks
            ],
        }


def make_qualification_report(
    *,
    task_id: str,
    suite_id: str,
    benchmark_revision: str,
    checks: tuple[QualificationCheck, ...] | list[QualificationCheck],
) -> QualificationReport:
    ordered = tuple(sorted(checks, key=lambda check: REQUIRED_QUALIFICATION_CHECKS.index(check.name)))
    names = [check.name for check in ordered]
    if len(names) != len(set(names)):
        raise ValueError("a qualification report cannot contain duplicate checks")
    payload = {
        "task_id": task_id,
        "suite_id": suite_id,
        "benchmark_revision": benchmark_revision,
        "checks": [
            {
                "name": check.name,
                "status": check.status.value,
                "measurements": check.measurements,
                "evidence_refs": check.evidence_refs,
            }
            for check in ordered
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report_id = f"flq-{hashlib.sha256(canonical.encode()).hexdigest()}"
    return QualificationReport(task_id, suite_id, benchmark_revision, ordered, report_id)
