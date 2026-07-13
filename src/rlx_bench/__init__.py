"""Clean-sheet RLX benchmark interfaces."""

from .actions import (
    ActionEnumerationError,
    ActionSpec,
    ConditionalHybridActionSpec,
    ContinuousActionSpec,
    EmbeddedCatalogActionSpec,
    FactoredDiscreteActionSpec,
    FlatDiscreteActionSpec,
    InvalidAction,
    make_action_spec,
)
from .audit import CausalAuditResult, audit_causal_contract
from .budget import BudgetExceeded, BudgetLedger, BudgetLimits, BudgetedEnv
from .factorlab import (
    EffectKind,
    FactorLabConfig,
    FactorLabEnv,
    FactorLabInspector,
    FactorLabWorld,
    NeuralTaskKernel,
    ObjectiveProtocol,
    derive_task_kernel,
    generate_world,
)
from .metrics import (
    constraint_metrics,
    coverage_metrics,
    hypervolume_2d,
    normalize_returns,
    pareto_mask,
)
from .independent_audit import IndependentAuditReport, run_independent_audit
from .oracle import ExactSolution, ParetoSolution, exact_pareto_front, exact_weighted_solution
from .suite import EvaluatorWorldSuite, PublicSuiteManifest, WorldBand, WorldSuiteSpec

__all__ = [
    "ActionEnumerationError",
    "ActionSpec",
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetLimits",
    "BudgetedEnv",
    "CausalAuditResult",
    "ConditionalHybridActionSpec",
    "ContinuousActionSpec",
    "EffectKind",
    "EmbeddedCatalogActionSpec",
    "EvaluatorWorldSuite",
    "ExactSolution",
    "FactorLabConfig",
    "FactorLabEnv",
    "FactorLabInspector",
    "FactorLabWorld",
    "FactoredDiscreteActionSpec",
    "FlatDiscreteActionSpec",
    "InvalidAction",
    "IndependentAuditReport",
    "NeuralTaskKernel",
    "ObjectiveProtocol",
    "ParetoSolution",
    "PublicSuiteManifest",
    "WorldBand",
    "WorldSuiteSpec",
    "audit_causal_contract",
    "constraint_metrics",
    "coverage_metrics",
    "derive_task_kernel",
    "exact_pareto_front",
    "exact_weighted_solution",
    "generate_world",
    "hypervolume_2d",
    "make_action_spec",
    "normalize_returns",
    "pareto_mask",
    "run_independent_audit",
]
