"""Training Pipeline (T) package for FTI credit risk models."""

from banking_fti.training.gates import (
    CohortSliceMetrics,
    CreditRiskGateConfig,
    CreditRiskGateResult,
    ValueSkewConfig,
    ValueSkewFinding,
    ValueSkewReport,
    assign_income_tier,
    check_value_skew,
    evaluate_credit_risk_gate,
)
from banking_fti.training.pipeline import TrainingPipeline, TrainingResult, run_stage1_training
from banking_fti.training.preprocessor import (
    DEFAULT_EXCLUDED_COLUMNS,
    PreprocessorBundle,
    TabularPreprocessor,
)

__all__ = [
    "DEFAULT_EXCLUDED_COLUMNS",
    "CohortSliceMetrics",
    "CreditRiskGateConfig",
    "CreditRiskGateResult",
    "PreprocessorBundle",
    "TabularPreprocessor",
    "TrainingPipeline",
    "TrainingResult",
    "ValueSkewConfig",
    "ValueSkewFinding",
    "ValueSkewReport",
    "assign_income_tier",
    "check_value_skew",
    "evaluate_credit_risk_gate",
    "run_stage1_training",
]
