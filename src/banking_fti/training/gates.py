"""Pre-training value-skew and Credit Risk / Cohort Fairness Validation Gates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score

from banking_fti.ports.model_registry import RegisteredModel
from banking_fti.training.preprocessor import DEFAULT_EXCLUDED_COLUMNS

_STD_FLOOR = 1e-6


@dataclass(frozen=True)
class ValueSkewConfig:
    """Thresholds for lightweight Stage 1 pre-training value-skew gating."""

    max_null_rate: float = 0.50
    max_numeric_z_shift: float = 3.5
    max_categorical_tvd: float = 0.45
    min_rows_per_split: int = 10


@dataclass(frozen=True)
class ValueSkewFinding:
    """Single value-skew or data-integrity violation between train and holdout splits."""

    column: str
    metric: str
    score: float
    threshold: float
    detail: str


@dataclass(frozen=True)
class ValueSkewReport:
    """Outcome of pre-training value-skew checks."""

    passed: bool
    findings: tuple[ValueSkewFinding, ...]
    checked_columns: tuple[str, ...]


@dataclass(frozen=True)
class CreditRiskGateConfig:
    """Thresholds for the Credit Risk & Cohort Fairness Validation Gate."""

    min_roc_auc: float = 0.75
    min_default_recall: float = 0.65
    max_brier_score: float = 0.25
    max_onnx_diff: float = 1e-5
    decision_threshold: float = 0.5
    min_slice_samples: int = 5
    min_cohort_recall: float = 0.50
    max_cohort_recall_gap: float = 0.25
    tie_tolerance: float = 0.005


@dataclass(frozen=True)
class CohortSliceMetrics:
    """Holdout recall summary for a single fairness cohort slice."""

    dimension: str
    cohort: str
    sample_count: int
    positive_count: int
    recall: float | None
    evaluated: bool


@dataclass(frozen=True)
class CreditRiskGateResult:
    """Detailed validation outcome across metrics, fairness, parity, and champion comparison."""

    passed: bool
    roc_auc: float
    default_recall: float
    brier_score: float
    onnx_max_abs_diff: float | None
    parity_passed: bool
    convergence_passed: bool
    nan_inf_passed: bool
    value_skew_passed: bool
    serving_smoke_passed: bool
    fairness_passed: bool
    champion_comparison_passed: bool
    min_cohort_recall: float
    max_cohort_recall_gap: float
    cohort_slices: tuple[CohortSliceMetrics, ...]
    reasons: tuple[str, ...]


def assign_income_tier(income: pd.Series) -> pd.Series:
    """Bucket annual income into canonical fairness cohorts (`low`, `medium`, `high`)."""
    values = pd.to_numeric(income, errors="coerce")
    tiers = np.where(
        values.isna(),
        "unknown",
        np.where(values < 50_000.0, "low", np.where(values < 100_000.0, "medium", "high")),
    )
    return pd.Series(tiers, index=income.index, dtype=object)


def check_value_skew(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    target_column: str = "default_label",
    exclude_columns: Sequence[str] = DEFAULT_EXCLUDED_COLUMNS,
    config: ValueSkewConfig | None = None,
) -> ValueSkewReport:
    """Verify split integrity, finiteness, null rates, and train-vs-holdout distribution skew."""
    cfg = config or ValueSkewConfig()
    findings: list[ValueSkewFinding] = []

    if len(train_df) < cfg.min_rows_per_split:
        findings.append(
            ValueSkewFinding(
                column="__split__",
                metric="train_rows",
                score=float(len(train_df)),
                threshold=float(cfg.min_rows_per_split),
                detail=f"train split has {len(train_df)} rows (< {cfg.min_rows_per_split})",
            )
        )
    if len(holdout_df) < cfg.min_rows_per_split:
        findings.append(
            ValueSkewFinding(
                column="__split__",
                metric="holdout_rows",
                score=float(len(holdout_df)),
                threshold=float(cfg.min_rows_per_split),
                detail=f"holdout split has {len(holdout_df)} rows (< {cfg.min_rows_per_split})",
            )
        )

    for split_name, split_df in (("train", train_df), ("holdout", holdout_df)):
        if target_column in split_df.columns:
            n_classes = int(split_df[target_column].dropna().nunique())
            if n_classes < 2:
                findings.append(
                    ValueSkewFinding(
                        column=target_column,
                        metric=f"{split_name}_classes",
                        score=float(n_classes),
                        threshold=2.0,
                        detail=f"{split_name} split has {n_classes} target classes (< 2)",
                    )
                )

    excluded = set(exclude_columns) | {target_column}
    checked_columns: list[str] = []
    for col in train_df.columns:
        if (
            col in excluded
            or col not in holdout_df.columns
            or pd.api.types.is_datetime64_any_dtype(train_df[col])
        ):
            continue
        checked_columns.append(col)

        for split_name, split_df in (("train", train_df), ("holdout", holdout_df)):
            if len(split_df) > 0:
                null_rate = float(split_df[col].isna().mean())
                if null_rate > cfg.max_null_rate:
                    findings.append(
                        ValueSkewFinding(
                            column=col,
                            metric=f"{split_name}_null_rate",
                            score=null_rate,
                            threshold=cfg.max_null_rate,
                            detail=f"{split_name} null_rate={null_rate:.3f} > {cfg.max_null_rate}",
                        )
                    )

        if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(
            holdout_df[col]
        ):
            tr_vals = train_df[col].to_numpy(dtype=np.float64, copy=False)
            ho_vals = holdout_df[col].to_numpy(dtype=np.float64, copy=False)
            inf_count = int(np.isinf(tr_vals).sum() + np.isinf(ho_vals).sum())
            if inf_count > 0:
                findings.append(
                    ValueSkewFinding(
                        column=col,
                        metric="inf_count",
                        score=float(inf_count),
                        threshold=0.0,
                        detail=f"found {inf_count} +/-Inf values in '{col}'",
                    )
                )
                continue

            tr_finite = tr_vals[np.isfinite(tr_vals)]
            ho_finite = ho_vals[np.isfinite(ho_vals)]
            if len(tr_finite) > 0 and len(ho_finite) > 0:
                tr_std = float(np.std(tr_finite, ddof=0))
                if tr_std > _STD_FLOOR:
                    z_shift = abs(float(np.mean(ho_finite)) - float(np.mean(tr_finite))) / tr_std
                    if z_shift > cfg.max_numeric_z_shift:
                        findings.append(
                            ValueSkewFinding(
                                column=col,
                                metric="standardized_mean_shift",
                                score=float(z_shift),
                                threshold=cfg.max_numeric_z_shift,
                                detail=(
                                    f"numeric z-shift={z_shift:.3f} > {cfg.max_numeric_z_shift}"
                                ),
                            )
                        )
        else:
            tr_cats = train_df[col].dropna().astype(str).value_counts(normalize=True)
            ho_cats = holdout_df[col].dropna().astype(str).value_counts(normalize=True)
            if not tr_cats.empty and not ho_cats.empty:
                all_keys = set(tr_cats.index) | set(ho_cats.index)
                tvd = 0.5 * sum(
                    abs(float(ho_cats.get(key, 0.0)) - float(tr_cats.get(key, 0.0)))
                    for key in all_keys
                )
                if tvd > cfg.max_categorical_tvd:
                    findings.append(
                        ValueSkewFinding(
                            column=col,
                            metric="total_variation_distance",
                            score=float(tvd),
                            threshold=cfg.max_categorical_tvd,
                            detail=f"categorical TVD={tvd:.3f} > {cfg.max_categorical_tvd}",
                        )
                    )

    return ValueSkewReport(
        passed=len(findings) == 0,
        findings=tuple(findings),
        checked_columns=tuple(checked_columns),
    )


def _evaluate_cohort_dimension(
    dimension: str,
    labels: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: CreditRiskGateConfig,
) -> tuple[list[CohortSliceMetrics], list[str], list[float], float]:
    slices: list[CohortSliceMetrics] = []
    reasons: list[str] = []
    evaluated_recalls: list[float] = []

    for cohort_val in sorted(labels.astype(str).unique()):
        mask = (labels.astype(str) == cohort_val).to_numpy()
        sample_count = int(mask.sum())
        positive_count = int((y_true[mask] == 1).sum())
        evaluated = sample_count >= cfg.min_slice_samples and positive_count >= 1
        recall: float | None = None
        if evaluated:
            recall = float(recall_score(y_true[mask], y_pred[mask], zero_division=0))
            evaluated_recalls.append(recall)
            if recall < cfg.min_cohort_recall:
                reasons.append(
                    f"cohort {dimension}={cohort_val} recall {recall:.3f} < {cfg.min_cohort_recall}"
                )
        slices.append(
            CohortSliceMetrics(
                dimension=dimension,
                cohort=cohort_val,
                sample_count=sample_count,
                positive_count=positive_count,
                recall=recall,
                evaluated=evaluated,
            )
        )

    gap = 0.0
    if len(evaluated_recalls) >= 2:
        gap = float(max(evaluated_recalls) - min(evaluated_recalls))
        if gap > cfg.max_cohort_recall_gap:
            reasons.append(f"cohort {dimension} recall gap {gap:.3f} > {cfg.max_cohort_recall_gap}")
    return slices, reasons, evaluated_recalls, gap


def evaluate_credit_risk_gate(
    *,
    holdout_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    onnx_max_abs_diff: float | None,
    convergence_passed: bool,
    nan_inf_passed: bool,
    value_skew_passed: bool,
    serving_smoke_passed: bool,
    current_champion: RegisteredModel | None = None,
    config: CreditRiskGateConfig | None = None,
) -> CreditRiskGateResult:
    """Evaluate all Credit Risk quality, parity, fairness, and Champion-vs-Challenger gates."""
    cfg = config or CreditRiskGateConfig()
    reasons: list[str] = []

    if len(y_true) == 0 or len(np.unique(y_true)) < 2 or not np.isfinite(y_prob).all():
        roc_auc = 0.0
        default_recall = 0.0
        brier = 1.0
        y_pred = np.zeros(len(y_true), dtype=np.int64)
        reasons.append("holdout target or probabilities invalid for metric computation")
    else:
        y_pred = (y_prob >= cfg.decision_threshold).astype(np.int64)
        roc_auc = float(roc_auc_score(y_true, y_prob))
        default_recall = float(recall_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))

    parity_passed = onnx_max_abs_diff is not None and onnx_max_abs_diff < cfg.max_onnx_diff

    if not nan_inf_passed:
        reasons.append("NaN or Inf detected in features, probabilities, or training loss")
    if not convergence_passed:
        reasons.append("training loss did not decrease across iterations")
    if not value_skew_passed:
        reasons.append("pre-training value-skew gate failed")
    if not parity_passed:
        reasons.append(f"ONNX parity diff {onnx_max_abs_diff} >= {cfg.max_onnx_diff}")
    if not serving_smoke_passed:
        reasons.append("ONNX serving smoke check failed")
    if roc_auc < cfg.min_roc_auc:
        reasons.append(f"holdout roc_auc {roc_auc:.3f} < {cfg.min_roc_auc}")
    if default_recall < cfg.min_default_recall:
        reasons.append(f"holdout default_recall {default_recall:.3f} < {cfg.min_default_recall}")
    if brier >= cfg.max_brier_score:
        reasons.append(f"holdout brier {brier:.3f} >= {cfg.max_brier_score}")

    # Cohort fairness slices across income_tier and home_ownership
    all_slices: list[CohortSliceMetrics] = []
    fairness_reasons: list[str] = []
    all_evaluated_recalls: list[float] = []
    dimension_gaps: list[float] = []

    if len(holdout_df) == len(y_true) and len(y_true) > 0:
        dimensions: list[tuple[str, pd.Series | None]] = [
            (
                "income_tier",
                assign_income_tier(holdout_df["income_annual"])
                if "income_annual" in holdout_df.columns
                else None,
            ),
            (
                "home_ownership",
                holdout_df["home_ownership"] if "home_ownership" in holdout_df.columns else None,
            ),
        ]
        for dimension_name, labels in dimensions:
            if labels is None:
                fairness_reasons.append(f"missing required fairness dimension '{dimension_name}'")
                continue
            dim_slices, dim_reasons, dim_recalls, dim_gap = _evaluate_cohort_dimension(
                dimension_name, labels, y_true, y_pred, cfg
            )
            all_slices.extend(dim_slices)
            fairness_reasons.extend(dim_reasons)
            if not dim_recalls:
                fairness_reasons.append(
                    f"no evaluable cohorts for fairness dimension '{dimension_name}'"
                )
            else:
                all_evaluated_recalls.extend(dim_recalls)
                dimension_gaps.append(dim_gap)

    fairness_passed = len(fairness_reasons) == 0
    reasons.extend(fairness_reasons)

    min_cohort_recall = min(all_evaluated_recalls) if all_evaluated_recalls else default_recall
    max_cohort_recall_gap = max(dimension_gaps) if dimension_gaps else 0.0

    # Champion vs. Challenger comparison (tie_tolerance without regressing recall or fairness)
    champ_reasons: list[str] = []
    if current_champion is not None:
        champ_auc = current_champion.metrics.get("roc_auc")
        champ_recall = current_champion.metrics.get("default_recall")
        champ_fair = current_champion.metrics.get("min_cohort_recall")
        champ_gap = current_champion.metrics.get("max_cohort_recall_gap")

        if champ_auc is not None and roc_auc < champ_auc - cfg.tie_tolerance:
            champ_reasons.append(
                f"challenger roc_auc {roc_auc:.4f} < champion {champ_auc:.4f} - {cfg.tie_tolerance}"
            )
        if champ_recall is not None and default_recall < champ_recall:
            champ_reasons.append(
                f"challenger default_recall {default_recall:.4f} < champion {champ_recall:.4f}"
            )
        if champ_fair is not None and min_cohort_recall < champ_fair:
            champ_reasons.append(
                f"challenger min_cohort_recall {min_cohort_recall:.4f} < champion {champ_fair:.4f}"
            )
        if champ_gap is not None and max_cohort_recall_gap > champ_gap:
            champ_reasons.append(
                "challenger max_cohort_recall_gap "
                f"{max_cohort_recall_gap:.4f} > champion {champ_gap:.4f}"
            )

    champion_comparison_passed = len(champ_reasons) == 0
    reasons.extend(champ_reasons)

    return CreditRiskGateResult(
        passed=len(reasons) == 0,
        roc_auc=roc_auc,
        default_recall=default_recall,
        brier_score=brier,
        onnx_max_abs_diff=onnx_max_abs_diff,
        parity_passed=parity_passed,
        convergence_passed=convergence_passed,
        nan_inf_passed=nan_inf_passed,
        value_skew_passed=value_skew_passed,
        serving_smoke_passed=serving_smoke_passed,
        fairness_passed=fairness_passed,
        champion_comparison_passed=champion_comparison_passed,
        min_cohort_recall=min_cohort_recall,
        max_cohort_recall_gap=max_cohort_recall_gap,
        cohort_slices=tuple(all_slices),
        reasons=tuple(reasons),
    )
