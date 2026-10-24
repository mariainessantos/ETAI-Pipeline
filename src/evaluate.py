"""
Evaluation on the development set: stratified k-fold cross-validation of the
whole pipeline on several metrics, out-of-fold probabilities and reports,
a fairness check and a paired comparison of two models.

The locked test set is never used here.

"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    Return a dictionary {metric_name: sklearn_scorer}.

    Supports standard scikit-learn scoring names such as:
    accuracy, balanced_accuracy, precision, recall, f1,
    roc_auc and average_precision.

    Also supports fbeta_<beta>, e.g. fbeta_2, which gives
    recall twice as much weight as precision.
    """
    scorers = {}

    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            scorers[name] = make_scorer(
                fbeta_score,
                beta=beta,
                zero_division=0,
            )
        else:
            scorers[name] = get_scorer(name)

    return scorers


# evaluate multiple metrics and return OOF probabilities.
def cross_validate_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorers: dict,
    selection_metric: str,
    n_jobs: int = 1,
):
    """
    Fit a fresh copy of the entire pipeline on each training fold.

    Preprocessing is fitted within each fold, preventing the validation
    fold from influencing its own imputation, encoding or scaling.

    Returns:
        fold_scores:
            One row per fold, containing:
            - train: training score for selection_metric
            - validation: validation score for selection_metric
            - gap: train minus validation score
            - one validation-score column for every configured metric

        oof_proba:
            Out-of-fold probability of the positive class for each row.
            Each row is predicted by a model that did not train on it.

    Requires a CV splitter that validates each row exactly once.
    """
    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric '{selection_metric}' is not present "
            "in the configured scorers."
        )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )

    fold_scores = pd.DataFrame({
        "fold": range(1, cv.get_n_splits() + 1),
        "train": scores[f"train_{selection_metric}"],
        "validation": scores[f"test_{selection_metric}"],
    })

    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X), dtype=float)

    for model, val_idx in zip(
        scores["estimator"],
        scores["indices"]["test"],
    ):
        oof_proba[val_idx] = model.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

    return fold_scores, oof_proba

# report all metrics, with a configurable title and selection metric.
def cv_report(
    fold_scores: pd.DataFrame,
    metrics: list,
    selection_metric: str,
    title: str,
) -> str:
    """
    Produce a per-fold table and mean / standard deviation for all metrics.
    Also report the mean train-validation gap for the selection metric.
    """
    lines = [
        (
            f"{title} ({len(fold_scores)} stratified folds, "
            f"selection metric: {selection_metric})"
        ),
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        ),
        "",
        "Metric means and standard deviations:",
        fold_scores[metrics]
        .agg(["mean", "std"])
        .T
        .to_string(float_format=lambda value: f"{value:.3f}"),
        (
            f"\n{selection_metric}: "
            f"train-validation gap mean = "
            f"{fold_scores['gap'].mean():+.3f}"
        ),
    ]

    text = "\n".join(lines)
    print(text)
    return text

def corrected_paired_test(
    scores_a,
    scores_b,
    test_train_ratio: float,
) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models
    evaluated on the same CV folds.

    The correction accounts for overlap between training folds.

    Returns:
        mean_diff: mean score difference (model A - model B)
        wins: number of folds in which A scores higher
        t: corrected t statistic
        p_value: two-sided p-value
    """
    diff = (
        np.asarray(scores_a, dtype=float)
        - np.asarray(scores_b, dtype=float)
    )

    k = len(diff)

    if k < 2:
        raise ValueError(
            "At least two paired fold scores are required."
        )

    se = np.sqrt(
        (1 / k + test_train_ratio) * diff.var(ddof=1)
    )

    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(
        2 * stats.t.sf(abs(t), df=k - 1)
    )

    return {
        "mean_diff": float(diff.mean()),
        "wins": f"{int((diff > 0).sum())}/{k}",
        "t": t,
        "p_value": p_value,
    }

def oof_classification_report(
    y_true,
    oof_proba,
    threshold: float = 0.5,
) -> str:
    """
    Confusion matrix and classification report using out-of-fold
    predictions from the development set.
    """
    y_pred = (
        np.asarray(oof_proba) >= threshold
    ).astype(int)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        index=["true 0", "true 1"],
        columns=["predicted 0", "predicted 1"],
    )

    text = (
        f"Out-of-fold predictions, development set "
        f"(threshold {threshold}):\n\n"
        f"{cm.to_string()}\n\n"
        f"{classification_report(y_true, y_pred, zero_division=0)}"
    )

    print(text)
    return text


def fairness_report(
    y_true,
    oof_proba,
    extras: pd.DataFrame,
    sensitive_attr: str = "race",
    threshold: float = 0.5,
) -> str:
    """
    Compare false positive rates across groups for:
      - our model, using out-of-fold predictions;
      - COMPAS, where score_text other than 'Low' is treated
        as a positive prediction.

    Both are evaluated on the same development rows.

    This is a descriptive fairness check, not a complete fairness audit.
    """
    df = extras.copy()

    if sensitive_attr not in df.columns:
        raise ValueError(
            f"Sensitive attribute '{sensitive_attr}' is missing "
            "from the extras DataFrame."
        )

    if "score_text" not in df.columns:
        raise ValueError(
            "'score_text' is missing from extras; required for "
            "comparison with COMPAS."
        )

    df[sensitive_attr] = (
        df[sensitive_attr]
        .astype(object)
        .fillna("unknown")
    )

    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = (
        np.asarray(oof_proba) >= threshold
    ).astype(int)

    df["y_pred_compas"] = (
        df["score_text"].notna()
        & (df["score_text"] != "Low")
    ).astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} "
        "(development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        f"Model classification threshold: {threshold}",
        "",
    ]

    for label, col in [
        ("Our model", "y_pred_model"),
        ("COMPAS's own score", "y_pred_compas"),
    ]:
        lines.append(f"  {label}:")

        for group, group_df in df.groupby(
            sensitive_attr,
            dropna=False,
        ):
            negatives = group_df[group_df["y_true"] == 0]

            if len(negatives) == 0:
                continue

            fpr = (negatives[col] == 1).mean()

            lines.append(
                f"    {str(group):<20s} "
                f"FPR = {fpr:.2f} "
                f"(n={len(negatives)})"
            )

        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
