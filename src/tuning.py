
"""
Hyperparameter tuning with Optuna, and nested cross-validation.

Tuning uses the development set only. Each candidate is scored using
cross-validation of the complete pipeline on the selection metric.

The best trial's score can be optimistic because it is the maximum
over multiple trials. Nested cross-validation provides a more honest
estimate of the tuning procedure's generalisation performance.
"""

import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


def tune_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorer,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """
    Runs an Optuna study on the development data.

    Each trial samples hyperparameters from search_space, fits a fresh
    copy of the pipeline through cross-validation, and returns the
    mean score for the selected metric.

    scorer can be a sklearn scorer callable or a scorer name string.

    Returns:
        best_pipeline: unfitted pipeline with the best parameters.
        study: Optuna study containing all trials and their scores.
    """

    # ADDED: allow either a scorer callable or a metric name.
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    def objective(trial):
        params = {}

        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )

            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )

            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    spec["choices"],
                )

            else:
                raise ValueError(
                    f"Unknown search-space type for {name}: {spec['type']}. "
                    "Options: int, float, categorical"
                )

        candidate = clone(pipeline).set_params(**params)

        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
        )

        # Store the variation between CV folds for each trial.
        trial.set_user_attr("std", float(scores.std(ddof=1)))

        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials)

    best_pipeline = clone(pipeline).set_params(**study.best_params)

    return best_pipeline, study


def nested_cross_validate(
    pipeline,
    X,
    y,
    outer_cv,
    inner_cv,
    scorers: dict,
    selection_metric: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """
    Estimates the performance of the complete tuning procedure.

    For each outer fold:
      1. Tune hyperparameters using only the outer training data.
      2. Fit the selected pipeline on the outer training data.
      3. Evaluate all configured metrics on the outer validation data.
      4. Store predicted probabilities for the outer validation rows.

    Returns:
        fold_scores: DataFrame with fold metrics, tuning scores,
                     and the selected hyperparameters.
        oof_proba: out-of-fold probabilities for the positive class.
    """

    # ADDED: validate the metric used to select hyperparameters.
    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric '{selection_metric}' is not in scorers. "
            f"Available metrics: {list(scorers)}"
        )

    rows = []

    # ADDED: store probabilities rather than hard class predictions.
    # This supports threshold-based classification and fairness analysis.
    oof_proba = np.empty(len(X), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Tune only on the outer training data.
        best_pipeline, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            random_state,
            n_jobs,
        )

        best_pipeline.fit(X_train, y_train)

        # Score the training data using the selection metric.
        train_score = scorers[selection_metric](
            best_pipeline,
            X_train,
            y_train,
        )

        # ADDED: evaluate every configured metric on the outer fold.
        val_scores = {
            name: scorer(best_pipeline, X_val, y_val)
            for name, scorer in scorers.items()
        }

        # ADDED: obtain probabilities for class 1 explicitly.
        classes = best_pipeline.classes_
        positive_class_indices = np.flatnonzero(classes == 1)

        if len(positive_class_indices) != 1:
            raise ValueError(
                "Expected class 1 as the positive class. "
                f"Found classes: {classes}"
            )

        positive_idx = positive_class_indices[0]

        oof_proba[val_idx] = best_pipeline.predict_proba(X_val)[
            :, positive_idx
        ]

        # Keep the original train/validation/gap structure.
        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": val_scores[selection_metric],
            "gap": train_score - val_scores[selection_metric],

            # ADDED: include all validation metrics.
            **val_scores,

            # Keep the best inner-CV score and chosen parameters.
            "inner_best": study.best_value,
            **study.best_params,
        })

    return pd.DataFrame(rows), oof_proba


def tuning_report(
    study,
    nested_scores: pd.DataFrame,
    selection_metric: str = "accuracy",
    top: int = 5,
) -> str:
    """
    Summarises the final Optuna study and the nested-CV optimism check.

    The best tuning score is compared with the mean outer-fold score.
    The outer-fold score is the more honest estimate of generalisation.
    """

    trials = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs")
    )

    trials = trials.rename(
        columns=lambda c: (
            c.replace("params_", "").replace("user_attrs_", "")
        )
    )

    trials = trials.rename(columns={
        "number": "trial",
        "value": f"mean {selection_metric}",
    })

    inner = nested_scores["inner_best"].mean()
    outer = nested_scores["validation"].mean()

    lines = [
        (
            f"Tuning on the whole development set "
            f"({len(study.trials)} Optuna trials, "
            f"metric: {selection_metric})"
        ),
        f"Best hyperparameters: {study.best_params}",
        (
            f"Best mean CV score:   {study.best_value:.3f} "
            "(maximum over trials -- potentially optimistic)"
        ),
        "",
        f"Top {top} trials:",
        trials.sort_values(
            f"mean {selection_metric}",
            ascending=False,
        ).head(top).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        ),
        "",
        "Optimism check (nested CV):",
        f"  Inner tuning score mean = {inner:.3f}",
        f"  Outer validation score mean = {outer:.3f}",
        f"  Optimism = {inner - outer:+.3f}",
        "  Report the outer validation score as the honest estimate.",
    ]

    # Preserve your original behaviour: print and return the report.
    text = "\n".join(lines)
    print(text)

    return text
