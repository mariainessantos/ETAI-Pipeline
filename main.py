
"""
Entry point: python main.py.

Load config -> load and clean data -> engineer features -> split data
-> lock the test set -> cross-validate the complete pipeline
-> [if tuning enabled: nested CV, then tune on all development rows]
-> out-of-fold classification and fairness reports
-> refit on all development rows -> scoreboard + full report.
"""

import shutil
import tempfile
from datetime import datetime

import optuna
import yaml
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    split_features_target,
    split_dev_test,
    drop_duplicate_rows,
    add_engineered_features,
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    fairness_report,
    cv_report,
    oof_classification_report,
)
from src.results import save_run, save_scores
from src.tuning import (
    nested_cross_validate,
    tune_pipeline,
    tuning_report,
)


def load_config(path: str = "config.yaml") -> dict:
    """Load the project configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    # LOAD + CLEAN DATA

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(
        df_clean,
        config["diagnostics"].get("id_column")
    )
    df_clean = add_engineered_features(df_clean)

    # SPLIT FEATURES + TARGET

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources
    )

    # LEAK-SAFE TRAIN/DEV/TEST SPLIT

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # CROSS-VALIDATION CONFIGURATION

    cv_config = config["cv"]

    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None
    )

    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    # FEATURE SELECTION CONFIGURATION

    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)

    print(
        f"Model: {model_type} | "
        f"Selection metric: {selection_metric} | "
        f"{cv_config['n_splits']}-fold CV | "
        f"Feature selection: "
        f"{'min_votes ' + str(selection_config.get('min_votes', 2)) if selection_enabled else 'off'}"
    )

    print(
        f"Data: {len(X_dev)} development rows | "
        f"{len(X_test)} test rows locked away (not used)"
    )

    # BUILD PIPELINE WITH CACHING

    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")

    pipeline = build_pipeline(
        config["preprocessing"],
        config["model"],
        selection_config,
        memory=cache_dir
    )

    floor_pipeline = build_pipeline(
        config["preprocessing"],
        {"type": "dummy"}
    )

    # BASELINE CROSS-VALIDATION

    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs
    )

    # DEFAULT MODEL CROSS-VALIDATION

    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scorers,
        selection_metric,
        n_jobs
    )

    floor = floor_scores["validation"].mean()
    untuned = fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")

    print(
        f"  {selection_metric}: "
        f"{untuned:.3f} ± "
        f"{fold_scores['validation'].std(ddof=1):.3f} | "
        f"train-validation gap: "
        f"{fold_scores['gap'].mean():+.3f} | "
        f"no-model floor: {floor:.3f}"
    )

    report = (
        f"Hyperparameters from config.yaml: "
        f"{config['model'].get('params')}\n\n"
        + cv_report(
            fold_scores,
            metrics,
            selection_metric,
            "Cross-validation"
        )
        + f"\n\nNo-model floor (dummy, same folds): "
          f"{selection_metric} = {floor:.3f}"
    )

    final_scores = fold_scores

    # HYPERPARAMETER TUNING (OPTIONAL)

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    study = None

    if tuning_enabled:

        search_spaces = tuning_config.get("search_spaces") or {}

        if model_type not in search_spaces:
            raise ValueError(
                f"Tuning is enabled but config.yaml has no "
                f"tuning.search_spaces for '{model_type}'. "
                f"Add one, or set tuning.enabled: false."
            )

        # Start with the model's search space.
        search_space = dict(search_spaces[model_type])

        # Merge feature-selection parameters into the model search space.
        if selection_enabled:
            search_space.update(
                selection_config.get("search_space") or {}
            )

        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed
        )

        print(
            f"\nTuning ({n_trials} trials, nested CV) ... "
            f"this takes a while",
            flush=True
        )

        # HONEST ESTIMATE USING NESTED CROSS-VALIDATION

        final_scores, oof_proba = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scorers,
            selection_metric,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs
        )

        print(
            f"  {selection_metric}: "
            f"{final_scores['validation'].mean():.3f} ± "
            f"{final_scores['validation'].std(ddof=1):.3f} "
            f"(honest estimate of the tuned model)"
        )

        report += "\n\n" + cv_report(
            final_scores,
            metrics,
            selection_metric,
            "Nested cross-validation "
            "(the tuning procedure, estimated honestly)"
        )

        # TUNE ON ALL DEVELOPMENT ROWS

        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs
        )

        print("  Chosen hyperparameters:")

        for name, value in study.best_params.items():

            display_name = name.replace("model__", "")

            if isinstance(value, float):
                print(f"    {display_name:<24s} {value:.4g}")
            else:
                print(f"    {display_name:<24s} {value}")

        report += "\n\n" + tuning_report(
            study,
            final_scores,
            selection_metric
        )

    # FINAL CROSS-VALIDATION METRICS

    print(
        f"\nAll metrics "
        f"({'tuned model' if tuning_enabled else 'default hyperparameters'}, "
        f"mean ± std)"
    )

    for name in metrics:
        print(
            f"  {name:<18s} "
            f"{final_scores[name].mean():.3f} ± "
            f"{final_scores[name].std(ddof=1):.3f}"
        )

    # OUT-OF-FOLD CLASSIFICATION + FAIRNESS REPORTS

    report += "\n\n" + oof_classification_report(
        y_dev,
        oof_proba
    )

    report += "\n" + fairness_report(
        y_dev,
        oof_proba,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"]
    )

    # CONFUSION MATRIX

    y_oof_pred = (oof_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_dev, y_oof_pred)

    report += "\n\nConfusion matrix (out-of-fold predictions):"
    report += "\n Predicted"
    report += "\n          0       1"
    report += f"\nActual 0  {cm[0, 0]:<7} {cm[0, 1]}"
    report += f"\nActual 1  {cm[1, 0]:<7} {cm[1, 1]}"

    balanced_acc = balanced_accuracy_score(
        y_dev,
        y_oof_pred
    )

    report += f"\n\nOOF balanced accuracy: {balanced_acc:.3f}"

    # REFIT ON ALL DEVELOPMENT ROWS

    pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {model_type} refit on all "
        f"{len(X_dev)} development rows."
    )

    # PRINT FEATURES KEPT BY FEATURE SELECTION

    if selection_enabled:

        all_features = pipeline.named_steps["prep"].get_feature_names_out()

        kept = pipeline.named_steps["select"].get_feature_names_out(
            all_features
        )

        refit += (
            f"\nFeatures kept: {len(kept)} of {len(all_features)}: "
            f"{', '.join(kept)}"
        )

    report += "\n" + refit + "\n"

    # LOCKED TEST SET

    locked = (
        f"Locked test set: {len(X_test)} rows set aside, "
        f"not evaluated. Development set: {len(X_dev)} rows."
    )

    print(locked)
    report += "\n" + locked + "\n"

    # SCOREBOARD

    selection_setting = (
        f"min_votes {pipeline.named_steps['select'].min_votes}"
        if selection_enabled
        else "off"
    )

    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuning_enabled,
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "feature_selection": selection_setting,
        "params": str(config["model"].get("params")),
        "tuned_params": (
            str(study.best_params) if study is not None else ""
        ),
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{
            f"fold_{i}": score
            for i, score in enumerate(
                final_scores["validation"],
                start=1
            )
        },
    }

    scoreboard_path = config.get(
        "output", {}
    ).get("scoreboard", "results/runs.csv")

    runs = save_scores(scoreboard_path, row)

    same = runs[
        (runs["data"] == row["data"])
        & (runs["metric"] == selection_metric)
    ]

    board = (
        same.sort_values("mean", ascending=False)[
            [
                "run",
                "model",
                "tuned",
                "feature_selection",
                "mean",
                "std",
                "untuned",
                "floor",
            ]
        ]
        .rename(columns={"mean": selection_metric})
        .to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}"
        )
    )

    print(
        f"\nScoreboard (runs on the same data and metric; best first)\n"
        f"{board}"
    )

    report += f"\nScoreboard:\n{board}\n"

    # CLEAN UP CACHE

    shutil.rmtree(cache_dir, ignore_errors=True)

    # SAVE FULL REPORT

    path = save_run(
        config.get("output", {}).get("results_dir", "results"),
        run,
        config,
        report
    )

    print(f"\n{refit}")
    print(
        f"Full report (per-fold tables, tuning, confusion matrix, "
        f"fairness): {path}"
    )


if __name__ == "__main__":
    main()
