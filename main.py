"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

load config -> load and clean data -> drop duplicate rows -> engineer features
-> split features/target -> lock test set -> cross-validate against a dummy baseline
-> [if tuning enabled: nested CV, then tune on all development rows]
-> out-of-fold classification and fairness reports
-> refit on all development rows -> scoreboard + save report

"""
import datetime

import optuna
import yaml 
from sklearn.metrics import ( balanced_accuracy_score, confusion_matrix, ) 
from sklearn.model_selection import ( StratifiedKFold, ) 
from sklearn.pipeline import Pipeline 
from src.data import load_data 
from src.preprocessing import ( clean_dataset, split_features_target, build_preprocessor, split_dev_test, drop_duplicate_rows, add_engineered_features, ) 
from src.model import build_model, build_pipeline
from src.evaluate import (build_scores, cross_validate_pipeline, fairness_report, cv_report, oof_classification_report, ) 
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    # LOAD + CLEAN DATA
    
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))
    df_clean = add_engineered_features(df_clean)

    # SPLIT FEATURES + TARGET

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # LEAK-SAFE TRAIN/DEV/TEST SPLIT

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # BUILD PIPELINE

    pipeline = build_pipeline(config["preprocessing"], config["model"]  )

    # CROSS-VALIDATION ON DEV SET

    cv_config = config["cv"]

    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)

    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scoreres = build_scores(metrics)

    n_jobs = cv_config.get("n_jobs", 1)

    print(
        f"Model: {model_type} | "
        f"Selection metric: {selection_metric} | "
        f"{cv_config['n_splits']}-fold CV"
    )
    print(
        f"Data: {len(X_dev)} development rows | "
        f"{len(X_test)} test rows locked away (not used)"
    )

    floor_pipeline = build_pipeline( config["preprocessing"], {"type": "dummy"})

    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs,
    )
    
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scorers,
        selection_metric,
        n_jobs,
    )

    floor = floor_scores["validation"].mean()
    untuned = fold_scores["validation"].mean()

    # NEW: default-model results and comparison with the no-model floor
    print("\nDefault hyperparameters (config.yaml)")
    print(
        f"  {selection_metric}: "
        f"{untuned:.3f} ± "
        f"{fold_scores['validation'].std(ddof=1):.3f}"
    )
    print(
        f"  Train-validation gap: "
        f"{fold_scores['gap'].mean():+.3f}"
    )
    print(f"  No-model floor: {floor:.3f}")

    report = (
        f"Hyperparameters from config.yaml: "
        f"{config['model'].get('params')}\n\n"
        + cv_report(
            fold_scores, metrics, selection_metric,
            "Cross-validation"
        )
        + f"\n\nNo-model floor (dummy, same folds): "
          f"{selection_metric} = {floor:.3f}"
    )

    # The scores used for the final report and scoreboard
    final_scores = fold_scores

    # HYPERPARAMETER TUNING (OPTIONAL)

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    study = None

    if tuning_enabled:
        # CHANGED: match the professor's config.yaml structure
        search_spaces = tuning_config.get("search_spaces") or {}

        if model_type not in search_spaces:
            raise ValueError(
                "Tuning is enabled but config.yaml has no "
                f"tuning.search_spaces entry for '{model_type}'."
            )

        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        print(
            f"\nNested CV: {cv.get_n_splits()} outer folds × "
            f"{inner_cv.get_n_splits()} inner folds × "
            f"{n_trials} Optuna trials"
        )

        # CHANGED: nested CV evaluates the tuning procedure honestly
        final_scores, y_oof = nested_cross_validate(
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
            n_jobs=n_jobs,
        )

        print(
            f"  Nested-CV {selection_metric}: "
            f"{final_scores['validation'].mean():.3f} ± "
            f"{final_scores['validation'].std(ddof=1):.3f}"
        )

        report += (
            "\n\n"
            + cv_report(
                final_scores,
                metrics,
                selection_metric,
                "Nested cross-validation "
                "(honest evaluation of the tuning procedure)",
            )
        )

        # Tune once on all development rows for the final model
        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        print("\nChosen hyperparameters:")
        for name, value in study.best_params.items():
            name = name.replace("model__", "")
            if isinstance(value, float):
                print(f"  {name:<24s} {value:.4g}")
            else:
                print(f"  {name:<24s} {value}")

        report += "\n\n" + tuning_report(
            study, final_scores, selection_metric
        )

    # NEW: report all configured metrics, not only the selection metric
    print(
        f"\nAll metrics "
        f"({'tuned model' if tuning_enabled else 'default model'}, "
        "mean ± SD)"
    )
    for name in metrics:
        print(
            f"  {name:<20s} "
            f"{final_scores[name].mean():.3f} ± "
            f"{final_scores[name].std(ddof=1):.3f}"
        )

    # OUT-OF-FOLD CLASSIFICATION + FAIRNESS REPORTS

    report += "\n\n" + oof_classification_report(y_dev, y_oof)

    report += "\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # Keep your additional confusion-matrix report
    cm = confusion_matrix(y_dev, y_oof)
    report += "\n\nConfusion matrix (out-of-fold predictions):"
    report += "\n Predicted"
    report += "\n          0       1"
    report += f"\nActual 0  {cm[0, 0]:<7} {cm[0, 1]}"
    report += f"\nActual 1  {cm[1, 0]:<7} {cm[1, 1]}"

    balanced_acc = balanced_accuracy_score(y_dev, y_oof)
    report += f"\n\nOOF balanced accuracy: {balanced_acc:.3f}"

    # REFIT ON ALL DEVELOPMENT ROWS

    pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {model_type} refit on all "
        f"{len(X_dev)} development rows."
    )
    report += "\n" + refit + "\n"

    # LOCKED TEST SET

    locked = (
        f"Locked test set: {len(X_test)} rows set aside, "
        f"not evaluated. Development set: {len(X_dev)} rows."
    )
    print(locked)
    report += "\n" + locked + "\n"

    # NEW: SAVE RUN TO SCOREBOARD

    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuning_enabled,
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": str(config["model"].get("params")),
        "tuned_params": (
            str(study.best_params) if tuning_enabled else ""
        ),
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{
            f"fold_{i}": score
            for i, score in enumerate(
                final_scores["validation"], start=1
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
                "run", "model", "tuned", "encoder", "scaler",
                "mean", "std", "untuned", "floor",
            ]
        ]
        .rename(columns={"mean": selection_metric})
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    print(
        "\nScoreboard (same data and selection metric; best first)\n"
        + board
    )
    report += f"\nScoreboard:\n{board}\n"

    # SAVE FULL REPORT

    results_dir = config.get(
        "output", {}
    ).get("results_dir", "results")

    # CHANGED: this assumes save_run accepts (results_dir, run, config, report)
    path = save_run(results_dir, run, config, report)

    print(f"\n{refit}")
    print(f"Full report saved to {path}")


if __name__ == "__main__":
    main()
