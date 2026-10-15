"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

load config -> load and clean data -> drop duplicate rows (training only) -> features/target
-> lock the test set -> cross-validate the pipeline on the development set
-> [if tuning is enabled: nested CV of the tuning procedure, then tuning on all development rows]
-> out-of-fold classification and fairness reports -> refit on all development rows -> save the report

"""
import optuna
import optuna
import yaml 
from sklearn.metrics import ( balanced_accuracy_score, confusion_matrix, ) 
from sklearn.model_selection import ( StratifiedKFold, ) 
from sklearn.pipeline import Pipeline 
from src.data import load_data 
from src.preprocessing import ( clean_dataset, split_features_target, build_preprocessor, split_dev_test, drop_duplicate_rows, add_engineered_features, ) 
from src.model import build_model, build_pipeline
from src.evaluate import (fairness_report, cv_report, oof_classification_report, cross_validate_pipeline, ) 
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

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
    
    scoring = cv_config.get("scoring", "accuracy")

    n_jobs = cv_config.get("n_jobs", 1)

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_config.get("n_jobs", n_jobs)
    )

    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)

    # CROSS-VALIDATION REPORT

    report = header + "\n" + cv_report(fold_scores, scoring)

    # HYPERPARAMETER TUNING (OPTIONAL)

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_enabled:
        model_type = config["model"]["type"]
        search_space = tuning_config.get("search_space") or {}

        if model_type not in search_space:
            raise ValueError(f"Model type {model_type} not found in tuning search space.")

        search_space = search_space[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed,)

        print(
            f"\nNested cross-validation: "
            f"{cv.get_n_splits()} outer folds x "
            f"{n_trials} trials x "
            f"{inner_cv.get_n_splits()} inner folds ..."
        )

        nested_scores, y_oof = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        report += (
            "\n\nNested cross-validation "
            "(honest evaluation of the tuning procedure):\n"
        )
        report += cv_report(nested_scores, scoring)

        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        report += "\n\n" + tuning_report(
            study,
            nested_scores,
            scoring,
        )

    else:
        pass

    # CLASSIFICATION + FAIRNESS REPORTS
    
    report += "\n\n" + oof_classification_report(y_dev, y_oof)

    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    cm = confusion_matrix(y_dev, y_oof)
    report += "\n\nConfusion matrix (out-of-fold predictions):" 
    report += "\n Predicted" 
    report += "\n 0 1" 
    report += f"\nActual 0 {cm[0, 0]:<7} {cm[0, 1]}" 
    report += f"\nActual 1 {cm[1, 0]:<7} {cm[1, 1]}"

    balanced_acc = balanced_accuracy_score(y_dev, y_oof)
    report += ( f"\n\nOOF balanced accuracy: " f"{balanced_acc:.3f}" )

    # FINAL MODEL

    final_model = pipeline.fit(X_dev, y_dev)

    refit = f"Final Model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if tuning_enabled:
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    # LOCKED TEST SET
    
    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_dev)} rows.")
    
    print(locked)
    report += "\n" + locked + "\n"

    # SAVE RESULTS

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")

if __name__ == "__main__":
    main()


