"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> drop duplicate rows (training only, week 4)
    -> split features/target
    -> set the final test set aside, locked (week 4)
    -> stratified k-fold cross-validation of preprocessing + model on the development set (week 4)
    -> out-of-fold classification report + fairness check
    -> refit the final model on the whole development set -> save results

"""
import yaml 
from sklearn.metrics import ( balanced_accuracy_score, confusion_matrix, ) 
from sklearn.model_selection import ( StratifiedKFold, ) 
from sklearn.pipeline import Pipeline 
from src.data import load_data 
from src.preprocessing import ( clean_dataset, split_features_target, build_preprocessor, split_dev_test, drop_duplicate_rows, ) 
from src.model import build_model 
from src.evaluate import ( evaluate, fairness_report, cv_report, oof_classification_report, cross_validate_pipeline, ) 
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # LOAD + CLEAN DATA
    
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

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

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    # CROSS-VALIDATION ON DEV SET

    cv_config = config["cv"]

    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    
    scoring = cv_config.get("scoring", "accuracy")

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_config.get("n_jobs", 1)
    )

    # CROSS-VALIDATION REPORT

    report = cv_report(fold_scores, scoring)

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


