"""
Saving each run's results to disk.

Writes each run's full report to a timestamped file in results/,
and appends a summary row to a CSV scoreboard for comparing runs.

"""
from asyncio import run
import os
from datetime import datetime
import pandas as pd

def save_scores(path: str, row: dict) -> pd.DataFrame:
    """
    Appends one run to the CSV at `path` (created if needed)
    and returns the complete scoreboard.
    """
    # Create the parent folder before reading/writing the file.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    runs = (
        pd.read_csv(path, dtype={"run": str})
        if os.path.exists(path)
        else pd.DataFrame()
    )

    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    runs.to_csv(path, index=False)

    return runs

def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<timestamp>.txt` (the
    folder is created if needed) and returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)

    run_id = str(run) if run is not None else datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    
    path = os.path.join(results_dir, f"run_{run_id}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    preprocessing = config.get("preprocessing", {})
    test_set = config.get("test_set", {})

    metrics = cv.get("metrics")
    selection_metric = cv.get(
        "selection_metric",
        cv.get("scoring", "accuracy")
    )

    if metrics is None:
        metrics_text = cv.get("scoring", "accuracy")
    else:
        metrics_text = ", ".join(metrics)

    if tuning.get("enabled"):
        n_trials = tuning.get("n_trials", "unspecified")
        n_splits = tuning.get("n_splits", "unspecified")
        random_state = tuning.get("random_state", "unspecified")

        tuning_line = (
            f"Optuna, {n_trials} trials x {n_splits} folds, "
            f"nested CV  random_state={random_state}"
        )
    else:
        tuning_line = "off"

    header = (
        f"Run: {run_id}\n"
        f"Model: {config['model']['type']}  "
        f"params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={preprocessing.get('encoder', 'unspecified')}  "
        f"scaler={preprocessing.get('scaler', 'unspecified')}\n"
        f"CV: {cv.get('n_splits', 'unspecified')} stratified folds  "
        f"shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state', 'unspecified')}\n"
        f"Metrics: {metrics_text}  "
        f"selection metric: {selection_metric}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={test_set.get('size', 'unspecified')}  "
        f"random_state={test_set.get('random_state', 'unspecified')}  "
        f"(not evaluated)\n"
        + "=" * 60
        + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path
