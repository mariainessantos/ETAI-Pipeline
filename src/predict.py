
"""
Prediction utilities for the Adult Income project.

Provides:
1. predict_new_data(): automatically predicts the configured
   prediction dataset during the training pipeline.
2. predict(): independently predicts a CSV using a saved model.
"""

import argparse
import os

import joblib
import pandas as pd
import yaml

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    add_engineered_features,
    split_features_target,
)


def predict_new_data(pipeline, config: dict):
    """
    Generate and save predictions for the configured new dataset.
    Used automatically by the training pipeline.
    """

    data_config = config["data"]
    output_config = config.get("output", {})
    diagnostics_config = config.get("diagnostics", {})

    prediction_path = data_config.get("prediction_path")
    predictions_path = output_config.get("predictions_path")

    if not prediction_path:
        return None

    if not predictions_path:
        raise ValueError(
            "A prediction file was configured, but "
            "output.predictions_path is missing."
        )

    id_column = diagnostics_config.get("id_column", "id")

    df = load_data(prediction_path)

    if id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' is missing from "
            f"{prediction_path}."
        )

    ids = df[id_column].copy()
    original_rows = len(df)

    df = clean_dataset(df, diagnostics_config)
    df = add_engineered_features(df)

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )

    X_new, _, _ = split_features_target(
        df,
        data_config,
        mnar_sources
    )

    probabilities = pipeline.predict_proba(X_new)[:, 1]

    threshold = output_config.get(
        "prediction_threshold", 0.5
    )

    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "id": ids.to_numpy(),
        "probability": probabilities,
        "prediction": predictions,
    })

    if len(results) != original_rows:
        raise ValueError(
            f"Prediction row count mismatch: "
            f"{len(results)} predictions for "
            f"{original_rows} input rows."
        )

    output_dir = os.path.dirname(predictions_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results.to_csv(predictions_path, index=False)

    print("\nNew data predictions:")
    print(f"  Input rows: {original_rows}")
    print(f"  Predictions generated: {len(results)}")
    print(f"  Positive predictions: {(predictions == 1).sum()}")
    print(f"  Negative predictions: {(predictions == 0).sum()}")
    print(f"  Saved to: {predictions_path}")

    return results


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """
    Generate predictions for every row in a new dataset.
    Used by the independent prediction script.
    """

    diagnostics_config = config["diagnostics"]
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    output_config = config.get("output", {})

    id_column = diagnostics_config["id_column"]

    if id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' is missing from the input data."
        )

    ids = df[id_column].copy()
    original_rows = len(df)

    clean = clean_dataset(df.copy(), diagnostics_config)
    clean = add_engineered_features(clean)

    mnar_sources = preprocessing_config.get(
        "mnar_indicator_sources", []
    )

    X, _, _ = split_features_target(
        clean,
        data_config,
        mnar_sources
    )

    probability = pipeline.predict_proba(X)[:, 1]

    threshold = output_config.get(
        "prediction_threshold", 0.5
    )

    prediction = (probability >= threshold).astype(int)

    results = pd.DataFrame({
        "id": ids.to_numpy(),
        "probability": probability,
        "prediction": prediction,
    })

    if len(results) != original_rows:
        raise ValueError(
            f"Expected {original_rows} predictions, "
            f"but generated {len(results)}."
        )

    return results


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file."""

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict Adult Income using a saved model."
    )

    parser.add_argument(
        "input",
        help="CSV containing new data (target column not required)."
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="predictions.csv",
        help="Path where predictions will be saved."
    )

    args = parser.parse_args()

    config = load_config("config_adult.yaml")

    model_path = config["output"]["model_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run 'python main_adult.py' first."
        )

    pipeline = joblib.load(model_path)

    df = load_data(args.input)

    predictions = predict(df, pipeline, config)

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    predictions.to_csv(args.output, index=False)

    print("\nPredictions generated successfully!")
    print(f"Input rows: {len(df)}")
    print(f"Rows predicted: {len(predictions)}")
    print(
        f"Positive predictions: "
        f"{(predictions['prediction'] == 1).sum()}"
    )
    print(
        f"Negative predictions: "
        f"{(predictions['prediction'] == 0).sum()}"
    )
    print(f"Saved to: {args.output}")
