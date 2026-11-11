
"""
Generate predictions for a new dataset using a fitted pipeline.

The input data is cleaned and transformed using the same
configuration as the training data.

Every input row is preserved.
"""

import os
import pandas as pd

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    add_engineered_features,
    split_features_target,
)


def predict_new_data(pipeline, config: dict):
    """
    Generate probabilities and class predictions for new data.

    Parameters
    ----------
    pipeline:
        A fitted sklearn pipeline.
    config:
        Dataset configuration dictionary.

    Returns
    -------
    pd.DataFrame or None
        Predictions containing id, probability and prediction.
    """

    data_config = config["data"]
    output_config = config.get("output", {})
    diagnostics_config = config.get("diagnostics", {})

    prediction_path = data_config.get("prediction_path")
    predictions_path = output_config.get("predictions_path")

    # Prediction is optional for datasets without a prediction file.
    if not prediction_path:
        return None

    if not predictions_path:
        raise ValueError(
            "A prediction file was configured, but "
            "output.predictions_path is missing."
        )

    id_column = diagnostics_config.get("id_column", "id")

    # LOAD NEW DATA
    df = load_data(prediction_path)

    if id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' is missing from "
            f"{prediction_path}."
        )

    # Preserve original IDs and row order.
    ids = df[id_column].copy()

    original_rows = len(df)

    # CLEAN DATA
    df = clean_dataset(df, diagnostics_config)
    df = add_engineered_features(df)

    # No duplicate removal here:
    # every input row must receive a prediction.

    # SEPARATE FEATURES
    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )

    X_new, _, _ = split_features_target(
        df,
        data_config,
        mnar_sources
    )

    # GENERATE PROBABILITIES
    probabilities = pipeline.predict_proba(X_new)[:, 1]

    threshold = output_config.get(
        "prediction_threshold", 0.5
    )

    predictions = (probabilities >= threshold).astype(int)

    # BUILD OUTPUT
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

    # SAVE RESULTS
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
