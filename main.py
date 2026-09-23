"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> split features/target
    -> leak-safe train/test split -> preprocess + train (week 3's encoder/scaler pair)
    -> evaluate (accuracy, fairness) -> save results
"""
from matplotlib import cm
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import confusion_matrix
import yaml
from sklearn.pipeline import Pipeline
from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, build_preprocessor, split_train_test
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run
from sklearn.model_selection import StratifiedKFold, cross_val_score


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # load + diagnose-and-clean (week 3): domain-rule/placeholder -> NaN, category
    # cleanup, de-duplication, redundant-column removal -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # leak-safe split: everything above this line is target/split-independent and may
    # see the whole dataset; everything below (imputation, encoding, scaling) is fit
    # only on the training fold, inside the Pipeline below
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    preprocessor = build_preprocessor(config["preprocessing"])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", build_model(config["model"])),
    ])
    pipeline.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_test_pred)
    report += "\n\nConfusion matrix:"
    report += "\n Predicted"
    report += "\n 0 1"
    report += f"\nActual 0 {cm[0, 0]:<7} {cm[0, 1]}"
    report += f"\nActual 1 {cm[1, 0]:<7} {cm[1, 1]}"

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    report += f"\n\nBalanced accuracy: {balanced_acc:.3f}"

    # ROC-AUC requires predicted probabilities.
    if hasattr(pipeline, "predict_proba"):
        y_test_proba = pipeline.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            y_test_proba,
        )

        report += f"\n\nROC-AUC: {auc:.3f}"

    # 5-fold cross-validation on TRAINING data only
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    cv_accuracy = cross_val_score(pipeline,X_train,y_train,cv=cv,scoring="accuracy",)
    cv_auc = cross_val_score(pipeline,X_train,y_train,cv=cv,scoring="roc_auc",)

    report += "\n\n5-fold cross-validation on training set:"

    report += (f"\nAccuracy scores: "
               f"{[round(x, 3) for x in cv_accuracy]}")
    report += (f"\nMean accuracy: "
               f"{cv_accuracy.mean():.3f}")
    report += (f"\nAccuracy std: "
               f"{cv_accuracy.std():.3f}")
    report += (f"\n\nROC-AUC scores: "
               f"{[round(x, 3) for x in cv_auc]}")
    report += (f"\nMean ROC-AUC: "
               f"{cv_auc.mean():.3f}")
    report += (f"\nROC-AUC std: "
               f"{cv_auc.std():.3f}")

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()


