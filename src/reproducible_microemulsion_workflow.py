from __future__ import annotations

"""
Reproducible microemulsion phase-class prediction workflow.

This script is designed as the final manuscript-specific implementation for
repository release. It addresses the reviewers' reproducibility concerns by:

1. Using only the seven numerical formulation variables described in the manuscript.
2. Applying a strict column check; no unrelated variables are allowed in the workflow.
3. Avoiding LabelEncoder, One-Hot Encoding, or categorical preprocessing.
4. Performing an 80:20 stratified train-test split before model development.
5. Applying scaling and SMOTE only inside training pipelines and cross-validation folds.
6. Selecting models only from repeated stratified cross-validation on the training set.
7. Keeping the hold-out test set separate for final evaluation only.
8. Computing Mutual Information, LinearSVC, and SHAP feature rankings using training data only.
9. Saving all tables, figures, JSON summaries, and manuscript-ready text outputs.
10. Saving figures at 600 dpi for journal submission.

Expected repository structure:

microemulsion-ml/
├── data/
│   └── dataset.csv
├── src/
│   └── reproducible_microemulsion_workflow.py
├── outputs/                 # generated after running the script
├── requirements.txt
├── README.md
└── run.sh

Example usage from repository root:

python src/reproducible_microemulsion_workflow.py --data data/dataset.csv --output outputs
"""

import argparse
import json
import math
import os
import random
import re
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 10
TOP_K_FEATURES = 6
TARGET_COLUMN = "FASA"
SMOTE_K_NEIGHBORS = 3
FIGURE_DPI = 600

FEATURE_COLUMNS = [
    "Oil (mPa.s)",
    "Oil Amount (g)",
    "Surfactant (HLB)",
    "Surfactant Amount (g)",
    "Water Phase (V)",
    "Water Phase Amount (g)",
    "Co-Surfactant (Ratio)",
]

INVALID_MISSING_TOKENS = ["-", "", " ", "NA", "N/A", "NaN", "nan", "None", "null"]


@dataclass(frozen=True)
class ModelSpec:
    """Container for a model configuration used in the manuscript workflow."""

    name: str
    estimator_factory: Callable[[pd.Series], object]
    uses_scaling: bool
    uses_smote: bool

    def build(self, y_train: pd.Series):
        """Build a leakage-safe sklearn/imblearn pipeline."""
        steps: List[Tuple[str, object]] = []
        if self.uses_scaling:
            steps.append(("scaler", StandardScaler()))
        if self.uses_smote:
            steps.append(("smote", make_smote(y_train)))
        steps.append(("classifier", self.estimator_factory(y_train)))
        if self.uses_smote:
            return ImbPipeline(steps=steps)
        return Pipeline(steps=steps)


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set all random seeds used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def ensure_output_dirs(base: Path) -> Dict[str, Path]:
    """Create and return the output directory structure."""
    paths = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "json": base / "json",
        "snippets": base / "snippets",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_filename(text: str) -> str:
    """Convert model or method names into stable file names."""
    text = text.lower().strip()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def validate_required_columns(df: pd.DataFrame, target_col: str) -> None:
    """Fail fast if the dataset does not match the manuscript-specific schema."""
    required = FEATURE_COLUMNS + [target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nExpected columns are: "
            + ", ".join(required)
        )


def load_dataset(path: Path, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Load, validate, clean, and return the manuscript-specific dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df_raw = pd.read_csv(path)
    df_raw = clean_column_names(df_raw)
    validate_required_columns(df_raw, target_col)

    # Use only manuscript-specific columns. Extra columns may exist in the CSV,
    # but they are intentionally ignored and documented in the output summary.
    ignored_columns = [col for col in df_raw.columns if col not in FEATURE_COLUMNS + [target_col]]
    df_raw_selected = df_raw[FEATURE_COLUMNS + [target_col]].copy()

    stats = {
        "dataset_path": str(path),
        "required_features": FEATURE_COLUMNS,
        "target_column": target_col,
        "ignored_extra_columns": ignored_columns,
        "n_rows_raw": int(len(df_raw_selected)),
        "n_duplicates_raw": int(df_raw_selected.duplicated().sum()),
        "random_state": RANDOM_STATE,
    }

    df_clean = df_raw_selected.replace(INVALID_MISSING_TOKENS, np.nan).copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    before_numeric_drop = len(df_clean)
    for col in FEATURE_COLUMNS:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")

    df_clean = df_clean.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
    stats["n_rows_after_cleaning"] = int(len(df_clean))
    stats["n_removed_duplicate_rows"] = int(stats["n_rows_raw"] - len(df_raw_selected.drop_duplicates()))
    stats["n_removed_invalid_or_missing_after_duplicate_removal"] = int(before_numeric_drop - len(df_clean))

    X = df_clean[FEATURE_COLUMNS].astype(float)
    y = df_clean[target_col].astype(int)

    class_distribution = y.value_counts().sort_index().to_dict()
    stats["class_distribution_cleaned"] = {str(k): int(v) for k, v in class_distribution.items()}

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    min_class_count = int(y.value_counts().min())
    if min_class_count < N_SPLITS:
        raise ValueError(
            f"The smallest class has {min_class_count} samples, which is less than N_SPLITS={N_SPLITS}. "
            "Use fewer CV folds or add more samples."
        )

    return df_clean, X, y, stats


def make_smote(y_train: pd.Series) -> SMOTE:
    """Create SMOTE with a safe k_neighbors value for the smallest class in a fold."""
    min_class_count = int(pd.Series(y_train).value_counts().min())
    if min_class_count <= 1:
        k_neighbors = 1
    else:
        k_neighbors = max(1, min(SMOTE_K_NEIGHBORS, min_class_count - 1))
    return SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)


def get_model_specs() -> List[ModelSpec]:
    """
    Define all manuscript model configurations.

    No grid search is performed in this script. The models use fixed,
    explicitly documented scikit-learn hyperparameters. Model selection is
    based on repeated stratified CV on the training set only.
    """
    return [
        ModelSpec(
            "Baseline SVM-linear (no scaling, no SMOTE)",
            lambda y: SVC(kernel="linear", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            uses_scaling=False,
            uses_smote=False,
        ),
        ModelSpec(
            "SVM-linear + scaling + SMOTE",
            lambda y: SVC(kernel="linear", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            uses_scaling=True,
            uses_smote=True,
        ),
        ModelSpec(
            "SVM-rbf + scaling + SMOTE",
            lambda y: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            uses_scaling=True,
            uses_smote=True,
        ),
        ModelSpec(
            "SVM-poly + scaling + SMOTE",
            lambda y: SVC(kernel="poly", C=1.0, gamma="scale", degree=3, probability=True, random_state=RANDOM_STATE),
            uses_scaling=True,
            uses_smote=True,
        ),
        ModelSpec(
            "SVM-sigmoid + scaling + SMOTE",
            lambda y: SVC(kernel="sigmoid", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            uses_scaling=True,
            uses_smote=True,
        ),
        ModelSpec(
            "KNN + scaling + SMOTE",
            lambda y: KNeighborsClassifier(n_neighbors=5, metric="minkowski", weights="uniform"),
            uses_scaling=True,
            uses_smote=True,
        ),
        ModelSpec(
            "GaussianNB + scaling + SMOTE",
            lambda y: GaussianNB(var_smoothing=1e-9),
            uses_scaling=True,
            uses_smote=True,
        ),
        ModelSpec(
            "DecisionTree + scaling + SMOTE",
            lambda y: DecisionTreeClassifier(
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_STATE,
            ),
            uses_scaling=True,
            uses_smote=True,
        ),
    ]


def scoring_dict() -> Dict[str, str]:
    """Return the metrics used for repeated stratified cross-validation."""
    return {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }


def summarize_cv_results(cv_results: Dict[str, np.ndarray], model_name: str) -> Dict[str, float | str]:
    """Summarize repeated CV results with mean, SD, and 95% CI."""
    row: Dict[str, float | str] = {"model": model_name}
    n_obs = len(cv_results["test_accuracy"])
    for metric_key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        values = np.asarray(cv_results[f"test_{metric_key}"], dtype=float)
        row[f"{metric_key}_mean"] = float(values.mean())
        row[f"{metric_key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row[f"{metric_key}_ci95"] = float(1.96 * values.std(ddof=1) / math.sqrt(n_obs)) if n_obs > 1 else 0.0

    fit_times = np.asarray(cv_results["fit_time"], dtype=float)
    row["training_time_mean"] = float(fit_times.mean())
    row["training_time_std"] = float(fit_times.std(ddof=1)) if len(fit_times) > 1 else 0.0
    return row


def evaluate_models_by_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    specs: List[ModelSpec],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Evaluate all models by repeated stratified CV on the training set only."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    raw_results = {}

    for spec in specs:
        pipe = spec.build(y_train)
        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_dict(),
            return_train_score=False,
            n_jobs=None,
        )
        rows.append(summarize_cv_results(cv_results, spec.name))
        raw_results[spec.name] = cv_results

    summary = pd.DataFrame(rows).sort_values(
        ["f1_macro_mean", "accuracy_mean", "precision_macro_mean", "recall_macro_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return summary, raw_results


def save_figure(fig_path: Path) -> None:
    """Save current matplotlib figure at journal-ready resolution."""
    plt.tight_layout()
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_model_comparison(cv_summary: pd.DataFrame, fig_path: Path) -> None:
    """Plot repeated-CV model accuracy with standard deviation error bars."""
    plot_df = cv_summary.copy().sort_values("accuracy_mean", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(plot_df["model"], plot_df["accuracy_mean"], xerr=plot_df["accuracy_std"], alpha=0.85)
    plt.xlabel("Mean accuracy (training-only repeated stratified CV)")
    plt.ylabel("Model")
    plt.title("All-feature model comparison using repeated stratified CV")
    save_figure(fig_path)


def evaluate_final_model_on_holdout(
    best_spec: ModelSpec,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fig_dir: Path,
    table_dir: Path,
) -> Tuple[object, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the selected model on training data and evaluate it once on hold-out test data."""
    model = best_spec.build(y_train)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    labels = np.sort(pd.concat([y_train, y_test]).unique())

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    holdout_summary = pd.DataFrame(
        [
            {
                "selected_model": best_spec.name,
                "accuracy": acc,
                "precision_macro": p_macro,
                "recall_macro": r_macro,
                "f1_macro": f1_macro,
                "training_time_seconds": training_time,
            }
        ]
    )

    classwise = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T.reset_index().rename(columns={"index": "label"})

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{label}" for label in labels], columns=[f"Pred_{label}" for label in labels])

    holdout_summary.to_csv(table_dir / "holdout_selected_model_summary.csv", index=False)
    classwise.to_csv(table_dir / "holdout_selected_model_classwise_report.csv", index=False)
    cm_df.to_csv(table_dir / "holdout_selected_model_confusion_matrix_counts.csv")

    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Hold-out confusion matrix: {best_spec.name}")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar(label="Count")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    save_figure(fig_dir / "holdout_selected_model_confusion_matrix.png")

    return model, holdout_summary, classwise, cm_df


def evaluate_all_models_on_holdout(
    specs: List[ModelSpec],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fig_dir: Path,
    table_dir: Path,
) -> pd.DataFrame:
    """Evaluate each predefined model on the hold-out test set for descriptive comparison."""
    rows = []
    labels = np.sort(pd.concat([y_train, y_test]).unique())

    for spec in specs:
        model = spec.build(y_train)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        rows.append(
            {
                "model": spec.name,
                "accuracy": acc,
                "precision_macro": p_macro,
                "recall_macro": r_macro,
                "f1_macro": f1_macro,
                "training_time_seconds": training_time,
            }
        )

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual_{label}" for label in labels], columns=[f"Pred_{label}" for label in labels])
        cm_df.to_csv(table_dir / f"holdout_confusion_matrix_{safe_filename(spec.name)}.csv")

    holdout_all = pd.DataFrame(rows).sort_values(
        ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        ascending=False,
    ).reset_index(drop=True)
    holdout_all.to_csv(table_dir / "holdout_all_model_summary.csv", index=False)

    plot_holdout_all_model_comparison(holdout_all, fig_dir / "holdout_all_model_comparison.png")
    plot_holdout_all_model_comparison_with_table(holdout_all, fig_dir / "holdout_all_model_comparison_with_table.png")
    return holdout_all


def plot_holdout_all_model_comparison(df: pd.DataFrame, fig_path: Path) -> None:
    """Plot hold-out metrics for all models."""
    plot_df = df.copy().sort_values("accuracy", ascending=True)
    y_pos = np.arange(len(plot_df))
    width = 0.18

    plt.figure(figsize=(14, 8))
    plt.barh(y_pos - 1.5 * width, plot_df["accuracy"], height=width, label="Accuracy")
    plt.barh(y_pos - 0.5 * width, plot_df["precision_macro"], height=width, label="Precision")
    plt.barh(y_pos + 0.5 * width, plot_df["recall_macro"], height=width, label="Recall")
    plt.barh(y_pos + 1.5 * width, plot_df["f1_macro"], height=width, label="F1-score")
    plt.yticks(y_pos, plot_df["model"])
    plt.xlabel("Metric value on hold-out test set")
    plt.ylabel("Model")
    plt.title("Hold-out performance comparison using all features")
    plt.legend()
    save_figure(fig_path)


def plot_holdout_all_model_comparison_with_table(df: pd.DataFrame, fig_path: Path) -> None:
    """Plot hold-out metrics with an embedded table."""
    plot_df = df.copy()
    display_df = pd.DataFrame(
        {
            "Model": plot_df["model"],
            "Accuracy": plot_df["accuracy"].map(lambda x: f"{x:.4f}"),
            "Precision": plot_df["precision_macro"].map(lambda x: f"{x:.4f}"),
            "Recall": plot_df["recall_macro"].map(lambda x: f"{x:.4f}"),
            "F1-score": plot_df["f1_macro"].map(lambda x: f"{x:.4f}"),
        }
    )

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.8])

    ax = fig.add_subplot(grid[0])
    x_pos = np.arange(len(plot_df))
    width = 0.18
    ax.bar(x_pos - 1.5 * width, plot_df["accuracy"], width=width, label="Accuracy")
    ax.bar(x_pos - 0.5 * width, plot_df["precision_macro"], width=width, label="Precision")
    ax.bar(x_pos + 0.5 * width, plot_df["recall_macro"], width=width, label="Recall")
    ax.bar(x_pos + 1.5 * width, plot_df["f1_macro"], width=width, label="F1-score")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Hold-out performance comparison using all features")
    ax.legend()

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def mi_ranking(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Compute Mutual Information feature ranking using training data only."""
    scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    return pd.DataFrame({"feature": X_train.columns, "score": scores}).sort_values("score", ascending=False).reset_index(drop=True)


def linear_svc_ranking(X_train: pd.DataFrame, y_train: pd.Series, penalty: str) -> pd.DataFrame:
    """Compute LinearSVC coefficient-based ranking using training data only."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if penalty == "l1":
        model = LinearSVC(penalty="l1", dual=False, C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    elif penalty == "l2":
        model = LinearSVC(penalty="l2", dual=True, C=1.0, max_iter=5000, random_state=RANDOM_STATE)
    else:
        raise ValueError("penalty must be 'l1' or 'l2'")

    model.fit(X_scaled, y_train)
    scores = np.mean(np.abs(model.coef_), axis=0)
    return pd.DataFrame({"feature": X_train.columns, "score": scores}).sort_values("score", ascending=False).reset_index(drop=True)


def mean_abs_shap_array(shap_values: object, n_features_expected: int) -> np.ndarray:
    """Convert SHAP outputs from different model types into mean absolute feature importance."""
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(values) for values in shap_values], axis=0)
        importance = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            importance = np.mean(np.abs(arr), axis=0)
        elif arr.ndim == 3:
            feature_axis = None
            for axis, size in enumerate(arr.shape):
                if size == n_features_expected:
                    feature_axis = axis
                    break
            if feature_axis is None:
                feature_axis = arr.ndim - 1
            moved = np.moveaxis(arr, feature_axis, -1)
            importance = np.mean(np.abs(moved), axis=tuple(range(moved.ndim - 1)))
        else:
            raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")

    importance = np.asarray(importance).reshape(-1)
    if importance.shape[0] != n_features_expected:
        raise ValueError(
            f"SHAP importance length mismatch: got {importance.shape[0]}, expected {n_features_expected}"
        )
    return importance


def shap_ranking(
    model_spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP ranking without using the hold-out test set.

    SHAP background samples are taken from training folds. Explained instances are
    validation-fold samples from the training set only. This avoids test-set leakage.
    """
    if not SHAP_AVAILABLE:
        return None

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_importances: List[np.ndarray] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]

        pipe = model_spec.build(y_fold_train)
        pipe.fit(X_fold_train, y_fold_train)

        background = X_fold_train.sample(min(40, len(X_fold_train)), random_state=RANDOM_STATE + fold_idx)
        explained = X_fold_valid.sample(min(20, len(X_fold_valid)), random_state=RANDOM_STATE + fold_idx)

        if "DecisionTree" in model_spec.name:
            transformed_explained = explained
            if "scaler" in pipe.named_steps:
                transformed_explained = pd.DataFrame(
                    pipe.named_steps["scaler"].transform(explained),
                    columns=X_train.columns,
                    index=explained.index,
                )
            classifier = pipe.named_steps["classifier"]
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_explained)
        else:
            predict_fn = lambda arr: pipe.predict_proba(pd.DataFrame(arr, columns=X_train.columns))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(explained, nsamples=100)

        fold_importances.append(mean_abs_shap_array(shap_values, X_train.shape[1]))

    scores = np.mean(np.vstack(fold_importances), axis=0)
    return pd.DataFrame({"feature": X_train.columns, "score": scores}).sort_values("score", ascending=False).reset_index(drop=True)


def plot_feature_ranking(ranking_df: pd.DataFrame, title: str, fig_path: Path) -> None:
    """Plot the top feature ranking."""
    top = ranking_df.head(TOP_K_FEATURES).iloc[::-1]
    plt.figure(figsize=(9, 5.5))
    plt.barh(top["feature"], top["score"], alpha=0.85)
    plt.xlabel("Importance score")
    plt.title(title)
    save_figure(fig_path)


def evaluate_feature_subsets_by_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_rankings: Dict[str, pd.DataFrame],
    specs: List[ModelSpec],
) -> pd.DataFrame:
    """Evaluate top-k feature subsets using repeated CV on training data only."""
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for ranking_name, ranking_df in feature_rankings.items():
        selected_features = ranking_df["feature"].head(TOP_K_FEATURES).tolist()
        X_subset = X_train[selected_features].copy()

        for spec in specs:
            pipe = spec.build(y_train)
            cv_results = cross_validate(
                pipe,
                X_subset,
                y_train,
                cv=cv,
                scoring=scoring_dict(),
                return_train_score=False,
                n_jobs=None,
            )
            row = summarize_cv_results(cv_results, spec.name)
            row["feature_method"] = ranking_name
            row["selected_features"] = "; ".join(selected_features)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["feature_method", "f1_macro_mean", "accuracy_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def evaluate_feature_subsets_on_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_rankings: Dict[str, pd.DataFrame],
    specs: List[ModelSpec],
) -> pd.DataFrame:
    """Evaluate top-k feature subsets on hold-out data for descriptive reporting only."""
    rows = []

    for ranking_name, ranking_df in feature_rankings.items():
        selected_features = ranking_df["feature"].head(TOP_K_FEATURES).tolist()
        X_train_subset = X_train[selected_features].copy()
        X_test_subset = X_test[selected_features].copy()

        for spec in specs:
            pipe = spec.build(y_train)
            start_time = time.time()
            pipe.fit(X_train_subset, y_train)
            training_time = time.time() - start_time
            y_pred = pipe.predict(X_test_subset)

            acc = accuracy_score(y_test, y_pred)
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            rows.append(
                {
                    "feature_method": ranking_name,
                    "selected_features": "; ".join(selected_features),
                    "model": spec.name,
                    "accuracy": acc,
                    "precision_macro": p_macro,
                    "recall_macro": r_macro,
                    "f1_macro": f1_macro,
                    "training_time_seconds": training_time,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["feature_method", "accuracy", "f1_macro", "precision_macro", "recall_macro"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)


def plot_subset_cv_comparison(subset_df: pd.DataFrame, feature_method: str, fig_path: Path) -> None:
    """Plot repeated-CV accuracy for one top-k feature subset method."""
    tmp = subset_df[subset_df["feature_method"] == feature_method].copy().sort_values("accuracy_mean", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(tmp["model"], tmp["accuracy_mean"], xerr=tmp["accuracy_std"], alpha=0.85)
    plt.xlabel("Mean accuracy (training-only repeated stratified CV)")
    plt.ylabel("Model")
    plt.title(f"Top-{TOP_K_FEATURES} feature subset: {feature_method}")
    save_figure(fig_path)


def plot_subset_holdout_comparison(subset_holdout_df: pd.DataFrame, feature_method: str, fig_path: Path) -> None:
    """Plot hold-out accuracy for one top-k feature subset method."""
    tmp = subset_holdout_df[subset_holdout_df["feature_method"] == feature_method].copy().sort_values("accuracy", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(tmp["model"], tmp["accuracy"], alpha=0.85)
    plt.xlabel("Hold-out accuracy")
    plt.ylabel("Model")
    plt.title(f"Hold-out performance using top-{TOP_K_FEATURES} features from {feature_method}")
    save_figure(fig_path)


def build_manuscript_snippets(
    stats: dict,
    cv_summary: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    ranking_map: Dict[str, pd.DataFrame],
    cm_df: pd.DataFrame,
    classwise: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write manuscript-ready numbers and reviewer-response text."""
    best_cv = cv_summary.iloc[0]
    holdout = holdout_summary.iloc[0]

    def top_features(name: str) -> str:
        return ", ".join(ranking_map[name]["feature"].head(TOP_K_FEATURES).tolist())

    lines = [
        "MANUSCRIPT-READY NUMBERS AND TEXT",
        "=================================",
        "",
        f"Cleaned dataset size: {stats['n_rows_after_cleaning']} samples.",
        f"Cleaned class distribution: {stats['class_distribution_cleaned']}.",
        f"Train-test split: stratified 80:20 with random_state={RANDOM_STATE}.",
        f"Model selection: repeated stratified {N_SPLITS}-fold cross-validation with {N_REPEATS} repeats on the training set only.",
        "No categorical encoding was used because all seven input variables are numerical.",
        "Scaling and SMOTE were applied only inside training pipelines and cross-validation folds.",
        "The hold-out test set was not used for feature ranking, SHAP background data, model selection, or preprocessing fitting.",
        "",
        f"Best model selected by training-only repeated CV: {best_cv['model']}.",
        (
            "Repeated stratified CV for the selected model: "
            f"accuracy = {best_cv['accuracy_mean']:.4f} ± {best_cv['accuracy_std']:.4f}; "
            f"precision = {best_cv['precision_macro_mean']:.4f} ± {best_cv['precision_macro_std']:.4f}; "
            f"recall = {best_cv['recall_macro_mean']:.4f} ± {best_cv['recall_macro_std']:.4f}; "
            f"F1-score = {best_cv['f1_macro_mean']:.4f} ± {best_cv['f1_macro_std']:.4f}."
        ),
        (
            "Final hold-out evaluation of the pre-selected model: "
            f"accuracy = {holdout['accuracy']:.4f}; "
            f"precision = {holdout['precision_macro']:.4f}; "
            f"recall = {holdout['recall_macro']:.4f}; "
            f"F1-score = {holdout['f1_macro']:.4f}."
        ),
        "",
        "Top feature rankings:",
        f"Mutual Information: {top_features('Mutual Information')}",
        f"LinearSVC-L1: {top_features('LinearSVC-L1')}",
        f"LinearSVC-L2: {top_features('LinearSVC-L2')}",
    ]

    for name in ranking_map:
        if name.startswith("SHAP"):
            lines.append(f"{name}: {top_features(name)}")

    lines.extend(
        [
            "",
            "Hold-out confusion matrix counts:",
            cm_df.to_string(),
            "",
            "Class-wise hold-out metrics:",
            classwise.to_string(index=False),
            "",
            "Suggested reviewer-response sentence:",
            (
                "We have replaced the previous repository with a clean manuscript-specific implementation. "
                "The final repository contains the exact dataset, source code, fixed random seed, environment file, "
                "and generated output tables and figures. The script uses only the seven numerical formulation variables "
                "described in the manuscript, applies SMOTE and scaling only within training pipelines, computes feature "
                "importance using training data only, and reserves the hold-out test set for final evaluation."
            ),
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_json_summary(
    stats: dict,
    cv_summary: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    holdout_all_df: pd.DataFrame,
    ranking_map: Dict[str, pd.DataFrame],
    subset_cv_df: pd.DataFrame,
    subset_holdout_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write a machine-readable JSON summary of the full workflow."""
    data = {
        "workflow": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv": f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS})",
            "top_k_features": TOP_K_FEATURES,
            "figure_dpi": FIGURE_DPI,
            "grid_search_used": False,
            "categorical_encoding_used": False,
            "smote_scope": "inside training pipelines and cross-validation folds only",
            "holdout_usage": "final evaluation only; not used for feature ranking or model selection",
        },
        "dataset_summary": stats,
        "best_model_training_only_cv": cv_summary.iloc[0].to_dict(),
        "selected_model_holdout_test": holdout_summary.iloc[0].to_dict(),
        "holdout_all_models": holdout_all_df.to_dict(orient="records"),
        "top_features": {
            name: df["feature"].head(TOP_K_FEATURES).tolist()
            for name, df in ranking_map.items()
        },
        "best_feature_subset_cv_row": subset_cv_df.iloc[0].to_dict() if len(subset_cv_df) else None,
        "best_feature_subset_holdout_row": subset_holdout_df.iloc[0].to_dict() if len(subset_holdout_df) else None,
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_protocol_summary(out_path: Path) -> None:
    """Write a transparent protocol summary for the repository."""
    protocol = textwrap.dedent(
        f"""
        Reproducible workflow summary
        =============================
        Random state: {RANDOM_STATE}
        Dataset: manuscript-specific dataset with seven numerical formulation variables and target column '{TARGET_COLUMN}'
        Train-test split: stratified 80:20
        Model selection: repeated stratified {N_SPLITS}-fold CV with {N_REPEATS} repeats on training data only
        Model configurations: fixed scikit-learn hyperparameters; no grid search in this version
        Categorical encoding: not used
        Scaling: fitted only within training pipelines/folds
        SMOTE: applied only inside training pipelines/folds, never before splitting
        Feature importance: Mutual Information, LinearSVC-L1, LinearSVC-L2, and SHAP computed using training data only
        SHAP: background samples and explained instances sampled only from training folds
        Hold-out test: used only after model selection for final evaluation and descriptive model comparison
        Figure resolution: {FIGURE_DPI} dpi
        """
    ).strip()
    out_path.write_text(protocol + "\n", encoding="utf-8")


def write_readme_template(out_path: Path) -> None:
    """Write a README template that can be committed with the repository."""
    readme = textwrap.dedent(
        """
        # Microemulsion Phase-Class Prediction

        This repository contains the manuscript-specific dataset and reproducible Python workflow for predicting visually assigned microemulsion phase classes from numerical formulation variables.

        ## Repository structure

        ```text
        microemulsion-ml/
        ├── data/
        │   └── dataset.csv
        ├── src/
        │   └── reproducible_microemulsion_workflow.py
        ├── outputs/
        ├── requirements.txt
        ├── README.md
        └── run.sh
        ```

        ## Dataset

        The workflow uses seven numerical formulation variables:

        1. Oil (mPa.s)
        2. Oil Amount (g)
        3. Surfactant (HLB)
        4. Surfactant Amount (g)
        5. Water Phase (V)
        6. Water Phase Amount (g)
        7. Co-Surfactant (Ratio)

        The target column is `FASA`, representing visually assigned formulation states.

        ## Reproducibility protocol

        - Random seed: 42
        - Stratified train-test split: 80:20
        - Model selection: repeated stratified 5-fold cross-validation with 10 repeats on the training set only
        - Scaling and SMOTE are applied only inside training pipelines and cross-validation folds
        - Feature importance is computed using training data only
        - The hold-out test set is used only for final evaluation
        - Figures are saved at 600 dpi

        ## Installation

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        ```

        ## Run

        ```bash
        bash run.sh
        ```

        or

        ```bash
        python src/reproducible_microemulsion_workflow.py --data data/dataset.csv --output outputs
        ```

        ## Outputs

        The script generates:

        - `outputs/tables/`: CSV tables for CV, hold-out metrics, confusion matrices, and feature rankings
        - `outputs/figures/`: 600 dpi figures
        - `outputs/json/`: machine-readable workflow summary
        - `outputs/snippets/`: manuscript-ready result text
        - `outputs/protocol_summary.txt`: reproducibility protocol
        """
    ).strip()
    out_path.write_text(readme + "\n", encoding="utf-8")


def write_requirements_template(out_path: Path) -> None:
    """Write a requirements.txt template."""
    requirements = textwrap.dedent(
        """
        numpy
        pandas
        scikit-learn
        imbalanced-learn
        matplotlib
        shap
        """
    ).strip()
    out_path.write_text(requirements + "\n", encoding="utf-8")


def write_run_script(out_path: Path) -> None:
    """Write a simple shell script for reproducing all outputs."""
    run_script = textwrap.dedent(
        """
        #!/usr/bin/env bash
        set -euo pipefail
        python src/reproducible_microemulsion_workflow.py --data data/dataset.csv --output outputs
        """
    ).strip()
    out_path.write_text(run_script + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible microemulsion phase-class prediction workflow")
    parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"), help="Path to dataset.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument(
        "--write-repo-templates",
        action="store_true",
        help="Write README_template.md, requirements.txt, and run.sh into the output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_STATE)
    dirs = ensure_output_dirs(args.output)

    df_clean, X, y, stats = load_dataset(args.data, TARGET_COLUMN)
    df_clean.to_csv(dirs["tables"] / "cleaned_dataset_used_for_modeling.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    split_summary = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_class_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()},
        "test_class_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()},
    }
    (dirs["json"] / "train_test_split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    specs = get_model_specs()

    cv_summary, _ = evaluate_models_by_cv(X_train, y_train, specs)
    cv_summary.to_csv(dirs["tables"] / "cv_model_summary_training_only.csv", index=False)
    plot_model_comparison(cv_summary, dirs["figures"] / "cv_all_feature_model_comparison.png")

    best_model_name = str(cv_summary.iloc[0]["model"])
    best_spec = next(spec for spec in specs if spec.name == best_model_name)

    holdout_all_df = evaluate_all_models_on_holdout(
        specs,
        X_train,
        X_test,
        y_train,
        y_test,
        dirs["figures"],
        dirs["tables"],
    )

    _, holdout_summary, classwise, cm_df = evaluate_final_model_on_holdout(
        best_spec,
        X_train,
        X_test,
        y_train,
        y_test,
        dirs["figures"],
        dirs["tables"],
    )

    mi_df = mi_ranking(X_train, y_train)
    l1_df = linear_svc_ranking(X_train, y_train, "l1")
    l2_df = linear_svc_ranking(X_train, y_train, "l2")

    ranking_map: Dict[str, pd.DataFrame] = {
        "Mutual Information": mi_df,
        "LinearSVC-L1": l1_df,
        "LinearSVC-L2": l2_df,
    }

    top3_model_names = cv_summary.head(3)["model"].tolist()
    top3_specs = [next(spec for spec in specs if spec.name == model_name) for model_name in top3_model_names]

    for rank, spec in enumerate(top3_specs, start=1):
        shap_result = shap_ranking(spec, X_train, y_train)
        if shap_result is not None:
            ranking_map[f"SHAP Rank {rank} ({spec.name})"] = shap_result

    for name, ranking_df in ranking_map.items():
        file_safe = safe_filename(name)
        ranking_df.to_csv(dirs["tables"] / f"feature_importance_{file_safe}.csv", index=False)
        plot_feature_ranking(
            ranking_df,
            f"Top-{TOP_K_FEATURES} feature importance: {name}",
            dirs["figures"] / f"feature_importance_{file_safe}.png",
        )

    subset_cv_df = evaluate_feature_subsets_by_cv(X_train, y_train, ranking_map, specs)
    subset_cv_df.to_csv(dirs["tables"] / "feature_subset_model_comparison_training_only.csv", index=False)

    for method in subset_cv_df["feature_method"].dropna().unique():
        plot_subset_cv_comparison(
            subset_cv_df,
            method,
            dirs["figures"] / f"subset_cv_comparison_{safe_filename(method)}.png",
        )

    subset_holdout_df = evaluate_feature_subsets_on_holdout(X_train, X_test, y_train, y_test, ranking_map, specs)
    subset_holdout_df.to_csv(dirs["tables"] / "feature_subset_model_comparison_holdout.csv", index=False)

    for method in subset_holdout_df["feature_method"].dropna().unique():
        plot_subset_holdout_comparison(
            subset_holdout_df,
            method,
            dirs["figures"] / f"subset_holdout_comparison_{safe_filename(method)}.png",
        )

    build_manuscript_snippets(
        stats,
        cv_summary,
        holdout_summary,
        ranking_map,
        cm_df,
        classwise,
        dirs["snippets"] / "manuscript_ready_numbers_and_response_text.txt",
    )

    build_json_summary(
        stats,
        cv_summary,
        holdout_summary,
        holdout_all_df,
        ranking_map,
        subset_cv_df,
        subset_holdout_df,
        dirs["json"] / "manuscript_summary.json",
    )

    write_protocol_summary(dirs["base"] / "protocol_summary.txt")

    if args.write_repo_templates:
        write_readme_template(dirs["base"] / "README_template.md")
        write_requirements_template(dirs["base"] / "requirements.txt")
        write_run_script(dirs["base"] / "run.sh")

    completion_log = textwrap.dedent(
        f"""
        Finished successfully.
        Dataset: {args.data.resolve()}
        Output directory: {args.output.resolve()}
        Selected model: {best_model_name}
        Figure DPI: {FIGURE_DPI}
        SHAP available: {SHAP_AVAILABLE}
        """
    ).strip()
    (dirs["logs"] / "run_completion_log.txt").write_text(completion_log + "\n", encoding="utf-8")
    print(completion_log)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
