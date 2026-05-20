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

Expected repository structure:

microemulsion-ml/
├── data/
│   ├── raw_dataset.csv
│   ├── cleaned_dataset_used_for_modeling.csv
│   └── data_dictionary.csv
├── src/
│   └── reproducible_microemulsion_workflow.py
├── outputs/                 # generated after running the script
│   ├── tables/
│   ├── figures/
│   ├── json/
│   ├── snippets/
│   └── logs/
├── requirements.txt
├── README.md
└── run.sh

Example usage from repository root:

python src/reproducible_microemulsion_workflow.py --data data/raw_dataset.csv --output outputs
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
    recall_score,
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
SMOTE_K_NEIGHBORS = 3
FIGURE_DPI = 600
DECISION_TREE_CCP_ALPHA = 0.006413

FEATURE_COLUMNS = [
    "Oil (mPa.s)",
    "Oil Amount (g)",
    "Surfactant (HLB)",
    "Surfactant Amount (g)",
    "Water-phase potential (V)",
    "Water Phase Amount (g)",
    "Co-Surfactant (Ratio)",
]

TARGET_COLUMN = "phase"

FORBIDDEN_TERMS = [
    "JUMLAH LOGAM",
    "LabelEncoder",
    "OneHotEncoder",
    "categorical",
    "emultant"
]

COLUMN_RENAME_MAP = {
    "FASA": "phase",

    # English legacy names
    "Water Phase (V)": "Water-phase potential (V)",
    "water phase (V)": "Water-phase potential (V)",
    "Water Phase Potential (V)": "Water-phase potential (V)",

    # Indonesian raw dataset column names
    "MINYAK (mPa.s)": "Oil (mPa.s)",
    "JUMLAH MINYAK (g)": "Oil Amount (g)",
    "SURFAKTAN (HLB)": "Surfactant (HLB)",
    "JUMLAH SURFAKTAN (g)": "Surfactant Amount (g)",
    "FASA AIR (V)": "Water-phase potential (V)",
    "JUMLAH FASA AIR (g)": "Water Phase Amount (g)",
    "Ko-Surfaktan (Rasio)": "Co-Surfactant (Ratio)",
}

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

def validate_no_forbidden_columns(df: pd.DataFrame) -> None:
    forbidden_hits = []
    for col in df.columns:
        for term in FORBIDDEN_TERMS:
            if term.lower() in str(col).lower():
                forbidden_hits.append(col)

    if forbidden_hits:
        raise ValueError(
            "Forbidden/unrelated column names detected in dataset: "
            + ", ".join(sorted(set(forbidden_hits)))
        )

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
    df_raw = df_raw.rename(columns=COLUMN_RENAME_MAP)
    validate_no_forbidden_columns(df_raw)
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
            "DecisionTree-pruned + scaling + SMOTE",
            lambda y: DecisionTreeClassifier(
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                ccp_alpha=DECISION_TREE_CCP_ALPHA,
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
        ["accuracy_mean", "f1_macro_mean", "precision_macro_mean", "recall_macro_mean"],
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
) -> Tuple[object, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
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
    classwise.to_csv(table_dir / "holdout_classwise_metrics.csv", index=False)
    cm_df.to_csv(table_dir / "holdout_confusion_matrix.csv")

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

    return model, holdout_summary, classwise, cm_df, y_pred

def bootstrap_holdout_ci(
    y_true: pd.Series,
    y_pred: np.ndarray,
    table_dir: Path,
    n_bootstrap: int = 5000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute stratified bootstrap 95% CI for hold-out metrics."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.array([1, 2, 3])

    idx_by_class = {label: np.where(y_true_arr == label)[0] for label in labels}
    rows = []

    def metric_row(y_t, y_p):
        p, r, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="macro", labels=labels, zero_division=0
        )
        recalls = recall_score(y_t, y_p, average=None, labels=labels, zero_division=0)
        return {
            "accuracy": accuracy_score(y_t, y_p),
            "precision_macro": p,
            "recall_macro": r,
            "f1_macro": f1,
            "phase1_recall": recalls[0],
            "phase2_recall": recalls[1],
            "phase3_recall": recalls[2],
        }

    point = metric_row(y_true_arr, y_pred_arr)

    for _ in range(n_bootstrap):
        boot_idx = []
        for label, idx in idx_by_class.items():
            boot_idx.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        boot_idx = np.asarray(boot_idx)
        rows.append(metric_row(y_true_arr[boot_idx], y_pred_arr[boot_idx]))

    boot_df = pd.DataFrame(rows)
    ci_rows = []
    for metric in point:
        ci_rows.append({
            "metric": metric,
            "point_estimate": point[metric],
            "ci95_low": float(np.percentile(boot_df[metric], 2.5)),
            "ci95_high": float(np.percentile(boot_df[metric], 97.5)),
        })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(table_dir / "holdout_bootstrap_ci_stratified.csv", index=False)
    return ci_df

def repeated_train_test_split_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    table_dir: Path,
    seeds: range = range(1, 51),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selected top models across repeated stratified train-test splits."""
    rows = []
    labels = np.array([1, 2, 3])

    top_models = [
        ("DecisionTree-pruned + scaling + SMOTE", lambda y_train: ModelSpec(
            "DecisionTree-pruned + scaling + SMOTE",
            lambda y: DecisionTreeClassifier(
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                ccp_alpha=DECISION_TREE_CCP_ALPHA,
                random_state=RANDOM_STATE,
            ),
            uses_scaling=True,
            uses_smote=True,
        ).build(y_train)),
        ("KNN + scaling + SMOTE", lambda y_train: ModelSpec(
            "KNN + scaling + SMOTE",
            lambda y: KNeighborsClassifier(n_neighbors=5, metric="minkowski", weights="uniform"),
            uses_scaling=True,
            uses_smote=True,
        ).build(y_train)),
        ("SVM-rbf + scaling + SMOTE", lambda y_train: ModelSpec(
            "SVM-rbf + scaling + SMOTE",
            lambda y: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            uses_scaling=True,
            uses_smote=True,
        ).build(y_train)),
    ]

    for seed in seeds:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
        )

        for model_name, builder in top_models:
            model = builder(y_train_s)
            model.fit(X_train_s, y_train_s)
            y_pred_s = model.predict(X_test_s)

            p, r, f1, _ = precision_recall_fscore_support(
                y_test_s, y_pred_s, average="macro", labels=labels, zero_division=0
            )
            recalls = recall_score(
                y_test_s, y_pred_s, average=None, labels=labels, zero_division=0
            )

            rows.append({
                "seed": seed,
                "model": model_name,
                "accuracy": accuracy_score(y_test_s, y_pred_s),
                "precision_macro": p,
                "recall_macro": r,
                "f1_macro": f1,
                "phase1_recall": recalls[0],
                "phase2_recall": recalls[1],
                "phase3_recall": recalls[2],
            })

    raw_df = pd.DataFrame(rows)
    summary_df = raw_df.groupby("model").agg({
        "accuracy": ["mean", "std", "min", "max"],
        "precision_macro": ["mean", "std", "min", "max"],
        "recall_macro": ["mean", "std", "min", "max"],
        "f1_macro": ["mean", "std", "min", "max"],
        "phase1_recall": ["mean", "std", "min", "max"],
        "phase2_recall": ["mean", "std", "min", "max"],
        "phase3_recall": ["mean", "std", "min", "max"],
    }).reset_index()

    summary_df.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]

    raw_df.to_csv(table_dir / "repeated_train_test_split_50seeds_raw.csv", index=False)
    summary_df.to_csv(table_dir / "repeated_train_test_split_50seeds_summary.csv", index=False)
    return raw_df, summary_df


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
        raise ImportError(
            "SHAP is required to reproduce manuscript feature-importance outputs. "
            "Install dependencies using: pip install -r requirements.txt"
        )

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

def summarize_smote_fold_distributions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report original and SMOTE-resampled class distributions within CV training folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        y_fold = y_train.iloc[train_idx]
        before = y_fold.value_counts().sort_index().to_dict()

        X_res, y_res = make_smote(y_fold).fit_resample(X_train.iloc[train_idx], y_fold)
        after = pd.Series(y_res).value_counts().sort_index().to_dict()

        rows.append({
            "fold": fold_id,
            "before_phase_1": before.get(1, 0),
            "before_phase_2": before.get(2, 0),
            "before_phase_3": before.get(3, 0),
            "after_phase_1": after.get(1, 0),
            "after_phase_2": after.get(2, 0),
            "after_phase_3": after.get(3, 0),
        })

    fold_df = pd.DataFrame(rows)

    summary_rows = []
    for phase in [1, 2, 3]:
        summary_rows.append({
            "class": f"Phase {phase}",
            "before_mean": fold_df[f"before_phase_{phase}"].mean(),
            "before_min": fold_df[f"before_phase_{phase}"].min(),
            "before_max": fold_df[f"before_phase_{phase}"].max(),
            "after_mean": fold_df[f"after_phase_{phase}"].mean(),
            "after_min": fold_df[f"after_phase_{phase}"].min(),
            "after_max": fold_df[f"after_phase_{phase}"].max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    fold_df.to_csv(table_dir / "smote_fold_distributions_50folds.csv", index=False)
    summary_df.to_csv(table_dir / "smote_fold_distribution_summary.csv", index=False)
    return fold_df, summary_df


def smote_sensitivity_analysis(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    table_dir: Path,
) -> pd.DataFrame:
    """Compare model performance with and without SMOTE using the same split."""
    rows = []
    labels = np.array([1, 2, 3])
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    base_models = [
        ("SVM-linear", lambda: SVC(kernel="linear", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ("SVM-rbf", lambda: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ("SVM-poly", lambda: SVC(kernel="poly", C=1.0, gamma="scale", degree=3, probability=True, random_state=RANDOM_STATE)),
        ("SVM-sigmoid", lambda: SVC(kernel="sigmoid", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
        ("KNN", lambda: KNeighborsClassifier(n_neighbors=5, metric="minkowski", weights="uniform")),
        ("GaussianNB", lambda: GaussianNB(var_smoothing=1e-9)),
        ("DecisionTree", lambda: DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2, ccp_alpha=DECISION_TREE_CCP_ALPHA, random_state=RANDOM_STATE)),
    ]

    for model_name, estimator_factory in base_models:
        for use_smote in [False, True]:
            steps = [("scaler", StandardScaler())]
            if use_smote:
                steps.append(("smote", make_smote(y_train)))
            steps.append(("classifier", estimator_factory()))

            pipe = ImbPipeline(steps) if use_smote else Pipeline(steps)

            cv_results = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring_dict(),
                n_jobs=None,
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            p, r, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", labels=labels, zero_division=0
            )
            recalls = recall_score(
                y_test, y_pred, average=None, labels=labels, zero_division=0
            )

            row = {
                "model": model_name,
                "scaling": True,
                "smote": use_smote,
                "accuracy_cv_mean": float(cv_results["test_accuracy"].mean()),
                "accuracy_cv_std": float(cv_results["test_accuracy"].std(ddof=1)),
                "precision_macro_cv_mean": float(cv_results["test_precision_macro"].mean()),
                "precision_macro_cv_std": float(cv_results["test_precision_macro"].std(ddof=1)),
                "recall_macro_cv_mean": float(cv_results["test_recall_macro"].mean()),
                "recall_macro_cv_std": float(cv_results["test_recall_macro"].std(ddof=1)),
                "f1_macro_cv_mean": float(cv_results["test_f1_macro"].mean()),
                "f1_macro_cv_std": float(cv_results["test_f1_macro"].std(ddof=1)),
                "holdout_accuracy": accuracy_score(y_test, y_pred),
                "holdout_precision_macro": p,
                "holdout_recall_macro": r,
                "holdout_f1_macro": f1,
                "holdout_phase1_recall": recalls[0],
                "holdout_phase2_recall": recalls[1],
                "holdout_phase3_recall": recalls[2],
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "smote_sensitivity_with_without.csv", index=False)
    return out

def decision_tree_pruning_sensitivity(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    table_dir: Path,
) -> pd.DataFrame:
    """Compare unpruned and cost-complexity-pruned Decision Tree models."""
    rows = []
    configs = [
        ("unpruned", 0.0),
        ("pruned", DECISION_TREE_CCP_ALPHA),
    ]

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    for label, alpha in configs:
        pipe = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote", make_smote(y_train)),
            ("classifier", DecisionTreeClassifier(
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                ccp_alpha=alpha,
                random_state=RANDOM_STATE,
            )),
        ])

        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_dict(),
            return_train_score=False,
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        clf = pipe.named_steps["classifier"]
        rows.append({
            "model": label,
            "ccp_alpha": alpha,
            "tree_depth": clf.get_depth(),
            "n_leaves": clf.get_n_leaves(),
            "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
            "cv_accuracy_std": cv_results["test_accuracy"].std(ddof=1),
            "cv_f1_macro_mean": cv_results["test_f1_macro"].mean(),
            "cv_f1_macro_std": cv_results["test_f1_macro"].std(ddof=1),
            "holdout_accuracy": accuracy_score(y_test, y_pred),
            "holdout_precision_macro": p,
            "holdout_recall_macro": r,
            "holdout_f1_macro": f1,
        })

    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "decision_tree_pruning_sensitivity.csv", index=False)
    return out

def summarize_decision_tree_structure(
    fitted_pipeline: object,
    table_dir: Path,
    prefix: str = "selected_decision_tree",
) -> pd.DataFrame:
    """Save fitted Decision Tree structural properties and terminal-node class distributions."""
    if "classifier" not in fitted_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a classifier step.")

    tree_model = fitted_pipeline.named_steps["classifier"]
    if not isinstance(tree_model, DecisionTreeClassifier):
        raise TypeError("Selected classifier is not a DecisionTreeClassifier.")

    tree = tree_model.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    n_node_samples = tree.n_node_samples
    values = tree.value.squeeze(axis=1) if tree.value.ndim == 3 else tree.value

    leaf_indices = np.where(children_left == children_right)[0]
    leaf_rows = []
    for node_id in leaf_indices:
        class_counts = values[node_id]
        leaf_rows.append({
            "node_id": int(node_id),
            "n_node_samples": int(n_node_samples[node_id]),
            "predicted_class": int(np.argmax(class_counts) + 1),
            "class1_count": float(class_counts[0]) if len(class_counts) > 0 else np.nan,
            "class2_count": float(class_counts[1]) if len(class_counts) > 1 else np.nan,
            "class3_count": float(class_counts[2]) if len(class_counts) > 2 else np.nan,
        })

    leaf_df = pd.DataFrame(leaf_rows)
    leaf_df.to_csv(table_dir / f"{prefix}_terminal_nodes.csv", index=False)

    summary = pd.DataFrame([{
        "model": prefix,
        "ccp_alpha": tree_model.ccp_alpha,
        "max_depth": int(tree_model.get_depth()),
        "number_of_leaves": int(tree_model.get_n_leaves()),
        "minimum_leaf_size": int(leaf_df["n_node_samples"].min()),
        "maximum_leaf_size": int(leaf_df["n_node_samples"].max()),
    }])
    summary.to_csv(table_dir / f"{prefix}_structure_summary.csv", index=False)
    return summary


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


def write_package_versions(json_dir: Path) -> None:
    """Write package versions used for reproducibility."""
    import platform
    import sklearn
    import imblearn
    import matplotlib

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "matplotlib": matplotlib.__version__,
        "shap_available": SHAP_AVAILABLE,
    }

    if SHAP_AVAILABLE:
        versions["shap"] = shap.__version__

    (json_dir / "package_versions.json").write_text(
        json.dumps(versions, indent=2),
        encoding="utf-8",
    )


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


def write_run_metadata(
    json_dir: Path,
    stats: dict,
    split_summary: dict,
    selected_model_name: str,
) -> None:
    metadata = {
        "raw_dataset_path": stats["dataset_path"],
        "cleaned_dataset_output": "outputs/tables/cleaned_dataset_used_for_modeling.csv",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "top_k_features": TOP_K_FEATURES,
        "smote_k_neighbors": SMOTE_K_NEIGHBORS,
        "figure_dpi": FIGURE_DPI,
        "selected_model": selected_model_name,
        "train_test_split_summary": split_summary,
        "workflow_order": [
            "data cleaning and duplicate removal",
            "stratified 80:20 train-test split",
            "repeated stratified CV on training set",
            "fold-specific StandardScaler for non-baseline models",
            "fold-specific SMOTE on training fold only",
            "classifier fitting and validation",
            "model selection from training-only CV",
            "final fitting on full training set",
            "training-only feature ranking and SHAP",
            "one-time hold-out evaluation"
        ],
        "holdout_policy": "Hold-out test set is never used for scaling fitting, SMOTE, feature selection, SHAP background, model selection, or cross-validation."
    }

    (json_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def write_model_configurations(specs: List[ModelSpec], json_dir: Path, y_reference: pd.Series) -> None:
    configs = []
    for spec in specs:
        estimator = spec.estimator_factory(y_reference)
        configs.append({
            "model": spec.name,
            "uses_scaling": spec.uses_scaling,
            "uses_smote": spec.uses_smote,
            "classifier": estimator.__class__.__name__,
            "hyperparameters": estimator.get_params(deep=False),
        })

    (json_dir / "model_configurations.json").write_text(
        json.dumps(configs, indent=2, default=str),
        encoding="utf-8",
    )


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
        │   └── raw_dataset.csv
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
        5. Water-phase potential (V)
        6. Water Phase Amount (g)
        7. Co-Surfactant (Ratio)

        The target column is 'phase', representing visually assigned formulation states.

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
        python src/reproducible_microemulsion_workflow.py --data data/raw_dataset.csv --output outputs
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

def write_data_dictionary(output_dir: Path) -> None:
    rows = [
        {
            "variable_name": "Oil (mPa.s)",
            "unit": "mPa.s",
            "type": "numerical feature",
            "description": "Oil viscosity descriptor",
            "used_in_model": True,
        },
        {
            "variable_name": "Oil Amount (g)",
            "unit": "g",
            "type": "numerical feature",
            "description": "Oil amount in formulation",
            "used_in_model": True,
        },
        {
            "variable_name": "Surfactant (HLB)",
            "unit": "dimensionless",
            "type": "numerical feature",
            "description": "Hydrophilic-lipophilic balance descriptor",
            "used_in_model": True,
        },
        {
            "variable_name": "Surfactant Amount (g)",
            "unit": "g",
            "type": "numerical feature",
            "description": "Surfactant amount in formulation",
            "used_in_model": True,
        },
        {
            "variable_name": "Water-phase potential (V)",
            "unit": "V",
            "type": "numerical feature",
            "description": "Electrical potential parameter of the water phase",
            "used_in_model": True,
        },
        {
            "variable_name": "Water Phase Amount (g)",
            "unit": "g",
            "type": "numerical feature",
            "description": "Water phase amount in formulation",
            "used_in_model": True,
        },
        {
            "variable_name": "Co-Surfactant (Ratio)",
            "unit": "ratio",
            "type": "numerical feature",
            "description": "Co-surfactant ratio in formulation",
            "used_in_model": True,
        },
        {
            "variable_name": "phase",
            "unit": "class label",
            "type": "target",
            "description": "Visually assigned formulation-state label",
            "used_in_model": True,
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "data_dictionary.csv", index=False)


def write_run_script(out_path: Path) -> None:
    """Write a simple shell script for reproducing all outputs."""
    run_script = textwrap.dedent(
        """
        #!/usr/bin/env bash
        set -euo pipefail
        python src/reproducible_microemulsion_workflow.py --data data/raw_dataset.csv --output outputs
        """
    ).strip()
    out_path.write_text(run_script + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible microemulsion phase-class prediction workflow")
    parser.add_argument("--data", type=Path, default=Path("data/raw_dataset.csv"), help="Path to raw_dataset.csv")
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
    write_package_versions(dirs["json"])

    df_clean, X, y, stats = load_dataset(args.data, TARGET_COLUMN)
    df_clean.to_csv(dirs["tables"] / "cleaned_dataset_used_for_modeling.csv", index=False)
    cleaned_data_path = args.data.parent / "cleaned_dataset_used_for_modeling.csv"
    df_clean.to_csv(cleaned_data_path, index=False)
    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    smote_fold_df, smote_fold_summary_df = summarize_smote_fold_distributions(
        X_train, y_train, dirs["tables"]
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

    repeated_split_raw_df, repeated_split_summary_df = repeated_train_test_split_sensitivity(
        X, y, dirs["tables"]
    )

    specs = get_model_specs()
    write_model_configurations(specs, dirs["json"], y_train)
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

    selected_model, holdout_summary, classwise, cm_df, y_pred_holdout = evaluate_final_model_on_holdout(
        best_spec,
        X_train,
        X_test,
        y_train,
        y_test,
        dirs["figures"],
        dirs["tables"],
    )

    if "DecisionTree" in best_model_name:
        tree_structure_df = summarize_decision_tree_structure(
            selected_model,
            dirs["tables"],
            prefix="selected_pruned_decision_tree",
        )

    bootstrap_ci_df = bootstrap_holdout_ci(y_test, y_pred_holdout, dirs["tables"])

    smote_sensitivity_df = smote_sensitivity_analysis(
        X_train, X_test, y_train, y_test, dirs["tables"]
    )

    dt_pruning_df = decision_tree_pruning_sensitivity(
        X_train, X_test, y_train, y_test, dirs["tables"]
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
        Dataset: {args.data}
        Output directory: {args.output}
        Selected model: {best_model_name}
        Figure DPI: {FIGURE_DPI}
        SHAP available: {SHAP_AVAILABLE}
        """
    ).strip()
    (dirs["logs"] / "run_completion_log.txt").write_text(completion_log + "\n", encoding="utf-8")
    print(completion_log)

    write_run_metadata(dirs["json"], stats, split_summary, best_model_name)
    write_data_dictionary(args.data.parent)
    write_data_dictionary(dirs["base"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
