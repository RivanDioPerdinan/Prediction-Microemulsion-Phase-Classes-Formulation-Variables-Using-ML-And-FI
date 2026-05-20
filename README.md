[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20306768.svg)](https://doi.org/10.5281/zenodo.20306768)

# Microemulsion Phase-Class Prediction

This repository contains the manuscript-specific dataset, source code, and reproducible Python workflow for predicting visually assigned microemulsion formulation states from numerical formulation variables.

This repository supports the manuscript:

**Prediction of Microemulsion Phase Classes from Formulation Variables Using Machine Learning and Feature Importance**

The workflow was designed to address reproducibility and validation requirements by providing the raw dataset, cleaned dataset, preprocessing and model-training code, fixed random seed, package versions, scripts to regenerate outputs, exact output tables, generated figures, uncertainty analysis, SMOTE sensitivity analysis, and Decision Tree pruning analysis.

---

## Repository structure

```text
microemulsion-ml/
├── data/
│   ├── raw_dataset.csv
│   ├── cleaned_dataset_used_for_modeling.csv
│   └── data_dictionary.csv
├── src/
│   └── reproducible_microemulsion_workflow.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── json/
│   ├── snippets/
│   └── logs/
├── requirements.txt
├── README.md
└── run.sh
```

Before running the workflow, the repository should contain at least:

```text
data/raw_dataset.csv
src/reproducible_microemulsion_workflow.py
requirements.txt
run.sh
```

After running the workflow, the script automatically generates:

```text
data/cleaned_dataset_used_for_modeling.csv
data/data_dictionary.csv
outputs/
```

---

## Dataset

The raw dataset is stored in:

```text
data/raw_dataset.csv
```

The cleaned dataset used for modelling is stored in:

```text
data/cleaned_dataset_used_for_modeling.csv
outputs/tables/cleaned_dataset_used_for_modeling.csv
```

The data dictionary is stored in:

```text
data/data_dictionary.csv
outputs/data_dictionary.csv
```

The raw dataset contains laboratory experimental formulation data. After duplicate removal and cleaning, the dataset used for modelling contains 208 samples.

The model uses only seven numerical formulation variables:

1. `Oil (mPa.s)`
2. `Oil Amount (g)`
3. `Surfactant (HLB)`
4. `Surfactant Amount (g)`
5. `Water-phase potential (V)`
6. `Water Phase Amount (g)`
7. `Co-Surfactant (Ratio)`

The target column used by the workflow is:

```text
phase
```

The phase labels represent visually assigned formulation states.

---

## Legacy column-name handling

The raw dataset may contain legacy column names such as:

```text
Water Phase (V)
FASA
```

During preprocessing, the workflow automatically maps these columns to the final manuscript terminology:

```text
Water Phase (V)  ->  Water-phase potential (V)
FASA             ->  phase
```

The variable `Water-phase potential (V)` represents an electrical-potential-related parameter measured in volts. It is not a volume measurement.

---

## Variables used by the model

The workflow uses only the seven numerical formulation variables listed above.

The workflow does **not** use:

- `LabelEncoder`
- `OneHotEncoder`
- categorical preprocessing
- unrelated variables from earlier code versions
- any variable outside the seven numerical formulation descriptors described in the manuscript

The source code performs a strict column check and ignores extra non-manuscript columns if present in the raw CSV.

---

## Reproducibility protocol

The workflow follows a leakage-controlled machine-learning protocol:

- Random seed: `42`
- Train-test split: stratified `80:20`
- Cross-validation: repeated stratified 5-fold cross-validation with 10 repeats on the training set only
- Baseline model: SVM-linear without StandardScaler and without SMOTE
- Non-baseline models: StandardScaler + SMOTE inside the training pipeline
- SMOTE is applied only to training folds
- SMOTE is never applied before splitting
- SMOTE is never applied to validation folds or the independent hold-out test set
- Feature ranking is computed using training/CV data only
- SHAP background samples and explained instances are sampled only from training folds
- The independent hold-out test set is used only once for final evaluation and descriptive comparison
- Figures are saved at 600 dpi

---

## Machine-learning workflow

The final workflow is implemented in the following order:

1. Data loading
2. Column-name cleaning and mapping
3. Duplicate removal and invalid/missing-value removal
4. Stratified 80:20 train-test split
5. Repeated stratified 5-fold cross-validation with 10 repeats on the training set
6. Fold-specific StandardScaler fitting for non-baseline models
7. Fold-specific SMOTE resampling on the training fold only
8. Classifier fitting and validation within each fold
9. Model selection based on training-set cross-validation performance
10. Final fitting of the selected model on the full training set
11. Training-only feature ranking and SHAP analysis
12. One-time evaluation on the independent hold-out test set
13. Bootstrap uncertainty analysis
14. Repeated train-test split sensitivity analysis
15. SMOTE sensitivity analysis
16. Decision Tree pruning and tree-structure analysis
17. Generation of tables, figures, JSON metadata, logs, and manuscript-ready snippets

---

## Model configurations

The evaluated models are:

1. Baseline SVM-linear without scaling and without SMOTE
2. SVM-linear with scaling and SMOTE
3. SVM-RBF with scaling and SMOTE
4. SVM-polynomial with scaling and SMOTE
5. SVM-sigmoid with scaling and SMOTE
6. KNN with scaling and SMOTE
7. Gaussian Naive Bayes with scaling and SMOTE
8. Cost-complexity-pruned Decision Tree with scaling and SMOTE

No grid search is performed. Fixed model parameters are documented in:

```text
outputs/json/model_configurations.json
```

---

## Installation

Create and activate a Python virtual environment.

For Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the workflow

From the repository root, run:

```bash
bash run.sh
```

Alternatively, run the Python script directly:

```bash
python src/reproducible_microemulsion_workflow.py --data data/raw_dataset.csv --output outputs
```

The script regenerates all output tables, figures, JSON summaries, logs, and manuscript-ready snippets.

Note: SHAP-based feature-importance calculations may require several minutes depending on hardware.

---

## Main outputs

### Cleaned data and data dictionary

```text
data/cleaned_dataset_used_for_modeling.csv
data/data_dictionary.csv
outputs/tables/cleaned_dataset_used_for_modeling.csv
outputs/data_dictionary.csv
```

### Cross-validation and model comparison

```text
outputs/tables/cv_model_summary_training_only.csv
outputs/tables/holdout_all_model_summary.csv
outputs/tables/holdout_selected_model_summary.csv
```

### Hold-out evaluation

```text
outputs/tables/holdout_classwise_metrics.csv
outputs/tables/holdout_confusion_matrix.csv
outputs/tables/holdout_selected_model_classwise_report.csv
outputs/tables/holdout_selected_model_confusion_matrix_counts.csv
```

### Feature importance

```text
outputs/tables/feature_importance_mutual_information.csv
outputs/tables/feature_importance_linearsvc_l1.csv
outputs/tables/feature_importance_linearsvc_l2.csv
outputs/tables/feature_importance_shap_*.csv
```

### Feature-subset model comparison

```text
outputs/tables/feature_subset_model_comparison_training_only.csv
outputs/tables/feature_subset_model_comparison_holdout.csv
```

### SMOTE analysis

```text
outputs/tables/smote_fold_distributions_50folds.csv
outputs/tables/smote_fold_distribution_summary.csv
outputs/tables/smote_sensitivity_with_without.csv
```

### Hold-out uncertainty and split sensitivity

```text
outputs/tables/holdout_bootstrap_ci_stratified.csv
outputs/tables/repeated_train_test_split_50seeds_raw.csv
outputs/tables/repeated_train_test_split_50seeds_summary.csv
```

### Decision Tree pruning and structure

```text
outputs/tables/decision_tree_pruning_sensitivity.csv
outputs/tables/selected_pruned_decision_tree_structure_summary.csv
outputs/tables/selected_pruned_decision_tree_terminal_nodes.csv
```

### JSON metadata

```text
outputs/json/package_versions.json
outputs/json/run_metadata.json
outputs/json/model_configurations.json
outputs/json/train_test_split_summary.json
outputs/json/manuscript_summary.json
```

### Figures

```text
outputs/figures/
```

All figures are saved at 600 dpi.

### Logs and manuscript snippets

```text
outputs/logs/run_completion_log.txt
outputs/snippets/manuscript_ready_numbers_and_response_text.txt
outputs/protocol_summary.txt
```

---

## Data leakage prevention

The workflow was designed to prevent information leakage:

1. The dataset is cleaned before modelling.
2. The cleaned dataset is split into training and independent hold-out test sets using stratified 80:20 splitting.
3. Model selection is performed only on the training set using repeated stratified cross-validation.
4. StandardScaler is fitted only within each training fold for non-baseline models.
5. SMOTE is applied only to the corresponding training fold.
6. Validation folds are not resampled.
7. The independent hold-out test set is never resampled.
8. Feature ranking is computed using training/CV data only.
9. SHAP background samples and explained instances are sampled only from training folds.
10. The hold-out test set is used only for final evaluation and descriptive comparison.

---

## Notes on SMOTE

SMOTE is used only as a training-fold resampling strategy to reduce class imbalance during model learning.

Because SMOTE generates synthetic samples by interpolation, synthetic formulations may be mathematically plausible but not always chemically realistic for constrained formulation chemistry. Therefore, SMOTE-based results are reported together with sensitivity analysis comparing model performance with and without SMOTE.

Relevant output files:

```text
outputs/tables/smote_fold_distribution_summary.csv
outputs/tables/smote_sensitivity_with_without.csv
```

---

## Notes on Decision Tree pruning

The final selected Decision Tree model uses cost-complexity pruning.

Relevant output files:

```text
outputs/tables/decision_tree_pruning_sensitivity.csv
outputs/tables/selected_pruned_decision_tree_structure_summary.csv
outputs/tables/selected_pruned_decision_tree_terminal_nodes.csv
```

These files report the comparison between unpruned and pruned Decision Tree models, including tree depth, number of leaves, minimum leaf size, and terminal-node class distributions.

---

## Notes on feature importance

The repository reports feature-importance results from:

- Mutual Information
- LinearSVC-L1
- LinearSVC-L2
- SHAP-based model explanations

Feature importance is interpreted as model-dependent predictive association within this dataset. It should not be interpreted as causal physicochemical proof.

---

## Package versions

Package versions used to run the workflow are saved automatically in:

```text
outputs/json/package_versions.json
```

This file records Python, NumPy, pandas, scikit-learn, imbalanced-learn, matplotlib, and SHAP availability/version information.

---

## Run metadata

Complete run metadata are saved in:

```text
outputs/json/run_metadata.json
```

This file records:

- raw dataset path
- cleaned dataset output
- target column
- feature columns
- random seed
- train-test split
- cross-validation setting
- SMOTE setting
- selected model
- workflow order
- hold-out policy

---

## Data and code availability

The raw dataset, cleaned dataset, data dictionary, preprocessing code, model-training code, random seed settings, package requirements, run script, generated tables, generated figures, JSON summaries, logs, and exact output files are included in this repository.

A stable archived version of this repository is available on Zenodo:

DOI: [10.5281/zenodo.20306768](https://doi.org/10.5281/zenodo.20306768)

---

## How to verify the repository

After running the workflow, verify that the following files exist:

```text
data/raw_dataset.csv
data/cleaned_dataset_used_for_modeling.csv
data/data_dictionary.csv

outputs/json/package_versions.json
outputs/json/run_metadata.json
outputs/json/model_configurations.json

outputs/tables/cv_model_summary_training_only.csv
outputs/tables/holdout_selected_model_summary.csv
outputs/tables/holdout_classwise_metrics.csv
outputs/tables/holdout_confusion_matrix.csv
outputs/tables/holdout_bootstrap_ci_stratified.csv
outputs/tables/repeated_train_test_split_50seeds_summary.csv
outputs/tables/smote_fold_distribution_summary.csv
outputs/tables/smote_sensitivity_with_without.csv
outputs/tables/decision_tree_pruning_sensitivity.csv
outputs/tables/selected_pruned_decision_tree_structure_summary.csv
outputs/tables/selected_pruned_decision_tree_terminal_nodes.csv

outputs/figures/
outputs/logs/run_completion_log.txt
```

If these files are present, the repository contains the data, code, metadata, and output files required to reproduce the manuscript results.