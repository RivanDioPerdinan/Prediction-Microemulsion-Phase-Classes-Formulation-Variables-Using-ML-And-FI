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
- SHAP background samples and explained instances are sampled only from training folds
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
