# Molecular GNN Property Ops

## Goal

Build a reproducible molecular property prediction system from SMILES strings. Future
milestones will cover graph neural networks, scaffold splits, calibration, uncertainty
analysis, FastAPI inference, and a Streamlit molecule explorer.

## Current Milestone

Milestone 1 provides the tested project foundation: standard project paths, logging,
validated CSV dataset loading, and command-line utilities. It does not include molecule
featurization or model training yet.

## Installation

Create and activate a Python 3.11+ virtual environment, then install the project in editable
mode:

```bash
python -m pip install -e .
```

## Tests and Linting

```bash
pytest -q
ruff check .
```

## Command Line

Create the expected data, report, artifact, and configuration directories:

```bash
molgnn-ops init-dirs
```

Inspect a CSV with a target column:

```bash
molgnn-ops inspect-csv data/raw/molecules.csv \
  --smiles-col smiles \
  --target-col target \
  --dataset-name example
```

The `--target-col` option may be omitted for an unlabeled dataset.
