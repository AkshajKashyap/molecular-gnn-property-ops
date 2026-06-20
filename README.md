# Molecular GNN Property Ops

## Goal

Build a reproducible molecular property prediction system from SMILES strings. Future
milestones will cover graph neural networks, scaffold splits, calibration, uncertainty
analysis, FastAPI inference, and a Streamlit molecule explorer.

## Current Milestone

Milestone 1 established project paths, logging, validated CSV loading, and command-line
utilities. Milestone 2 adds reproducible dataset preparation and persistent random or
scaffold split metadata. It does not include graph featurization or model training yet.

## Milestone 2 Splits

A random split shuffles individual rows with a fixed seed. It is useful as a simple baseline,
but closely related molecules can appear in multiple partitions.

A scaffold split groups molecules by their Bemis-Murcko scaffold and places each complete
group in only one partition, reducing structural leakage. RDKit is available for the supported
Python environment and supplies the normal scaffold implementation. If RDKit cannot be
imported, the code returns an explicitly prefixed `fallback_shape:` key based on normalized
SMILES token topology. That fallback is deterministic but is not chemically exact and should
not be reported as a Bemis-Murcko scaffold.

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

Prepare a CSV with reproducible split metadata:

```bash
molgnn-ops prepare-csv data/raw/molecules.csv data/processed/molecules.csv \
  --smiles-col smiles \
  --target-col target \
  --dataset-name example \
  --split-strategy scaffold \
  --seed 42
```
