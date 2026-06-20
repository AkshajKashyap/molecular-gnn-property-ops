# Molecular GNN Property Ops

## Goal

Build a reproducible molecular property prediction system from SMILES strings. Future
milestones will cover graph neural networks, scaffold splits, calibration, uncertainty
analysis, FastAPI inference, and a Streamlit molecule explorer.

## Current Milestone

Milestone 1 established project paths, logging, validated CSV loading, and command-line
utilities. Milestone 2 adds reproducible dataset preparation and persistent random or
scaffold split metadata. Milestone 3 converts valid SMILES into molecular graph records. It
does not include tensor conversion. Milestone 4 adds classical Morgan fingerprint baselines
before any neural network training.

## Milestone 2 Splits

A random split shuffles individual rows with a fixed seed. It is useful as a simple baseline,
but closely related molecules can appear in multiple partitions.

A scaffold split groups molecules by their Bemis-Murcko scaffold and places each complete
group in only one partition, reducing structural leakage. RDKit is available for the supported
Python environment and supplies the normal scaffold implementation. If RDKit cannot be
imported, the code returns an explicitly prefixed `fallback_shape:` key based on normalized
SMILES token topology. That fallback is deterministic but is not chemically exact and should
not be reported as a Bemis-Murcko scaffold.

## Milestone 3 Graph Featurization

Molecular graph featurization represents each atom as a node with numeric chemical features
and each bond as two directed edges with bond features. The current atom features include
element, degree, formal charge, aromaticity, hybridization, hydrogen count, and ring status.
Bond features include bond type, conjugation, and ring status.

Graphs are written as JSON Lines so every molecule and its features remain easy to inspect
before a later milestone introduces framework-specific tensor formats.

## Milestone 4 Fingerprint Baselines

Morgan fingerprints encode the circular atom environments around each molecule as a fixed
binary vector. Logistic regression, ridge regression, and random forests can use these vectors
directly, providing fast and credible reference models.

Classical baselines matter before GNNs because they show whether added graph-model complexity
actually improves held-out performance. Models are selected on the validation split and the
chosen model is evaluated once on the test split. Each run saves metrics, validation/test
predictions, a joblib model artifact, and a Markdown report.

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

Featurize a prepared CSV into molecular graph records:

```bash
molgnn-ops featurize-csv \
  data/processed/example_prepared.csv \
  data/processed/example_graphs.jsonl
```

Create Morgan fingerprints from a prepared CSV:

```bash
molgnn-ops fingerprint-csv \
  data/processed/example_prepared.csv \
  data/processed/example_fingerprints.npz
```

Train and evaluate the classical baseline models:

```bash
molgnn-ops train-fingerprint-baseline \
  data/processed/example_fingerprints.npz \
  artifacts/baselines/example
```
