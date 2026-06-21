# Molecular GNN Property Ops

## Goal

Build a reproducible molecular property prediction system from SMILES strings. Future
milestones will cover inference services, an interactive molecule explorer, deployment,
and final model documentation.

## Current Milestone

Milestone 1 established project paths, logging, validated CSV loading, and command-line
utilities. Milestone 2 adds reproducible dataset preparation and persistent random or
scaffold split metadata. Milestone 3 converts valid SMILES into molecular graph records. It
does not include tensor conversion. Milestone 4 adds classical Morgan fingerprint baselines
before any neural network training. Milestone 5 adds a reproducible real-data benchmark
workflow using ESOL/Delaney. Milestone 6 adds benchmark diagnostics and seeded split
comparison. Milestone 7 adds the first small GCN and GIN molecular graph baselines.
Milestone 8 compares those architectures across repeated seeds and reports their
aggregate performance alongside the fingerprint baseline when it is available.
Milestone 9 adds validation-calibrated GNN ensemble uncertainty and molecular error
analysis without treating the resulting intervals as formal guarantees.
Milestone 9.5 completes that work with stable sample identity and multiple GCNs trained
against one immutable scaffold split.

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

## Milestone 5 ESOL Benchmark

ESOL/Delaney is a small aqueous-solubility regression benchmark containing molecular SMILES
and measured log solubility. It is the first real dataset used to exercise the complete
classical workflow:

1. Download the registered source CSV.
2. Prepare deterministic random or scaffold splits.
3. Generate Morgan fingerprints.
4. Train and select the classical regression baselines.
5. Save metrics, predictions, a model, a Markdown report, and a benchmark summary.

Benchmark runs are written under `artifacts/benchmarks/<dataset>/seed_<seed>/`. Unit tests use
local synthetic files and do not depend on internet access.

## Milestone 6 Diagnostics and Split Comparison

Diagnostics come before GNNs so target shifts, scaffold concentration, prediction outliers,
and structural novelty are visible before adding model complexity. The diagnostics workflow
summarizes errors and train-test Morgan similarity, lists the worst predictions, and writes
four inspectable plots plus JSON and Markdown reports.

Random splits can place close structural analogues in both training and test sets. Scaffold
splits keep scaffold groups separate and are therefore expected to be harder when test
molecules are structurally less similar to training molecules. Multi-seed comparison helps
separate that split effect from one favorable or unfavorable random seed.

## Milestone 7 First GNN Baselines

A graph convolutional network (GCN) averages and transforms neighboring atom features. A
graph isomorphism network (GIN) uses learned neighborhood aggregation designed to distinguish
more graph structures. Both baselines use global mean pooling and a small regression head.

These models intentionally remain simple. The scaffold-split Morgan random forest is still
the comparison point, and GNN results are reported without assuming that added complexity
will improve RMSE. Bond features are preserved in the graph data, but these first GCN/GIN
operators use graph connectivity and atom features only.

## Milestone 8 Repeated-Seed GNN Comparison

Repeated model seeds expose sensitivity to parameter initialization, batch ordering, and
dropout. A single favorable result is weaker evidence than a consistent mean and standard
deviation across repeated runs. The split seed is recorded separately.

The comparison workflow runs GCN and GIN under the same configuration, preserves every run,
and writes CSV, JSON, Markdown, and matplotlib figures. It reports mean and standard deviation
for test RMSE, MAE, and R2 without hiding weak GIN results or unstable seeds. When nearby
fingerprint split-comparison metrics exist, the report includes them in a separate baseline
section.

## Milestone 9 GNN Uncertainty and Error Analysis

A deep ensemble averages predictions from repeated GCN runs. The ensemble mean is the point
prediction, while the sample standard deviation across members measures model disagreement as
a practical estimate of epistemic uncertainty. Ensemble disagreement captures only part of
predictive uncertainty, especially with just three models.

Regression prediction intervals are calibrated on validation residuals by scaling ensemble
disagreement to target empirical coverage levels. They are evaluated on test molecules using
empirical coverage and mean interval width (sharpness), not classification reliability
diagrams. These validation-calibrated intervals are not guaranteed confidence intervals and
may lose nominal coverage under scaffold or other distribution shifts.

Selective prediction sorts test molecules from lowest to highest uncertainty and reports the
RMSE obtained when retaining 25%, 50%, 75%, or 100% of predictions. Molecular descriptor
groups, uncertainty buckets, and the largest ensemble errors provide descriptive failure
analysis without claiming causal relationships. Every ensemble member must contain the same
validation and test sample IDs, splits, canonical structures, and targets.

## Milestone 9.5 Fixed-Split Ensemble

The original uncertainty attempt was rejected because its repeated runs used different
scaffold partitions and aligned ambiguous rows by SMILES. That failure remains visible as an
experimental-integrity safeguard: predictions from different validation/test populations
must never be combined.

`split_seed` controls only the immutable train/validation/test assignment. `model_seed`
controls parameter initialization, batch order, dropout, and other training randomness. The
fixed-split workflow prepares and featurizes ESOL once with one split seed, then reuses the
same graph JSONL for every model seed.

SMILES is not a unique observation identifier: ESOL contains repeated canonical molecules,
including measurements with conflicting targets. Each source row therefore receives a stable
`sample_id` derived from the dataset name and original source-row index. The ID and canonical
SMILES are propagated through prepared CSV, graph JSONL, PyG metadata, and prediction CSVs.
Duplicate measurements remain separate and are audited; conflicting targets are never
silently removed or averaged.

## Installation

Create and activate a Python 3.11+ virtual environment, then install the project in editable
mode:

```bash
python -m pip install -e .
```

For CPU-only GNN development, install the verified PyTorch CPU wheel before the optional GNN
extra:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[gnn]"
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
  --split-seed 42
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

List and download registered benchmark datasets:

```bash
molgnn-ops list-datasets
molgnn-ops download-dataset esol
```

Run the complete ESOL fingerprint benchmark:

```bash
molgnn-ops run-fingerprint-benchmark esol \
  --split-strategy scaffold \
  --seed 42 \
  --overwrite
```

Generate diagnostics for the existing ESOL seed-42 benchmark:

```bash
molgnn-ops diagnose-benchmark \
  artifacts/benchmarks/esol/seed_42/prepared.csv \
  artifacts/benchmarks/esol/seed_42/baseline/predictions.csv \
  artifacts/benchmarks/esol/seed_42/diagnostics
```

Compare random and scaffold splits across three seeds:

```bash
molgnn-ops compare-splits esol artifacts/benchmarks/esol/split_comparison \
  --seeds 42,43,44 \
  --split-strategies random,scaffold \
  --overwrite
```

Train directly from an existing graph JSONL file:

```bash
molgnn-ops train-gnn-regressor data/processed/example_graphs.jsonl artifacts/gnn/example \
  --model-name gcn \
  --seed 42 \
  --epochs 50
```

Run the complete ESOL GNN workflow:

```bash
molgnn-ops run-gnn-benchmark esol artifacts/benchmarks/esol/gnn_gcn_seed_42 \
  --split-strategy scaffold \
  --model-name gcn \
  --seed 42 \
  --epochs 50 \
  --overwrite
```

Equivalent reproducible scripts are available for both architectures:

```bash
bash scripts/run_esol_gcn_benchmark.sh
bash scripts/run_esol_gin_benchmark.sh
```

Compare both architectures on the ESOL scaffold split across three seeds:

```bash
molgnn-ops compare-gnns esol artifacts/benchmarks/esol/gnn_comparison \
  --models gcn,gin \
  --seeds 42,43,44 \
  --split-strategy scaffold \
  --epochs 50 \
  --overwrite
```

The reproducible wrapper runs the same command:

```bash
bash scripts/run_esol_gnn_comparison.sh
```

Analyze repeated GCN prediction files as an uncertainty ensemble:

```bash
molgnn-ops analyze-gnn-uncertainty \
  artifacts/benchmarks/esol/gnn_uncertainty \
  artifacts/benchmarks/esol/gnn_comparison/gcn/seed_42/training/predictions.csv \
  artifacts/benchmarks/esol/gnn_comparison/gcn/seed_43/training/predictions.csv \
  artifacts/benchmarks/esol/gnn_comparison/gcn/seed_44/training/predictions.csv \
  --target-coverages 0.80,0.90,0.95
```

The reproducible ESOL wrapper validates every input file before running:

```bash
bash scripts/run_esol_gcn_uncertainty.sh
```

The currently generated Milestone 8 files use seed-dependent scaffold partitions and ESOL
also contains duplicate `(smiles, split)` keys, including conflicting target measurements.
The strict ensemble loader rejects those legacy artifacts instead of silently intersecting or
deduplicating the test set. A real uncertainty report therefore requires ensemble members
trained with different initialization seeds on one fixed prepared split and aligned by stable
sample ID.

Generate and evaluate the valid fixed-split ESOL GCN ensemble:

```bash
molgnn-ops run-fixed-split-ensemble \
  esol \
  artifacts/benchmarks/esol/fixed_split_gcn \
  --split-strategy scaffold \
  --split-seed 42 \
  --model-seeds 42,43,44 \
  --model-name gcn \
  --epochs 50 \
  --overwrite
```

The reproducible wrapper runs the same workflow:

```bash
bash scripts/run_esol_fixed_split_gcn_ensemble.sh
```
