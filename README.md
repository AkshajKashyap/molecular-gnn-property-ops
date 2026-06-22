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
Milestone 10 adds validation-only model promotion and FastAPI molecular solubility inference.
Milestone 11 adds a Streamlit molecule explorer and training-set applicability context.
Milestone 12 adds reproducible CPU containers, Docker Compose operations, and CI quality and
model-free service smoke checks.

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

## Milestone 10 Model Promotion and FastAPI Inference

Promotion ranks candidate training runs using validation metrics only. Test performance is
copied into the manifest after selection for final reporting, but it never influences which
checkpoint is deployed. The promoted package contains its own checkpoint, architecture and
feature dimensions, normalization metadata, candidate ranking, and selection report, so it
does not depend on the original training directory.

The API accepts one SMILES string or an ordered batch of up to 100 entries. It canonicalizes
valid molecules, predicts ESOL log solubility, and converts that result to molar solubility.
Invalid batch entries return individual errors without failing valid entries. Model metadata,
validation metrics, post-selection test metrics, split seeds, and limitations are available
from `/model`; internal filesystem paths are not exposed.

The earlier ensemble analysis found that disagreement did not rank ESOL errors reliably and
required very wide intervals for coverage. This API therefore exposes no uncertainty value.
It is a research and portfolio service, not a medical, pharmaceutical, laboratory-grade, or
safety-critical prediction system.

## Milestone 11 Molecule Explorer and Applicability Context

The promoted package now includes a compact reference index containing every training sample,
including duplicate canonical molecules with distinct sample IDs and conflicting measured
targets. Morgan fingerprints encode local atom environments, and Tanimoto similarity ranks
the query against those training fingerprints. The nearest molecules help users inspect
whether the query resembles examples seen during training.

Similarity is applicability context, not uncertainty, probability, or a reliability
guarantee. High similarity does not guarantee an accurate solubility estimate, similar
molecules can have different measured properties, and low similarity does not prove that a
prediction is wrong. Deterministic warnings flag low structural similarity and molecular
descriptors outside the observed training ranges without exposing the failed ensemble
disagreement signal.

The Streamlit explorer renders the query molecule, displays predicted log and molar
solubility, summarizes molecular descriptors and applicability warnings, and lists or renders
the nearest training molecules. The same information is available programmatically from
`POST /predict/context`; existing prediction endpoints remain unchanged.

## Milestone 12 Docker and CI

One Python 3.13 slim image packages the CLI, FastAPI service, Streamlit explorer, RDKit,
CPU PyTorch, and PyG. It runs as a non-root user and deliberately contains no datasets,
training checkpoints, or promoted registry artifacts. API and dashboard containers mount
the host registry read-only and use the same image with different commands.

GitHub Actions runs Ruff and the full unit suite on Python 3.13. A separate Docker workflow
builds the CPU image, validates runtime imports and Compose configuration, and verifies that
the API can report healthy model-free status without downloading data or training a model.
See [the operations guide](docs/operations.md) for configuration and troubleshooting.

Promote a model, build the image, and start both services:

```bash
bash scripts/promote_esol_gcn.sh
bash scripts/docker_build.sh
docker compose up --detach
```

Use alternate host ports without changing container ports:

```bash
API_PORT=8010 DASHBOARD_PORT=8502 docker compose up --detach
```

Run the operational smoke checks and stop the services:

```bash
bash scripts/docker_smoke.sh
docker compose down
```

The combined image is intentionally substantial because RDKit, CPU Torch/PyG, FastAPI,
and Streamlit share one reproducible runtime. It does not include CUDA.

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

Install the GNN and API extras together for model serving:

```bash
python -m pip install -e ".[gnn,api]"
```

Install all model-serving and dashboard extras together:

```bash
python -m pip install -e ".[gnn,api,dashboard]"
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

Promote the fixed-split candidate with the best validation RMSE:

```bash
bash scripts/promote_esol_gcn.sh
```

Run one CLI prediction:

```bash
molgnn-ops predict-smiles \
  artifacts/registry/esol-gcn-v1/manifest.json \
  "CCO"
```

Serve the promoted model on port 8000:

```bash
bash scripts/serve_esol_api.sh
```

Inspect health and public model metadata:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model
```

Request a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO"}'
```

Batch prediction is available at `POST /predict/batch` with a JSON body such as
`{"smiles":["CCO","CCN","invalid"]}`.

Re-promote the ESOL model with its training reference index, then launch the explorer:

```bash
bash scripts/promote_esol_gcn.sh
bash scripts/run_esol_dashboard.sh
```

Run Streamlit directly with a custom port:

```bash
molgnn-ops run-dashboard \
  artifacts/registry/esol-gcn-v1/manifest.json \
  --port 8501
```

Request prediction plus applicability context from the API:

```bash
curl -X POST http://localhost:8000/predict/context \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO","top_k":5}'
```
