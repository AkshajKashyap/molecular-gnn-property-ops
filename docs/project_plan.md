# Project Plan

## Completed

### Milestone 1: Project foundation

- Package and project directory structure
- Validated CSV loading
- Logging and command-line utilities
- Unit tests and lint configuration

### Milestone 2: Dataset preparation and split metadata

- Deterministic random splits
- Bemis-Murcko scaffold grouping with a clearly labeled token fallback
- Prepared CSV output with persistent train, validation, and test labels
- Dataset preparation summaries and CLI workflow

### Milestone 3: Molecular graph featurization

- RDKit SMILES validation and canonicalization
- Deterministic atom and bond feature vectors
- Directed molecular graph construction
- Inspectable JSONL graph persistence and CLI workflow

### Milestone 4: Classical fingerprint baselines

- RDKit Morgan fingerprint generation and compressed NPZ datasets
- Logistic/ridge and random forest reference models
- Validation-based model selection and held-out test evaluation
- Inspectable metrics, predictions, model artifacts, and Markdown reports

### Milestone 5: Real benchmark workflow

- Registered ESOL/Delaney benchmark metadata and cached download support
- End-to-end preparation, fingerprinting, baseline training, and reporting
- Reproducible seed-specific benchmark artifact directories
- Network-free unit tests and a repeatable shell entry point

### Milestone 6: Benchmark diagnostics and split comparison

- Target, prediction-error, scaffold, and train-test similarity diagnostics
- Inspectable matplotlib figures and Markdown diagnostics report
- Multi-seed random versus scaffold fingerprint baseline comparison
- Reproducible diagnostic and comparison shell entry points

### Milestone 7: First molecular GNN baselines

- PyTorch Geometric conversion of inspectable molecular graph records
- Minimal GCN and GIN graph-level regression models
- Validation-selected training with early stopping and held-out evaluation
- Reproducible ESOL GNN workflows and fingerprint-baseline comparison

### Milestone 8: Repeated-seed GNN comparison

- Repeated GCN and GIN benchmarks under matched hyperparameters
- Mean and standard deviation for held-out RMSE, MAE, and R2
- Optional nearby fingerprint baseline integration
- Inspectable CSV, JSON, Markdown, and matplotlib comparison artifacts

### Milestone 9: GNN uncertainty and molecular error analysis

- GNN ensemble mean and disagreement-based uncertainty
- Validation-calibrated regression prediction intervals
- Empirical coverage, interval width, and selective prediction evaluation
- Descriptor groups, uncertainty buckets, and worst-prediction analysis

Completion required a fixed-split repair after the first real run correctly rejected
seed-dependent partitions and ambiguous SMILES-only alignment.

### Milestone 9.5: Fixed-split ensemble generation

- Stable source-row sample IDs and canonical SMILES across all data representations
- Duplicate-molecule audit that preserves and reports conflicting measurements
- Independent split and model seeds
- One immutable prepared split and graph dataset reused by every ensemble member
- Sample-ID-aligned uncertainty evaluation with validation-only interval calibration

## Future milestones

10. Inference API.
11. Interactive dashboard.
12. Docker and CI.
13. Final model card and portfolio documentation.
