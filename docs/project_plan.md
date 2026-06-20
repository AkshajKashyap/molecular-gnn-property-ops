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

## Future milestones

7. GCN/GIN/MPNN models.
8. Random split vs scaffold split GNN evaluation.
9. Calibration and uncertainty analysis.
10. Extended error analysis reports.
11. FastAPI inference endpoint.
12. Streamlit molecule explorer.
13. Docker, CI, and final portfolio polish.
