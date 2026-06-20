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

## Future milestones

6. GCN/GIN/MPNN models.
7. Random split vs scaffold split model evaluation.
8. Calibration and uncertainty analysis.
9. Error analysis reports.
10. FastAPI inference endpoint.
11. Streamlit molecule explorer.
12. Docker, CI, and final portfolio polish.
