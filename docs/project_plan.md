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

## Future milestones

4. Fingerprint baseline.
5. GCN/GIN/MPNN models.
6. Random split vs scaffold split model evaluation.
7. Calibration and uncertainty analysis.
8. Error analysis reports.
9. FastAPI inference endpoint.
10. Streamlit molecule explorer.
11. Docker, CI, and final portfolio polish.
