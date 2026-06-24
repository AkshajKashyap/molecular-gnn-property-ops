# Model Card: ESOL GCN Solubility Regressor

## Model Overview

- Model name: `esol-gcn-v1`
- Task: molecular aqueous solubility regression from SMILES strings
- Dataset: ESOL/Delaney
- Target definition: measured log solubility in mols per litre
- Architecture: graph convolutional network (GCN) with global mean pooling and a regression head
- Input representation: RDKit-validated molecular graph from canonicalized SMILES
- Output: predicted log solubility and a derived molar solubility value, `10 ** logS`
- Intended use: educational, research, and portfolio demonstration of molecular ML operations
- Out-of-scope use: medical, pharmaceutical, clinical, laboratory, regulatory, safety-critical, or autonomous decision-making

The model is intentionally modest. Its value is the complete and reproducible workflow around
data preparation, realistic evaluation, promotion, inference, applicability context, and
honest uncertainty reporting.

## Training And Selection

- Split strategy: scaffold split
- Split seed: 42
- Promoted model seed: 43
- Train/validation/test counts: 790 / 169 / 169
- Selection rule: choose the candidate with the lowest validation RMSE
- Test usage: test metrics are reported only after validation-based selection
- Hidden dimension: 64
- Message-passing layers: 3
- Dropout: 0.1
- Epoch budget: 50
- Batch size: 32
- Optimizer settings: learning rate 0.001, weight decay 0.0001

The promoted fixed-split candidate had validation RMSE 1.3420. Its post-selection test
metrics were RMSE 1.3502, MAE 1.0385, and R2 0.6441.

## Benchmark Comparison

| Method | Split | Seeds | Test RMSE mean | RMSE std |
| --- | --- | --- | ---: | ---: |
| Morgan fingerprint random forest | scaffold | 42, 43, 44 | 1.8480 | 0.0214 |
| GCN | scaffold | 42, 43, 44 | 1.3395 | 0.0738 |
| GIN | scaffold | 42, 43, 44 | 1.4499 | 0.1372 |
| Promoted fixed-split GCN | scaffold | split 42, model 43 | 1.3502 | N/A |

The best repeated-split single GCN run used seed 44 and reached test RMSE 1.2391. That
single run was not used for deployment selection because the promoted model was chosen from
fixed-split candidates using validation RMSE only.

## Data Quality

- Dataset size: 1,128 rows
- Duplicate canonical-SMILES groups: 11
- Conflicting-target duplicate groups: 6
- Reference index size: 790 training samples

SMILES strings are not unique observation identifiers. The workflow assigns stable sample IDs
from dataset name and source-row index, then carries those IDs through prepared CSVs, graph
JSONL, training metadata, and prediction CSVs. This prevents incorrect row alignment when
canonical SMILES repeat. Conflicting measurements are audited and preserved as distinct
observations rather than silently averaged, deleted, or merged.

## Uncertainty And Applicability

A fixed-split three-member GCN ensemble was evaluated as a possible uncertainty signal.
Model-seed test RMSE values were 1.4148, 1.3502, and 1.7675; the ensemble RMSE was 1.4497.
Validation-calibrated intervals achieved nominal coverage only with very large mean widths.
The uncertainty-error correlations were approximately -0.016 for both Pearson and rank
correlation.

The conclusion is negative and important: ensemble disagreement did not rank prediction
errors effectively in this setup. The API and dashboard therefore do not expose confidence
or uncertainty estimates.

Applicability context is reported separately. The service can compare a query molecule to
training molecules using Morgan fingerprint Tanimoto similarity and descriptor-range checks.
This is descriptive context, not confidence.

## Limitations

- ESOL is a small dataset, so learned relationships may not generalize broadly.
- Scaffold splits are intentionally hard and expose structural generalization limits.
- The dataset covers a limited molecular domain and target definition.
- Log-solubility measurements may contain experimental noise.
- Structurally similar compounds can have substantially different solubility.
- Similarity is not confidence, probability, or a guarantee of accuracy.
- Predicted molar solubility can become physically implausible for out-of-domain molecules.
- The model should not be used for autonomous decisions.
- The system is not medical, pharmaceutical, clinical, or laboratory guidance.

## Ethics And Responsible Use

Appropriate uses include education, software engineering demonstration, molecular ML
workflow exploration, and reproducibility practice. Unsupported uses include high-stakes
screening, claims about biological activity or safety, clinical decisions, material release
decisions, or substituting for experimental validation.

Any real scientific or engineering decision should require independent domain review and
experimental measurement.

## Reproducibility

Primary commands:

```bash
pytest -q
ruff check .
bash scripts/promote_esol_gcn.sh
molgnn-ops predict-smiles artifacts/registry/esol-gcn-v1/manifest.json "CCO"
bash scripts/generate_demo.sh
bash scripts/docker_build.sh
bash scripts/docker_smoke_test.sh
```

Runtime and packaging:

- Package version: 1.0.0
- Python target: 3.13 in CI and Docker
- Key dependencies: RDKit, scikit-learn, PyTorch CPU, PyTorch Geometric, FastAPI, Streamlit
- Docker: CPU-only, non-root runtime, no generated artifacts baked into the image
- Model manifest: records architecture, feature dimensions, target normalization, split seed,
  model seed, validation metrics, post-selection test metrics, package version, and git commit

Seed handling is explicit: `split_seed` controls dataset partitioning, while `model_seed`
controls training randomness. They are not interchangeable.

