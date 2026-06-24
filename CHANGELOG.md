# Changelog

## 1.0.0 - 2026-06-24

Initial portfolio release.

### Added

- ESOL/Delaney benchmark ingestion, preparation, scaffold splits, and duplicate audits
- RDKit graph featurization and Morgan fingerprint generation
- Classical fingerprint baselines and GCN/GIN molecular regression baselines
- Repeated-seed benchmark comparison with tracked portfolio summaries
- Fixed-split ensemble uncertainty validation with the negative result preserved
- Validation-based model promotion with self-contained manifests and checkpoints
- FastAPI inference and Streamlit molecule explorer with applicability context
- Docker, Docker Compose, and GitHub Actions quality and Docker smoke workflows
- Model card, architecture documentation, methodology notes, release checklist, license,
  citation metadata, and contribution guide

### Benchmark Highlights

- Fingerprint random forest scaffold RMSE: 1.8480 +/- 0.0214
- GCN scaffold RMSE: 1.3395 +/- 0.0738
- GIN scaffold RMSE: 1.4499 +/- 0.1372
- Promoted fixed-split GCN validation RMSE: 1.3420
- Promoted fixed-split GCN test RMSE: 1.3502

### Experimental Integrity

- Model promotion uses validation metrics only.
- Test metrics are reported after selection.
- Duplicate and conflicting-target molecules are audited rather than silently averaged.
- Ensemble disagreement was not a reliable uncertainty signal and is not exposed.

### Known Limitations

- ESOL is small and domain-limited.
- Applicability similarity is not confidence.
- Predictions are not medical, pharmaceutical, clinical, or laboratory guidance.
- Out-of-domain molecules can produce implausible molar solubility values.

