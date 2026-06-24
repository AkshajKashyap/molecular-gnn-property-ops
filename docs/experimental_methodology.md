# Experimental Methodology

## Why Scaffold Splits Matter

Random splits are useful sanity checks, but they can place close structural analogues in
training and test partitions. Molecular property prediction often cares about whether a
model generalizes to new chemical series. Scaffold splits group molecules by Bemis-Murcko
scaffold and hold out whole scaffold groups, making the test set structurally harder.

That difference showed up clearly. The three-seed fingerprint random forest reached mean
test RMSE 1.1622 on random splits, but 1.8480 on scaffold splits. The scaffold result is the
more credible stress test for structural generalization.

## Repeated Seeds Versus Ensembles

Repeated-seed benchmark comparison asks whether an architecture is consistently strong under
matched training settings. It reports mean and standard deviation across independent runs.
In this project, GCN outperformed GIN and the fingerprint baseline on the ESOL scaffold
benchmark:

- GCN RMSE: 1.3395 +/- 0.0738
- GIN RMSE: 1.4499 +/- 0.1372
- Fingerprint random forest RMSE: 1.8480 +/- 0.0214

The fixed-split deep ensemble was a different experiment. It held one partition fixed and
varied only model seeds to test whether model disagreement could rank prediction errors.
That uncertainty signal failed, so it is not exposed in the service.

## Validation Selects Models

Model promotion uses validation RMSE. The test split is reserved for final reporting after a
candidate has already been selected. This protects the test set from becoming an informal
tuning target.

The promoted model is the fixed-split GCN with model seed 43. It was selected with validation
RMSE 1.3420. Its post-selection test metrics were RMSE 1.3502, MAE 1.0385, and R2 0.6441.

## Why The First Uncertainty Attempt Was Rejected

An early uncertainty attempt combined repeated runs that were not guaranteed to share the
same validation and test samples. It also risked aligning predictions by SMILES, which is not
a unique observation identifier in ESOL. That was rejected rather than papered over.

The repaired workflow assigns stable sample IDs and keeps the split fixed across ensemble
members. The resulting ensemble still did not produce useful error ranking: both Pearson and
rank uncertainty-error correlations were approximately -0.016. Selective prediction did not
improve error ranking, and interval coverage required wide intervals.

## Why Keep Negative Results

Negative or disappointing results are part of the project record. They prevent overstated
claims, explain why uncertainty is not exposed by the API, and make the final system more
credible. The project reports what worked, what did not, and what should not be inferred.

