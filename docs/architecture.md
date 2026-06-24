# Architecture

## Component Overview

```mermaid
flowchart LR
    raw[ESOL CSV] --> prep[Dataset preparation]
    prep --> audit[Duplicate and split audits]
    prep --> graphs[RDKit graph features]
    prep --> fps[Morgan fingerprints]
    fps --> classical[Classical baselines]
    graphs --> gnn[GCN and GIN training]
    classical --> compare[Benchmark comparison]
    gnn --> compare
    gnn --> ensemble[Fixed-split ensemble analysis]
    compare --> promote[Validation-based promotion]
    promote --> registry[Self-contained registry package]
    registry --> api[FastAPI inference]
    registry --> dashboard[Streamlit explorer]
```

## Data Flow

```mermaid
flowchart TD
    A[download] --> B[prepare]
    B --> C[audit duplicate molecules and split metadata]
    C --> D1[graph features]
    C --> D2[fingerprint features]
    D1 --> E1[train GCN and GIN]
    D2 --> E2[train fingerprint baselines]
    E1 --> F[evaluate repeated seeds]
    E2 --> F
    F --> G[promote using validation metrics]
    G --> H[serve API and dashboard]
```

Generated benchmark artifacts live under `artifacts/` and are ignored by Git. Small portfolio
summaries live under `reports/portfolio/` and are tracked so the repository remains readable
without requiring large local model artifacts.

## Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant RDKit
    participant GCN
    User->>API: SMILES
    API->>RDKit: validate and canonicalize
    RDKit-->>API: canonical SMILES and molecular graph
    API->>GCN: atom features and edge index
    GCN-->>API: predicted logS
    API-->>User: logS and 10 ** logS mol/L
```

The API rejects blank or invalid SMILES. It does not expose uncertainty values because the
ensemble disagreement experiment did not support a reliable confidence story.

## Applicability Flow

```mermaid
flowchart LR
    smiles[SMILES] --> fp[Morgan fingerprint]
    fp --> tanimoto[Tanimoto neighbor search]
    smiles --> descriptors[Descriptor calculation]
    descriptors --> ranges[Training descriptor ranges]
    tanimoto --> warnings[Applicability warnings]
    ranges --> warnings
```

The nearest-neighbor context uses the training reference index packaged with the promoted
model. Similarity and descriptor-range warnings are descriptive applicability context, not
confidence.

## Promoted Registry Structure

```text
artifacts/registry/esol-gcn-v1/
  manifest.json
  candidate_ranking.csv
  selection_report.md
  featurization_config.json
  models/esol-gcn-v1/
    checkpoint.pt
    reference_index.npz
```

The registry package is self-contained for inference. It records the checkpoint location
relative to the manifest, atom and edge feature dimensions, architecture settings,
normalization metadata, validation metrics, and post-selection test metrics.

## API, Dashboard, And Containers

```mermaid
flowchart LR
    registry[(Read-only registry mount)] --> api[FastAPI container]
    registry --> dashboard[Streamlit container]
    api --> user1[HTTP clients]
    dashboard --> user2[Browser users]
```

The API and dashboard use the same Docker image with different commands. The image contains
runtime dependencies but not generated models or datasets. Compose mounts the registry
read-only into each service.

## Tracked Versus Generated

Tracked in Git:

- source code, tests, scripts, CI workflows
- docs and release metadata
- small portfolio summaries in `reports/portfolio/`

Generated and ignored:

- downloaded datasets
- trained checkpoints
- benchmark artifacts
- promoted registry contents
- demo outputs

## Seed Semantics

`split_seed` controls the train/validation/test partition. `model_seed` controls parameter
initialization, batch order, dropout, and other training randomness. Repeated-split
benchmarks vary seeds for benchmark comparison, while the fixed-split ensemble keeps the
partition immutable and varies only model seeds.

## Why Stable IDs Matter For Uncertainty

The uncertainty pipeline requires identical sample IDs, canonical molecules, targets, and
split labels across every ensemble member. ESOL contains duplicate canonical molecules,
including conflicting measurements. Aligning predictions by SMILES alone can combine the
wrong rows. Stable sample IDs prevent that failure.

