# Operations Guide

This guide covers local and containerized operation of the promoted ESOL model. The
service is a research and portfolio application, not a medical, pharmaceutical,
laboratory-grade, or safety-critical system.

## Prerequisites

- Python 3.13 for host development
- Docker Engine with Docker Compose v2 for containers
- A promoted model package at
  `artifacts/registry/esol-gcn-v1/manifest.json` for prediction and dashboard use

Create the promoted package from existing fixed-split training runs:

```bash
bash scripts/promote_esol_gcn.sh
```

Promotion ranks candidates by validation RMSE only. It does not retrain a model and does
not use test metrics for selection.

## Host Quickstart

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[gnn,api,dashboard]"
bash scripts/promote_esol_gcn.sh
```

Start the API and dashboard in separate terminals:

```bash
bash scripts/serve_esol_api.sh
bash scripts/run_esol_dashboard.sh
```

The defaults are `http://localhost:8000` for FastAPI and
`http://localhost:8501` for Streamlit.

## Docker Quickstart

The image contains the CLI, API, dashboard, RDKit, CPU PyTorch, and PyG. It contains no
datasets, checkpoints, or promoted registry artifacts. The same non-root image runs both
services, with the host registry mounted read-only.

```bash
bash scripts/docker_build.sh
bash scripts/promote_esol_gcn.sh
docker compose up --detach
docker compose ps
```

Stop and remove the service containers and network:

```bash
docker compose down
```

To run a single foreground service instead:

```bash
bash scripts/docker_run_api.sh
bash scripts/docker_run_dashboard.sh
```

Run import checks and, when the promoted manifest exists, live API checks:

```bash
bash scripts/docker_smoke.sh
```

## Configuration

Service commands resolve configuration in this order: explicit CLI argument, environment
variable, then built-in default. Relevant variables are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MOLGNN_MANIFEST_PATH` | Promoted manifest loaded by API or dashboard | unset |
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `DASHBOARD_HOST` | Streamlit bind address | `0.0.0.0` |
| `DASHBOARD_PORT` | Streamlit port | `8501` |
| `IMAGE_NAME` | Image used by scripts and Compose | `molecular-gnn-property-ops:latest` |

In Compose, `API_PORT` and `DASHBOARD_PORT` customize host ports; container ports remain
8000 and 8501. For example:

```bash
API_PORT=8010 DASHBOARD_PORT=8502 docker compose up --detach
```

The Compose manifest path is
`/app/artifacts/registry/esol-gcn-v1/manifest.json`, backed by the read-only host mount
`./artifacts/registry:/app/artifacts/registry:ro`.

## Health and Logs

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO"}'
docker compose logs --follow api
docker compose logs --follow dashboard
```

FastAPI can intentionally start without a manifest. In that mode `/health` returns 200
with `model_loaded: false`, while model-dependent endpoints return 503. The dashboard
requires a configured model and fails at startup when its manifest is absent.

## Troubleshooting

**The manifest is missing.** Run `bash scripts/promote_esol_gcn.sh` and confirm the
registry exists on the host. Registry artifacts are deliberately gitignored and excluded
from the image.

**A configured manifest path is wrong.** The API fails startup clearly rather than falling
back to model-free mode. Check the host path and container mount.

**A port is already in use.** Override the host mapping, for example
`API_PORT=8010 DASHBOARD_PORT=8502 docker compose up --detach`.

**The image is large.** RDKit, CPU PyTorch, PyG, scikit-learn, FastAPI, and Streamlit share
one reusable image. This favors reproducibility over minimum size; no CUDA runtime is
installed.

**The container cannot use a GPU.** CPU is the supported default. This milestone does not
configure CUDA or GPU passthrough.

## CI Scope

The quality workflow installs Python 3.13 CPU dependencies, runs Ruff, and runs the unit
suite. The Docker workflow builds the image, checks runtime imports, validates Compose,
and starts a model-free API for health and 503 behavior. CI does not download benchmark
data, train models, require a GPU, publish images, or fabricate a registry artifact.
