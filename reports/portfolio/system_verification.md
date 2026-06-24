# System Verification Summary

- Package version: 1.0.0
- Tests: 142 passing tests
- Lint: Ruff clean
- Docker image: approximately 526 MiB CPU image
- Compose: API and dashboard services verified healthy in Docker Compose
- CPU-only behavior: CPU PyTorch and PyG; no CUDA required
- Registry mount design: promoted registry mounted read-only into API and dashboard containers

## Service Checks

- GET /health verified for loaded and model-free API states
- GET /model verified for promoted model metadata
- POST /predict verified with CCO
- POST /predict/batch verified with mixed valid and invalid inputs
- Streamlit dashboard health endpoint verified
- RDKit molecule rendering verified inside the dashboard container
- GitHub Actions quality workflow runs Ruff and pytest
- GitHub Actions Docker workflow builds image and checks model-free API behavior
