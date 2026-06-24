# Release Checklist

Use this checklist before publishing `v1.0.0`.

- Run `pytest -q`
- Run `ruff check .`
- Run `molgnn-ops --version`
- Run `molgnn-ops project-info`
- Promote the ESOL GCN model with `bash scripts/promote_esol_gcn.sh`
- Build Docker with `bash scripts/docker_build.sh`
- Validate Compose with `docker compose config`
- Run Docker smoke checks with `bash scripts/docker_smoke_test.sh`
- Generate demo artifacts with `bash scripts/generate_demo.sh`
- Review `docs/model_card.md`
- Review `docs/architecture.md`
- Review `docs/experimental_methodology.md`
- Review `reports/portfolio/`
- Confirm README links resolve
- Confirm generated artifacts remain ignored
- Confirm `git status --short` is clean
- Push the release commit
- Create an annotated tag with `git tag -a v1.0.0 -m "molecular-gnn-property-ops v1.0.0"`
- Push the tag with `git push origin v1.0.0`
- Optionally create a GitHub release from the tag

