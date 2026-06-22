.PHONY: install test lint check promote api dashboard docker-build docker-up docker-down docker-smoke

install:
	python -m pip install "torch>=2.12" --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[gnn,api,dashboard]"

test:
	pytest -q

lint:
	ruff check .

check: lint test

promote:
	bash scripts/promote_esol_gcn.sh

api:
	bash scripts/serve_esol_api.sh

dashboard:
	bash scripts/run_esol_dashboard.sh

docker-build:
	bash scripts/docker_build.sh

docker-up:
	docker compose up --detach

docker-down:
	docker compose down

docker-smoke:
	bash scripts/docker_smoke.sh
