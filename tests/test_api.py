from pathlib import Path

import httpx
import pytest

from molgnn_ops import api as api_module
from molgnn_ops.model_registry import promote_model


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_api_endpoints_and_model_load_once(
    promoted_manifest_path: Path,
    monkeypatch,
) -> None:
    real_loader = api_module.load_promoted_model
    load_calls = []

    def tracked_loader(path):
        load_calls.append(path)
        return real_loader(path)

    monkeypatch.setattr(api_module, "load_promoted_model", tracked_loader)
    app = api_module.create_app(promoted_manifest_path)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            model_info = await client.get("/model")
            prediction = await client.post("/predict", json={"smiles": "OCC"})
            batch = await client.post(
                "/predict/batch",
                json={"smiles": ["CCO", "invalid", "CCN"]},
            )
            invalid = await client.post("/predict", json={"smiles": "not-a-smiles"})
            oversized = await client.post(
                "/predict/batch",
                json={"smiles": ["CCO"] * 101},
            )
            context = await client.post(
                "/predict/context",
                json={"smiles": "CCO", "top_k": 2},
            )
            oversized_context = await client.post(
                "/predict/context",
                json={"smiles": "CCO", "top_k": 21},
            )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_id": "synthetic-gcn-v1",
    }
    assert model_info.status_code == 200
    assert model_info.json()["model_seed"] == 12
    assert model_info.json()["benchmark_summary"]["uncertainty_exposed"] is False
    assert model_info.json()["reference_index_available"] is True
    assert model_info.json()["reference_set_size"] == 3
    assert "checkpoint_path" not in model_info.json()
    assert prediction.status_code == 200
    assert prediction.json()["canonical_smiles"] == "CCO"
    assert batch.status_code == 200
    assert [item["success"] for item in batch.json()] == [True, False, True]
    assert invalid.status_code == 400
    assert oversized.status_code == 422
    assert context.status_code == 200
    assert context.json()["applicability"]["maximum_similarity"] == 1.0
    assert len(context.json()["nearest_training_molecules"]) == 2
    assert oversized_context.status_code == 422
    assert len(load_calls) == 1


@pytest.mark.anyio
async def test_api_without_model_returns_service_unavailable() -> None:
    app = api_module.create_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            model_info = await client.get("/model")
            prediction = await client.post("/predict", json={"smiles": "CCO"})

    assert health.json()["model_loaded"] is False
    assert model_info.status_code == 503
    assert prediction.status_code == 503


@pytest.mark.anyio
async def test_context_endpoint_requires_reference_index(
    tmp_path: Path,
    synthetic_candidate_dirs: list[Path],
) -> None:
    registry_dir = tmp_path / "minimal-registry"
    promote_model(
        synthetic_candidate_dirs,
        registry_dir,
        model_id="minimal-model",
        include_reference_index=False,
    )
    app = api_module.create_app(registry_dir / "manifest.json")
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/predict/context",
                json={"smiles": "CCO", "top_k": 5},
            )

    assert response.status_code == 409
    assert "no training reference index" in response.json()["detail"]
