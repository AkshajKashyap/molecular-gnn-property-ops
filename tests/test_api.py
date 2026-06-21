from pathlib import Path

import httpx
import pytest

from molgnn_ops import api as api_module


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

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_id": "synthetic-gcn-v1",
    }
    assert model_info.status_code == 200
    assert model_info.json()["model_seed"] == 12
    assert model_info.json()["benchmark_summary"]["uncertainty_exposed"] is False
    assert "checkpoint_path" not in model_info.json()
    assert prediction.status_code == 200
    assert prediction.json()["canonical_smiles"] == "CCO"
    assert batch.status_code == 200
    assert [item["success"] for item in batch.json()] == [True, False, True]
    assert invalid.status_code == 400
    assert oversized.status_code == 422
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
