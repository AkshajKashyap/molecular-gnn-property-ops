from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status

from molgnn_ops.api_schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from molgnn_ops.inference import (
    LoadedPromotedModel,
    load_promoted_model,
    predict_smiles,
    predict_smiles_batch,
)


def _require_model(request: Request) -> LoadedPromotedModel:
    loaded_model = getattr(request.app.state, "loaded_model", None)
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No promoted model is configured",
        )
    return loaded_model


def _model_info(loaded_model: LoadedPromotedModel) -> ModelInfoResponse:
    manifest = loaded_model.manifest
    limitations = [
        "This model is a research benchmark trained on the small ESOL dataset.",
        "Predictions outside similar chemical space may be unreliable.",
        "It is not intended for medical, pharmaceutical, laboratory, or safety-critical use.",
        "Ensemble uncertainty was not reliable and is not exposed by this API.",
    ]
    return ModelInfoResponse(
        model_id=manifest.model_id,
        model_type=manifest.model_type,
        dataset_name=manifest.dataset_name,
        task_type=manifest.task_type,
        target_name=manifest.target_name,
        created_at=manifest.created_at.isoformat(),
        split_strategy=manifest.split_strategy,
        split_seed=manifest.split_seed,
        model_seed=manifest.model_seed,
        validation_metrics=manifest.validation_metrics,
        test_metrics=manifest.test_metrics,
        known_limitations=limitations,
        benchmark_summary={
            "selection_basis": "validation metrics only",
            "validation_metrics": manifest.validation_metrics,
            "post_selection_test_metrics": manifest.test_metrics,
            "uncertainty_exposed": False,
        },
    )


def create_app(manifest_path: Path | None = None) -> FastAPI:
    """Create a FastAPI application that loads one promoted model at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.loaded_model = (
            load_promoted_model(manifest_path) if manifest_path is not None else None
        )
        yield

    app = FastAPI(
        title="Molecular Solubility Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(request: Request) -> dict:
        loaded_model = getattr(request.app.state, "loaded_model", None)
        return {
            "status": "ok",
            "model_loaded": loaded_model is not None,
            "model_id": (
                loaded_model.manifest.model_id if loaded_model is not None else None
            ),
        }

    @app.get("/model", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        return _model_info(_require_model(request))

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        try:
            prediction = predict_smiles(payload.smiles, _require_model(request))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return PredictionResponse.model_validate(prediction)

    @app.post("/predict/batch", response_model=list[BatchPredictionItem])
    async def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
    ) -> list[BatchPredictionItem]:
        results = predict_smiles_batch(payload.smiles, _require_model(request))
        return [BatchPredictionItem.model_validate(result) for result in results]

    return app
