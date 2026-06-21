from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=10000)

    @field_validator("smiles")
    @classmethod
    def reject_blank_smiles(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SMILES must not be blank")
        return value


class BatchPredictionRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    input_smiles: str
    canonical_smiles: str
    predicted_log_solubility: float
    predicted_solubility_mol_per_litre: float
    model_id: str
    model_type: str
    dataset_name: str
    warnings: list[str]


class BatchPredictionItem(BaseModel):
    input_smiles: str
    success: bool
    prediction: PredictionResponse | None = None
    error: str | None = None


class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    dataset_name: str
    task_type: str
    target_name: str
    created_at: str
    split_strategy: str
    split_seed: int
    model_seed: int
    validation_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]
    known_limitations: list[str]
    benchmark_summary: dict[str, Any]
