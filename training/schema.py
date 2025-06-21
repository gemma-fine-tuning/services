from pydantic import BaseModel
from typing import Literal, Optional


class ModelConfig(BaseModel):
    method: Literal["LoRA", "QLoRA", "RL"]
    model_id: str
    lora_rank: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]
    learning_rate: float
    batch_size: int
    epochs: int
    max_seq_length: int
    gradient_accumulation_steps: int
    provider: Literal["unsloth", "huggingface"] = "huggingface"


class TrainRequest(BaseModel):
    # This struct is shared between the API and the backend service
    processed_dataset_id: str
    model_config: ModelConfig


class TrainResponse(BaseModel):
    job_id: str
    adapter_path: str
    model_id: str


class InferenceRequest(BaseModel):
    prompt: str


class InferenceResponse(BaseModel):
    result: str
