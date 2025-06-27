from pydantic import BaseModel
from typing import Literal, Optional


class TrainingConfig(BaseModel):
    method: Literal["Full", "LoRA", "QLoRA", "RL"]
    base_model_id: str
    lora_rank: Optional[int]
    lora_alpha: Optional[int]
    lora_dropout: Optional[float]
    learning_rate: float
    # Seems like batch size = 8 will cause OOM on the L4 on Cloud Run
    # I haven't enforced a check yet but for prod we should probably limit this
    batch_size: int = 4
    epochs: int
    # Default this to -1 instead of None to avoid operator errors
    max_steps: Optional[int] = -1
    max_seq_length: Optional[int]  # used to load pretrained models
    packing: bool = True  # whether to pack sequences for training
    gradient_accumulation_steps: int
    use_fa2: bool = False  # FA2 is only available when provider is "huggingface"
    provider: Literal["unsloth", "huggingface"] = "huggingface"


class WandbConfig(BaseModel):
    api_key: str
    # project is defaulted to "huggingface" if not provided
    project: Optional[str] = None
    log_model: Optional[Literal["false", "checkpoint", "end"]] = "end"


class TrainRequest(BaseModel):
    # This struct is shared between the API and the backend service
    processed_dataset_id: str
    # NOTE: This is marked optional for dev but in deployment it should be required
    hf_token: Optional[str] = None
    training_config: TrainingConfig
    export: Literal["gcs", "hfhub"] = "gcs"
    # If export is hfhub, this is the Hugging Face repo ID to push the model to
    hf_repo_id: Optional[str] = None

    # Weights & Biases logging configuration
    wandb_config: Optional[WandbConfig] = None


class TrainResponse(BaseModel):
    job_id: str
    adapter_path: str
    base_model_id: str


class SubmitTrainResponse(BaseModel):
    job_id: str
    wandb_url: Optional[str] = None


class JobStatusResponse(BaseModel):
    status: Literal["queued", "preparing", "training", "completed", "failed"]
    wandb_url: Optional[str] = None
    adapter_path: Optional[str] = None
    base_model_id: Optional[str] = None
    error: Optional[str] = None


class InferenceRequest(BaseModel):
    # HF Token must be provided for Gemma models
    hf_token: Optional[str] = None
    # The user should specify this so we don't have to look it up
    storage_type: Literal["gcs", "hfhub"]
    prompt: str


class InferenceResponse(BaseModel):
    result: str
