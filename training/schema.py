from pydantic import BaseModel
from typing import Literal, Optional, List


class EvaluationMetrics(BaseModel):
    """
    Evaluation metrics structure to hold results after training.
    This is used to store metrics like accuracy, loss, etc.
    """

    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    eval_loss: Optional[float] = None


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
    max_seq_length: Optional[int] = None  # used to load pretrained models
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
    job_name: str
    # This struct is shared between the API and the backend service
    processed_dataset_id: str  # this is dataset_name for now
    # Dataset modality: "text" for text-only, "vision" for text+images
    modality: Literal["text", "vision"] = "text"
    # NOTE: This is marked optional for dev but in deployment it should be required
    hf_token: Optional[str] = None
    training_config: TrainingConfig
    export: Literal["gcs", "hfhub"] = "gcs"
    # If export is hfhub, this is the Hugging Face repo ID to push the model to
    hf_repo_id: Optional[str] = None

    # Weights & Biases logging configuration
    wandb_config: Optional[WandbConfig] = None


class JobSubmitResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_name: str
    status: Literal["queued", "preparing", "training", "completed", "failed"]
    modality: Optional[Literal["text", "vision"]] = "text"
    wandb_url: Optional[str] = None
    processed_dataset_id: Optional[str] = None
    adapter_path: Optional[str] = None
    base_model_id: Optional[str] = None
    # Evaluation metrics recorded after training
    metrics: Optional[EvaluationMetrics] = None
    error: Optional[str] = None


class JobListEntry(BaseModel):
    job_id: str
    job_name: str = "unnamed job"
    modality: Optional[Literal["text", "vision"]] = "text"
    # "unknown" is a fallback for jobs that don't have a status but are listed
    status: Literal[
        "queued", "preparing", "training", "completed", "failed", "unknown"
    ] = "unknown"


class JobListResponse(BaseModel):
    jobs: List[JobListEntry]
