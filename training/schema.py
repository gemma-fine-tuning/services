from pydantic import BaseModel
from typing import Literal, Optional, List


class ExportConfig(BaseModel):
    """Configuration for model export"""

    format: Literal["adapter", "merged", "gguf"] = "adapter"
    quantization: Optional[
        Literal[
            # For merged models
            "none",
            "fp16",
            # "q8",
            "q4",
            # TODO: For now we don't check this we assume the frontend is aware of valid config
            # for GGUF format (Unsloth)
            "f16",
            "not_quantized",
            "fast_quantized",
            "quantized",
            "q8_0",
            "q4_k_m",  # recommended for Unsloth
            "q5_k_m",  # recommended for Unsloth
            "q2_k",
        ]
    ] = "none"
    destination: Literal["gcs", "hfhub"] = "gcs"
    hf_repo_id: Optional[str] = None


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
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05

    learning_rate: float = 2e-4
    # Seems like batch size = 8 will cause OOM on the L4 on Cloud Run
    # I haven't enforced a check yet but for prod we should probably limit this
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    # Default this to -1 instead of None to avoid operator errors
    max_steps: Optional[int] = -1

    # NOTE: Packing only works with FA2
    packing: bool = False  # whether to pack sequences for training
    use_fa2: bool = False  # FA2 is only available when provider is "huggingface"

    provider: Literal["unsloth", "huggingface"] = "huggingface"

    max_seq_length: Optional[int] = None  # used to load pretrained models
    lr_scheduler_type: Optional[str] = "linear"
    save_strategy: Optional[str] = "epoch"
    logging_steps: Optional[int] = 10

    # NOTE: Only specify eval_strategy if you actually provide eval_dataset
    eval_strategy: Optional[str] = "no"  # "no", "steps", "epoch"
    eval_steps: Optional[int] = 50  # Required if eval_strategy="steps"

    # Metrics configuration
    evaluation_metrics: Optional[List[str]] = ["accuracy", "perplexity"]
    # Set to True to enable batch evaluation mode for metrics computation
    batch_eval_metrics: Optional[bool] = False


class WandbConfig(BaseModel):
    api_key: str
    # project is defaulted to "huggingface" if not provided
    project: Optional[str] = None
    log_model: Optional[Literal["false", "checkpoint", "end"]] = "end"


class TrainRequest(BaseModel):
    # This struct is shared between the API and the backend service
    processed_dataset_id: str
    hf_token: str
    training_config: TrainingConfig
    export_config: ExportConfig

    # TODO: Duplicate field with TrainingConfig, consider refactoring
    modality: Literal["text", "vision"] = "text"

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
