# Training Service

FastAPI service for managing fine-tuning jobs on Gemma models.

## Structure

- **`app.py`** - FastAPI application with job management endpoints
- **`job_manager.py`** - Job state management with Firestore
- **`schema.py`** - Request/response models

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

## Endpoints

### POST `/train`

Start a new training job.

**Request:**

```json
{
  "processed_dataset_id": "dataset_abc123",
  "job_name": "My Training Job",
  "hf_token": "hf_...",
  "modality": "text" | "vision",
  "training_config": {
    "method": "LoRA" | "QLoRA" | "Full" | "RL",
    "base_model_id": "google/gemma-2b",
    "learning_rate": 0.0001,
    "batch_size": 4,
    "epochs": 3,
    "gradient_accumulation_steps": 4,
    "provider": "unsloth" | "huggingface",
    "eval_strategy": "no" | "steps" | "epoch",
    "eval_steps": 50,
    "evaluation_metrics": ["accuracy", "perplexity"],
    "batch_eval_metrics": false
  },
  "export_config": {
    "format": "adapter" | "merged" | "gguf",
    "quantization": "none" | "fp16" | "q4" | "f16" | "q8_0" | "q4_k_m" | "q5_k_m" | "q2_k",
    "destination": "gcs" | "hfhub",
    "hf_repo_id": "user/model-name"
  },
  "wandb_config": {
    "api_key": "wandb_...",
    "project": "my-project"
  }
}
```

**Response:**

```json
{
  "job_id": "training_abc123_gemma-2b_def456"
}
```

### GET `/jobs`

List all jobs.

**Response:**

```json
{
  "jobs": [
    {
      "job_id": "training_abc123_gemma-2b_def456",
      "job_name": "My Training Job",
      "status": "queued" | "preparing" | "training" | "completed" | "failed" | "unknown",
      "modality": "text" | "vision",
    }
  ]
}
```

### GET `/jobs/{job_id}/status`

Get training job status.

**Response:**

```json
{
  "job_name": "My Training Job",
  "status": "queued" | "preparing" | "training" | "completed" | "failed",
  "modality": "text" | "vision",
  "wandb_url": "https://wandb.ai/...",
  "adapter_path": "gs://bucket/path",
  "base_model_id": "google/gemma-2b",
  "error": "Error message if failed"
}
```

### GET `/health`

Health check endpoint.

## Job Lifecycle

1. **Submit** → Job queued in Firestore
2. **Start** → Cloud Run job triggered
3. **Track** → Status updates via Firestore
4. **Complete** → Model exported to GCS/HF Hub

## Export Formats

### Adapter Export

- **Format**: `adapter`
- **Quantization**: Not applicable
- **Use case**: Lightweight LoRA/QLoRA adapter weights only
- **Storage**: ~MB in size, fast upload/download

### Merged Export

- **Format**: `merged`
- **Quantization**: `none`, `fp16`, `q4` (4-bit)
- **Use case**: Full standalone model for deployment
- **Providers**:
  - Unsloth: `merged_16bit` (vLLM compatible), `merged_4bit`
  - HuggingFace: Standard merge with PEFT, supports FP16 quantization

### GGUF Export

- **Format**: `gguf`
- **Quantization**: `f16`, `q8_0`, `q4_k_m`, `q5_k_m`, `q2_k`
- **Use case**: CPU-optimized inference (llama.cpp, Ollama)
- **Providers**:
  - Unsloth: Native GGUF export with quantization
  - HuggingFace: Not implemented (requires llama.cpp conversion)

## Evaluation

Configure evaluation during training:

- **eval_strategy**: When to run evaluation (`no`, `steps`, `epoch`)
- **eval_steps**: Evaluation frequency when using `steps` strategy
- **evaluation_metrics**: Metrics to compute (`accuracy`, `perplexity`)
- **batch_eval_metrics**: Enable batch evaluation mode for better performance

Supported metrics:

- **Accuracy**: Token-level classification accuracy
- **Perplexity**: Model confidence metric (2^loss)
- **Loss**: Training/validation loss

## Configuration

- **Environment**: `PROJECT_ID`, `REGION`, `GCS_CONFIG_BUCKET_NAME`
- **Storage**: Training configs stored in GCS
- **Monitoring**: Weights & Biases integration
- **Port**: 8080 (default)
