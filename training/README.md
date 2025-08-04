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
  "adapter_path": "gs://bucket/adapters/job123/ or gs://bucket/merged_models/job123/",
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

## Configuration

- **Environment**: `PROJECT_ID`, `REGION`, `GCS_CONFIG_BUCKET_NAME`
- **Storage**: Training configs stored in GCS
- **Monitoring**: Weights & Biases integration
- **Port**: 8080 (default)
