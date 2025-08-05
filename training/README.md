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
  "training_config": {
    "method": "LoRA" | "QLoRA" | "Full" | "RL",
    "base_model_id": "google/gemma-2b",
    "learning_rate": 0.0001,
    "batch_size": 4,
    "epochs": 3,
    "gradient_accumulation_steps": 4,
    "provider": "unsloth" | "huggingface",
    "modality": "text" | "vision",
    "eval_strategy": "no" | "steps" | "epoch",
    "eval_steps": 50,
    "evaluation_metrics": true,
    "batch_eval_metrics": false
  },
  "export_config": {
    "format": "adapter" | "merged",
    "quantization": "none" | "f16" | "bf16" | "q8_0" | "q4_k_m",
    "destination": "gcs" | "hfhub",
    "hf_repo_id": "user/model-name",
    "include_gguf": false,
    "gguf_quantization": "none" | "f16" | "bf16" | "q8_0" | "q4_k_m"
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
      "base_model_id": "google/gemma-1b-it",
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
  "adapter_path": "gs://bucket/trained_adapters/job123/ or gs://bucket/merged_models/job123/",
  "base_model_id": "google/gemma-2b",
  "gguf_path": "gs://bucket/gguf_models/job123/model-q8_0.gguf",
  "metrics": {
    "accuracy": 0.95,
    "perplexity": 1.23,
    "eval_loss": 0.156
  },
  "error": "Error message if failed"
}
```

### GET `/get-download-url`

Generate a signed URL for downloading files from GCS.

**Parameters:**

- `file_path` (required): Path to the file in GCS bucket (e.g., `gguf_models/job123/model.gguf`)

**Response:**

```json
{
  "download_url": "https://storage.googleapis.com/...",
  "filename": "model.gguf",
  "expires_at": "2025-08-05T15:30:00Z"
}
```

**Usage Example:**

```javascript
// Extract blob path from gguf_path in job status
const blobPath = job.gguf_path.replace(/^gs:\/\/[^\/]+\//, "");

// Get signed download URL
const response = await fetch(
  `/get-download-url?file_path=${encodeURIComponent(blobPath)}`
);
const { download_url } = await response.json();

// Download the file
window.open(download_url, "_blank");
```

### GET `/health`

Health check endpoint.

## Export Configuration

The export configuration now supports:

- **Primary Format**: Choose between `adapter` or `merged` model formats
- **Optional GGUF**: Set `include_gguf: true` to also export a GGUF file alongside the primary model
- **Separate Quantization**: Use `gguf_quantization` to specify different quantization for GGUF files

### Example: Adapter with GGUF Export

```json
{
  "export_config": {
    "format": "adapter",
    "quantization": "q4_k_m",
    "destination": "gcs",
    "include_gguf": true,
    "gguf_quantization": "q8_0"
  }
}
```

This will export both:

- Primary adapter model with q4_k_m quantization
- GGUF file with q8_0 quantization

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
